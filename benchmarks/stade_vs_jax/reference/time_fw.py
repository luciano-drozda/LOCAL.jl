"""Measures one benchmark case in JAX (jit): runtime or memory of the primal or of the adjoint (primal plus vector-Jacobian product).
   python time_fw.py --case CASEDIR --roles SMALLCASEDIR [--phase time|mem_primal|mem_adjoint] [--warmup 5 --reps 30]
CASEDIR holds meta.json and the input binaries (tools/dump_inputs.jl).
SMALLCASEDIR is the small case of the same kernel: its meta.json gives the outputs (deps) and the differentiated arrays (indeps),
the same roles that STADE uses. Prints one JSON line. Run one case and one phase per process.
Phases:
  time        timed calls of the primal and of the adjoint (host clock around one call and block_until_ready). No memory reading.
  mem_primal  only the inputs are on the device. First call, 3 calls, memory reading. No timing.
  mem_adjoint the inputs and the cotangents are on the device. First call, 3 calls, memory reading. No timing."""
import argparse, gc, json, os, platform, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def load_case(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    args = {}
    for e in meta["args"]:
        n = e["name"]
        if e["kind"] == "scalar":
            args[n] = e["value"]
        else:
            args[n] = np.fromfile(os.path.join(d, n + ".in.bin"), dtype=np.float64 if e["kind"] == "float" else np.int64)
    return meta, args


def stats(ts):
    a = np.sort(np.asarray(ts))
    return {"median": float(np.median(a)), "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)),
            "min": float(a[0]), "max": float(a[-1]), "n": int(len(a))}


def smi_mb():
    # device memory in use, from the driver via nvidia-smi (MiB -> MB); the same quantity STADE reads with cuMemGetInfo
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=20).decode().split()[0]
        return float(o) * 1.048576
    except Exception:
        return None


def smi_driver():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=20).decode().split()[0]
    except Exception:
        return None


def _sub(a, b):
    return None if a is None or b is None else a - b


def versions(jax):
    import jaxlib
    d = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "python": platform.python_version(), "numpy": np.__version__,
         "driver": smi_driver(), "kit_hash": os.environ.get("KIT_HASH", ""), "stade_commit": os.environ.get("STADE_COMMIT", "")}
    try:
        d["platform_version"] = jax.devices()[0].client.platform_version.replace("\n", " ")[:200]
    except Exception:
        d["platform_version"] = None
    return d


def run_jax(meta, args, roles, warmup, reps, phase):
    import jax, jax.numpy as jnp
    from jax_ref import FUNCS
    f = FUNCS[meta["kernel"]]
    keys = [k for k in roles["indeps"] if k in args and isinstance(args[k], np.ndarray) and args[k].dtype == np.float64]
    jax.block_until_ready(jnp.zeros(1) + 1)       # context and runtime start-up
    mem0 = smi_mb()
    dev = {k: (jax.device_put(v) if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    jax.block_until_ready([v for v in dev.values() if not isinstance(v, (int, float))])
    scal = {k: v for k, v in dev.items() if not isinstance(v, jax.Array)}
    arrs = {k: v for k, v in dev.items() if isinstance(v, jax.Array)}
    # output shapes without running the function (an eager probe would grow the allocator's high-water mark)
    shp = jax.eval_shape(lambda A: f({**scal, **A}), arrs)
    outk = [k for k in roles["deps"] if k in shp]
    xs = tuple(dev[k] for k in keys)

    def outs(xs):
        a = dict(dev); a.update(zip(keys, xs))
        r = f(a)
        return tuple(r[k] for k in outk)

    res = {"phase": phase, "outputs": outk, "differentiated": keys, "versions": versions(jax),
           "input_mb": sum(v.nbytes for v in args.values() if isinstance(v, np.ndarray)) / 1e6}
    if phase != "time":
        res["mem_context_mb"] = mem0
        res["mem_inputs_mb"] = _sub(smi_mb(), mem0)
    primal = jax.jit(outs)
    adjoint = jax.jit(lambda xs, cot: jax.vjp(outs, xs)[1](cot)[0])
    cot = None
    if phase != "mem_primal":
        rng = np.random.default_rng(7)
        oshapes = [shp[k].shape for k in outk]
        cot = tuple(jax.device_put(rng.standard_normal(int(np.prod(sh)))).reshape(sh) for sh in oshapes)
        jax.block_until_ready(cot)
    if phase == "mem_adjoint":
        res["mem_setup_mb"] = _sub(smi_mb(), mem0)
    todo = {"time": (("primal", primal, (xs,)), ("adjoint", adjoint, (xs, cot))),
            "mem_primal": (("primal", primal, (xs,)),),
            "mem_adjoint": (("adjoint", adjoint, (xs, cot)),)}[phase]
    for name, fn, a in todo:
        t0 = time.perf_counter(); r = fn(*a); jax.block_until_ready(r); first = time.perf_counter() - t0
        if name == "adjoint":
            g = [np.asarray(x) for x in r]
            res["grad_norm"] = {k: float(np.linalg.norm(x)) for k, x in zip(keys, g)}
            res["grad_finite"] = bool(all(np.isfinite(x).all() for x in g))
            del g
        r = None   # the first result must not stay on the device during the memory calls
        if phase == "time":
            for _ in range(warmup):
                jax.block_until_ready(fn(*a))
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter() - t0)
            res[name] = stats(ts); res[name]["first_call_s"] = first
        else:
            for _ in range(3):
                jax.block_until_ready(fn(*a))
            gc.collect()
            res["mem_peak_%s_mb" % name] = _sub(smi_mb(), mem0)
            res["first_%s_call_s" % name] = first
    if phase != "time":
        st = jax.devices()[0].memory_stats() or {}
        res["peak_mb"] = st.get("peak_bytes_in_use", 0) / 1e6
        res["pool_mb"] = st.get("pool_bytes", 0) / 1e6
    res["device"] = str(jax.devices()[0])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True); p.add_argument("--roles", required=True)
    p.add_argument("--phase", default="time", choices=["time", "mem_primal", "mem_adjoint"])
    p.add_argument("--warmup", type=int, default=5); p.add_argument("--reps", type=int, default=30)
    o = p.parse_args()
    meta, args = load_case(o.case)
    rm = json.load(open(os.path.join(o.roles, "meta.json")))
    roles = {"deps": rm["deps"], "indeps": rm["indeps"]}
    out = {"framework": "jax", "variant": "jit", "kernel": meta["kernel"], "size": meta["size"], "phase": o.phase}
    try:
        out.update(run_jax(meta, args, roles, o.warmup, o.reps, o.phase))
        out["ok"] = True
    except Exception as e:
        out["ok"] = False; out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
