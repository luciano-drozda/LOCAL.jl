"""Times one benchmark case in JAX or PyTorch: primal, and primal plus vector-Jacobian product (the same work as the STADE adjoint).
   python time_fw.py --fw jax|torch --variant NAME --case CASEDIR --roles SMALLCASEDIR [--warmup 5 --reps 30]
CASEDIR holds meta.json and the input binaries (tools/dump_inputs.jl, or a case from tools/dump_case.jl).
SMALLCASEDIR is the small case of the same kernel: its meta.json gives the outputs (deps) and the differentiated arrays (indeps),
the same roles that STADE used. Prints one JSON line. Run one case per process, so the peak memory is clean.
Variants: jax:jit, torch:eager, torch:compile."""
import argparse, json, os, subprocess, sys, time
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


def _sub(a, b):
    return None if a is None or b is None else a - b


def run_jax(meta, args, roles, warmup, reps, variant):
    import jax, jax.numpy as jnp
    from jax_ref import FUNCS
    f = FUNCS[meta["kernel"]]
    keys = [k for k in roles["indeps"] if k in args and isinstance(args[k], np.ndarray) and args[k].dtype == np.float64]
    jax.block_until_ready(jnp.zeros(1) + 1)       # context and runtime start-up
    mem0 = smi_mb()
    dev = {k: (jax.device_put(v) if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    jax.block_until_ready([v for v in dev.values() if not isinstance(v, (int, float))])
    mem_inputs = _sub(smi_mb(), mem0)
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

    primal = jax.jit(outs)
    adjoint = jax.jit(lambda xs, cot: jax.vjp(outs, xs)[1](cot)[0])
    rng = np.random.default_rng(7)
    oshapes = [shp[k].shape for k in outk]
    cot = tuple(jax.device_put(rng.standard_normal(int(np.prod(sh)))).reshape(sh) for sh in oshapes)
    res = {"mem_context_mb": mem0, "mem_inputs_mb": mem_inputs}
    res.update({"outputs": outk, "differentiated": keys})
    for name, fn, a in (("primal", primal, (xs,)), ("adjoint", adjoint, (xs, cot))):
        t0 = time.perf_counter(); r = fn(*a); jax.block_until_ready(r); first = time.perf_counter() - t0
        for _ in range(warmup):
            jax.block_until_ready(fn(*a))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter() - t0)
        res[name] = stats(ts); res[name]["first_call_s"] = first
        if name == "primal":
            res["peak_primal_mb"] = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6
            res["mem_peak_primal_mb"] = _sub(smi_mb(), mem0)
        if name == "adjoint":
            g = [np.asarray(x) for x in r]
            res["grad_norm"] = {k: float(np.linalg.norm(x)) for k, x in zip(keys, g)}
            res["grad_finite"] = bool(all(np.isfinite(x).all() for x in g))
    res["mem_peak_adjoint_mb"] = _sub(smi_mb(), mem0)
    st = jax.devices()[0].memory_stats() or {}
    res["peak_mb"] = st.get("peak_bytes_in_use", 0) / 1e6
    res["pool_mb"] = st.get("pool_bytes", 0) / 1e6
    res["device"] = str(jax.devices()[0])
    return res


def run_torch(meta, args, roles, warmup, reps, variant):
    import torch
    from torch_ref import FUNCS
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    f = FUNCS[meta["kernel"]]
    keys = [k for k in roles["indeps"] if k in args and isinstance(args[k], np.ndarray) and args[k].dtype == np.float64]
    a = {k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    for k in keys:
        a[k].requires_grad_(True)
    fn = f
    if variant == "compile":
        fn = torch.compile(f, mode="max-autotune-no-cudagraphs")
    with torch.no_grad():
        probe = fn(a)
    outk = [k for k in roles["deps"] if k in probe]
    del probe
    rng = np.random.default_rng(7)   # the same cotangents as the JAX run, so the gradients can be compared

    def prim():
        with torch.no_grad():
            r = fn(a)
            return [r[k] for k in outk]

    cots = [torch.from_numpy(rng.standard_normal(o.numel())).to(dev).reshape(o.shape) for o in prim()]

    def adj():
        r = fn(a)
        return torch.autograd.grad([r[k] for k in outk], [a[k] for k in keys], grad_outputs=cots, allow_unused=True)

    torch.cuda.reset_peak_memory_stats()
    res = {"outputs": outk, "differentiated": keys}
    for name, call in (("primal", prim), ("adjoint", adj)):
        t0 = time.perf_counter(); r = call(); torch.cuda.synchronize(); first = time.perf_counter() - t0
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); call(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e-3)
        res[name] = stats(ts); res[name]["first_call_s"] = first
        if name == "adjoint":
            gr = [x for x in r]
            res["grad_norm"] = {k: (float(torch.linalg.norm(x)) if x is not None else 0.0) for k, x in zip(keys, gr)}
            res["grad_finite"] = bool(all(bool(torch.isfinite(x).all()) for x in gr if x is not None))
    res["peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    res["device"] = torch.cuda.get_device_name(0)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fw", required=True); p.add_argument("--variant", required=True)
    p.add_argument("--case", required=True); p.add_argument("--roles", required=True)
    p.add_argument("--warmup", type=int, default=5); p.add_argument("--reps", type=int, default=30)
    o = p.parse_args()
    meta, args = load_case(o.case)
    rm = json.load(open(os.path.join(o.roles, "meta.json")))
    roles = {"deps": rm["deps"], "indeps": rm["indeps"]}
    out = {"framework": o.fw, "variant": o.variant, "kernel": meta["kernel"], "size": meta["size"]}
    try:
        out.update((run_jax if o.fw == "jax" else run_torch)(meta, args, roles, o.warmup, o.reps, o.variant))
        out["ok"] = True
    except Exception as e:
        out["ok"] = False; out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
