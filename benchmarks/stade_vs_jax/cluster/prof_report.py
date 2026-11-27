"""Builds the report of the profile job: per-kernel device times from nsys, and counters from ncu.
Usage: python3 prof_report.py --results FILE [--time-ref FILE] --out OUTDIR
  --results   the result file of the profile job (phases prof_primal, prof_adjoint, ncu_adjoint)
  --time-ref  a result file with time runs (FULLT1), for the time of one call without a profiler. Optional.
Writes OUTDIR/PROFILE.md and OUTDIR/prof_rows.json.
Each profile run records 3 calls. A value "per call" is the total over the 3 calls divided by 3.
The kernels of the reset between calls (names that start with gpu_fill_kernel_) are not counted in the calls. The report lists them apart.
The host time of a call during profiling includes the cost of the profiler, so it is larger than the time of the benchmark."""
import argparse, csv, io, json, math, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import report as R

NCALL = 3
RESET = ("gpu_fill_kernel_",)


KROW = re.compile(r'^([\d.]+),(\d+),(\d+),([\d.]+),([\d.]+),(\d+),(\d+),([\d.]+),(.*)$')


def kern_rows(txt):
    """Rows of the kernel table of nsys. A regular expression, not a CSV reader: the last column is a long name with commas, and an old
    result file has names that are cut inside their quotes, which would make a CSV reader merge the following rows."""
    out = []
    for l in (txt or "").splitlines():
        m = KROW.match(l.strip())
        if m: out.append(dict(name=m.group(9).strip('"'), total=float(m.group(2)), n=int(m.group(3)), avg=float(m.group(4)), pct=float(m.group(1))))
    return out


def calls_of(prog):
    """The times of the 3 profiled calls from the result of the program (STADE: inside results, JAX: at the top)."""
    prog = prog or {}
    for q in (prog.get("results") or [prog]):
        if isinstance(q, dict) and q.get("prof_calls_s"): return q["prof_calls_s"]
    return []


def short(n, w=70):
    n = n.replace("CuDeviceArray<Float64, (int)1, (int)1>", "A").replace("CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<(int)1, Tuple<OneTo<Int64>>>, NDRange<(int)1, DynamicSize, DynamicSize, CartesianIndices<(int)1, Tuple<OneTo<Int64>>>, CartesianIndices<(int)1, Tuple<OneTo<Int64>>>>>", "CM")
    n = n.replace("|", "/")
    return n if len(n) <= w else n[:w - 3] + "..."


def med(v):
    v = sorted(v); return v[len(v) // 2] if v else None


def nsys_summary(x):
    """x = the result of one nsys run. Returns the numbers per call."""
    K = kern_rows(x.get("kern_csv")); calls = calls_of(x.get("prog"))
    work = [k for k in K if not k["name"].startswith(RESET)]; rst = [k for k in K if k["name"].startswith(RESET)]
    t = sum(k["total"] for k in work) / NCALL / 1e6   # ms per call
    api = {}
    for row in csv.DictReader(io.StringIO(x.get("api_csv") or "")):
        try: api[row["Name"]] = (float(row["Total Time (ns)"]), int(row["Num Calls"]), float(row["Avg (ns)"]))
        except (KeyError, ValueError): pass
    launch = next((v for k, v in api.items() if "LaunchKernel" in k), None)
    return dict(kernel_ms=t, launches=sum(k["n"] for k in work) / NCALL, distinct=len(work), wall_ms=(med(calls) or 0) * 1e3 if calls else None,
                reset_launches=sum(k["n"] for k in rst) / NCALL, work=sorted(work, key=lambda k: -k["total"]), capture=x.get("capture"),
                launch_avg_us=launch[2] / 1e3 if launch else None, launch_calls=launch[1] / NCALL if launch else None)


def ncu_summary(x):
    rows = []
    for row in csv.DictReader(io.StringIO(x.get("ncu_csv") or "")):
        if not (row.get("ID") or "").strip().isdigit(): continue
        rows.append(row)
    agg = {}
    for r in rows:
        nm = r.get("Kernel Name", "?")
        if nm.startswith(RESET): continue
        a = agg.setdefault(nm, dict(n=0, v={}))
        a["n"] += 1
        for k, v in r.items():
            if k == "ID": continue
            try: a["v"][k] = a["v"].get(k, 0.0) + float(v.replace(",", ""))
            except (ValueError, AttributeError): pass
    return agg, len(rows)


def col(a, key):
    for k, v in a["v"].items():
        if k.startswith(key): return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--time-ref"); ap.add_argument("--out", required=True)
    o = ap.parse_args()
    runs, jobs = R.collect([o.results])
    P = []; w = P.append
    w("# Profile of the adjoint and the primal on a V100\n")
    for j in jobs:
        v = j.get("versions") or {}
        w(f"- Job on {j.get('host')} (Slurm {j.get('slurm_job') or '-'}), {R.g((j.get('seconds') or 0) / 60, 1)} min. STADE commit {v.get('stade_commit', '-')[:9]}, kit {v.get('kit_hash', '-')}.")
    vs = {}
    for r in runs:
        for a, b in (((r.get("result") or {}).get("prog") or {}).get("versions") or {}).items():
            if b and a not in ("stade_commit", "kit_hash"): vs[a] = b
    w("- versions: " + ", ".join(f"{a} {b}" for a, b in sorted(vs.items())))
    w("\nEach profile run records 3 calls of one case between `cuProfilerStart` and `cuProfilerStop`. Values per call are the total of the 3 calls divided by 3. "
      "The kernels of the reset between calls (`gpu_fill_kernel_`) are not counted. The host time of a call includes the cost of the profiler.\n")
    bad = [r for r in runs if r.get("status") not in ("", None, "ok") or r.get("exit") not in (0, None)]
    nsys = {}; ncu = {}; ncu_info = {}; rows = []
    for r in runs:
        x = r.get("result") or {}
        key = (r["fw"], r["config"], r["kernel"], r["size"])
        if r["phase"] in ("prof_primal", "prof_adjoint") and x.get("kern_csv"):
            nsys[key + (r["phase"],)] = nsys_summary(x)
        elif r["phase"] == "ncu_adjoint" and x.get("ncu_csv"):
            ncu[key] = ncu_summary(x); ncu_info[key] = (x.get('ncu_head') or '').splitlines()[:1]
    ref = {}
    if o.time_ref:
        tr, _ = R.collect([o.time_ref]); st, jx = R.index(tr)
        for (cfg, k, ph), sizes in st.items():
            if ph != "time": continue
            for s, x in sizes.items():
                ref[("stade", cfg, k, s)] = (R.med_ms(x, "primal_wall"), R.med_ms(x, "adjoint_wall"))
        for (k, s, ph), x in jx.items():
            if ph == "time" and x.get("ok"): ref[("jax", "J", k, s)] = (R.med_ms(x, "primal"), R.med_ms(x, "adjoint"))
    cases = sorted({(k[2], k[3]) for k in nsys})
    w("## 1. Device work per call\n")
    w("Kernel ms = the sum of the device times of all kernels of one call. Wall ms = the host time of one call during profiling. Busy = kernel ms over wall ms. "
      "Bench ms = the time of one call without a profiler (job FULLT1, host clock).\n")
    w("| kernel | size | call | framework | launches | kernel ms | wall ms | busy | bench ms | kernel ms / JAX kernel ms | launches / JAX launches |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in cases:
        for ph in ("prof_primal", "prof_adjoint"):
            jx = nsys.get(("jax", "J", k, s, ph))
            for fw, cfg in (("stade", "C"), ("stade", "A"), ("jax", "J")):
                v = nsys.get((fw, cfg, k, s, ph))
                if not v: continue
                bm = ref.get((fw, cfg, k, s.replace("bench_", "bench_")), (None, None))
                bm = bm[0] if ph == "prof_primal" else bm[1]
                busy = v["kernel_ms"] / v["wall_ms"] if v["wall_ms"] else None
                rk = v["kernel_ms"] / jx["kernel_ms"] if jx and fw == "stade" else None
                rl = v["launches"] / jx["launches"] if jx and fw == "stade" and jx["launches"] else None
                w(f"| {k} | {s[6:]} | {ph[5:]} | {fw.upper() if fw == 'jax' else 'STADE ' + cfg} | {v['launches']:.0f} | {R.g(v['kernel_ms'])} | {R.g(v['wall_ms'])} | "
                  f"{R.g(busy, 2)} | {R.g(bm)} | {R.g(rk, 2)} | {R.g(rl, 2)} |")
                rows.append(dict(kernel=k, size=s, call=ph[5:], fw=fw, config=cfg, **{a: b for a, b in v.items() if a not in ("work",)}, bench_ms=bm))
    w("\n## 2. Largest kernels\n")
    w("The 6 kernels with the largest device time of each profile run.\n")
    for k, s in cases:
        for ph in ("prof_primal", "prof_adjoint"):
            for fw, cfg in (("stade", "C"), ("stade", "A"), ("jax", "J")):
                v = nsys.get((fw, cfg, k, s, ph))
                if not v: continue
                w(f"\n**{k} {s[6:]}, {ph[5:]}, {fw.upper() if fw == 'jax' else 'STADE ' + cfg}** ({v['launches']:.0f} launches, {v['distinct']} distinct kernels, capture {v['capture']}, "
                  f"reset kernels per call {v['reset_launches']:.0f})\n")
                w("| kernel | per call | ms per call | share | avg us |\n|---|---:|---:|---:|---:|")
                tot = sum(a["total"] for a in v["work"]) or 1
                for a in v["work"][:6]:
                    w(f"| `{short(a['name'])}` | {a['n'] / NCALL:.1f} | {a['total'] / NCALL / 1e6:.4f} | {a['total'] / tot:.0%} | {a['avg'] / 1e3:.1f} |")
    if ncu:
        w("\n## 3. Counters of ncu\n")
        w("ncu runs each kernel alone with the GPU clocks locked, so its times differ from the times of the benchmark. Use the counters, and use the times only to compare kernels with each other. "
          "`atom` and `red` are the warp instructions (SASS) of atomic operations and of atomic reductions on global memory. "
          "Values are per call (the mean of the 3 calls).\n")
        for key, (agg, n) in sorted(ncu.items()):
            w(f"\n**{key[2]} {key[3][6:]}, adjoint, STADE {key[1]}** ({n} kernel launches recorded; {' '.join(ncu_info.get(key, []))})\n")
            w("| kernel | per call | time us | atom warp inst | red warp inst | red warp inst (other counter) |\n|---|---:|---:|---:|---:|---:|")
            for nm, a in sorted(agg.items(), key=lambda kv: -(col(kv[1], "gpu__time_duration") or 0))[:10]:
                c = a["n"] / NCALL; f = lambda key: (col(a, key) or 0) / NCALL
                w(f"| `{short(nm)}` | {c:.1f} | {f('gpu__time_duration') / 1e3:.1f} | {f('smsp__sass_inst_executed_op_global_atom'):.3g} | "
                  f"{f('smsp__sass_inst_executed_op_global_red'):.3g} | {f('smsp__inst_executed_op_global_red'):.3g} |")
    w("\n## 4. Checks\n")
    chk = []
    for r in bad: chk.append(("FAIL", f"RUN {r['fw']} {r['config']} {r['kernel']} {r['size']} {r['phase']}: status {r.get('status')}, exit {r.get('exit')}"))
    for key, v in nsys.items():
        if v["capture"] == "none": chk.append(("NOTE", f"{key[0]} {key[2]} {key[3]} {key[4]}: no capture range worked, the trace holds the whole process"))
        elif v["launches"] < 1: chk.append(("FAIL", f"{key}: no kernel in the capture range"))
    for r in runs:
        x = (r.get("result") or {}); p = x.get("prog") or {}
        if r["phase"].startswith(("prof", "ncu")) and r.get("status") in ("", None, "ok") and not calls_of(p):
            chk.append(("FAIL", f"{r['fw']} {r['kernel']} {r['size']} {r['phase']}: the program result has no prof_calls_s"))
    nf = [c for c in chk if c[0] == "FAIL"]
    w(f"{len(nf)} failed checks, {len(chk) - len(nf)} notes.\n")
    for st, t in chk: w(f"- {st}: {t}")
    os.makedirs(o.out, exist_ok=True)
    open(os.path.join(o.out, "PROFILE.md"), "w").write("\n".join(P) + "\n")
    json.dump(rows, open(os.path.join(o.out, "prof_rows.json"), "w"), indent=0)
    print(f"{len(nsys)} nsys runs, {len(ncu)} ncu runs, {len(nf)} failed checks, {len(chk) - len(nf)} notes")


if __name__ == "__main__":
    main()
