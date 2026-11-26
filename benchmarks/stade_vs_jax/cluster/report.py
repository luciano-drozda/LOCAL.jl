"""Builds the report of one benchmark from the result files of one or more jobs (the output of build_job.py payloads).
Usage: python3 report.py --results FILE [--results FILE ...] --out OUTDIR [--tag TEXT] [--headline wall|events]
Writes OUTDIR/REPORT.md and OUTDIR/rows.json. Gap = STADE / JAX. Below 1 means STADE is smaller or faster.
Headline STADE timer: wall = host clock with a device synchronization (like the JAX timer, default), events = CUDA events.
Best = the smaller adjoint time of the two STADE configs (C = defaults, A = best opt-in).
The acceptance checks G1 to G9 of PROCEDURE.md section 7 are evaluated here. A FAIL is listed in the report; the owner decides on the case."""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gpu"))
from specs import SZ

POOL_MIN_MB = 34.0     # the CUDA.jl pool minimum chunk: STADE readings below it do not show the true use
G1_LIMIT = 1e-10


def load(path):
    d = json.loads(open(path).read().strip().splitlines()[-1])
    d = d.get("result") or d
    return json.loads(d) if isinstance(d, str) else d


def geo(v):
    v = [x for x in v if x]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else None


def ratio(a, b):
    return (a / b) if a and b else None


def g(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def med_ms(r, f):
    st = (r or {}).get(f)
    return st["median"] * 1e3 if isinstance(st, dict) else None


def collect(paths):
    """Returns (runs, jobs). A run is the record of one process, with fw, config, kernel, size, phase, status, exit, gpu_before, gpu, result."""
    runs, jobs = [], []
    for p in paths:
        d = load(p); jobs.append({k: d.get(k) for k in ("host", "slurm_job", "seconds", "versions")})
        runs += d["runs"]
    return runs, jobs


def index(runs):
    """stade[(config, kernel, phase)] = {size: result of one size}. jax[(kernel, size, phase)] = result."""
    stade, jax = {}, {}
    for r in runs:
        res = r.get("result")
        if not res: continue
        if r["fw"] == "stade":
            stade[(r["config"], r["kernel"], r["phase"])] = {x["size"]: x for x in res.get("results", [])}
        else:
            jax[(r["kernel"], r["size"], r["phase"])] = res
    return stade, jax


def checks(runs, stade, jax):
    """Returns a list of (id, status, text). status is FAIL or NOTE. An empty list means every check passed."""
    out = []
    add = lambda i, st, t: out.append((i, st, t))
    for (cfg, k, ph), sizes in stade.items():
        for s, x in sizes.items():
            tag = f"STADE {cfg} {k} {s} {ph}"
            if "error" in x: add("RUN", "FAIL", f"{tag}: {x['error'][:120]}"); continue
            if s in ("tiny", "small") and ph == "time":                                            # G1
                for f in ("primal_relerr", "grad_relerr"):
                    if x.get(f) is not None and x[f] > G1_LIMIT: add("G1", "FAIL", f"{tag}: {f} = {x[f]:.2e}")
            if x.get("elements_changed") is not None and x["elements_changed"] <= 0: add("G2", "FAIL", f"{tag}: elements_changed = 0")
            if ph == "time" and str(s).startswith("bench_"):                                         # G5
                for f in ("primal", "adjoint"):
                    n = (x.get(f) or {}).get("n")
                    if n not in (30, 10, 5): add("G5", "NOTE", f"{tag}: {f} n = {n}")
            if ph != "time" and x.get("mem_inputs_mb") is not None and x.get("input_mb") is not None and x["mem_inputs_mb"] < x["input_mb"] - 1.0:   # G6
                add("G6", "FAIL" if x["input_mb"] >= POOL_MIN_MB else "NOTE", f"{tag}: mem_inputs {x['mem_inputs_mb']:.1f} MB below input {x['input_mb']:.1f} MB")
            if ph != "time" and x.get("live_inputs_mb") is not None and x.get("input_mb") is not None and abs(x["live_inputs_mb"] - x["input_mb"]) > 1.0:
                add("G6", "FAIL", f"{tag}: live inputs {x['live_inputs_mb']:.1f} MB against input {x['input_mb']:.1f} MB")
            if ph == "mem_adjoint" and x.get("mem_setup_mb") is not None and x.get("device_mb_estimate") is not None:                           # G7
                est = x["device_mb_estimate"]
                if abs(x["mem_setup_mb"] - est) > max(0.05 * est, POOL_MIN_MB): add("G7", "FAIL", f"{tag}: setup {x['mem_setup_mb']:.1f} MB, parts {est:.1f} MB")
                if x.get("live_setup_mb") is not None and abs(x["live_setup_mb"] - est) > max(0.01 * est, 1.0): add("G7", "FAIL", f"{tag}: live setup {x['live_setup_mb']:.1f} MB, parts {est:.1f} MB")
    for (k, s, ph), x in jax.items():
        tag = f"JAX {k} {s} {ph}"
        if not x.get("ok"): add("RUN", "FAIL" if "RESOURCE_EXHAUSTED" not in str(x.get("error")) else "NOTE", f"{tag}: {str(x.get('error'))[:120]}"); continue
        if x.get("grad_finite") is False: add("G3", "FAIL", f"{tag}: gradients not finite")                                                      # G3
        if ph == "time" and (x.get("primal") or {}).get("n") != 30: add("G5", "NOTE", f"{tag}: n = {(x.get('primal') or {}).get('n')}")
        if ph != "time" and x.get("mem_inputs_mb") is not None and x.get("input_mb") is not None and x["mem_inputs_mb"] < x["input_mb"] - 1.0:
            add("G6", "FAIL" if x["input_mb"] >= POOL_MIN_MB else "NOTE", f"{tag}: mem_inputs {x['mem_inputs_mb']:.1f} MB below input {x['input_mb']:.1f} MB")
        if ph == "mem_adjoint":                                                                                                                  # G8
            p = jax.get((k, s, "mem_primal"))
            if p and p.get("ok") and x.get("mem_peak_adjoint_mb") is not None and p.get("mem_peak_primal_mb") is not None \
                    and x["mem_peak_adjoint_mb"] < p["mem_peak_primal_mb"] - 1.0:
                add("G8", "FAIL", f"{tag}: adjoint peak {x['mem_peak_adjoint_mb']:.1f} MB below primal peak {p['mem_peak_primal_mb']:.1f} MB")
    ea = {(k, s): x.get("elements_changed") for (cfg, k, ph), sz in stade.items() if cfg == "C" and ph == "time" for s, x in sz.items()}
    for (cfg, k, ph), sz in stade.items():
        if cfg == "A" and ph == "time":
            for s, x in sz.items():
                if (k, s) in ea and ea[(k, s)] != x.get("elements_changed"): add("G2", "FAIL", f"STADE {k} {s}: elements_changed differ between C and A")
    for r in runs:
        tag = f"{r['fw']} {r['config']} {r['kernel']} {r.get('size') or ''} {r['phase']}".replace("  ", " ")
        if r.get("status"): add("RUN", "FAIL", f"{tag}: status {r['status']}")
        elif r.get("exit") not in (0, None) or not r.get("result"): add("RUN", "FAIL", f"{tag}: exit {r.get('exit')}, no result. {str(r.get('stderr_tail'))[-150:]}")
        gb = r.get("gpu_before") or {}
        if gb.get("busy") is True: add("G4", "FAIL", f"{tag}: GPU busy before the run ({gb.get('apps') or gb.get('mem_used_mib')})")                       # G4
        if (r.get("gpu") or {}).get("throttle_bad"): add("G9", "FAIL", f"{tag}: throttle reasons {r['gpu'].get('throttle_or')}")                       # G9
    return out


def build_rows(stade, jax, head):
    rows = []
    pf = "primal_wall" if head == "wall" else "primal"; af = "adjoint_wall" if head == "wall" else "adjoint"
    for k in SZ:
        for s in SZ[k]:
            sl = s
            r = dict(k=k, s=s[6:])
            sc = (stade.get(("C", k, "time")) or {}).get(sl); sa = (stade.get(("A", k, "time")) or {}).get(sl)
            r["pd"] = med_ms(sc, pf); r["ad"] = med_ms(sc, af); r["pd_ev"] = med_ms(sc, "primal"); r["ad_ev"] = med_ms(sc, "adjoint")
            r["ab"] = med_ms(sa, af); r["ab_ev"] = med_ms(sa, "adjoint")
            jt = jax.get((k, s, "time"))
            if jt and jt.get("ok"): r["jp"] = med_ms(jt, "primal"); r["ja"] = med_ms(jt, "adjoint")
            mp = (stade.get(("C", k, "mem_primal")) or {}).get(sl); ma = (stade.get(("C", k, "mem_adjoint")) or {}).get(sl); mb = (stade.get(("A", k, "mem_adjoint")) or {}).get(sl)
            r["spk_p"] = (mp or {}).get("mem_peak_primal_mb"); r["spk_d"] = (ma or {}).get("mem_peak_adjoint_mb"); r["spk_b"] = (mb or {}).get("mem_peak_adjoint_mb")
            r["spe_p"] = (mp or {}).get("peak_exact_primal_mb"); r["spe_d"] = (ma or {}).get("peak_exact_adjoint_mb"); r["spe_b"] = (mb or {}).get("peak_exact_adjoint_mb")
            r["sstack"] = (ma or {}).get("stack_mb"); r["sgrad"] = (ma or {}).get("grad_mb"); r["sin"] = (ma or mp or {}).get("input_mb")
            jmp = jax.get((k, s, "mem_primal")); jma = jax.get((k, s, "mem_adjoint"))
            if jmp and jmp.get("ok"): r["jpk_p"] = jmp.get("mem_peak_primal_mb"); r["jown_p"] = jmp.get("peak_mb")
            elif jmp: r["jerr_p"] = jmp.get("error")
            if jma and jma.get("ok"): r["jpk_a"] = jma.get("mem_peak_adjoint_mb"); r["jown_a"] = jma.get("peak_mb")
            elif jma: r["jerr_a"] = jma.get("error")
            if jt and not jt.get("ok"): r["jerr"] = jt.get("error")
            if r["ad"] and r["ab"]: r["best"] = min(r["ab"], r["ad"])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=""); ap.add_argument("--headline", default="wall", choices=["wall", "events"])
    o = ap.parse_args()
    runs, jobs = collect(o.results); stade, jax = index(runs)
    rows = build_rows(stade, jax, o.headline); chk = checks(runs, stade, jax)
    os.makedirs(o.out, exist_ok=True)
    json.dump(rows, open(os.path.join(o.out, "rows.json"), "w"), indent=1)
    out = []; P = out.append
    hl = "host clock with device synchronization" if o.headline == "wall" else "CUDA events"
    P(f"# STADE against JAX on a V100 {o.tag}\n")
    for j in jobs:
        v = j.get("versions") or {}
        P(f"- Job on {j.get('host')} (Slurm {j.get('slurm_job') or '-'}), {g((j.get('seconds') or 0) / 60, 1)} min. STADE commit {v.get('stade_commit', '-')[:9]}{' (dirty)' if v.get('stade_dirty') else ''}, kit {v.get('kit_hash', '-')}.")
    vs = {}
    for r in runs:
        x = r.get("result") or {}
        for k2, v2 in (x.get("versions") or {}).items():
            if v2: vs.setdefault(r["fw"], {})[k2] = v2
    for fw, d in vs.items():
        P(f"- {fw}: " + ", ".join(f"{a} {b}" for a, b in sorted(d.items()) if a not in ("stade_commit", "kit_hash")))
    P(f"\nSTADE config C = defaults. Config A = primal_outputs = false with fresh_gradient. JAX = jit. Gap = STADE / JAX. Headline STADE timer: {hl}. Failed checks are listed in section 7.\n")
    tp = [r for r in rows if r.get("jp") and r.get("pd")]
    P("## 1. Runtime of the primal\n")
    P("| kernel | size | STADE ms | STADE events ms | JAX ms | gap |\n|---|---|---:|---:|---:|---:|")
    for r in sorted(tp, key=lambda r: -(r["pd"] / r["jp"])):
        P(f"| {r['k']} | {r['s']} | {r['pd']:.4f} | {g(r.get('pd_ev'), 4)} | {r['jp']:.4f} | {r['pd']/r['jp']:.2f} |")
    gp = [r["pd"] / r["jp"] for r in tp]
    P(f"\n{len(tp)} cases. Geometric-mean gap **{g(geo(gp))}**. STADE slower than JAX in {sum(x > 1 for x in gp)} cases.\n")
    ta = [r for r in rows if r.get("ja") and r.get("ad") and r.get("best")]
    P("## 2. Runtime of the adjoint\n")
    P("| kernel | size | STADE C ms | STADE C events ms | STADE best ms | JAX ms | gap C | gap best |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(ta, key=lambda r: -(r["best"] / r["ja"])):
        P(f"| {r['k']} | {r['s']} | {r['ad']:.3f} | {g(r.get('ad_ev'))} | {r['best']:.3f} | {r['ja']:.3f} | {r['ad']/r['ja']:.2f} | {r['best']/r['ja']:.2f} |")
    gd = [r["ad"] / r["ja"] for r in ta]; gb = [r["best"] / r["ja"] for r in ta]
    P(f"\n{len(ta)} cases. Geometric-mean gap: C **{g(geo(gd))}**, best **{g(geo(gb))}**. STADE slower than JAX: C {sum(x > 1 for x in gd)}, best {sum(x > 1 for x in gb)}.\n")
    P("## 3. Ratio adjoint time over primal time\n")
    P("| kernel | size | STADE C | JAX |\n|---|---|---:|---:|")
    for r in rows:
        if r.get("ad") and r.get("pd") and r.get("ja") and r.get("jp"):
            P(f"| {r['k']} | {r['s']} | {r['ad']/r['pd']:.2f} | {r['ja']/r['jp']:.2f} |")
    P("\n## 4. Memory of the primal\n")
    P("Only the inputs are on the device. Two readings. **Exact**: STADE = arrays alive in the pool + bytes that one call allocates; JAX = XLA live-bytes peak (`JAX own`). **Driver**: device memory in use above the baseline, as nvidia-smi shows it. The driver reading is rounded up by the allocators (the CUDA.jl pool chunk is 32 MiB = 33.6 MB, and the XLA pool grows in regions). It also holds one-time library memory. Use the exact gap for small cases. A driver gap is computed only if both readings are at least 34 MB.\n")
    P("| kernel | size | inputs MB | STADE exact MB | JAX own MB | gap exact | STADE driver MB | JAX driver MB | gap driver |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    mp = [r for r in rows if r.get("spk_p") is not None or r.get("jpk_p") is not None]
    sigp, sigpe = [], []
    for r in mp:
        ge = ratio(r.get("spe_p"), r.get("jown_p"))
        ok = r.get("spk_p") is not None and r.get("jpk_p") is not None and r["spk_p"] >= POOL_MIN_MB and r["jpk_p"] >= POOL_MIN_MB
        gap = ratio(r.get("spk_p"), r.get("jpk_p")) if ok else None
        if gap: sigp.append(gap)
        if ge: sigpe.append(ge)
        P(f"| {r['k']} | {r['s']} | {g(r.get('sin'), 1)} | {g(r.get('spe_p'), 1)} | {g(r.get('jown_p'), 1)} | {g(ge, 2)} | {g(r.get('spk_p'), 0)} | {g(r.get('jpk_p'), 0)} | {g(gap, 2)} |")
    P(f"\nGeometric-mean primal memory gap: exact **{g(geo(sigpe), 2)}** over {len(sigpe)} cases, driver **{g(geo(sigp), 2)}** over {len(sigp)} cases.\n")
    P("## 5. Memory of the adjoint\n")
    P("Only the inputs, the gradient buffers and the stacks (STADE) or the cotangents (JAX) are on the device. Exact and driver readings as in section 4.\n")
    P("| kernel | size | STADE stacks MB | STADE C exact MB | STADE A exact MB | JAX own MB | gap C exact | gap A exact | STADE C driver MB | STADE A driver MB | JAX driver MB | gap C driver |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ma = [r for r in rows if r.get("spk_d") is not None or r.get("jpk_a") is not None]
    sig, sige, sigeb = [], [], []
    for r in ma:
        gce, gae = ratio(r.get("spe_d"), r.get("jown_a")), ratio(r.get("spe_b"), r.get("jown_a"))
        ok = r.get("spk_d") is not None and r.get("jpk_a") is not None and r["spk_d"] >= POOL_MIN_MB and r["jpk_a"] >= POOL_MIN_MB
        gd_ = ratio(r.get("spk_d"), r.get("jpk_a")) if ok else None
        if gd_: sig.append(gd_)
        if gce: sige.append(gce)
        if gae: sigeb.append(gae)
        P(f"| {r['k']} | {r['s']} | {g(r.get('sstack'), 1)} | {g(r.get('spe_d'), 1)} | {g(r.get('spe_b'), 1)} | {g(r.get('jown_a'), 1)} | {g(gce, 2)} | {g(gae, 2)} | {g(r.get('spk_d'), 0)} | {g(r.get('spk_b'), 0)} | {g(r.get('jpk_a'), 0)} | {g(gd_, 2)} |")
    P(f"\nGeometric-mean adjoint memory gap: exact C **{g(geo(sige), 2)}**, exact A **{g(geo(sigeb), 2)}** over {len(sige)} cases; driver C **{g(geo(sig), 2)}** over {len(sig)} cases.\n")
    P("## 6. Summary\n")
    P(f"- Primal runtime: gap {g(geo(gp))}, STADE slower in {sum(x > 1 for x in gp)} of {len(gp)} cases.")
    P(f"- Adjoint runtime: gap C {g(geo(gd))}, best {g(geo(gb))}; STADE C slower in {sum(x > 1 for x in gd)} of {len(gd)} cases.")
    P(f"- Primal memory: exact gap {g(geo(sigpe), 2)} over {len(sigpe)} cases; driver gap {g(geo(sigp), 2)} over {len(sigp)} cases.")
    P(f"- Adjoint memory: exact gap C {g(geo(sige), 2)}, A {g(geo(sigeb), 2)} over {len(sige)} cases; driver gap C {g(geo(sig), 2)} over {len(sig)} cases.\n")
    P("## 7. Checks and invalid cases\n")
    fails = [c for c in chk if c[1] == "FAIL"]; notes = [c for c in chk if c[1] == "NOTE"]
    P(f"{len(fails)} failed checks, {len(notes)} notes.\n")
    for i, st, t in sorted(chk): P(f"- {st} {i}: {t}")
    if not chk: P("- All checks passed (G1 to G9).")
    P("\n## 8. Limits\n")
    for t in ("One GPU model (V100, 16 GB), float64, batch size one. JAX appears as jit only.",
              "The two runtime timers differ. The caches are not flushed, and STADE copies its inputs between calls.",
              "The driver readings are rounded up by the allocators and hold one-time library memory. The exact readings are not. JAX driver readings come from nvidia-smi, and the JAX pool never shrinks.",
              "One job gives one run. The noise limit of 5 % is a working value. The STADE checkpoint option is off."):
        P(f"- {t}")
    open(os.path.join(o.out, "REPORT.md"), "w").write("\n".join(out) + "\n")
    print(f"{len(runs)} runs, {len(ta)} adjoint, {len(tp)} primal, {len(mp)} primal-memory, {len(ma)} adjoint-memory rows; {len(fails)} failed checks, {len(notes)} notes")


if __name__ == "__main__":
    main()
