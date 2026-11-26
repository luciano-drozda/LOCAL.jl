"""Builds the report of the option ablation: the adjoint time of STADE with the configs C, P, G and A, case by case.
Usage: python3 ablation_report.py --results FILE [--ref FILE ...] --out OUTDIR [--headline wall|events]
  C = defaults, P = primal_outputs = false, G = fresh_gradient = true, A = both (P and G).
  --results   the result file of the ablation job (time phase, STADE only)
  --stade-dir the folder with the generated scripts (all/KERNEL_CONFIG.jl). With it, the report lists the configs that generate the same code
              as another config. Their time ratios show the noise of the job.
  --ref       result files of a full job with JAX time runs (FULLT1), used for the JAX column. Optional.
Writes OUTDIR/ABLATION.md and OUTDIR/ablation_rows.json. The checks of report.py apply (G1, G2, G4, G9 and failed runs)."""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "gpu"))
import report as R
from specs import SZ

CFGS = ("C", "P", "G", "A")
FAM = {"heat3d_step": "stencil", "heat3d_multi": "stencil", "heat1d_multi": "stencil", "wave2d_step": "stencil", "burgers2d_step": "stencil",
       "swe2d_step": "flow", "euler2d_step": "flow", "ufv_scatter_step": "unstructured", "ufv_gather_step": "unstructured", "mpnn_gather": "unstructured",
       "mlp3": "NN", "mlp3_push": "NN", "transformer_block": "NN", "unet2": "NN"}
TIE = 0.03   # a change below 3 % of the C time is a tie (the noise of one job is up to 3.3 %)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--ref", action="append", default=[]); ap.add_argument("--out", required=True)
    ap.add_argument("--stade-dir"); ap.add_argument("--headline", default="wall", choices=["wall", "events"])
    o = ap.parse_args()
    runs, jobs = R.collect([o.results])
    stade, _ = R.index(runs)
    _, jax = R.index(R.collect(o.ref)[0]) if o.ref else ({}, {})
    chk = R.checks(runs, stade, {})
    af = "adjoint_wall" if o.headline == "wall" else "adjoint"; pf = "primal_wall" if o.headline == "wall" else "primal"
    kernels = [k for k in SZ if ("C", k, "time") in stade]
    # G2 for the ablation: every config changes the same number of elements as C
    for k in kernels:
        for s, x in stade[("C", k, "time")].items():
            for c in CFGS[1:]:
                y = (stade.get((c, k, "time")) or {}).get(s)
                if y is not None and y.get("elements_changed") != x.get("elements_changed"):
                    chk.append(("G2", "FAIL", f"STADE {k} {s}: elements_changed differ between C and {c}"))
    rows = []
    for k in kernels:
        for s in SZ[k]:
            r = dict(k=k, s=s[6:], fam=FAM.get(k, "other"))
            for c in CFGS:
                x = (stade.get((c, k, "time")) or {}).get(s)
                r[c] = R.med_ms(x, af); r["p" + c] = R.med_ms(x, pf)
            jt = jax.get((k, s, "time"))
            if jt and jt.get("ok"): r["J"] = R.med_ms(jt, "adjoint")
            rows.append(r)
    rows = [r for r in rows if r.get("C")]
    P = []; w = P.append
    w(f"# Ablation of the adjoint options of STADE on a V100\n")
    for j in jobs:
        v = j.get("versions") or {}
        w(f"- Job on {j.get('host')} (Slurm {j.get('slurm_job') or '-'}), {R.g((j.get('seconds') or 0) / 60, 1)} min. STADE commit {v.get('stade_commit', '-')[:9]}{' (dirty)' if v.get('stade_dirty') else ''}, kit {v.get('kit_hash', '-')}.")
    vs = {}
    for r in runs:
        for a, b in ((r.get("result") or {}).get("versions") or {}).items():
            if b and a not in ("stade_commit", "kit_hash"): vs[a] = b
    w("- stade: " + ", ".join(f"{a} {b}" for a, b in sorted(vs.items())) + "\n")
    w("Config C = defaults. P = `primal_outputs = false`. G = `fresh_gradient = true`. A = P and G together. The tables show the adjoint time in ms "
      f"({'host clock with device synchronization' if o.headline == 'wall' else 'CUDA events'}). A ratio below 1 means the option makes the adjoint faster than C. "
      f"A change below {TIE:.0%} of the C time is a tie.\n")
    w("## 1. Adjoint time by case\n")
    w("| kernel | size | C ms | P ms | G ms | A ms | P/C | G/C | A/C | JAX ms | gap C | gap P | gap G | gap A |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rt = lambda c: R.ratio(r.get(c), r["C"])
        gp = lambda c: R.ratio(r.get(c), r.get("J"))
        w(f"| {r['k']} | {r['s']} | {R.g(r['C'])} | {R.g(r.get('P'))} | {R.g(r.get('G'))} | {R.g(r.get('A'))} | {R.g(rt('P'), 2)} | {R.g(rt('G'), 2)} | {R.g(rt('A'), 2)} | "
          f"{R.g(r.get('J'))} | {R.g(gp('C'), 2)} | {R.g(gp('P'), 2)} | {R.g(gp('G'), 2)} | {R.g(gp('A'), 2)} |")
    w("\n## 2. Which option gives the gain\n")
    w("Each case with an A gain above the tie level (A/C below 1 minus the tie) falls in one class. **P** means P gives most of the gain and G gives little. **G** is the reverse. "
      "**both** means each option gives a part. **none** means neither option alone gives a gain although A does. **mix** means one option alone is slower than C.\n")
    cls = {}
    for r in rows:
        if not all(r.get(c) for c in CFGS): continue
        a, p, g_ = (r["C"] - r["A"]) / r["C"], (r["C"] - r["P"]) / r["C"], (r["C"] - r["G"]) / r["C"]
        if a <= TIE: c = "no gain" if min(p, g_) >= -TIE else "mix"
        elif min(p, g_) < -TIE: c = "mix"
        elif p > TIE and g_ <= TIE: c = "P"
        elif g_ > TIE and p <= TIE: c = "G"
        elif p > TIE and g_ > TIE: c = "both"
        else: c = "none"
        r["cls"] = c; r["gain"] = (a, p, g_); cls.setdefault(c, []).append(r)
    for c in ("P", "G", "both", "none", "mix", "no gain"):
        L = cls.get(c, [])
        w(f"- **{c}**: {len(L)} cases" + (": " + ", ".join(f"{r['k']} {r['s']}" for r in L) if L else ""))
    w("\n| kernel | size | A gain | P gain | G gain | class |\n|---|---|---:|---:|---:|---|")
    for r in rows:
        if "cls" in r:
            a, p, g_ = r["gain"]
            w(f"| {r['k']} | {r['s']} | {a:.0%} | {p:.0%} | {g_:.0%} | {r['cls']} |")
    w("\n## 3. Geometric-mean ratio to C, by family\n")
    w("| family | cases | P/C | G/C | A/C |\n|---|---:|---:|---:|---:|")
    for f in ("stencil", "flow", "unstructured", "NN", "all"):
        L = [r for r in rows if (f == "all" or r["fam"] == f) and all(r.get(c) for c in CFGS)]
        if L: w(f"| {f} | {len(L)} | " + " | ".join(R.g(R.geo([r[c] / r["C"] for r in L]), 3) for c in ("P", "G", "A")) + " |")
    if any(r.get("J") for r in rows):
        w("\n## 4. Geometric-mean gap to JAX\n")
        w("| set | cases | C | P | G | A |\n|---|---:|---:|---:|---:|---:|")
        L = [r for r in rows if r.get("J") and all(r.get(c) for c in CFGS)]
        w(f"| all with JAX | {len(L)} | " + " | ".join(R.g(R.geo([r[c] / r["J"] for r in L]), 3) for c in CFGS) + " |")
        w(f"| STADE slower than JAX | | " + " | ".join(str(sum(r[c] > r["J"] for r in L)) for c in CFGS) + " |")
    if o.stade_dir:
        w("\n## 5. Configs with the same generated code\n")
        w("A config pair with the same code is the same program. The time ratio of such a pair is noise, so this table measures the noise of the job on the real cases.\n")
        w("| kernel | same code | size | ratio |\n|---|---|---|---:|")
        noise = []
        for k in kernels:
            f = {c: os.path.join(o.stade_dir, f"{k}_{c}.jl") for c in CFGS}
            same = [(a, b) for i, a in enumerate(CFGS) for b in CFGS[i + 1:] if os.path.isfile(f[a]) and os.path.isfile(f[b]) and open(f[a], "rb").read() == open(f[b], "rb").read()]
            for a, b in same:
                for r in [r for r in rows if r["k"] == k and r.get(a) and r.get(b)]:
                    noise.append(abs(r[b] / r[a] - 1)); w(f"| {k} | {a} = {b} | {r['s']} | {r[b] / r[a]:.3f} |")
            if not same: w(f"| {k} | none | | |")
        if noise:
            noise.sort()
            w(f"\n{len(noise)} pairs. Median change {noise[len(noise) // 2]:.1%}, largest {noise[-1]:.1%}.")
    w("\n## 6. Checks\n")
    nf = [c for c in chk if c[1] == "FAIL"]
    w(f"{len(nf)} failed checks, {len(chk) - len(nf)} notes.\n")
    for i, st, t in chk: w(f"- {st} {i}: {t}")
    os.makedirs(o.out, exist_ok=True)
    open(os.path.join(o.out, "ABLATION.md"), "w").write("\n".join(P) + "\n")
    json.dump(rows, open(os.path.join(o.out, "ablation_rows.json"), "w"), indent=0)
    print(f"{len(rows)} cases, {len(nf)} failed checks, {len(chk) - len(nf)} notes; classes " + ", ".join(f"{c}={len(L)}" for c, L in cls.items()))


if __name__ == "__main__":
    main()
