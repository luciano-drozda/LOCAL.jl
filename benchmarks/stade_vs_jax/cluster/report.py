"""Builds the report of one benchmark from the result files of the STADE job and of the JAX job.
Usage: python3 report.py --stade STADE_RESULT.json --jax JAX_RESULT.json [--jax MORE.json ...] --out OUTDIR [--tag TEXT]
Writes OUTDIR/REPORT.md and OUTDIR/rows.json. Gap = STADE / JAX. Below 1 means STADE is smaller or faster.
Best = the smaller adjoint time of the two STADE configs (C = defaults, A = best opt-in).
Primal memory is not in this report. The harness gives a valid primal memory reading only in its own phase (see PROCEDURE.md, section 5)."""
import argparse, json, math, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gpu"))
from specs import SZ

ap = argparse.ArgumentParser()
ap.add_argument("--stade", required=True); ap.add_argument("--jax", action="append", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
o = ap.parse_args()

def last_json(path):
    return json.loads(open(path).read().strip().splitlines()[-1])

jax, jax_nodes = {}, set()
for p in o.jax:
    d = last_json(p); d = d.get("result") or d
    for l in d["log"].split("\n"):
        if l.startswith("{"):
            try: r = json.loads(l)
            except Exception: continue
            jax[(r["kernel"], r["size"])] = r
        elif l.startswith("node:"): jax_nodes.add(l.split()[1])
d = last_json(o.stade); stade_host = d.get("host")
A, C, bad, secs = {}, {}, [], 0.0
for r in d["runs"]:
    secs += r["seconds"]
    if r["exit"] != 0 or not r.get("result"):
        bad.append((r["name"], r["exit"], r["stderr_tail"][-200:])); continue
    cfg = A if r["name"].endswith("_A") else C
    for x in r["result"]["results"]:
        if isinstance(x.get("adjoint"), dict) and str(x["size"]).startswith("bench_"):
            cfg[(r["result"]["kernel"], x["size"])] = x

keys = [(k, s) for k in SZ for s in SZ[k]]
g = lambda x, nd=3: "-" if x is None else f"{x:.{nd}f}"
def geo(v):
    v = [x for x in v if x]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else None
ratio = lambda a, b: (a / b) if a and b else None
ms = lambda st, f: None if not st or not isinstance(st.get(f), dict) else st[f]["median"] * 1e3
rows = []
for (k, s) in keys:
    j, a, c = jax.get((k, s)), A.get((k, s)), C.get((k, s))
    r = dict(k=k, s=s[6:])
    if j and j.get("ok"):
        r.update(jp=ms(j, "primal"), ja=ms(j, "adjoint"), jpk=j.get("mem_peak_adjoint_mb"), jown=j.get("peak_mb"))
    elif j: r["jerr"] = j.get("error")
    if a: r.update(ab=ms(a, "adjoint"), ea=a.get("elements_changed"), spk_b=a.get("mem_peak_adjoint_mb"))
    if c: r.update(ad=ms(c, "adjoint"), pd=ms(c, "primal"), ec=c.get("elements_changed"), spk_d=c.get("mem_peak_adjoint_mb"),
                   sstack=c.get("stack_mb"), pre=c.get("primal_relerr"), gre=c.get("grad_relerr"))
    if r.get("ad") and r.get("ab"): r["best"] = min(r["ab"], r["ad"])
    rows.append(r)
os.makedirs(o.out, exist_ok=True)
json.dump(rows, open(os.path.join(o.out, "rows.json"), "w"), indent=1)

out = []; P = out.append
P(f"# STADE against JAX on a V100 {o.tag}\n")
P(f"STADE job node: {stade_host}. JAX job node: {', '.join(sorted(jax_nodes)) or '-'}. STADE config C = defaults. Config A = primal_outputs = false with fresh_gradient. JAX = jit. Gap = STADE / JAX.\n")
tp = [r for r in rows if r.get("jp") and r.get("pd")]
P("## 1. Runtime of the primal\n")
P("| kernel | size | STADE ms | JAX ms | gap |\n|---|---|---:|---:|---:|")
for r in sorted(tp, key=lambda r: -(r["pd"] / r["jp"])):
    P(f"| {r['k']} | {r['s']} | {r['pd']:.4f} | {r['jp']:.4f} | {r['pd']/r['jp']:.2f} |")
gp = [r["pd"] / r["jp"] for r in tp]
P(f"\n{len(tp)} cases. Geometric-mean gap **{g(geo(gp))}**. STADE slower than JAX in {sum(x > 1 for x in gp)} cases.\n")
ta = [r for r in rows if r.get("ja") and r.get("ad") and r.get("best")]
P("## 2. Runtime of the adjoint\n")
P("| kernel | size | STADE C ms | STADE best ms | JAX ms | gap C | gap best |\n|---|---|---:|---:|---:|---:|---:|")
for r in sorted(ta, key=lambda r: -(r["best"] / r["ja"])):
    P(f"| {r['k']} | {r['s']} | {r['ad']:.3f} | {r['ab']:.3f} | {r['ja']:.3f} | {r['ad']/r['ja']:.2f} | {r['best']/r['ja']:.2f} |")
gd = [r["ad"] / r["ja"] for r in ta]; gb = [r["best"] / r["ja"] for r in ta]
P(f"\n{len(ta)} cases. Geometric-mean gap: C **{g(geo(gd))}**, best **{g(geo(gb))}**. STADE slower than JAX: C {sum(x > 1 for x in gd)}, best {sum(x > 1 for x in gb)}.\n")
P("## 3. Ratio adjoint time over primal time\n")
P("| kernel | size | STADE C | JAX |\n|---|---|---:|---:|")
for r in rows:
    if r.get("ad") and r.get("pd") and r.get("ja") and r.get("jp"):
        P(f"| {r['k']} | {r['s']} | {r['ad']/r['pd']:.2f} | {r['ja']/r['jp']:.2f} |")
P("\n## 4. Memory of the adjoint\n")
P("Readings come from the driver, above the baseline of the same process. STADE values below about 34 MB show the pool minimum chunk. `JAX own` is the XLA live-bytes peak.\n")
P("| kernel | size | STADE stacks MB | STADE C MB | STADE A MB | JAX MB | JAX own MB | gap C | gap C / own |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
mrows = [r for r in rows if r.get("spk_d") is not None or r.get("jpk") is not None]
for r in mrows:
    sp = r.get("spk_d")
    P(f"| {r['k']} | {r['s']} | {g(r.get('sstack'),0)} | {g(sp,0)} | {g(r.get('spk_b'),0)} | {g(r.get('jpk'),0)} | {g(r.get('jown'),0)} | {g(ratio(sp, r.get('jpk')),2)} | {g(ratio(sp, r.get('jown')),2)} |")
sig = [r for r in mrows if r.get("spk_d") and r.get("jpk") and r["spk_d"] > 40]
P(f"\nGeometric-mean memory gap (driver reading) over the {len(sig)} cases with a STADE peak above 40 MB: **{g(geo([ratio(r['spk_d'], r['jpk']) for r in sig]),2)}**. Against the XLA live-bytes peak: **{g(geo([ratio(r['spk_d'], r['jown']) for r in sig]),2)}**.")
worse = [(r['k'], r['s'], r['spk_d'], r['jpk']) for r in sig if r['spk_d'] > r['jpk']]
P("STADE peak above the JAX driver reading: " + (", ".join(f"{k} {s} ({a:.0f} against {b:.0f} MB)" for k, s, a, b in worse) or "no case") + ".\n")
P("## 5. Checks\n")
diff = [(r['k'], r['s'], r.get('ea'), r.get('ec')) for r in rows if r.get('ea') != r.get('ec')]
P("- Elements changed, config A against config C (must be equal): " + (str(diff) if diff else "equal in all cases") + ".")
P("- Elements changed, smallest value over all cases (must be above 0): " + str(min([x for r in rows for x in (r.get('ea'), r.get('ec')) if x is not None] or [None])) + ".")
P("- STADE failed runs: " + (str(bad) if bad else "none") + f". STADE process time {secs/60:.1f} min.")
P("- JAX cases without a result: " + (str([(r['k'], r['s'], r.get('jerr')) for r in rows if not r.get('ja')]) or "none") + ".")
open(os.path.join(o.out, "REPORT.md"), "w").write("\n".join(out) + "\n")
print(f"{len(ta)} adjoint cases, {len(tp)} primal cases, geomean gap C {geo(gd):.3f} best {geo(gb):.3f}")
