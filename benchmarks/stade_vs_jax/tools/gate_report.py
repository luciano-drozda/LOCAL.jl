"""Compares two configs of one time job, case by case, over the repeated runs of each.
Usage: python3 tools/gate_report.py RESULT.json [--base C] [--new G] [--out FILE]
For every kernel and size: the median adjoint time (ms) of each run of the base config and of the new config, their medians, the change new/base - 1, and the spread of the base
(max/min - 1 over its runs). The checks columns compare grad_relerr and elements_changed of the two configs (same count of elements written in every run, and a relative error against the CPU below 1e-9 in every run)."""
import argparse, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "cluster")); sys.path.insert(0, os.path.join(HERE, "..", "gpu"))
import report as R
from specs import SZ

ap = argparse.ArgumentParser(); ap.add_argument("result"); ap.add_argument("--base", default="C"); ap.add_argument("--new", default="G"); ap.add_argument("--out")
o = ap.parse_args()
runs = R.load(o.result)["runs"]
rec = {}
for r in runs:
    res = r.get("result")
    if not res or r["fw"] != "stade": continue
    for x in res.get("results", []):
        rec.setdefault((r["kernel"], x["size"], r["config"]), []).append(x)
def ms(x, f): return x[f]["median"] * 1e3
lines = ["| kernel | size | base runs (ms) | new runs (ms) | base med | new med | change | base spread | same output |", "|---|---|---|---|---|---|---|---|---|"]
ch = []
for k in SZ:
    for s in ["tiny"] + list(SZ[k]):
        b = rec.get((k, s, o.base)); n = rec.get((k, s, o.new))
        if not b or not n: continue
        bv = [ms(x, "adjoint") for x in b]; nv = [ms(x, "adjoint") for x in n]
        bm, nm = st.median(bv), st.median(nv)
        # atomic adds land in any order, so the gradient differs between two runs of one script in the last bits: the check is the size of the error and the count of elements written
        same = all(x.get("elements_changed") == b[0].get("elements_changed") for x in b + n) and max(x.get("grad_relerr", 0) for x in b + n) < 1e-9
        if s != "tiny": ch.append((k, s, nm / bm - 1))
        lines.append(f"| {k} | {s} | {' '.join(f'{v:.3f}' for v in bv)} | {' '.join(f'{v:.3f}' for v in nv)} | {bm:.3f} | {nm:.3f} | {100*(nm/bm-1):+.1f} % | {100*(max(bv)/min(bv)-1):.1f} % | {'yes' if same else 'NO'} |")
out = "\n".join(lines) + "\n"
print(out)
if o.out: open(o.out, "w").write(out)
