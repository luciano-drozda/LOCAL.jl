"""Recomputes the timing statistics that DISCUSSION.md quotes, from a rows.json of cluster/report.py.
Usage: python3 tools/discussion_stats.py ROWS.json"""
import json, math, sys
rows = json.load(open(sys.argv[1]))
FAM = {"Stencil": ["heat3d_step", "heat3d_multi", "heat1d_multi", "wave2d_step", "burgers2d_step"], "Flow": ["swe2d_step", "euler2d_step"],
       "Unstructured": ["ufv_scatter_step", "ufv_gather_step", "mpnn_gather"], "NN": ["mlp3", "mlp3_push", "transformer_block", "unet2"]}
def gm(v): return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
for r in rows:
    r["gp"] = r["pd"] / r["jp"] if r.get("jp") else None
    r["gpe"] = r["pd_ev"] / r["jp"] if r.get("jp") else None
    r["gc"] = r["ad"] / r["ja"] if r.get("ja") else None
    r["gce"] = r["ad_ev"] / r["ja"] if r.get("ja") else None
    r["ga"] = r["ab"] / r["ja"] if r.get("ja") else None
    r["gb"] = min(r["ad"], r["ab"]) / r["ja"] if r.get("ja") else None
    r["gbe"] = min(r["ad_ev"], r["ab_ev"]) / r["ja"] if r.get("ja") else None
P = [r for r in rows if r["gp"]]; A = [r for r in rows if r["gc"]]
print(f"primal: n={len(P)} gm={gm([r['gp'] for r in P]):.3f} event gm={gm([r['gpe'] for r in P]):.3f} slower={sum(r['gp']>1 for r in P)} faster={sum(r['gp']<1 for r in P)}")
print(f"adjoint C: n={len(A)} gm={gm([r['gc'] for r in A]):.3f} event gm={gm([r['gce'] for r in A]):.3f} faster={sum(r['gc']<1 for r in A)} slower={sum(r['gc']>1 for r in A)} >5%slower={sum(r['gc']>1.05 for r in A)} >5%faster={sum(r['gc']<0.95 for r in A)}")
print(f"adjoint best: gm={gm([r['gb'] for r in A]):.3f} event gm={gm([r['gbe'] for r in A]):.3f} >5%slower={sum(r['gb']>1.05 for r in A)} >5%faster={sum(r['gb']<0.95 for r in A)} slower={sum(r['gb']>1 for r in A)}")
print(f"A never slower than C: {all(r['ab'] <= r['ad']*1.0 for r in rows)}; A saves >10%: {sum(r['ab'] < 0.9*r['ad'] for r in rows)} of {len(rows)}")
print("family: primal gm (n, range) | adjoint gap C | best")
for f, ks in FAM.items():
    p = [r for r in P if r["k"] in ks]; a = [r for r in A if r["k"] in ks]
    print(f"  {f:13s} primal n={len(p)} gm={gm([r['gp'] for r in p]):.2f} range {min(r['gp'] for r in p):.2f}-{max(r['gp'] for r in p):.2f} | C {gm([r['gc'] for r in a]):.2f} best {gm([r['gb'] for r in a]):.2f}")
print("primal slower:", [(r['k'], r['s'], round(r['gp'], 2)) for r in P if r['gp'] > 1])
print("small-size primal gap (smallest of each non-NN kernel):", end=" ")
first = {}
for r in P:
    if r["k"] in sum([FAM[f] for f in ("Stencil", "Flow", "Unstructured")], []) and r["k"] not in first: first[r["k"]] = r
print(f"{gm([r['gp'] for r in first.values()]):.2f}")
print("adjoint C slower cases (gap C, gap A, best):")
for r in sorted([r for r in A if r["gc"] > 1], key=lambda r: -r["gc"]): print(f"  {r['k']} {r['s']}: C {r['gc']:.2f} A {r['ga']:.2f} best {r['gb']:.2f}")
print(f"ratio adj/primal: STADE C gm={gm([r['ad']/r['pd'] for r in A]):.2f} JAX gm={gm([r['ja']/r['jp'] for r in A]):.2f}")
for k in ("mpnn_gather", "mlp3", "mlp3_push", "unet2"):
    print(" ", k, [(r['s'], round(r['ad']/r['pd'], 2), round(r['ja']/r['jp'], 2)) for r in A if r['k'] == k])
print("adjoint gaps by case (C / A):")
for r in A:
    if r['k'] in ("swe2d_step", "euler2d_step", "mpnn_gather", "ufv_scatter_step", "ufv_gather_step", "unet2", "transformer_block", "mlp3", "mlp3_push"): print(f"  {r['k']} {r['s']}: C {r['gc']:.2f} A {r['ga']:.2f} best {r['gb']:.2f}")
print("--- extra")
nonnn = sum([FAM[f] for f in ("Stencil", "Flow", "Unstructured")], [])
last = {}
for r in P:
    if r["k"] in nonnn: last[r["k"]] = r
print("primal gap at the largest size of each non-NN kernel:", round(gm([r["gp"] for r in last.values()]), 2))
sm = [r for r in P if r["jp"] < 0.15]; lg = [r for r in P if r["jp"] >= 0.15]
print(f"JAX primal < 0.15 ms: n={len(sm)} gm={gm([r['gp'] for r in sm]):.2f}; others n={len(lg)} gm={gm([r['gp'] for r in lg]):.2f}")
print("adjoint gaps at 1e5 (C):", {f"{r['k']}": round(r['gc'], 2) for r in A if r['s'] == '1e5' and r['k'] in ("burgers2d_step", "heat3d_step", "wave2d_step", "ufv_scatter_step", "ufv_gather_step")})
for r in A:
    if r['k'] in ("swe2d_step", "euler2d_step", "unet2", "ufv_scatter_step") : print(f"  {r['k']} {r['s']}: C {r['gc']:.3f} A {r['ga']:.3f}  ja {r['ja']:.3f} ad {r['ad']:.3f} ab {r['ab']:.3f} ratio JAX adj/primal {r['ja']/r['jp']:.2f}")
print("mlp3 A ratio ab/pd:", [(r['s'], round(r['ab']/r['pd'], 2), round(r['ad']/r['pd'], 2)) for r in A if r['k'] == 'mlp3'])
print("mpnn times:", [(r['s'], round(r['ad'], 3), round(r['ab'], 3), round(r['ja'], 3), round(r['pd'], 3), round(r['jp'], 3)) for r in A if r['k'] == 'mpnn_gather'])
