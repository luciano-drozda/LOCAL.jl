"""Compares two runs of the same time job, case by case: the run-to-run noise of STADE and of JAX.
Usage: python3 repeat_report.py --first FILE --second FILE --out OUTDIR [--headline wall|events]
Writes OUTDIR/REPEAT.md and OUTDIR/repeat_rows.json. Both files are result files of the same plan (for example FULLT1 and FULLT2).
Change = second / first - 1 for the median time of one case, one framework and one quantity (primal, adjoint C, adjoint A).
The report also gives the gap to JAX and the counts of the main statements of DISCUSSION.md in each run, so that a reader sees what a repeat changes."""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "gpu"))
import report as R
from specs import SZ


def rows_of(path, head):
    runs, jobs = R.collect([path]); stade, jax = R.index(runs)
    pf = "primal_wall" if head == "wall" else "primal"; af = "adjoint_wall" if head == "wall" else "adjoint"
    d = {}
    for k in SZ:
        for s in SZ[k]:
            sc = (stade.get(("C", k, "time")) or {}).get(s); sa = (stade.get(("A", k, "time")) or {}).get(s); jt = jax.get((k, s, "time"))
            r = {"sp": R.med_ms(sc, pf), "sc": R.med_ms(sc, af), "sa": R.med_ms(sa, af)}
            if jt and jt.get("ok"): r["jp"] = R.med_ms(jt, "primal"); r["ja"] = R.med_ms(jt, "adjoint")
            d[(k, s[6:])] = r
    return d, runs, jobs, stade, jax


def pct(v, q):
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True); ap.add_argument("--second", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--headline", default="wall", choices=["wall", "events"])
    o = ap.parse_args()
    a, ra, ja_, sa_, xa = rows_of(o.first, o.headline); b, rb, jb_, sb_, xb = rows_of(o.second, o.headline)
    names = {"sp": "STADE primal", "sc": "STADE adjoint C", "sa": "STADE adjoint A", "jp": "JAX primal", "ja": "JAX adjoint"}
    ch = {q: [] for q in names}; rows = []
    for key in a:
        r = {"k": key[0], "s": key[1]}
        for q in names:
            x, y = a[key].get(q), b.get(key, {}).get(q)
            if x and y:
                r[q] = (x, y, y / x - 1); ch[q].append((abs(y / x - 1), y / x - 1, key, x))
        rows.append(r)
    P = []; w = P.append
    w("# Repeat of the time job: run-to-run change\n")
    for tag, jobs in (("First", ja_), ("Second", jb_)):
        for j in jobs:
            v = j.get("versions") or {}
            w(f"- {tag}: Slurm {j.get('slurm_job')} on {j.get('host')}, {R.g((j.get('seconds') or 0) / 60, 1)} min, STADE commit {v.get('stade_commit', '-')[:9]}, kit {v.get('kit_hash', '-')}.")
    w(f"\nChange = second / first - 1 for the median time of one case ({'host clock' if o.headline == 'wall' else 'CUDA events'}). `|change|` is the absolute value. "
      "Cases under 0.1 ms and cases of 0.1 ms or more are separate, because the launch cost dominates the small cases.\n")
    w("## 1. Size of the change\n")
    w("| quantity | cases | median abs change | 90th percentile | largest | share above 5 % | cases of 0.1 ms or more: median | : largest |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for q, nm in names.items():
        L = sorted(ch[q])
        if not L: continue
        big = sorted(c for c in L if c[3] >= 0.1)
        v = [c[0] for c in L]
        w(f"| {nm} | {len(L)} | {pct(v, .5):.2%} | {pct(v, .9):.2%} | {v[-1]:.1%} | {sum(x > .05 for x in v) / len(v):.0%} | "
          + (f"{pct([c[0] for c in big], .5):.2%} | {big[-1][0]:.1%} |" if big else "- | - |"))
    w("\n## 2. Largest changes\n")
    w("| quantity | kernel | size | first ms | change |\n|---|---|---|---:|---:|")
    for q, nm in names.items():
        for c in sorted(ch[q], reverse=True)[:4]:
            w(f"| {nm} | {c[2][0]} | {c[2][1]} | {c[3]:.3f} | {c[1]:+.1%} |")
    w("\n## 3. The headline numbers in each run\n")
    w("| statement | first | second |\n|---|---:|---:|")
    def stats(d):
        t = [(r, k) for k, r in d.items() if r.get("jp") and r.get("ja") and r.get("sp") and r.get("sc") and r.get("sa")]
        gp = [r["sp"] / r["jp"] for r, _ in t]; gc = [r["sc"] / r["ja"] for r, _ in t]; gb = [min(r["sc"], r["sa"]) / r["ja"] for r, _ in t]
        return len(t), R.geo(gp), sum(x > 1 for x in gp), R.geo(gc), sum(x > 1 for x in gc), R.geo(gb), sum(x > 1 for x in gb), \
            sum(x > 1.05 for x in gc), sum(x < .95 for x in gc), sum(x > 1.05 for x in gb), sum(x < .95 for x in gb)
    st1, st2 = stats(a), stats(b)
    for i, nm, f in ((0, "cases", "{}"), (1, "primal gap (geometric mean)", "{:.3f}"), (2, "primal: STADE slower", "{}"), (3, "adjoint gap C", "{:.3f}"), (4, "adjoint C: STADE slower", "{}"),
                     (5, "adjoint gap best", "{:.3f}"), (6, "adjoint best: STADE slower", "{}"), (7, "adjoint C: more than 5 % slower", "{}"), (8, "adjoint C: more than 5 % faster", "{}"),
                     (9, "adjoint best: more than 5 % slower", "{}"), (10, "adjoint best: more than 5 % faster", "{}")):
        w(f"| {nm} | {f.format(st1[i])} | {f.format(st2[i])} |")
    # classification flips: cases that switch side of the 5 % band in the adjoint C gap
    def band(r):
        g = r["sc"] / r["ja"]; return "slower" if g > 1.05 else "faster" if g < .95 else "tie"
    flips = [(k, band(a[k]), band(b[k])) for k in a if all(a[k].get(q) and b[k].get(q) for q in ("sc", "ja")) and band(a[k]) != band(b[k])]
    w(f"\nCases that change class (more than 5 % slower, tie, more than 5 % faster) in the adjoint gap C: {len(flips)}.\n")
    for k, x, y in flips: w(f"- {k[0]} {k[1]}: {x} to {y}")
    w("\n## 4. Checks\n")
    chk = R.checks(rb, sb_, xb); nf = [c for c in chk if c[1] == "FAIL"]
    w(f"Second run: {len(nf)} failed checks, {len(chk) - len(nf)} notes.\n")
    for i, st, t in chk: w(f"- {st} {i}: {t}")
    os.makedirs(o.out, exist_ok=True)
    open(os.path.join(o.out, "REPEAT.md"), "w").write("\n".join(P) + "\n")
    json.dump(rows, open(os.path.join(o.out, "repeat_rows.json"), "w"), indent=0)
    print(f"{len(a)} cases; adjoint C change: median {pct(sorted(c[0] for c in ch['sc']), .5):.2%}; JAX adjoint: median {pct(sorted(c[0] for c in ch['ja']), .5):.2%}; flips {len(flips)}; failed checks {len(nf)}")


if __name__ == "__main__":
    main()
