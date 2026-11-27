# Full benchmark of 2026-11-27

This folder repeats the time phase of `../full_2026-11-26/` with STADE v0.6.39, after the changes of the warp-level atomic adds (v0.6.33 to v0.6.39, `../experiments_2026-11-27/`).

| Job | Slurm | Phases | Files |
|---|---|---|---|
| FULLT3 | 2101858 | `time` (STADE C and A, JAX): 70 runs, 45 min, krakengpu1 | `FULLT3_time.json`, `plan_time.tsv`, `versions_time.json` |
| FULLM1 | 2100097 | `mem_primal`, `mem_adjoint`: 126 runs, 54 min (2026-11-26, not repeated) | `../full_2026-11-26/FULLM1_mem.json`, `plan_mem.tsv`, `versions_mem.json` |

FULLT3 ran STADE commit 8913babc8 (v0.6.39) and the kit with hash 3d3e6fa193a59b9d. `versions_time.json` lists all three phases because the kit was built with all of them, but the job ran only `time` (70 runs: 28 STADE and 42 JAX). The memory readings in `REPORT.md` come from FULLM1, made with the earlier commit 823fa703e. The memory phase was not repeated, because the changes add no array. This was not measured.

`DISCUSSION.md` interprets the results. It is the text of `../full_2026-11-26/DISCUSSION.md` with every timing figure recomputed, plus section 4.6 on the effect of the changes. `REPORT.md` and `rows.json` come from both jobs. Rebuild them with:

```bash
python3 cluster/report.py --results results/full_2026-11-27/FULLT3_time.json --results results/full_2026-11-26/FULLM1_mem.json --out OUTDIR
```

`tools/discussion_stats.py results/full_2026-11-27/rows.json` recomputes the statistics quoted in `DISCUSSION.md` (gaps, counts, family tables, ratios).

Result: 0 failed checks, 8 notes (six G6 notes for the JAX allocator granularity on the 1e5 cases, two notes for the JAX out-of-memory at euler2d_step 16e6).

Main changes against FULLT1 (adjoint, config C): mpnn_gather -31 to -32 %, euler2d_step -28 to -41 %, swe2d_step -3 to -4 %, mlp3 +0.7 to +2.1 %. The primal gap is 0.53 (0.53 before), the adjoint gap with C is 0.79 (0.84), and with the best of C and A it is 0.62 (0.64). STADE is faster in 34 of 41 primal cases and 24 of 41 adjoint cases with C.
