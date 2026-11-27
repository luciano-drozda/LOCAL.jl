# Full benchmark of 2026-11-26

| Job | Slurm | Phases | Files |
|---|---|---|---|
| FULLT1 | 2100092 | `time` (STADE C and A, JAX): 70 runs, 45 min | `FULLT1_time.json`, `plan_time.tsv`, `versions_time.json` |
| FULLM1 | 2100097 | `mem_primal`, `mem_adjoint` (STADE C and A, JAX): 126 runs, 54 min | `FULLM1_mem.json`, `plan_mem.tsv`, `versions_mem.json` |

`DISCUSSION.md` interprets the results. `REPORT.md` and `rows.json` come from both jobs. Rebuild them with:

```bash
python3 cluster/report.py --results results/full_2026-11-26/FULLT1_time.json --results results/full_2026-11-26/FULLM1_mem.json --out OUTDIR
```

Both jobs ran the same STADE source and the same kit (hash 5a6d98c18367b4a4). The recorded commits differ (d1663351b and 823fa703e), and the code folders do not (checked with `git diff`).

Result: 0 failed checks, 8 notes (six G6 notes for JAX allocator granularity on the 1e5 cases, two notes for the JAX out-of-memory at euler2d_step 16e6).

A later time job with STADE v0.6.39 is in `../full_2026-11-27/` (job FULLT3, with a refreshed `DISCUSSION.md`). The timing figures in this folder are for the code before the changes of the warp-level atomic adds.
