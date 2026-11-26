# Smoke jobs of 2026-11-26

Two one-kernel jobs on the V100, to test the harness before the full benchmark. They are tests of the harness. They are not benchmark results.

| Job | Slurm | Content | Status |
|---|---|---|---|
| `SMOKE1.json` | 2099932 | `heat3d_step`, sizes 1e5 and 1e6, all phases | Kit commit 5b92f1a. It ran before the memory fixes: its memory readings are not valid (rounding to the pool chunk, one-time memory at the first size). Its runtime values are valid. |
| `SMOKE2.json` | 2100088 | `heat3d_step` and `mpnn_gather`, sizes 1e5 and 1e6, all phases | Kit commit [v0.6.5] and later. `REPORT_SMOKE2.md` is its report. |

Build the report again with `python3 cluster/report.py --results results/smoke_2026-11-26/SMOKE2.json --out OUTDIR`.
