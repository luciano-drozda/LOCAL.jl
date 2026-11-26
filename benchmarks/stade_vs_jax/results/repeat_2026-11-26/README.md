# Repeat of the full time job, 2026-11-26

Job FULLT2 (Slurm 2100137, V100, same node as FULLT1): the same plan as FULLT1, 70 runs, 45 min. 0 failed checks, 1 note (the JAX out-of-memory at euler2d_step 16e6).

| File | Content |
|---|---|
| `FULLT2_time.json`, `plan_time.tsv`, `versions_time.json` | the result, the plan and the versions of the job |
| `REPEAT.md`, `repeat_rows.json` | the comparison with FULLT1, case by case |

Rebuild the comparison with:

```bash
python3 cluster/repeat_report.py --first results/full_2026-11-26/FULLT1_time.json --second results/repeat_2026-11-26/FULLT2_time.json --out OUTDIR
```

Result: the median change of one case is 0.1 to 0.5 %. The largest change is 6.3 % (STADE primal, transformer_block d512_T512). The headline gaps are equal to three digits, and no case changes class. The discussion is in section 2 of `../full_2026-11-26/DISCUSSION.md`.
