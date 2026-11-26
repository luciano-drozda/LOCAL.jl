# Option ablation of 2026-11-26

Job ABL1 (Slurm 2100116, V100, 46 min): 36 STADE runs, time phase, 9 kernels at all sizes, configs C, P, G and A. C = defaults, P = `primal_outputs = false`, G = `fresh_gradient = true`, A = P and G together.

| File | Content |
|---|---|
| `ABL1_time.json` | the result of the job |
| `plan.tsv`, `versions.json` | the plan and the versions of the job |
| `ABLATION.md`, `ablation_rows.json` | the report |

Rebuild the report with:

```bash
python3 cluster/ablation_report.py --results results/ablation_2026-11-26/ABL1_time.json --ref results/full_2026-11-26/FULLT1_time.json --stade-dir work/abl/stade/all --out OUTDIR
```

`work/abl/stade/all` holds the generated scripts. Generate them with `CONFIGS="C A P G" bash cluster/gen_stade_scripts.sh work/abl/stade "heat1d_multi heat3d_step heat3d_multi mlp3 mlp3_push ufv_gather_step transformer_block wave2d_step unet2"`. Without `--stade-dir`, the report leaves out the table of configs with the same code.

Result: 0 failed checks, 0 notes. `primal_outputs = false` gives most of the gain of config A. `fresh_gradient` gives the rest for mlp3, mlp3_push, wave2d_step and heat3d_step. The discussion is in section 4.3 of `../full_2026-11-26/DISCUSSION.md`.
