# Profile of four cases, 2026-11-27

Four cases were profiled on one V100 (float64): mpnn_gather 1e6, transformer_block d1024_T1024, mlp3 13M and swe2d_step 1e6. Each run records 3 calls between `cuProfilerStart` and `cuProfilerStop`.

| Job | Slurm | Tool | Files | Result |
|---|---|---|---|---|
| PROF3 | 2100561 | `nsys` (STADE C and A, JAX): 19 runs, 18 min | `PROF3_nsys.json`, `plan_nsys.tsv`, `versions_nsys.json` | Complete. Per-kernel device times for all 19 runs. |
| PROF8 | 2100628 | `ncu`, application replay (STADE C, mpnn_gather 1e6 and mlp3 13M) | `PROF8_ncu_application_replay.json`, `plan_ncu.tsv` | Partial. ncu recorded only the first 1 launch (mlp3) and the first 2 launches (mpnn_gather) of the range. |
| PROF9 | 2100641 | `ncu`, kernel replay, one-pass counters | `PROF9_ncu_kernel_replay.json` | Failed. The second kernel of the range gets `LaunchFailed` (exit 9). |

`PROF3_nsys.json` holds the 19 nsys runs of job PROF3. The two ncu runs of that job failed (a table that was cut, and `--launch-count` stopped the program), so they are not in the file.

Earlier jobs that did not give a result: PROF1 (2100559, `nsys` was not on the PATH of the job), PROF2 (2100560, stale STADE scripts and JAX kernels inside CUDA graphs), PROF4 (2100564), PROF5 (2100583), PROF6 (2100590) and PROF7 (2100606). PROF4 to PROF7 were ncu attempts. Their causes are in the commits v0.6.19 to v0.6.24 of this repository.

## ncu on the STADE kernels: what failed

- Kernel replay (the default) ends with `LaunchFailed` and exit code 9 for the second kernel of the range, with 11 metrics and with 4 metrics that fit in one pass (PROF4, PROF5, PROF9).
- Application replay (PROF8) runs the program once for each pass and records only the first 1 or 2 kernels of the range. The program ends with exit 0, and the ncu log has no message.
- The cause is not known. The kernels profiled without error show 0 atomic instructions, which is correct for them (no atomic in their code).
- No counter exists for the three kernels of mpnn_gather that use atomics. The discussion (section 4.5) uses the source of these kernels and their `nsys` times instead.

## Files

- `PROFILE.md` and `prof_rows.json`: the report, built with
  ```bash
  python3 cluster/prof_report.py --results results/profile_2026-11-27/PROF3_nsys.json results/profile_2026-11-27/PROF8_ncu_application_replay.json \
      --time-ref results/full_2026-11-26/FULLT1_time.json --out OUTDIR
  ```
- The recorded STADE commits are f394a706a (PROF3), e955a38db (PROF8) and 74a47e082 (PROF9). The kit hashes are in the `versions_*.json` files and in the job files.
  The STADE source (`src/STADE.jl`) is the same in all of them. Only the profile kit and the runner changed between the jobs.
- Reading the table of kernels: the `nsys` summary lists each kernel once. The kernels of the reset between calls (`gpu_fill_kernel_`) are not counted in the calls.
- The host time of a call during profiling includes the cost of the profiler. For JAX it is 1.1 to 35 times the time of the benchmark (`nsys --cuda-graph-trace=node`). Use the column "bench ms" for the time of a call, and use the kernel times for the device work.
