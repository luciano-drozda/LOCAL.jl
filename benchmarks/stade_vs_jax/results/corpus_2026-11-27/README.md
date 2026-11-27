# GPU corpus validation of 2026-11-27

`test/validate_corpus_gpu.jl` runs the whole corpus (`test/val-corpus`, 164 kernels) on the device: for each kernel, the adjoint and the HVP, with the CUDA backend and with the JACC backend, against the CPU result of the baseline. The count is 164 x 2 x 2 = 656 checks. The job driver is `cluster/corpus_driver.jl`, and `tools/build_corpus_job.py` builds the payload (the source `src/STADE.jl` and `test/` are bundled into the job).

| Job | Slurm | Call | Result |
|---|---|---|---|
| CORP1 | 2101831 | `fuse_ii_loops = false` (default options, `warp_runs = true`) | 652 of 656 checks passed, 4 not ok, 4426 s, krakengpu1 |

File: `CORP1_default.json` (the JSON line of the driver: host, seconds, exit code, call, summary, result lines, log tail). The run used STADE v0.6.39 (commit 8913babc8).

The four checks that are not ok are all `idm_dup_write` (adjoint and HVP, CUDA and JACC). This kernel is the documented counter-example of the plain-write contract: its source writes one element twice, and `test/validate_ownership.jl` expects the ownership check to flag it (`expected_flagged = ["idm_dup_write"]`). The gradient error on the device is 0.4 to 0.7 against 1e-16 on the CPU. No other kernel fails. The result lines in the JSON list the kernels more than once (the driver collects the progress lines and the final list), so count checks with the summary line and not with the number of lines.

The device run covers the three-way gate and the run helper only through what the corpus kernels contain. The structural test `test/validate_warp_atomics.jl` (1127 calls, 216 kernel plans) covers the shapes without a GPU.
