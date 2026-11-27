# GPU corpus validation of 2026-11-27

`test/validate_corpus_gpu.jl` runs the whole corpus (`test/val-corpus`, 164 kernels) on the device: for each kernel, the adjoint and the HVP, with the CUDA backend and with the JACC backend, against the CPU result of the baseline. The count is 164 x 2 x 2 = 656 checks. The job driver is `cluster/corpus_driver.jl`, and `tools/build_corpus_job.py` builds the payload (the source `src/STADE.jl` and `test/` are bundled into the job).

| Job | Slurm | Call | Result |
|---|---|---|---|
| CORP1 | 2101831 | `fuse_ii_loops = false` (default options, `warp_runs = true`) | 652 of 656 checks passed, 4 not ok, 4426 s, krakengpu1 |
| CORP2 | 2101867 | `fuse_ii_loops = true, run_jacc = false` | CUDA run: 326 of 328 checks passed, 2 not ok; 4140 s |
| CORP3 | 2101891 | `warp_runs = false, run_jacc = false` | CUDA run: 326 of 328 checks passed, 2 not ok; 3924 s |

Files: `CORP1_default.json`, `CORP2_fuse_ii.json`, `CORP3_warp_runs_false.json` (the JSON line of the driver: host, seconds, exit code, call, summary, result lines, log tail). All runs used STADE v0.6.39 (commit 8913babc8) on krakengpu1.

CORP2 and CORP3 each hold two summary lines. The first (652 of 656) is the default run that the include started (see below), and the second is the requested call (CUDA only, 164 x 2 = 328 checks). `fuse_ii_loops = true` is the option of the default configuration C of the benchmark. `warp_runs = false` sends the `:S` and `:X` adds back to the full-warp helper, so CORP3 tests the switch on the device.

In every run the checks that are not ok are `idm_dup_write` only (adjoint and HVP, CUDA and JACC; CUDA only in the requested runs of CORP2 and CORP3). This kernel is the documented counter-example of the plain-write contract: its source writes one element twice, and `test/validate_ownership.jl` expects the ownership check to flag it (`expected_flagged = ["idm_dup_write"]`). The gradient error on the device is 0.4 to 0.7 against 1e-16 on the CPU. No other kernel fails. The JSON lists 1312 result lines and two summary lines, because `test/validate_corpus_gpu.jl` ends with a bare call `validate_corpus_gpu()`. Including the file ran the corpus once with the default options, and the driver then ran the requested call. In CORP1 both runs used the default options, so the file holds two independent default runs, and both gave 652 of 656. The driver was changed afterwards so that the include does not run the corpus (`cluster/corpus_driver.jl`, not yet used in a job). CORP2 and CORP3 each hold one default run first and the requested run second.

The device run covers the three-way gate and the run helper only through what the corpus kernels contain. The structural test `test/validate_warp_atomics.jl` (1127 calls, 216 kernel plans) covers the shapes without a GPU.
