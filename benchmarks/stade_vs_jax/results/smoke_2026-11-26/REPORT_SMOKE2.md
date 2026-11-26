# STADE against JAX on a V100 (smoke 2)

- Job on krakengpu1.cluster (Slurm 2100088), 20.8 min. STADE commit ef9a194f8, kit 14bbad85bccd001e.
- stade: cuda_jl 6.3.1, driver 12.4.0, julia 1.11.9, runtime 12.4.0
- jax: driver 550.90.12, jax 0.12.0, jaxlib 0.12.0, numpy 2.5.4, platform_version cuda 12090, python 3.12.7

STADE config C = defaults. Config A = primal_outputs = false with fresh_gradient. JAX = jit. Gap = STADE / JAX. Headline STADE timer: host clock with device synchronization. Failed checks are listed in section 7.

## 1. Runtime of the primal

| kernel | size | STADE ms | STADE events ms | JAX ms | gap |
|---|---|---:|---:|---:|---:|
| heat3d_step | 1e6 | 0.0582 | 0.0536 | 0.0985 | 0.59 |
| mpnn_gather | 1e6 | 3.4984 | 3.3859 | 14.0637 | 0.25 |
| mpnn_gather | 1e5 | 0.3650 | 0.3644 | 1.6311 | 0.22 |
| heat3d_step | 1e5 | 0.0180 | 0.0149 | 0.0911 | 0.20 |

4 cases. Geometric-mean gap **0.284**. STADE slower than JAX in 0 cases.

## 2. Runtime of the adjoint

| kernel | size | STADE C ms | STADE C events ms | STADE best ms | JAX ms | gap C | gap best |
|---|---|---:|---:|---:|---:|---:|---:|
| mpnn_gather | 1e6 | 39.697 | 39.581 | 39.587 | 36.832 | 1.08 | 1.07 |
| mpnn_gather | 1e5 | 4.143 | 4.031 | 4.124 | 4.357 | 0.95 | 0.95 |
| heat3d_step | 1e6 | 0.109 | 0.105 | 0.071 | 0.149 | 0.73 | 0.48 |
| heat3d_step | 1e5 | 0.029 | 0.026 | 0.022 | 0.116 | 0.25 | 0.19 |

4 cases. Geometric-mean gap: C **0.660**, best **0.551**. STADE slower than JAX: C 1, best 1.

## 3. Ratio adjoint time over primal time

| kernel | size | STADE C | JAX |
|---|---|---:|---:|
| heat3d_step | 1e5 | 1.62 | 1.27 |
| heat3d_step | 1e6 | 1.88 | 1.51 |
| mpnn_gather | 1e5 | 11.35 | 2.67 |
| mpnn_gather | 1e6 | 11.35 | 2.62 |

## 4. Memory of the primal

Only the inputs are on the device. Two readings. **Exact**: STADE = arrays alive in the pool + bytes that one call allocates; JAX = XLA live-bytes peak (`JAX own`). **Driver**: device memory in use above the baseline, as nvidia-smi shows it. The driver reading is rounded up by the allocators (the CUDA.jl pool chunk is 32 MiB = 33.6 MB, and the XLA pool grows in regions). It also holds one-time library memory. Use the exact gap for small cases. A driver gap is computed only if both readings are at least 34 MB.

| kernel | size | inputs MB | STADE exact MB | JAX own MB | gap exact | STADE driver MB | JAX driver MB | gap driver |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heat3d_step | 1e5 | 1.6 | 1.6 | 2.9 | 0.54 | 34 | 4 | - |
| heat3d_step | 1e6 | 16.0 | 16.0 | 24.8 | 0.65 | 34 | 34 | - |
| mpnn_gather | 1e5 | 88.8 | 88.8 | 477.5 | 0.19 | 101 | 577 | 0.17 |
| mpnn_gather | 1e6 | 888.0 | 888.0 | 3719.6 | 0.24 | 906 | 4494 | 0.20 |

Geometric-mean primal memory gap: exact **0.35** over 4 cases, driver **0.19** over 2 cases.

## 5. Memory of the adjoint

Only the inputs, the gradient buffers and the stacks (STADE) or the cotangents (JAX) are on the device. Exact and driver readings as in section 4.

| kernel | size | STADE stacks MB | STADE C exact MB | STADE A exact MB | JAX own MB | gap C exact | gap A exact | STADE C driver MB | STADE A driver MB | JAX driver MB | gap C driver |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heat3d_step | 1e5 | 0.0 | 3.1 | 3.1 | 4.4 | 0.70 | 0.70 | 34 | 34 | 4 | - |
| heat3d_step | 1e6 | 0.0 | 32.0 | 32.0 | 41.6 | 0.77 | 0.77 | 34 | 34 | 67 | - |
| mpnn_gather | 1e5 | 38.4 | 210.5 | 210.5 | 516.3 | 0.41 | 0.41 | 235 | 235 | 577 | 0.41 |
| mpnn_gather | 1e6 | 384.0 | 2104.1 | 2104.1 | 4176.4 | 0.50 | 0.50 | 2114 | 2114 | 8789 | 0.24 |

Geometric-mean adjoint memory gap: exact C **0.58**, exact A **0.58** over 4 cases; driver C **0.31** over 2 cases.

## 6. Summary

- Primal runtime: gap 0.284, STADE slower in 0 of 4 cases.
- Adjoint runtime: gap C 0.660, best 0.551; STADE C slower in 1 of 4 cases.
- Primal memory: exact gap 0.35 over 4 cases; driver gap 0.19 over 2 cases.
- Adjoint memory: exact gap C 0.58, A 0.58 over 4 cases; driver gap C 0.31 over 2 cases.

## 7. Checks and invalid cases

0 failed checks, 2 notes.

- NOTE G6: JAX heat3d_step bench_1e5 mem_adjoint: mem_inputs 0.0 MB below input 1.6 MB
- NOTE G6: JAX heat3d_step bench_1e5 mem_primal: mem_inputs 0.0 MB below input 1.6 MB

## 8. Limits

- One GPU model (V100, 16 GB), float64, batch size one. JAX appears as jit only.
- The two runtime timers differ. The caches are not flushed, and STADE copies its inputs between calls.
- The driver readings are rounded up by the allocators and hold one-time library memory. The exact readings are not. JAX driver readings come from nvidia-smi, and the JAX pool never shrinks.
- One job gives one run. The noise limit of 5 % is a working value. The STADE checkpoint option is off.
