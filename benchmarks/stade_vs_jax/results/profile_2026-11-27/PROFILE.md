# Profile of the adjoint and the primal on a V100

- Job on krakengpu1.cluster (Slurm 2100561), 18.2 min. STADE commit f394a706a, kit a8f9bc53e2e9f2ce.
- Job on krakengpu1.cluster (Slurm 2100628), 14.0 min. STADE commit e955a38db, kit d96d762ed671d613.
- versions: cuda_jl 6.3.1, driver 12.4.0, jax 0.12.0, jaxlib 0.12.0, julia 1.11.9, numpy 2.5.4, platform_version cuda 12090, python 3.12.7, runtime 12.4.0

Each profile run records 3 calls of one case between `cuProfilerStart` and `cuProfilerStop`. Values per call are the total of the 3 calls divided by 3. The kernels of the reset between calls (`gpu_fill_kernel_`) are not counted. The host time of a call includes the cost of the profiler.

## 1. Device work per call

Kernel ms = the sum of the device times of all kernels of one call. Wall ms = the host time of one call during profiling. Busy = kernel ms over wall ms. Bench ms = the time of one call without a profiler (job FULLT1, host clock).

| kernel | size | call | framework | launches | kernel ms | wall ms | busy | bench ms | kernel ms / JAX kernel ms | launches / JAX launches |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp3 | 13M | primal | STADE C | 3 | 0.133 | 0.154 | 0.86 | 0.145 | 1.01 | 0.50 |
| mlp3 | 13M | primal | JAX | 6 | 0.132 | 7.536 | 0.02 | 0.217 | - | - |
| mlp3 | 13M | adjoint | STADE C | 5 | 0.600 | 0.733 | 0.82 | 0.720 | 1.70 | 0.42 |
| mlp3 | 13M | adjoint | STADE A | 5 | 0.491 | 0.512 | 0.96 | 0.609 | 1.39 | 0.42 |
| mlp3 | 13M | adjoint | JAX | 12 | 0.353 | 10.660 | 0.03 | 0.472 | - | - |
| mpnn_gather | 1e6 | primal | STADE C | 3 | 3.503 | 4.370 | 0.80 | 3.496 | 0.24 | 0.25 |
| mpnn_gather | 1e6 | primal | JAX | 12 | 14.695 | 20.802 | 0.71 | 14.074 | - | - |
| mpnn_gather | 1e6 | adjoint | STADE C | 5 | 40.110 | 39.699 | 1.01 | 39.689 | 1.12 | 0.14 |
| mpnn_gather | 1e6 | adjoint | STADE A | 5 | 39.412 | 39.550 | 1.00 | 39.586 | 1.10 | 0.14 |
| mpnn_gather | 1e6 | adjoint | JAX | 36 | 35.919 | 38.841 | 0.92 | 36.812 | - | - |
| swe2d_step | 1e6 | primal | STADE C | 1 | 0.136 | 0.184 | 0.74 | 0.156 | 0.22 | 0.10 |
| swe2d_step | 1e6 | primal | JAX | 10 | 0.623 | 12.465 | 0.05 | 0.681 | - | - |
| swe2d_step | 1e6 | adjoint | STADE C | 1 | 0.725 | 3.994 | 0.18 | 0.802 | 0.13 | 0.11 |
| swe2d_step | 1e6 | adjoint | JAX | 9 | 5.410 | 24.590 | 0.22 | 5.517 | - | - |
| transformer_block | d1024_T1024 | primal | STADE C | 15 | 6.711 | 14.130 | 0.47 | 6.312 | 1.12 | 0.60 |
| transformer_block | d1024_T1024 | primal | JAX | 25 | 5.970 | 10.693 | 0.56 | 6.115 | - | - |
| transformer_block | d1024_T1024 | adjoint | STADE C | 41 | 23.919 | 24.749 | 0.97 | 22.267 | 1.44 | 0.69 |
| transformer_block | d1024_T1024 | adjoint | STADE A | 39 | 21.926 | 24.688 | 0.89 | 20.999 | 1.32 | 0.66 |
| transformer_block | d1024_T1024 | adjoint | JAX | 59 | 16.633 | 35.352 | 0.47 | 17.392 | - | - |

## 2. Largest kernels

The 6 kernels with the largest device time of each profile run.


**mlp3 13M, primal, STADE C** (3 launches, 3 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_mlp3_2_s_(A, A, A, Int64, Int64, A)` | 1.0 | 0.0451 | 34% | 45.1 |
| `cuda_kernel_mlp3_1_s_(A, A, Int64, Int64, A, A)` | 1.0 | 0.0441 | 33% | 44.1 |
| `cuda_kernel_mlp3_3_s_(A, A, Int64, Int64, A, A)` | 1.0 | 0.0440 | 33% | 44.0 |

**mlp3 13M, primal, JAX** (6 launches, 4 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `input_reduce_fusion_1` | 2.0 | 0.0823 | 62% | 41.1 |
| `input_reduce_fusion_2` | 1.0 | 0.0416 | 32% | 41.6 |
| `loop_tanh_fusion_1` | 2.0 | 0.0057 | 4% | 2.9 |
| `wrapped_add` | 1.0 | 0.0020 | 2% | 2.0 |

**mlp3 13M, adjoint, STADE C** (5 launches, 5 distinct kernels, capture range, reset kernels per call 7)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_mlp3_b_3_s_(A, A, A, A, Int64, Int64, A, A,` | 1.0 | 0.1904 | 32% | 190.4 |
| `cuda_kernel_mlp3_b_5_s_(A, A, A, Int64, Int64, A, A, A,` | 1.0 | 0.1574 | 26% | 157.4 |
| `cuda_kernel_mlp3_b_4_s_(A, A, A, A, A, Int64, Int64, A` | 1.0 | 0.1572 | 26% | 157.2 |
| `cuda_kernel_mlp3_b_1_s_(A, A, Int64, Int64, A, A)` | 1.0 | 0.0500 | 8% | 50.0 |
| `cuda_kernel_mlp3_b_2_s_(A, A, A, Int64, Int64, A)` | 1.0 | 0.0447 | 7% | 44.7 |

**mlp3 13M, adjoint, STADE A** (5 launches, 5 distinct kernels, capture range, reset kernels per call 7)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_mlp3_b_4_s_(A, A, A, A, A, Int64, Int64, A,` | 1.0 | 0.1328 | 27% | 132.8 |
| `cuda_kernel_mlp3_b_5_s_(A, A, A, Int64, Int64, A, A, A,` | 1.0 | 0.1328 | 27% | 132.8 |
| `cuda_kernel_mlp3_b_3_s_(A, A, A, A, Int64, Int64, A, A,` | 1.0 | 0.1301 | 27% | 130.1 |
| `cuda_kernel_mlp3_b_1_s_(A, A, Int64, Int64, A, A)` | 1.0 | 0.0500 | 10% | 50.0 |
| `cuda_kernel_mlp3_b_2_s_(A, A, A, Int64, Int64, A)` | 1.0 | 0.0449 | 9% | 44.9 |

**mlp3 13M, adjoint, JAX** (12 launches, 11 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `input_reduce_fusion_1` | 2.0 | 0.0838 | 24% | 41.9 |
| `input_reduce_fusion_4` | 1.0 | 0.0491 | 14% | 49.1 |
| `input_transpose_fusion_1` | 1.0 | 0.0440 | 12% | 44.0 |
| `input_transpose_fusion_2` | 1.0 | 0.0433 | 12% | 43.3 |
| `input_reduce_fusion_2` | 1.0 | 0.0415 | 12% | 41.5 |
| `input_reduce_fusion_3` | 1.0 | 0.0414 | 12% | 41.4 |

**mpnn_gather 1e6, primal, STADE C** (3 launches, 3 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_mpnn_gather_2_(A, A, A, A, A, Int64, Int64, CuDeviceArr...` | 1.0 | 1.9314 | 55% | 1931.4 |
| `cuda_kernel_mpnn_gather_3_(A, A, A, A, Int64, Int64, Int64, A, CuDe...` | 1.0 | 0.9282 | 26% | 928.2 |
| `cuda_kernel_mpnn_gather_1_(A, A, A, Int64, Int64, Int64, A, A)` | 1.0 | 0.6433 | 18% | 643.3 |

**mpnn_gather 1e6, primal, JAX** (12 launches, 5 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `volta_dgemm_64x64_nn` | 6.0 | 5.6608 | 39% | 943.5 |
| `input_scatter_fusion` | 1.0 | 3.3619 | 23% | 3361.9 |
| `loop_add_fusion` | 1.0 | 2.7051 | 18% | 2705.1 |
| `void cublasLt::epilogue::impl::globalKernel<(int)8, (int)32, double...` | 3.0 | 2.6554 | 18% | 885.1 |
| `loop_broadcast_fusion` | 1.0 | 0.3122 | 2% | 312.2 |

**mpnn_gather 1e6, adjoint, STADE C** (5 launches, 5 distinct kernels, capture range, reset kernels per call 9)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_mpnn_gather_b_4_p_(A, A, A, A, A, CuDeviceArray<Float64...` | 1.0 | 15.0872 | 38% | 15087.2 |
| `cuda_kernel_mpnn_gather_b_3_p_(A, A, A, A, A, CuDeviceArray<Float64...` | 1.0 | 13.4530 | 34% | 13453.0 |
| `cuda_kernel_mpnn_gather_b_5_p_(A, A, A, A, Int64, Int64, Int64, A, ...` | 1.0 | 8.5239 | 21% | 8523.9 |
| `cuda_kernel_mpnn_gather_b_2_(A, A, A, A, A, Int64, Int64, CuDeviceA...` | 1.0 | 2.4397 | 6% | 2439.7 |
| `cuda_kernel_mpnn_gather_b_1_(A, A, A, Int64, Int64, Int64, A, A)` | 1.0 | 0.6065 | 2% | 606.5 |

**mpnn_gather 1e6, adjoint, STADE A** (5 launches, 5 distinct kernels, capture range, reset kernels per call 9)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_mpnn_gather_b_4_p_(A, A, A, A, A, CuDeviceArray<Float64...` | 1.0 | 14.9290 | 38% | 14929.0 |
| `cuda_kernel_mpnn_gather_b_3_p_(A, A, A, A, A, CuDeviceArray<Float64...` | 1.0 | 13.0564 | 33% | 13056.4 |
| `cuda_kernel_mpnn_gather_b_5_p_(A, A, A, A, Int64, Int64, Int64, A, ...` | 1.0 | 8.4007 | 21% | 8400.7 |
| `cuda_kernel_mpnn_gather_b_2_(A, A, A, A, A, Int64, Int64, CuDeviceA...` | 1.0 | 2.4321 | 6% | 2432.1 |
| `cuda_kernel_mpnn_gather_b_1_(A, A, A, Int64, Int64, Int64, A, A)` | 1.0 | 0.5934 | 2% | 593.4 |

**mpnn_gather 1e6, adjoint, JAX** (36 launches, 16 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `input_scatter_fusion` | 2.0 | 6.9409 | 19% | 3470.4 |
| `volta_dgemm_64x64_nn` | 6.0 | 5.3473 | 15% | 891.2 |
| `volta_dgemm_64x64_tn` | 6.0 | 5.3251 | 15% | 887.5 |
| `void dgemm_largek<(bool)0, (bool)1, (int)4, (int)4, (int)4, (int)3,...` | 5.0 | 4.6474 | 13% | 929.5 |
| `input_scatter_fusion_2` | 1.0 | 3.0604 | 9% | 3060.4 |
| `loop_select_fusion` | 1.0 | 2.9436 | 8% | 2943.6 |

**swe2d_step 1e6, primal, STADE C** (1 launches, 1 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_swe2d_step_1_(A, A, A, A, A, A, Int64)` | 1.0 | 0.1358 | 100% | 135.8 |

**swe2d_step 1e6, primal, JAX** (10 launches, 7 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `loop_concatenate_multiply_fusion` | 1.0 | 0.2114 | 34% | 211.4 |
| `loop_subtract_fusion_1` | 3.0 | 0.1435 | 23% | 47.8 |
| `loop_divide_fusion` | 1.0 | 0.0734 | 12% | 73.4 |
| `input_concatenate_fusion_2` | 1.0 | 0.0628 | 10% | 62.8 |
| `input_concatenate_fusion_4` | 2.0 | 0.0458 | 7% | 22.9 |
| `input_concatenate_fusion_1` | 1.0 | 0.0439 | 7% | 43.9 |

**swe2d_step 1e6, adjoint, STADE C** (1 launches, 1 distinct kernels, capture range, reset kernels per call 3)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `cuda_kernel_swe2d_step_b_1_(A, A, A, A, A, A, CuDevice` | 1.0 | 0.7252 | 100% | 725.2 |

**swe2d_step 1e6, adjoint, JAX** (9 launches, 8 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `loop_add_compare_concatenate_divide_maximum_minimum_multiply_subtra...` | 1.0 | 2.8356 | 52% | 2835.6 |
| `loop_add_fusion` | 1.0 | 2.1966 | 41% | 2196.6 |
| `input_concatenate_fusion_2` | 1.0 | 0.1496 | 3% | 149.6 |
| `loop_broadcast_fusion` | 1.0 | 0.0676 | 1% | 67.6 |
| `loop_divide_fusion` | 1.0 | 0.0518 | 1% | 51.8 |
| `input_concatenate_fusion_3` | 2.0 | 0.0452 | 1% | 22.6 |

**transformer_block d1024_T1024, primal, STADE C** (15 launches, 12 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `volta_dgemm_128x64_nn` | 4.0 | 4.8728 | 73% | 1218.2 |
| `cuda_kernel_transformer_block_6_t_(A, Int64, Int64, A, Int64, A, A,...` | 1.0 | 0.7212 | 11% | 721.2 |
| `cuda_kernel_transformer_block_4_t_(Int64, Int64, Int64, Int64, A, A...` | 1.0 | 0.6381 | 10% | 638.1 |
| `cuda_kernel_transformer_block_5_s_(A, Int64, A, Int64)` | 1.0 | 0.1653 | 2% | 165.3 |
| `cuda_kernel_transformer_block_10_fa_(A, Int64, Int64, A, Int64, A, ...` | 1.0 | 0.0896 | 1% | 89.6 |
| `cuda_kernel_transformer_block_3_fa_(A, Int64, Int64, A, Int64, A, A...` | 1.0 | 0.0653 | 1% | 65.3 |

**transformer_block d1024_T1024, primal, JAX** (25 launches, 20 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `volta_dgemm_128x64_nn` | 4.0 | 4.4252 | 74% | 1106.3 |
| `volta_dgemm_64x64_tn` | 1.0 | 0.4027 | 7% | 402.7 |
| `volta_dgemm_64x64_nn` | 1.0 | 0.4021 | 7% | 402.1 |
| `input_exponential_reduce_fusion` | 1.0 | 0.3366 | 6% | 336.6 |
| `loop_multiply_fusion` | 1.0 | 0.1115 | 2% | 111.5 |
| `void cublasLt::epilogue::impl::globalKernel<(int)8, (int)32, double...` | 2.0 | 0.0507 | 1% | 25.4 |

**transformer_block d1024_T1024, adjoint, STADE C** (41 launches, 31 distinct kernels, capture range, reset kernels per call 16)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `volta_dgemm_128x64_nn` | 5.0 | 6.5941 | 28% | 1318.8 |
| `volta_dgemm_128x64_nt` | 4.0 | 5.0896 | 21% | 1272.4 |
| `volta_dgemm_128x64_tn` | 4.0 | 4.7301 | 20% | 1182.5 |
| `cuda_kernel_transformer_block_b_6_t_(A, Int64, Int64, A, Int64, A, ...` | 1.0 | 0.7266 | 3% | 726.6 |
| `cuda_kernel_transformer_block_b_17_f_t_(A, Int64, Int64, A, A, Int6...` | 1.0 | 0.6805 | 3% | 680.5 |
| `cuda_kernel_transformer_block_b_19_a_(Int64, Int64, Int64, Int64, A...` | 1.0 | 0.6519 | 3% | 651.9 |

**transformer_block d1024_T1024, adjoint, STADE A** (39 launches, 30 distinct kernels, capture range, reset kernels per call 16)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `volta_dgemm_128x64_nn` | 4.0 | 5.0486 | 23% | 1262.1 |
| `volta_dgemm_128x64_nt` | 4.0 | 4.9435 | 23% | 1235.9 |
| `volta_dgemm_128x64_tn` | 4.0 | 4.6043 | 21% | 1151.1 |
| `cuda_kernel_transformer_block_b_6_t_(A, Int64, Int64, A, Int64, A, ...` | 1.0 | 0.7081 | 3% | 708.1 |
| `cuda_kernel_transformer_block_b_17_f_t_(A, Int64, Int64, A, A, Int6...` | 1.0 | 0.6642 | 3% | 664.2 |
| `cuda_kernel_transformer_block_b_19_a_(Int64, Int64, Int64, Int64, A...` | 1.0 | 0.6524 | 3% | 652.4 |

**transformer_block d1024_T1024, adjoint, JAX** (59 launches, 44 distinct kernels, capture range, reset kernels per call 0)

| kernel | per call | ms per call | share | avg us |
|---|---:|---:|---:|---:|
| `volta_dgemm_128x64_nt` | 4.0 | 4.5011 | 27% | 1125.3 |
| `volta_dgemm_128x64_nn` | 3.0 | 3.1414 | 19% | 1047.1 |
| `volta_dgemm_128x64_tn` | 3.0 | 2.6909 | 16% | 897.0 |
| `volta_dgemm_64x64_tn` | 3.0 | 2.3154 | 14% | 771.8 |
| `volta_dgemm_64x64_nn` | 2.0 | 0.8008 | 5% | 400.4 |
| `volta_dgemm_64x64_nt` | 2.0 | 0.7933 | 5% | 396.6 |

## 3. Counters of ncu

ncu runs each kernel alone with the GPU clocks locked, so its times differ from the times of the benchmark. Use the counters, and use the times only to compare kernels with each other. `atom` and `red` are the warp instructions (SASS) of atomic operations and of atomic reductions on global memory. Values are the mean over the launches that ncu recorded (the column says how many). Some launches of a profiled range may be missing: see the README of the job folder.


**mlp3 13M, adjoint, STADE C** (1 kernel launches recorded; ncu table: 11 rows, 1 distinct IDs, 1 launches)

| kernel | launches recorded | time us | atom warp inst | red warp inst | red warp inst (other counter) |
|---|---:|---:|---:|---:|---:|
| `cuda_kernel_mlp3_b_1_s_(A, A, Int64, Int64, A, Cu` | 1 | 48.4 | 0 | 0 | 0 |

**mpnn_gather 1e6, adjoint, STADE C** (2 kernel launches recorded; ncu table: 22 rows, 2 distinct IDs, 2 launches)

| kernel | launches recorded | time us | atom warp inst | red warp inst | red warp inst (other counter) |
|---|---:|---:|---:|---:|---:|
| `cuda_kernel_mpnn_gather_b_2_(A, A, A, CuDeviceArr` | 1 | 2434.3 | 0 | 0 | 0 |
| `cuda_kernel_mpnn_gather_b_1_(A, A, A, Int64, Int6` | 1 | 637.2 | 0 | 0 | 0 |

## 4. Checks

0 failed checks, 0 notes.

