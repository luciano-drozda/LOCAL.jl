# STADE against JAX on a V100 full 2026-11-26 (FULLT1 time, FULLM1 memory)

- Job on krakengpu1.cluster (Slurm 2100092), 45.1 min. STADE commit d1663351b, kit 5a6d98c18367b4a4.
- Job on krakengpu1.cluster (Slurm 2100097), 54.4 min. STADE commit 823fa703e, kit 5a6d98c18367b4a4.
- stade: cuda_jl 6.3.1, driver 12.4.0, julia 1.11.9, runtime 12.4.0
- jax: driver 550.90.12, jax 0.12.0, jaxlib 0.12.0, numpy 2.5.4, platform_version cuda 12090, python 3.12.7

STADE config C = defaults. Config A = primal_outputs = false with fresh_gradient. JAX = jit. Gap = STADE / JAX. Headline STADE timer: host clock with device synchronization. Failed checks are listed in section 7.

## 1. Runtime of the primal

| kernel | size | STADE ms | STADE events ms | JAX ms | gap |
|---|---|---:|---:|---:|---:|
| unet2 | c16_n256 | 2.5240 | 2.4083 | 1.7497 | 1.44 |
| unet2 | c32_n128 | 2.2575 | 2.1485 | 1.6034 | 1.41 |
| unet2 | c64_n64 | 2.4384 | 2.3279 | 1.9190 | 1.27 |
| transformer_block | d1024_T1024 | 6.3121 | 6.2563 | 6.1152 | 1.03 |
| unet2 | c128_n32 | 3.1545 | 3.0429 | 3.0794 | 1.02 |
| transformer_block | d2048_T1024 | 18.9946 | 18.9340 | 18.6974 | 1.02 |
| mlp3_push | 79M | 0.8722 | 0.7629 | 0.8768 | 0.99 |
| transformer_block | d512_T512 | 1.1613 | 1.0513 | 1.1695 | 0.99 |
| mlp3 | 79M | 0.8425 | 0.7332 | 0.8625 | 0.98 |
| ufv_gather_step | 4e6 | 2.2906 | 2.1801 | 2.3533 | 0.97 |
| wave2d_step | 16e6 | 0.7562 | 0.6457 | 0.8130 | 0.93 |
| heat3d_multi | 1e6 | 0.1815 | 0.1770 | 0.2073 | 0.88 |
| ufv_gather_step | 1e6 | 0.2945 | 0.2899 | 0.3541 | 0.83 |
| heat3d_step | 8e6 | 0.2950 | 0.2910 | 0.3616 | 0.82 |
| burgers2d_step | 16e6 | 0.7927 | 0.6817 | 1.0150 | 0.78 |
| mlp3_push | 13M | 0.1574 | 0.1536 | 0.2287 | 0.69 |
| ufv_scatter_step | 1e6 | 0.5487 | 0.4398 | 0.7992 | 0.69 |
| mlp3 | 13M | 0.1449 | 0.1403 | 0.2167 | 0.67 |
| heat3d_step | 1e6 | 0.0567 | 0.0528 | 0.0989 | 0.57 |
| mlp3 | 3M | 0.0682 | 0.0635 | 0.1310 | 0.52 |
| mlp3_push | 3M | 0.0662 | 0.0625 | 0.1297 | 0.51 |
| heat1d_multi | 1e5_t8 | 0.0808 | 0.0768 | 0.1614 | 0.50 |
| wave2d_step | 1e6 | 0.0622 | 0.0578 | 0.1272 | 0.49 |
| ufv_scatter_step | 4e6 | 3.4720 | 3.3651 | 7.1946 | 0.48 |
| burgers2d_step | 1e6 | 0.0674 | 0.0630 | 0.1461 | 0.46 |
| heat1d_multi | 1e6_t32 | 1.6240 | 1.5142 | 3.7411 | 0.43 |
| ufv_scatter_step | 1e5 | 0.0432 | 0.0378 | 0.1054 | 0.41 |
| heat1d_multi | 1e6_t8 | 0.3986 | 0.3941 | 1.0109 | 0.39 |
| euler2d_step | 1e6 | 0.5766 | 0.4668 | 1.5038 | 0.38 |
| heat1d_multi | 1e6_t2 | 0.1210 | 0.1168 | 0.3243 | 0.37 |
| euler2d_step | 1e5 | 0.0693 | 0.0655 | 0.1918 | 0.36 |
| heat3d_multi | 1e5 | 0.0397 | 0.0365 | 0.1119 | 0.35 |
| ufv_gather_step | 1e5 | 0.0335 | 0.0292 | 0.0973 | 0.34 |
| mpnn_gather | 1e6 | 3.4956 | 3.3858 | 14.0742 | 0.25 |
| mpnn_gather | 1e5 | 0.3697 | 0.3668 | 1.5741 | 0.23 |
| swe2d_step | 1e6 | 0.1565 | 0.1515 | 0.6809 | 0.23 |
| swe2d_step | 1e5 | 0.0339 | 0.0290 | 0.1482 | 0.23 |
| swe2d_step | 16e6 | 1.9870 | 1.8769 | 8.8504 | 0.22 |
| heat3d_step | 1e5 | 0.0182 | 0.0148 | 0.0907 | 0.20 |
| burgers2d_step | 1e5 | 0.0203 | 0.0166 | 0.1033 | 0.20 |
| wave2d_step | 1e5 | 0.0170 | 0.0135 | 0.0948 | 0.18 |

41 cases. Geometric-mean gap **0.533**. STADE slower than JAX in 6 cases.

## 2. Runtime of the adjoint

| kernel | size | STADE C ms | STADE C events ms | STADE best ms | JAX ms | gap C | gap best |
|---|---|---:|---:|---:|---:|---:|---:|
| mlp3 | 13M | 0.720 | 0.609 | 0.609 | 0.472 | 1.52 | 1.29 |
| ufv_gather_step | 1e6 | 1.637 | 1.529 | 1.444 | 1.149 | 1.43 | 1.26 |
| transformer_block | d1024_T1024 | 22.267 | 22.150 | 20.999 | 17.392 | 1.28 | 1.21 |
| transformer_block | d2048_T1024 | 66.879 | 66.772 | 61.692 | 52.733 | 1.27 | 1.17 |
| unet2 | c16_n256 | 13.260 | 13.156 | 13.237 | 11.359 | 1.17 | 1.17 |
| mlp3_push | 79M | 3.193 | 3.082 | 2.477 | 2.164 | 1.48 | 1.14 |
| mpnn_gather | 1e6 | 39.689 | 39.581 | 39.586 | 36.812 | 1.08 | 1.08 |
| transformer_block | d512_T512 | 3.779 | 3.668 | 3.589 | 3.397 | 1.11 | 1.06 |
| mlp3 | 79M | 3.125 | 3.012 | 2.251 | 2.138 | 1.46 | 1.05 |
| ufv_gather_step | 4e6 | 11.142 | 11.030 | 9.924 | 9.562 | 1.17 | 1.04 |
| mpnn_gather | 1e5 | 4.143 | 4.034 | 4.123 | 4.163 | 1.00 | 0.99 |
| mlp3 | 3M | 0.270 | 0.265 | 0.237 | 0.244 | 1.11 | 0.97 |
| mlp3_push | 13M | 0.682 | 0.570 | 0.455 | 0.486 | 1.41 | 0.94 |
| unet2 | c128_n32 | 7.144 | 7.034 | 7.123 | 7.811 | 0.91 | 0.91 |
| heat3d_multi | 1e6 | 0.383 | 0.380 | 0.239 | 0.262 | 1.46 | 0.91 |
| unet2 | c32_n128 | 7.999 | 7.887 | 7.989 | 8.808 | 0.91 | 0.91 |
| unet2 | c64_n64 | 6.661 | 6.550 | 6.653 | 7.743 | 0.86 | 0.86 |
| heat1d_multi | 1e6_t32 | 3.132 | 3.022 | 1.630 | 1.910 | 1.64 | 0.85 |
| wave2d_step | 16e6 | 2.874 | 2.765 | 2.020 | 2.375 | 1.21 | 0.85 |
| heat3d_step | 8e6 | 0.788 | 0.678 | 0.388 | 0.475 | 1.66 | 0.82 |
| euler2d_step | 1e5 | 0.528 | 0.418 | 0.375 | 0.492 | 1.07 | 0.76 |
| heat1d_multi | 1e6_t8 | 0.905 | 0.794 | 0.416 | 0.592 | 1.53 | 0.70 |
| heat1d_multi | 1e5_t8 | 0.162 | 0.155 | 0.092 | 0.136 | 1.19 | 0.68 |
| mlp3_push | 3M | 0.190 | 0.185 | 0.165 | 0.253 | 0.75 | 0.65 |
| burgers2d_step | 16e6 | 2.653 | 2.542 | 1.660 | 2.888 | 0.92 | 0.58 |
| wave2d_step | 1e6 | 0.200 | 0.197 | 0.145 | 0.254 | 0.79 | 0.57 |
| heat3d_multi | 1e5 | 0.081 | 0.077 | 0.056 | 0.101 | 0.80 | 0.55 |
| euler2d_step | 1e6 | 3.769 | 3.656 | 3.392 | 6.151 | 0.61 | 0.55 |
| ufv_scatter_step | 1e6 | 1.442 | 1.332 | 1.001 | 1.818 | 0.79 | 0.55 |
| heat1d_multi | 1e6_t2 | 0.242 | 0.237 | 0.137 | 0.255 | 0.95 | 0.54 |
| heat3d_step | 1e6 | 0.115 | 0.111 | 0.071 | 0.149 | 0.77 | 0.48 |
| burgers2d_step | 1e6 | 0.191 | 0.188 | 0.128 | 0.279 | 0.69 | 0.46 |
| ufv_gather_step | 1e5 | 0.070 | 0.066 | 0.054 | 0.124 | 0.56 | 0.44 |
| swe2d_step | 1e5 | 0.090 | 0.085 | 0.086 | 0.213 | 0.42 | 0.40 |
| ufv_scatter_step | 4e6 | 9.427 | 9.321 | 5.944 | 15.381 | 0.61 | 0.39 |
| ufv_scatter_step | 1e5 | 0.088 | 0.084 | 0.058 | 0.165 | 0.54 | 0.35 |
| burgers2d_step | 1e5 | 0.035 | 0.032 | 0.030 | 0.124 | 0.28 | 0.24 |
| heat3d_step | 1e5 | 0.030 | 0.027 | 0.022 | 0.103 | 0.29 | 0.22 |
| swe2d_step | 16e6 | 10.782 | 10.673 | 9.602 | 47.389 | 0.23 | 0.20 |
| wave2d_step | 1e5 | 0.028 | 0.025 | 0.024 | 0.123 | 0.23 | 0.20 |
| swe2d_step | 1e6 | 0.802 | 0.691 | 0.730 | 5.517 | 0.15 | 0.13 |

41 cases. Geometric-mean gap: C **0.836**, best **0.643**. STADE slower than JAX: C 19, best 10.

## 3. Ratio adjoint time over primal time

| kernel | size | STADE C | JAX |
|---|---|---:|---:|
| heat3d_step | 1e5 | 1.62 | 1.13 |
| heat3d_step | 1e6 | 2.02 | 1.51 |
| heat3d_step | 8e6 | 2.67 | 1.31 |
| heat3d_multi | 1e5 | 2.05 | 0.91 |
| heat3d_multi | 1e6 | 2.11 | 1.27 |
| heat1d_multi | 1e5_t8 | 2.00 | 0.84 |
| heat1d_multi | 1e6_t2 | 2.00 | 0.79 |
| heat1d_multi | 1e6_t8 | 2.27 | 0.59 |
| heat1d_multi | 1e6_t32 | 1.93 | 0.51 |
| wave2d_step | 1e5 | 1.63 | 1.30 |
| wave2d_step | 1e6 | 3.22 | 2.00 |
| wave2d_step | 16e6 | 3.80 | 2.92 |
| burgers2d_step | 1e5 | 1.73 | 1.20 |
| burgers2d_step | 1e6 | 2.84 | 1.91 |
| burgers2d_step | 16e6 | 3.35 | 2.84 |
| swe2d_step | 1e5 | 2.64 | 1.43 |
| swe2d_step | 1e6 | 5.12 | 8.10 |
| swe2d_step | 16e6 | 5.43 | 5.35 |
| euler2d_step | 1e5 | 7.61 | 2.56 |
| euler2d_step | 1e6 | 6.54 | 4.09 |
| ufv_scatter_step | 1e5 | 2.04 | 1.56 |
| ufv_scatter_step | 1e6 | 2.63 | 2.27 |
| ufv_scatter_step | 4e6 | 2.72 | 2.14 |
| ufv_gather_step | 1e5 | 2.08 | 1.27 |
| ufv_gather_step | 1e6 | 5.56 | 3.24 |
| ufv_gather_step | 4e6 | 4.86 | 4.06 |
| mlp3 | 3M | 3.95 | 1.86 |
| mlp3 | 13M | 4.97 | 2.18 |
| mlp3 | 79M | 3.71 | 2.48 |
| mlp3_push | 3M | 2.88 | 1.95 |
| mlp3_push | 13M | 4.34 | 2.12 |
| mlp3_push | 79M | 3.66 | 2.47 |
| mpnn_gather | 1e5 | 11.21 | 2.64 |
| mpnn_gather | 1e6 | 11.35 | 2.62 |
| transformer_block | d512_T512 | 3.25 | 2.90 |
| transformer_block | d1024_T1024 | 3.53 | 2.84 |
| transformer_block | d2048_T1024 | 3.52 | 2.82 |
| unet2 | c16_n256 | 5.25 | 6.49 |
| unet2 | c32_n128 | 3.54 | 5.49 |
| unet2 | c64_n64 | 2.73 | 4.03 |
| unet2 | c128_n32 | 2.26 | 2.54 |

## 4. Memory of the primal

Only the inputs are on the device. Two readings. **Exact**: STADE = arrays alive in the pool + bytes that one call allocates; JAX = XLA live-bytes peak (`JAX own`). **Driver**: device memory in use above the baseline, as nvidia-smi shows it. The driver reading is rounded up by the allocators (the CUDA.jl pool chunk is 32 MiB = 33.6 MB, and the XLA pool grows in regions). It also holds one-time library memory. Use the exact gap for small cases. A driver gap is computed only if both readings are at least 34 MB.

| kernel | size | inputs MB | STADE exact MB | JAX own MB | gap exact | STADE driver MB | JAX driver MB | gap driver |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heat3d_step | 1e5 | 1.6 | 1.6 | 2.9 | 0.54 | 34 | 4 | - |
| heat3d_step | 1e6 | 16.0 | 16.0 | 24.8 | 0.65 | 34 | 34 | - |
| heat3d_step | 8e6 | 128.0 | 128.0 | 198.2 | 0.65 | 134 | 268 | 0.50 |
| heat3d_multi | 1e5 | 1.6 | 1.6 | 4.4 | 0.35 | 34 | 4 | - |
| heat3d_multi | 1e6 | 16.0 | 16.0 | 41.6 | 0.39 | 34 | 67 | - |
| heat1d_multi | 1e5_t8 | 1.6 | 1.6 | 4.5 | 0.36 | 34 | 4 | - |
| heat1d_multi | 1e6_t2 | 16.0 | 16.0 | 41.6 | 0.39 | 34 | 67 | - |
| heat1d_multi | 1e6_t8 | 16.0 | 16.0 | 41.6 | 0.39 | 34 | 69 | - |
| heat1d_multi | 1e6_t32 | 16.0 | 16.0 | 41.6 | 0.39 | 34 | 69 | - |
| wave2d_step | 1e5 | 3.2 | 3.2 | 4.5 | 0.71 | 34 | 4 | - |
| wave2d_step | 1e6 | 32.0 | 32.0 | 41.6 | 0.77 | 34 | 67 | - |
| wave2d_step | 16e6 | 512.0 | 512.0 | 664.9 | 0.77 | 537 | 1074 | 0.50 |
| burgers2d_step | 1e5 | 3.2 | 3.2 | 5.3 | 0.60 | 34 | 4 | - |
| burgers2d_step | 1e6 | 32.0 | 32.0 | 49.6 | 0.65 | 34 | 67 | - |
| burgers2d_step | 16e6 | 512.0 | 512.0 | 792.9 | 0.65 | 537 | 1074 | 0.50 |
| swe2d_step | 1e5 | 4.8 | 4.8 | 14.7 | 0.33 | 34 | 13 | - |
| swe2d_step | 1e6 | 48.0 | 48.0 | 134.2 | 0.36 | 67 | 134 | 0.50 |
| swe2d_step | 16e6 | 768.0 | 768.0 | 1969.7 | 0.39 | 772 | 2147 | 0.36 |
| euler2d_step | 1e5 | 6.4 | 6.4 | 43.8 | 0.15 | 34 | 46 | - |
| euler2d_step | 1e6 | 64.0 | 64.0 | 233.3 | 0.27 | 67 | 268 | 0.25 |
| euler2d_step | 16e6 | 1024.0 | 1024.0 | 3249.7 | 0.32 | 1040 | 4297 | 0.24 |
| ufv_scatter_step | 1e5 | 7.2 | 7.2 | 11.9 | 0.60 | 34 | 17 | - |
| ufv_scatter_step | 1e6 | 72.0 | 72.0 | 107.5 | 0.67 | 101 | 161 | 0.62 |
| ufv_scatter_step | 4e6 | 288.0 | 288.0 | 430.0 | 0.67 | 302 | 640 | 0.47 |
| ufv_gather_step | 1e5 | 8.0 | 8.0 | 10.3 | 0.78 | 34 | 15 | - |
| ufv_gather_step | 1e6 | 80.0 | 80.0 | 91.5 | 0.87 | 101 | 159 | 0.63 |
| ufv_gather_step | 4e6 | 320.0 | 320.0 | 366.0 | 0.87 | 336 | 635 | 0.53 |
| mlp3 | 3M | 25.2 | 25.2 | 25.2 | 1.00 | 34 | 34 | - |
| mlp3 | 13M | 100.8 | 100.8 | 100.8 | 1.00 | 134 | 134 | 1.00 |
| mlp3 | 79M | 629.4 | 629.4 | 749.6 | 0.84 | 638 | 1076 | 0.59 |
| mlp3_push | 3M | 25.2 | 25.2 | 25.3 | 1.00 | 34 | 34 | - |
| mlp3_push | 13M | 100.8 | 100.8 | 100.9 | 1.00 | 134 | 134 | 1.00 |
| mlp3_push | 79M | 629.5 | 629.5 | 749.8 | 0.84 | 638 | 1074 | 0.59 |
| mpnn_gather | 1e5 | 88.8 | 88.8 | 477.5 | 0.19 | 101 | 577 | 0.17 |
| mpnn_gather | 1e6 | 888.0 | 888.0 | 3719.6 | 0.24 | 906 | 4494 | 0.20 |
| transformer_block | d512_T512 | 69.3 | 88.2 | 234.9 | 0.38 | 145 | 277 | 0.52 |
| transformer_block | d1024_T1024 | 344.2 | 419.7 | 717.5 | 0.58 | 614 | 1095 | 0.56 |
| transformer_block | d2048_T1024 | 889.7 | 1040.7 | 1598.5 | 0.65 | 1386 | 2179 | 0.64 |
| unet2 | c16_n256 | 61.8 | 61.8 | 484.5 | 0.13 | 67 | 543 | 0.12 |
| unet2 | c32_n128 | 33.4 | 33.4 | 283.5 | 0.12 | 34 | 543 | - |
| unet2 | c64_n64 | 29.7 | 29.7 | 270.5 | 0.11 | 36 | 279 | 0.13 |
| unet2 | c128_n32 | 67.5 | 67.5 | 270.5 | 0.25 | 101 | 279 | 0.36 |

Geometric-mean primal memory gap: exact **0.47** over 42 cases, driver **0.42** over 23 cases.

## 5. Memory of the adjoint

Only the inputs, the gradient buffers and the stacks (STADE) or the cotangents (JAX) are on the device. Exact and driver readings as in section 4.

| kernel | size | STADE stacks MB | STADE C exact MB | STADE A exact MB | JAX own MB | gap C exact | gap A exact | STADE C driver MB | STADE A driver MB | JAX driver MB | gap C driver |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heat3d_step | 1e5 | 0.0 | 3.1 | 3.1 | 4.4 | 0.70 | 0.70 | 34 | 34 | 4 | - |
| heat3d_step | 1e6 | 0.0 | 32.0 | 32.0 | 41.6 | 0.77 | 0.77 | 34 | 34 | 67 | - |
| heat3d_step | 8e6 | 0.0 | 256.0 | 256.0 | 332.4 | 0.77 | 0.77 | 268 | 268 | 537 | 0.50 |
| heat3d_multi | 1e5 | 0.0 | 3.1 | 3.1 | 5.2 | 0.60 | 0.60 | 34 | 34 | 6 | - |
| heat3d_multi | 1e6 | 0.0 | 32.0 | 32.0 | 49.6 | 0.65 | 0.65 | 34 | 34 | 69 | - |
| heat1d_multi | 1e5_t8 | 0.0 | 3.2 | 3.2 | 7.7 | 0.42 | 0.42 | 34 | 34 | 15 | - |
| heat1d_multi | 1e6_t2 | 0.0 | 32.0 | 32.0 | 73.6 | 0.44 | 0.44 | 34 | 34 | 136 | - |
| heat1d_multi | 1e6_t8 | 0.0 | 32.0 | 32.0 | 73.6 | 0.44 | 0.44 | 34 | 34 | 136 | - |
| heat1d_multi | 1e6_t32 | 0.0 | 32.0 | 32.0 | 73.6 | 0.44 | 0.44 | 34 | 34 | 136 | - |
| wave2d_step | 1e5 | 0.0 | 6.4 | 6.4 | 7.9 | 0.81 | 0.81 | 34 | 34 | 17 | - |
| wave2d_step | 1e6 | 0.0 | 64.0 | 64.0 | 75.1 | 0.85 | 0.85 | 67 | 67 | 138 | 0.48 |
| wave2d_step | 16e6 | 0.0 | 1024.0 | 1024.0 | 1201.7 | 0.85 | 0.85 | 1040 | 1040 | 2152 | 0.48 |
| burgers2d_step | 1e5 | 0.0 | 6.4 | 6.4 | 8.7 | 0.74 | 0.74 | 34 | 34 | 17 | - |
| burgers2d_step | 1e6 | 0.0 | 64.0 | 64.0 | 83.1 | 0.77 | 0.77 | 67 | 67 | 138 | 0.48 |
| burgers2d_step | 16e6 | 0.0 | 1024.0 | 1024.0 | 1329.7 | 0.77 | 0.77 | 1040 | 1040 | 2152 | 0.48 |
| swe2d_step | 1e5 | 0.0 | 9.6 | 9.6 | 29.5 | 0.33 | 0.33 | 34 | 34 | 34 | - |
| swe2d_step | 1e6 | 0.0 | 96.0 | 96.0 | 391.5 | 0.25 | 0.25 | 101 | 101 | 463 | 0.22 |
| swe2d_step | 16e6 | 0.0 | 1536.0 | 1536.0 | 5361.7 | 0.29 | 0.29 | 1544 | 1544 | 6447 | 0.24 |
| euler2d_step | 1e5 | 0.0 | 12.8 | 12.8 | 84.2 | 0.15 | 0.15 | 34 | 34 | 101 | - |
| euler2d_step | 1e6 | 0.0 | 128.0 | 128.0 | 703.1 | 0.18 | 0.18 | 134 | 134 | 816 | 0.16 |
| euler2d_step | 16e6 | 0.0 | 2048.0 | 2048.0 | - | - | - | 2080 | 2080 | - | - |
| ufv_scatter_step | 1e5 | 0.0 | 11.2 | 11.2 | 17.9 | 0.63 | 0.63 | 34 | 34 | 36 | - |
| ufv_scatter_step | 1e6 | 0.0 | 112.0 | 112.0 | 147.5 | 0.76 | 0.76 | 134 | 134 | 298 | 0.45 |
| ufv_scatter_step | 4e6 | 0.0 | 448.0 | 448.0 | 590.0 | 0.76 | 0.76 | 470 | 470 | 1179 | 0.40 |
| ufv_gather_step | 1e5 | 0.0 | 12.8 | 12.8 | 16.3 | 0.79 | 0.79 | 34 | 34 | 31 | - |
| ufv_gather_step | 1e6 | 0.0 | 128.0 | 128.0 | 158.6 | 0.81 | 0.81 | 134 | 134 | 294 | 0.46 |
| ufv_gather_step | 4e6 | 0.0 | 512.0 | 512.0 | 634.4 | 0.81 | 0.81 | 537 | 537 | 1172 | 0.46 |
| mlp3 | 3M | 0.0 | 50.4 | 50.4 | 50.5 | 1.00 | 1.00 | 67 | 67 | 71 | 0.94 |
| mlp3 | 13M | 0.0 | 201.6 | 201.6 | 201.6 | 1.00 | 1.00 | 235 | 235 | 273 | 0.86 |
| mlp3 | 79M | 0.0 | 1258.9 | 1258.9 | 1493.9 | 0.84 | 0.84 | 1275 | 1275 | 2154 | 0.59 |
| mlp3_push | 3M | 0.0 | 50.5 | 50.5 | 50.5 | 1.00 | 1.00 | 67 | 67 | 71 | 0.94 |
| mlp3_push | 13M | 0.0 | 201.6 | 201.6 | 201.7 | 1.00 | 1.00 | 235 | 235 | 273 | 0.86 |
| mlp3_push | 79M | 0.0 | 1259.0 | 1259.0 | 1494.1 | 0.84 | 0.84 | 1275 | 1275 | 2154 | 0.59 |
| mpnn_gather | 1e5 | 38.4 | 210.5 | 210.5 | 516.3 | 0.41 | 0.41 | 235 | 235 | 577 | 0.41 |
| mpnn_gather | 1e6 | 384.0 | 2104.1 | 2104.1 | 4176.4 | 0.50 | 0.50 | 2114 | 2114 | 8789 | 0.24 |
| transformer_block | d512_T512 | 0.0 | 222.5 | 220.4 | 287.3 | 0.77 | 0.77 | 346 | 346 | 547 | 0.63 |
| transformer_block | d1024_T1024 | 0.0 | 1158.2 | 1149.8 | 1242.2 | 0.93 | 0.93 | 1554 | 1554 | 2171 | 0.72 |
| transformer_block | d2048_T1024 | 0.0 | 2718.9 | 2702.2 | 3054.8 | 0.89 | 0.88 | 3500 | 3500 | 4329 | 0.81 |
| unet2 | c16_n256 | 54.5 | 232.6 | 232.6 | 528.5 | 0.44 | 0.44 | 201 | 201 | 547 | 0.37 |
| unet2 | c32_n128 | 27.3 | 121.3 | 121.3 | 304.1 | 0.40 | 0.40 | 101 | 101 | 547 | 0.18 |
| unet2 | c64_n64 | 13.6 | 86.6 | 86.6 | 270.5 | 0.32 | 0.32 | 103 | 103 | 283 | 0.36 |
| unet2 | c128_n32 | 6.8 | 147.5 | 147.5 | 270.5 | 0.55 | 0.55 | 168 | 168 | 552 | 0.30 |

Geometric-mean adjoint memory gap: exact C **0.59**, exact A **0.59** over 41 cases; driver C **0.45** over 27 cases.

## 6. Summary

- Primal runtime: gap 0.533, STADE slower in 6 of 41 cases.
- Adjoint runtime: gap C 0.836, best 0.643; STADE C slower in 19 of 41 cases.
- Primal memory: exact gap 0.47 over 42 cases; driver gap 0.42 over 23 cases.
- Adjoint memory: exact gap C 0.59, A 0.59 over 41 cases; driver gap C 0.45 over 27 cases.

## 7. Checks and invalid cases

0 failed checks, 8 notes.

- NOTE G6: JAX heat1d_multi bench_1e5_t8 mem_adjoint: mem_inputs 0.0 MB below input 1.6 MB
- NOTE G6: JAX heat1d_multi bench_1e5_t8 mem_primal: mem_inputs 0.0 MB below input 1.6 MB
- NOTE G6: JAX heat3d_multi bench_1e5 mem_adjoint: mem_inputs 0.0 MB below input 1.6 MB
- NOTE G6: JAX heat3d_multi bench_1e5 mem_primal: mem_inputs 0.0 MB below input 1.6 MB
- NOTE G6: JAX heat3d_step bench_1e5 mem_adjoint: mem_inputs 0.0 MB below input 1.6 MB
- NOTE G6: JAX heat3d_step bench_1e5 mem_primal: mem_inputs 0.0 MB below input 1.6 MB
- NOTE RUN: JAX euler2d_step bench_16e6 mem_adjoint: JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 7.99GiB with allocator GPU_0_bfc on device 0
- NOTE RUN: JAX euler2d_step bench_16e6 time: JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 7.99GiB with allocator GPU_0_bfc on device 0

## 8. Limits

- One GPU model (V100, 16 GB), float64, batch size one. JAX appears as jit only.
- The two runtime timers differ. The caches are not flushed, and STADE copies its inputs between calls.
- The driver readings are rounded up by the allocators and hold one-time library memory. The exact readings are not. JAX driver readings come from nvidia-smi, and the JAX pool never shrinks.
- One job gives one run. The noise limit of 5 % is a working value. The STADE checkpoint option is off.
