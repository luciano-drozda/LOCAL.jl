# STADE against JAX on a V100 

- Job on krakengpu1.cluster (Slurm 2101858), 45.0 min. STADE commit 8913babc8, kit 3d3e6fa193a59b9d.
- Job on krakengpu1.cluster (Slurm 2100097), 54.4 min. STADE commit 823fa703e, kit 5a6d98c18367b4a4.
- stade: cuda_jl 6.3.1, driver 12.4.0, julia 1.11.9, runtime 12.4.0
- jax: driver 550.90.12, jax 0.12.0, jaxlib 0.12.0, numpy 2.5.4, platform_version cuda 12090, python 3.12.7

STADE config C = defaults. Config A = primal_outputs = false with fresh_gradient. JAX = jit. Gap = STADE / JAX. Headline STADE timer: host clock with device synchronization. Failed checks are listed in section 7.

## 1. Runtime of the primal

| kernel | size | STADE ms | STADE events ms | JAX ms | gap |
|---|---|---:|---:|---:|---:|
| unet2 | c16_n256 | 2.5175 | 2.4087 | 1.7491 | 1.44 |
| unet2 | c32_n128 | 2.2388 | 2.1290 | 1.6055 | 1.39 |
| unet2 | c64_n64 | 2.4417 | 2.3309 | 1.9225 | 1.27 |
| unet2 | c128_n32 | 3.1556 | 3.0443 | 3.0840 | 1.02 |
| transformer_block | d2048_T1024 | 18.9927 | 18.9289 | 18.6791 | 1.02 |
| transformer_block | d512_T512 | 1.1850 | 1.0739 | 1.1698 | 1.01 |
| transformer_block | d1024_T1024 | 6.3163 | 6.2634 | 6.2597 | 1.01 |
| mlp3_push | 79M | 0.8709 | 0.7619 | 0.8769 | 0.99 |
| mlp3 | 79M | 0.8430 | 0.7342 | 0.8613 | 0.98 |
| ufv_gather_step | 4e6 | 2.2894 | 2.1791 | 2.3573 | 0.97 |
| wave2d_step | 16e6 | 0.7545 | 0.6449 | 0.8208 | 0.92 |
| heat3d_multi | 1e6 | 0.1817 | 0.1766 | 0.2098 | 0.87 |
| ufv_gather_step | 1e6 | 0.2948 | 0.2904 | 0.3566 | 0.83 |
| heat3d_step | 8e6 | 0.2950 | 0.2904 | 0.3615 | 0.82 |
| burgers2d_step | 16e6 | 0.7928 | 0.6835 | 1.0283 | 0.77 |
| mlp3_push | 13M | 0.1570 | 0.1536 | 0.2295 | 0.68 |
| ufv_scatter_step | 1e6 | 0.5391 | 0.4291 | 0.8031 | 0.67 |
| mlp3 | 13M | 0.1452 | 0.1403 | 0.2194 | 0.66 |
| heat3d_step | 1e6 | 0.0576 | 0.0529 | 0.0999 | 0.58 |
| mlp3 | 3M | 0.0683 | 0.0635 | 0.1315 | 0.52 |
| mlp3_push | 3M | 0.0664 | 0.0625 | 0.1315 | 0.50 |
| heat1d_multi | 1e5_t8 | 0.0817 | 0.0781 | 0.1623 | 0.50 |
| ufv_scatter_step | 4e6 | 3.5366 | 3.4229 | 7.1839 | 0.49 |
| wave2d_step | 1e6 | 0.0624 | 0.0585 | 0.1275 | 0.49 |
| burgers2d_step | 1e6 | 0.0673 | 0.0634 | 0.1485 | 0.45 |
| heat1d_multi | 1e6_t32 | 1.6420 | 1.5319 | 3.8445 | 0.43 |
| euler2d_step | 1e6 | 0.6166 | 0.5064 | 1.5305 | 0.40 |
| heat1d_multi | 1e6_t8 | 0.4036 | 0.3985 | 1.0374 | 0.39 |
| ufv_scatter_step | 1e5 | 0.0409 | 0.0369 | 0.1063 | 0.39 |
| euler2d_step | 1e5 | 0.0696 | 0.0648 | 0.1932 | 0.36 |
| heat3d_multi | 1e5 | 0.0397 | 0.0362 | 0.1106 | 0.36 |
| heat1d_multi | 1e6_t2 | 0.1196 | 0.1154 | 0.3334 | 0.36 |
| ufv_gather_step | 1e5 | 0.0336 | 0.0296 | 0.0993 | 0.34 |
| mpnn_gather | 1e6 | 3.4931 | 3.3822 | 14.1736 | 0.25 |
| mpnn_gather | 1e5 | 0.3746 | 0.3720 | 1.6063 | 0.23 |
| swe2d_step | 1e5 | 0.0342 | 0.0294 | 0.1486 | 0.23 |
| swe2d_step | 1e6 | 0.1557 | 0.1522 | 0.6910 | 0.23 |
| swe2d_step | 16e6 | 1.9865 | 1.8747 | 8.9716 | 0.22 |
| burgers2d_step | 1e5 | 0.0204 | 0.0169 | 0.1033 | 0.20 |
| heat3d_step | 1e5 | 0.0175 | 0.0141 | 0.0944 | 0.18 |
| wave2d_step | 1e5 | 0.0175 | 0.0134 | 0.0954 | 0.18 |

41 cases. Geometric-mean gap **0.530**. STADE slower than JAX in 7 cases.

## 2. Runtime of the adjoint

| kernel | size | STADE C ms | STADE C events ms | STADE best ms | JAX ms | gap C | gap best |
|---|---|---:|---:|---:|---:|---:|---:|
| mlp3 | 13M | 0.727 | 0.617 | 0.623 | 0.474 | 1.53 | 1.31 |
| ufv_gather_step | 1e6 | 1.527 | 1.417 | 1.442 | 1.149 | 1.33 | 1.26 |
| transformer_block | d1024_T1024 | 22.267 | 22.174 | 20.935 | 17.605 | 1.26 | 1.19 |
| transformer_block | d2048_T1024 | 66.848 | 66.737 | 61.661 | 52.862 | 1.26 | 1.17 |
| unet2 | c16_n256 | 13.274 | 13.158 | 13.243 | 11.373 | 1.17 | 1.16 |
| mlp3_push | 79M | 3.207 | 3.099 | 2.481 | 2.156 | 1.49 | 1.15 |
| mlp3 | 79M | 3.148 | 3.036 | 2.286 | 2.131 | 1.48 | 1.07 |
| transformer_block | d512_T512 | 3.783 | 3.665 | 3.577 | 3.409 | 1.11 | 1.05 |
| ufv_gather_step | 4e6 | 10.717 | 10.606 | 9.832 | 9.495 | 1.13 | 1.04 |
| mlp3 | 3M | 0.275 | 0.271 | 0.244 | 0.242 | 1.14 | 1.01 |
| mlp3_push | 13M | 0.682 | 0.572 | 0.453 | 0.485 | 1.40 | 0.93 |
| heat3d_multi | 1e6 | 0.386 | 0.382 | 0.240 | 0.261 | 1.48 | 0.92 |
| unet2 | c128_n32 | 7.138 | 7.023 | 7.120 | 7.837 | 0.91 | 0.91 |
| unet2 | c32_n128 | 8.006 | 7.893 | 7.995 | 8.822 | 0.91 | 0.91 |
| unet2 | c64_n64 | 6.659 | 6.547 | 6.652 | 7.748 | 0.86 | 0.86 |
| heat1d_multi | 1e6_t32 | 3.175 | 3.064 | 1.655 | 1.992 | 1.59 | 0.83 |
| wave2d_step | 16e6 | 2.872 | 2.763 | 2.010 | 2.447 | 1.17 | 0.82 |
| heat3d_step | 8e6 | 0.785 | 0.675 | 0.388 | 0.487 | 1.61 | 0.80 |
| mpnn_gather | 1e6 | 27.077 | 26.965 | 27.037 | 37.071 | 0.73 | 0.73 |
| heat1d_multi | 1e6_t8 | 0.915 | 0.805 | 0.421 | 0.613 | 1.49 | 0.69 |
| mpnn_gather | 1e5 | 2.840 | 2.730 | 2.840 | 4.240 | 0.67 | 0.67 |
| heat1d_multi | 1e5_t8 | 0.158 | 0.153 | 0.091 | 0.138 | 1.15 | 0.66 |
| mlp3_push | 3M | 0.191 | 0.186 | 0.166 | 0.254 | 0.75 | 0.65 |
| euler2d_step | 1e5 | 0.310 | 0.306 | 0.289 | 0.481 | 0.64 | 0.60 |
| heat3d_multi | 1e5 | 0.081 | 0.077 | 0.058 | 0.102 | 0.80 | 0.57 |
| burgers2d_step | 16e6 | 2.657 | 2.546 | 1.659 | 2.918 | 0.91 | 0.57 |
| wave2d_step | 1e6 | 0.199 | 0.197 | 0.147 | 0.260 | 0.77 | 0.56 |
| heat1d_multi | 1e6_t2 | 0.242 | 0.238 | 0.140 | 0.264 | 0.92 | 0.53 |
| ufv_scatter_step | 1e6 | 1.390 | 1.281 | 0.957 | 1.827 | 0.76 | 0.52 |
| heat3d_step | 1e6 | 0.116 | 0.112 | 0.070 | 0.152 | 0.76 | 0.46 |
| burgers2d_step | 1e6 | 0.192 | 0.188 | 0.129 | 0.282 | 0.68 | 0.46 |
| ufv_gather_step | 1e5 | 0.072 | 0.068 | 0.057 | 0.127 | 0.57 | 0.45 |
| ufv_scatter_step | 4e6 | 9.716 | 9.587 | 6.139 | 15.389 | 0.63 | 0.40 |
| euler2d_step | 1e6 | 2.698 | 2.585 | 2.462 | 6.248 | 0.43 | 0.39 |
| swe2d_step | 1e5 | 0.086 | 0.082 | 0.084 | 0.215 | 0.40 | 0.39 |
| ufv_scatter_step | 1e5 | 0.082 | 0.078 | 0.056 | 0.167 | 0.49 | 0.33 |
| burgers2d_step | 1e5 | 0.035 | 0.032 | 0.030 | 0.140 | 0.25 | 0.21 |
| heat3d_step | 1e5 | 0.029 | 0.026 | 0.022 | 0.106 | 0.27 | 0.21 |
| wave2d_step | 1e5 | 0.027 | 0.024 | 0.024 | 0.121 | 0.23 | 0.20 |
| swe2d_step | 16e6 | 10.417 | 10.303 | 9.224 | 47.778 | 0.22 | 0.19 |
| swe2d_step | 1e6 | 0.778 | 0.668 | 0.706 | 5.570 | 0.14 | 0.13 |

41 cases. Geometric-mean gap: C **0.789**, best **0.615**. STADE slower than JAX: C 17, best 10.

## 3. Ratio adjoint time over primal time

| kernel | size | STADE C | JAX |
|---|---|---:|---:|
| heat3d_step | 1e5 | 1.64 | 1.13 |
| heat3d_step | 1e6 | 2.01 | 1.52 |
| heat3d_step | 8e6 | 2.66 | 1.35 |
| heat3d_multi | 1e5 | 2.05 | 0.92 |
| heat3d_multi | 1e6 | 2.12 | 1.24 |
| heat1d_multi | 1e5_t8 | 1.94 | 0.85 |
| heat1d_multi | 1e6_t2 | 2.03 | 0.79 |
| heat1d_multi | 1e6_t8 | 2.27 | 0.59 |
| heat1d_multi | 1e6_t32 | 1.93 | 0.52 |
| wave2d_step | 1e5 | 1.57 | 1.27 |
| wave2d_step | 1e6 | 3.19 | 2.04 |
| wave2d_step | 16e6 | 3.81 | 2.98 |
| burgers2d_step | 1e5 | 1.73 | 1.36 |
| burgers2d_step | 1e6 | 2.85 | 1.90 |
| burgers2d_step | 16e6 | 3.35 | 2.84 |
| swe2d_step | 1e5 | 2.51 | 1.45 |
| swe2d_step | 1e6 | 4.99 | 8.06 |
| swe2d_step | 16e6 | 5.24 | 5.33 |
| euler2d_step | 1e5 | 4.45 | 2.49 |
| euler2d_step | 1e6 | 4.38 | 4.08 |
| ufv_scatter_step | 1e5 | 2.00 | 1.57 |
| ufv_scatter_step | 1e6 | 2.58 | 2.27 |
| ufv_scatter_step | 4e6 | 2.75 | 2.14 |
| ufv_gather_step | 1e5 | 2.15 | 1.28 |
| ufv_gather_step | 1e6 | 5.18 | 3.22 |
| ufv_gather_step | 4e6 | 4.68 | 4.03 |
| mlp3 | 3M | 4.03 | 1.84 |
| mlp3 | 13M | 5.01 | 2.16 |
| mlp3 | 79M | 3.73 | 2.47 |
| mlp3_push | 3M | 2.87 | 1.93 |
| mlp3_push | 13M | 4.34 | 2.12 |
| mlp3_push | 79M | 3.68 | 2.46 |
| mpnn_gather | 1e5 | 7.58 | 2.64 |
| mpnn_gather | 1e6 | 7.75 | 2.62 |
| transformer_block | d512_T512 | 3.19 | 2.91 |
| transformer_block | d1024_T1024 | 3.53 | 2.81 |
| transformer_block | d2048_T1024 | 3.52 | 2.83 |
| unet2 | c16_n256 | 5.27 | 6.50 |
| unet2 | c32_n128 | 3.58 | 5.49 |
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

- Primal runtime: gap 0.530, STADE slower in 7 of 41 cases.
- Adjoint runtime: gap C 0.789, best 0.615; STADE C slower in 17 of 41 cases.
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
