# Ablation of the adjoint options of STADE on a V100

- Job on krakengpu1.cluster (Slurm 2100116), 45.8 min. STADE commit ac89177f8, kit bace4aff63e3eb38.
- stade: cuda_jl 6.3.1, driver 12.4.0, julia 1.11.9, runtime 12.4.0

Config C = defaults. P = `primal_outputs = false`. G = `fresh_gradient = true`. A = P and G together. The tables show the adjoint time in ms (host clock with device synchronization). A ratio below 1 means the option makes the adjoint faster than C. A change below 3% of the C time is a tie.

## 1. Adjoint time by case

| kernel | size | C ms | P ms | G ms | A ms | P/C | G/C | A/C | JAX ms | gap C | gap P | gap G | gap A |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heat3d_step | 1e5 | 0.029 | 0.022 | 0.029 | 0.022 | 0.75 | 0.99 | 0.77 | 0.103 | 0.28 | 0.21 | 0.28 | 0.22 |
| heat3d_step | 1e6 | 0.115 | 0.074 | 0.109 | 0.072 | 0.65 | 0.95 | 0.63 | 0.149 | 0.77 | 0.50 | 0.73 | 0.48 |
| heat3d_step | 8e6 | 0.788 | 0.410 | 0.763 | 0.387 | 0.52 | 0.97 | 0.49 | 0.475 | 1.66 | 0.86 | 1.60 | 0.81 |
| heat3d_multi | 1e5 | 0.081 | 0.055 | 0.082 | 0.052 | 0.68 | 1.01 | 0.64 | 0.101 | 0.80 | 0.55 | 0.81 | 0.51 |
| heat3d_multi | 1e6 | 0.385 | 0.237 | 0.385 | 0.237 | 0.62 | 1.00 | 0.62 | 0.262 | 1.46 | 0.90 | 1.47 | 0.90 |
| heat1d_multi | 1e5_t8 | 0.161 | 0.091 | 0.160 | 0.092 | 0.56 | 0.99 | 0.57 | 0.136 | 1.18 | 0.67 | 1.18 | 0.68 |
| heat1d_multi | 1e6_t2 | 0.241 | 0.140 | 0.240 | 0.138 | 0.58 | 0.99 | 0.57 | 0.255 | 0.95 | 0.55 | 0.94 | 0.54 |
| heat1d_multi | 1e6_t8 | 0.906 | 0.415 | 0.901 | 0.416 | 0.46 | 0.99 | 0.46 | 0.592 | 1.53 | 0.70 | 1.52 | 0.70 |
| heat1d_multi | 1e6_t32 | 3.128 | 1.629 | 3.124 | 1.630 | 0.52 | 1.00 | 0.52 | 1.910 | 1.64 | 0.85 | 1.64 | 0.85 |
| wave2d_step | 1e5 | 0.027 | 0.024 | 0.026 | 0.023 | 0.90 | 0.99 | 0.87 | 0.123 | 0.22 | 0.20 | 0.21 | 0.19 |
| wave2d_step | 1e6 | 0.202 | 0.154 | 0.173 | 0.146 | 0.77 | 0.86 | 0.72 | 0.254 | 0.79 | 0.61 | 0.68 | 0.57 |
| wave2d_step | 16e6 | 2.875 | 2.156 | 2.469 | 2.019 | 0.75 | 0.86 | 0.70 | 2.375 | 1.21 | 0.91 | 1.04 | 0.85 |
| ufv_gather_step | 1e5 | 0.070 | 0.055 | 0.069 | 0.055 | 0.78 | 0.99 | 0.79 | 0.124 | 0.57 | 0.44 | 0.56 | 0.45 |
| ufv_gather_step | 1e6 | 1.637 | 1.427 | 1.653 | 1.444 | 0.87 | 1.01 | 0.88 | 1.149 | 1.43 | 1.24 | 1.44 | 1.26 |
| ufv_gather_step | 4e6 | 11.142 | 9.923 | 11.184 | 9.924 | 0.89 | 1.00 | 0.89 | 9.562 | 1.17 | 1.04 | 1.17 | 1.04 |
| mlp3 | 3M | 0.270 | 0.258 | 0.252 | 0.237 | 0.96 | 0.93 | 0.88 | 0.244 | 1.11 | 1.06 | 1.03 | 0.97 |
| mlp3 | 13M | 0.720 | 0.684 | 0.653 | 0.610 | 0.95 | 0.91 | 0.85 | 0.472 | 1.53 | 1.45 | 1.38 | 1.29 |
| mlp3 | 79M | 3.125 | 2.888 | 2.501 | 2.258 | 0.92 | 0.80 | 0.72 | 2.138 | 1.46 | 1.35 | 1.17 | 1.06 |
| mlp3_push | 3M | 0.190 | 0.190 | 0.164 | 0.165 | 1.00 | 0.86 | 0.87 | 0.253 | 0.75 | 0.75 | 0.65 | 0.65 |
| mlp3_push | 13M | 0.681 | 0.681 | 0.453 | 0.455 | 1.00 | 0.67 | 0.67 | 0.486 | 1.40 | 1.40 | 0.93 | 0.94 |
| mlp3_push | 79M | 3.194 | 3.193 | 2.477 | 2.477 | 1.00 | 0.78 | 0.78 | 2.164 | 1.48 | 1.48 | 1.14 | 1.14 |
| transformer_block | d512_T512 | 3.779 | 3.583 | 3.779 | 3.613 | 0.95 | 1.00 | 0.96 | 3.397 | 1.11 | 1.05 | 1.11 | 1.06 |
| transformer_block | d1024_T1024 | 22.275 | 21.003 | 22.269 | 20.950 | 0.94 | 1.00 | 0.94 | 17.392 | 1.28 | 1.21 | 1.28 | 1.20 |
| transformer_block | d2048_T1024 | 66.879 | 61.695 | 66.887 | 61.696 | 0.92 | 1.00 | 0.92 | 52.733 | 1.27 | 1.17 | 1.27 | 1.17 |
| unet2 | c16_n256 | 13.261 | 13.249 | 13.264 | 13.247 | 1.00 | 1.00 | 1.00 | 11.359 | 1.17 | 1.17 | 1.17 | 1.17 |
| unet2 | c32_n128 | 8.001 | 7.988 | 7.999 | 7.992 | 1.00 | 1.00 | 1.00 | 8.808 | 0.91 | 0.91 | 0.91 | 0.91 |
| unet2 | c64_n64 | 6.664 | 6.654 | 6.663 | 6.653 | 1.00 | 1.00 | 1.00 | 7.743 | 0.86 | 0.86 | 0.86 | 0.86 |
| unet2 | c128_n32 | 7.146 | 7.128 | 7.144 | 7.129 | 1.00 | 1.00 | 1.00 | 7.811 | 0.91 | 0.91 | 0.91 | 0.91 |

## 2. Which option gives the gain

Each case with an A gain above the tie level (A/C below 1 minus the tie) falls in one class. **P** means P gives most of the gain and G gives little. **G** is the reverse. **both** means each option gives a part. **none** means neither option alone gives a gain although A does. **mix** means one option alone is slower than C.

- **P**: 14 cases: heat3d_step 1e5, heat3d_multi 1e5, heat3d_multi 1e6, heat1d_multi 1e5_t8, heat1d_multi 1e6_t2, heat1d_multi 1e6_t8, heat1d_multi 1e6_t32, wave2d_step 1e5, ufv_gather_step 1e5, ufv_gather_step 1e6, ufv_gather_step 4e6, transformer_block d512_T512, transformer_block d1024_T1024, transformer_block d2048_T1024
- **G**: 3 cases: mlp3_push 3M, mlp3_push 13M, mlp3_push 79M
- **both**: 7 cases: heat3d_step 1e6, heat3d_step 8e6, wave2d_step 1e6, wave2d_step 16e6, mlp3 3M, mlp3 13M, mlp3 79M
- **none**: 0 cases
- **mix**: 0 cases
- **no gain**: 4 cases: unet2 c16_n256, unet2 c32_n128, unet2 c64_n64, unet2 c128_n32

| kernel | size | A gain | P gain | G gain | class |
|---|---|---:|---:|---:|---|
| heat3d_step | 1e5 | 23% | 25% | 1% | P |
| heat3d_step | 1e6 | 37% | 35% | 5% | both |
| heat3d_step | 8e6 | 51% | 48% | 3% | both |
| heat3d_multi | 1e5 | 36% | 32% | -1% | P |
| heat3d_multi | 1e6 | 38% | 38% | -0% | P |
| heat1d_multi | 1e5_t8 | 43% | 44% | 1% | P |
| heat1d_multi | 1e6_t2 | 43% | 42% | 1% | P |
| heat1d_multi | 1e6_t8 | 54% | 54% | 1% | P |
| heat1d_multi | 1e6_t32 | 48% | 48% | 0% | P |
| wave2d_step | 1e5 | 13% | 10% | 1% | P |
| wave2d_step | 1e6 | 28% | 23% | 14% | both |
| wave2d_step | 16e6 | 30% | 25% | 14% | both |
| ufv_gather_step | 1e5 | 21% | 22% | 1% | P |
| ufv_gather_step | 1e6 | 12% | 13% | -1% | P |
| ufv_gather_step | 4e6 | 11% | 11% | -0% | P |
| mlp3 | 3M | 12% | 4% | 7% | both |
| mlp3 | 13M | 15% | 5% | 9% | both |
| mlp3 | 79M | 28% | 8% | 20% | both |
| mlp3_push | 3M | 13% | 0% | 14% | G |
| mlp3_push | 13M | 33% | -0% | 33% | G |
| mlp3_push | 79M | 22% | 0% | 22% | G |
| transformer_block | d512_T512 | 4% | 5% | -0% | P |
| transformer_block | d1024_T1024 | 6% | 6% | 0% | P |
| transformer_block | d2048_T1024 | 8% | 8% | -0% | P |
| unet2 | c16_n256 | 0% | 0% | -0% | no gain |
| unet2 | c32_n128 | 0% | 0% | 0% | no gain |
| unet2 | c64_n64 | 0% | 0% | 0% | no gain |
| unet2 | c128_n32 | 0% | 0% | 0% | no gain |

## 3. Geometric-mean ratio to C, by family

| family | cases | P/C | G/C | A/C |
|---|---:|---:|---:|---:|
| stencil | 12 | 0.634 | 0.965 | 0.621 |
| unstructured | 3 | 0.846 | 1.000 | 0.851 |
| NN | 13 | 0.972 | 0.912 | 0.883 |
| all | 28 | 0.798 | 0.944 | 0.756 |

## 4. Geometric-mean gap to JAX

| set | cases | C | P | G | A |
|---|---:|---:|---:|---:|---:|
| all with JAX | 28 | 1.011 | 0.806 | 0.954 | 0.765 |
| STADE slower than JAX | | 17 | 11 | 16 | 9 |

## 5. Configs with the same generated code

A config pair with the same code is the same program. The time ratio of such a pair is noise, so this table measures the noise of the job on the real cases.

| kernel | same code | size | ratio |
|---|---|---|---:|
| heat3d_step | none | | |
| heat3d_multi | C = G | 1e5 | 1.008 |
| heat3d_multi | C = G | 1e6 | 1.001 |
| heat3d_multi | P = A | 1e5 | 0.940 |
| heat3d_multi | P = A | 1e6 | 1.000 |
| heat1d_multi | C = G | 1e5_t8 | 0.995 |
| heat1d_multi | C = G | 1e6_t2 | 0.993 |
| heat1d_multi | C = G | 1e6_t8 | 0.994 |
| heat1d_multi | C = G | 1e6_t32 | 0.999 |
| heat1d_multi | P = A | 1e5_t8 | 1.015 |
| heat1d_multi | P = A | 1e6_t2 | 0.989 |
| heat1d_multi | P = A | 1e6_t8 | 1.003 |
| heat1d_multi | P = A | 1e6_t32 | 1.001 |
| wave2d_step | none | | |
| ufv_gather_step | none | | |
| mlp3 | none | | |
| mlp3_push | C = P | 3M | 0.999 |
| mlp3_push | C = P | 13M | 1.001 |
| mlp3_push | C = P | 79M | 1.000 |
| mlp3_push | G = A | 3M | 1.006 |
| mlp3_push | G = A | 13M | 1.003 |
| mlp3_push | G = A | 79M | 1.000 |
| transformer_block | C = G | d512_T512 | 1.000 |
| transformer_block | C = G | d1024_T1024 | 1.000 |
| transformer_block | C = G | d2048_T1024 | 1.000 |
| transformer_block | P = A | d512_T512 | 1.008 |
| transformer_block | P = A | d1024_T1024 | 0.997 |
| transformer_block | P = A | d2048_T1024 | 1.000 |
| unet2 | C = G | c16_n256 | 1.000 |
| unet2 | C = G | c32_n128 | 1.000 |
| unet2 | C = G | c64_n64 | 1.000 |
| unet2 | C = G | c128_n32 | 1.000 |
| unet2 | P = A | c16_n256 | 1.000 |
| unet2 | P = A | c32_n128 | 1.000 |
| unet2 | P = A | c64_n64 | 1.000 |
| unet2 | P = A | c128_n32 | 1.000 |

32 pairs. Median change 0.1%, largest 6.0%.

## 6. Checks

0 failed checks, 0 notes.

