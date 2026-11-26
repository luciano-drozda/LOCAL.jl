# Repeat of the time job: run-to-run change

- First: Slurm 2100092 on krakengpu1.cluster, 45.1 min, STADE commit d1663351b, kit 5a6d98c18367b4a4.
- Second: Slurm 2100137 on krakengpu1.cluster, 44.5 min, STADE commit 46c192a51, kit bace4aff63e3eb38.

Change = second / first - 1 for the median time of one case (host clock). `|change|` is the absolute value. Cases under 0.1 ms and cases of 0.1 ms or more are separate, because the launch cost dominates the small cases.

## 1. Size of the change

| quantity | cases | median abs change | 90th percentile | largest | share above 5 % | cases of 0.1 ms or more: median | : largest |
|---|---:|---:|---:|---:|---:|---:|---:|
| STADE primal | 42 | 0.49% | 2.73% | 6.3% | 2% | 0.09% | 6.3% |
| STADE adjoint C | 42 | 0.11% | 0.63% | 1.4% | 0% | 0.08% | 1.4% |
| STADE adjoint A | 42 | 0.13% | 1.24% | 2.1% | 0% | 0.04% | 1.1% |
| JAX primal | 41 | 0.22% | 1.36% | 1.9% | 0% | 0.18% | 1.9% |
| JAX adjoint | 41 | 0.21% | 1.58% | 3.0% | 0% | 0.21% | 3.0% |

## 2. Largest changes

| quantity | kernel | size | first ms | change |
|---|---|---|---:|---:|
| STADE primal | transformer_block | d512_T512 | 1.161 | +6.3% |
| STADE primal | swe2d_step | 1e5 | 0.034 | -4.8% |
| STADE primal | heat3d_step | 1e5 | 0.018 | +3.6% |
| STADE primal | wave2d_step | 1e5 | 0.017 | +2.8% |
| STADE adjoint C | wave2d_step | 1e6 | 0.200 | -1.4% |
| STADE adjoint C | heat1d_multi | 1e6_t2 | 0.242 | -1.4% |
| STADE adjoint C | heat3d_step | 1e5 | 0.030 | -0.8% |
| STADE adjoint C | burgers2d_step | 1e5 | 0.035 | +0.7% |
| STADE adjoint A | heat1d_multi | 1e5_t8 | 0.092 | -2.1% |
| STADE adjoint A | burgers2d_step | 1e5 | 0.030 | +1.8% |
| STADE adjoint A | swe2d_step | 1e5 | 0.086 | +1.6% |
| STADE adjoint A | heat3d_step | 1e5 | 0.022 | -1.5% |
| JAX primal | mlp3_push | 3M | 0.130 | +1.9% |
| JAX primal | ufv_gather_step | 1e5 | 0.097 | +1.9% |
| JAX primal | heat3d_multi | 1e5 | 0.112 | +1.7% |
| JAX primal | mlp3 | 3M | 0.131 | -1.5% |
| JAX adjoint | heat3d_multi | 1e5 | 0.101 | +3.0% |
| JAX adjoint | heat3d_step | 1e6 | 0.149 | +2.5% |
| JAX adjoint | ufv_gather_step | 1e5 | 0.124 | +2.2% |
| JAX adjoint | mlp3 | 3M | 0.244 | -1.6% |

## 3. The headline numbers in each run

| statement | first | second |
|---|---:|---:|
| cases | 41 | 41 |
| primal gap (geometric mean) | 0.533 | 0.533 |
| primal: STADE slower | 6 | 7 |
| adjoint gap C | 0.836 | 0.835 |
| adjoint C: STADE slower | 19 | 19 |
| adjoint gap best | 0.643 | 0.642 |
| adjoint best: STADE slower | 10 | 10 |
| adjoint C: more than 5 % slower | 19 | 19 |
| adjoint C: more than 5 % faster | 21 | 21 |
| adjoint best: more than 5 % slower | 9 | 9 |
| adjoint best: more than 5 % faster | 29 | 29 |

Cases that change class (more than 5 % slower, tie, more than 5 % faster) in the adjoint gap C: 0.


## 4. Checks

Second run: 0 failed checks, 1 notes.

- NOTE RUN: JAX euler2d_step bench_16e6 time: JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 7.99GiB with allocator GPU_0_bfc on device 0
