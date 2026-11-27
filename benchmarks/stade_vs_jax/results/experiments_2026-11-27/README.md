# Experiments of 2026-11-27

## Half-warp reduction for the atomic adds of the mpnn_gather adjoint (job ATOM1)

Question: do the atomic adds of the three adjoint kernels `b_3_p`, `b_4_p` and `b_5_p` limit the mpnn_gather adjoint (section 4.5 of `DISCUSSION.md`)?

Change: in the generated script of config C, the helper `__stade_wa_add!` for Float64 reduces over the half-warp (16 lanes) when the 16 lanes add to one address, and then does one atomic add. In these kernels, 16 consecutive lanes add to the same address in most of the global adds. The original helper reduces only when all 32 lanes have the same address, which does not happen here. The script is made by `tools/patch_halfwarp.py`. This is an experiment on the generated script, and it is not a change of `STADE.jl`.

Job ATOM1 (Slurm 2100686, one V100, 340 s): the time phase of mpnn_gather, order C, X, C, X (C = generated code of STADE, X = with the half-warp reduction). The size bench_1e5 is verified against the CPU in both.

| size | C ms (two runs) | X ms (two runs) | X / C | gradient error against the CPU, C and X |
|---|---|---|---:|---|
| tiny | 0.052, 0.052 | 0.062, 0.063 | 1.20 | 1.1e-15 both |
| bench_1e5 | 4.143, 4.144 | 3.569, 3.570 | 0.86 | 3.7e-12 and 2.9e-12 (C), 2.5e-12 and 4.2e-12 (X) |
| bench_1e6 | 39.683, 39.678 | 34.017, 34.017 | 0.86 | not verified (the number of changed elements and of non-zero gradient values is the same for C and X) |

The adjoint time is the median of the host clock with device synchronization. The primal time is the same for C and X (3.494 ms at 1e6).

Reading:
- The change makes the adjoint 14 % faster at 1e5 and 1e6, and the result agrees with the CPU like the result of C does (the rounding differs from run to run in both, as the order of the atomic adds is not fixed).
- The change removes most of the global atomic adds (about 1.1e9 of 1.25e9 by the count of `DISCUSSION.md`, if all half-warps are active and uniform), and it saves 5.7 ms. So the collisions of the atomic adds are a part of the time, but they are not the main limit. The other 34 ms are not explained.
- The extra check costs 0.01 ms for the tiny size, where the new path is never used. A version for `STADE.jl` has to apply the half-warp reduction only where it can match (the index does not change with the lane inside a half-warp).

## Without atomic adds, timing only (job ATOM2)

Question: how much time do the atomic adds cost at most? Answer: replace every atomic add of the generated script by a plain add (variant N, 44 places, made by `tools/patch_noatomic.py`) and time it. The result of N is wrong (lost updates and races), so the time of N is only a lower bound for any version that keeps the atomic adds. It can be too low if the compiler keeps a sum in a register, and N may do less work if a lost update skips a later one (the gradient of N has 27.7e6 non-zero values at 1e6 against 40.0e6 for C, while the number of changed elements is the same).

Job ATOM2 (Slurm 2100711, one V100, 417 s): the time phase of mpnn_gather in the order C, N, X, C, N (C = unchanged code, N = no atomic adds, X = half-warp reduction of ATOM1).

| size | C ms (two runs) | X ms | N ms (two runs) | X / C | N / C |
|---|---|---:|---|---:|---:|
| bench_1e5 | 4.141, 4.143 | 3.568 | 2.749, 2.749 | 0.86 | 0.66 |
| bench_1e6 | 39.683, 39.683 | 34.017 | 26.307, 26.308 | 0.86 | 0.66 |

C and X agree with ATOM1 to 3 digits. The two runs of C and of N are the same to 0.001 ms.

Reading:
- The atomic adds cost at most 13.4 ms of 39.7 ms (34 %) at 1e6. No change of the atomic adds alone can give more than that, and the real limit is lower (N is a lower bound).
- The half-warp reduction (X) gets 5.7 ms of the 13.4 ms, which is 43 %. The other 7.7 ms is in the atomic adds that X does not change. Job ATOM3 (next section) separates them: the scatter into `asrcb` is not the cause.
- Even without atomic adds, the adjoint takes 26.3 ms. These three kernels do not reach the speed of memory or of arithmetic, and the cause is not known.

## Which atomic adds remain: shared memory, scatter or reduced global adds (job ATOM3)

Question: after the half-warp reduction (X), where is the rest of the atomic cost? Variants of X (made by `tools/patch_split.py`) in which one group of call sites of the helper `__stade_wa_add!` becomes a plain add (wrong result, timing only). A plain add also removes the checks of the helper (`active_mask`, vote and shuffle of the index), so a gain below is the atomic add and the check together.

| variant | plain adds | call sites |
|---|---|---:|
| S | the adds into arrays of shared memory (`__sh_*`) | 10 |
| Y | the scatter into `asrcb` (random addresses) | 2 |
| R | the adds into `aggb`, `hb` and `efb` (the global adds that X already cuts to a sixteenth) | 21 |
| Z | S, Y and R together (the block-final atomic adds stay) | 33 |

Job ATOM3 (Slurm 2100719, one V100, 672 s): the time phase of mpnn_gather, order C, X, S, Y, R, Z, X, S. The repeats of X and S agree to 0.002 ms.

| size | C | X | S | Y | R | Z | N (ATOM2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bench_1e5 ms | 4.143 | 3.568 | 2.891 | 3.504 | 2.757 | 2.334 | 2.749 |
| bench_1e6 ms | 39.685 | 34.016 | 27.273 | 33.368 | 26.156 | 22.576 | 26.307 |
| 1e6, change from X | | | -6.74 | -0.65 | -7.86 | -11.44 | |

Reading (1e6):
- The scatter into `asrcb` costs 0.65 ms. It is not the cause of the remaining cost.
- The shared-memory adds (6.7 ms) and the reduced global adds (7.9 ms) each cost about as much as the other. The gains do not add up (15.3 ms for S, Y and R separately, against 11.4 ms for Z), because the groups overlap in time: when one is removed, another becomes the limit.
- The version without any of these adds and without the helper checks (Z) takes 22.6 ms. That is 57 % of the time of C. So the atomic adds and the checks around them cost 17.1 ms (43 %) of the 39.7 ms of C, and the half-warp reduction removes 5.7 ms of it.
- Z is 3.7 ms faster than N (26.3 ms, job ATOM2). N keeps the helper calls and only removes the atomic add inside them. So the check of the helper in C (the full-warp test, which never succeeds in these kernels) costs about 3.7 ms by itself.
- The test cannot separate the atomic add from the helper check inside S and R. The cause of the cost of the shared-memory adds was not measured. One possible cause is that an atomic add of a float64 on shared memory is slow on this GPU, which was not checked.

## Gate of the warp helper, Part A (job GATE1)

Job GATE1 (Slurm 2100933, one V100, 4356 s): the time phase of 8 kernels, run in the order C, G, C, G, C, G for each kernel. C = the scripts of the commit before v0.6.33 (helper on every atomic add of a named array). G = the scripts of v0.6.33 (`cgen_warp_gate`: the helper only where the index can be equal in the lanes of a warp). Both sets were generated from the same sources with `gen_stade_scripts.sh`, config C. The C scripts come from `git show HEAD~1:src/STADE.jl`, and the job records the commit of G (d94a8a5). All 48 runs ended with exit 0. Table made with `python3 tools/gate_report.py GATE1_time.json`. Times are the adjoint, median of the call, in ms. Files: `GATE1_time.json`, `GATE1_table.md`, `plan_gate1.tsv`, `versions_gate1.json`.

Helper calls in the generated scripts (the three definitions included), C to G: euler2d_step 39 to 0, swe2d_step 18 to 0, mpnn_gather 58 to 32, transformer_block 90 to 51, mlp3_push 27 to 15, ufv_gather_step 9 to 6, unet2 101 to 91, mlp3 15 to 15 (unchanged). Over the corpus (adjoint and HVP) the count falls from 2168 to 1553.

| kernel | size | C (ms) | G (ms) | change | spread of C |
|---|---|---:|---:|---:|---:|
| euler2d_step | 1e5 | 0.421 | 0.305 | -27.6 % | 0.1 % |
| euler2d_step | 1e6 | 3.652 | 2.587 | -29.1 % | 0.1 % |
| euler2d_step | 16e6 | 60.055 | 42.199 | -29.7 % | 0.0 % |
| swe2d_step | 1e5 | 0.085 | 0.082 | -3.6 % | 1.2 % |
| swe2d_step | 1e6 | 0.691 | 0.667 | -3.6 % | 0.0 % |
| swe2d_step | 16e6 | 10.668 | 10.302 | -3.4 % | 0.0 % |
| mpnn_gather | 1e5 | 4.034 | 4.015 | -0.5 % | 0.0 % |
| mpnn_gather | 1e6 | 39.575 | 39.549 | -0.1 % | 0.0 % |
| transformer_block | d1024_T1024 | 22.163 | 22.143 | -0.1 % | 0.1 % |
| unet2 | c16_n256 | 13.150 | 13.145 | -0.0 % | 0.0 % |
| mlp3_push, mlp3, ufv_gather_step | all | | | within +0.1 % | |

The full table (every size, every run) is in `GATE1_table.md`.

Reading:
- The gate removes the helper from euler2d_step and swe2d_step completely, and euler2d_step is 28 to 30 % faster at every size. This is the case predicted by the JACC path (36 scatter sites, all with an index that differs in adjacent lanes).
- mpnn_gather does not gain. The scope document expected up to 3.7 ms (9 %). The gate changed 26 of its 55 call sites and the 1e6 time moved by 0.03 ms. So the check cost of the experiment ATOM3 (3.7 ms) is most likely in the sites that the gate keeps (not measured per site): the sites whose index is the same in runs of lanes (class S) or anything else (class X). The sites of class D are cheap. This is the first measurement that says where the cost is, and it points at Part B (a group reduction for S and X sites).
- transformer_block, unet2, mlp3, mlp3_push and ufv_gather_step do not change by more than the spread of C (the spread is 0.0 to 1.2 % for the sizes above 1e5). No case is slower.
- Output: every run writes the same number of elements as the baseline run of its kernel, and the error against the CPU is below 1e-9 in all 48 runs (the check of the table, column "same output"). Bitwise equality is not testable on the device, because atomic adds land in any order. The bit-equality is shown by the structural test: with the calls replaced by plain atomics, the gated output equals the output of `warp_atomics = false` (`test/validate_warp_atomics.jl`, 1127 calls, 216 kernel plans).
- Not done: the GPU corpus (`validate_corpus_gpu.jl`) on the V100. The 8 kernels of this job passed the verification of the time phase at the size `tiny` and at `bench_1e5`.

| kernel | size | base runs (ms) | new runs (ms) | base med | new med | change | base spread | same output |
|---|---|---|---|---|---|---|---|---|
| swe2d_step | tiny | 0.024 0.022 0.023 | 0.020 0.019 0.021 | 0.023 | 0.020 | -9.1 % | 4.7 % | yes |
| swe2d_step | bench_1e5 | 0.085 0.085 0.086 | 0.081 0.082 0.082 | 0.085 | 0.082 | -3.6 % | 1.2 % | yes |
| swe2d_step | bench_1e6 | 0.691 0.691 0.691 | 0.667 0.670 0.667 | 0.691 | 0.667 | -3.6 % | 0.0 % | yes |
| swe2d_step | bench_16e6 | 10.668 10.668 10.668 | 10.303 10.302 10.302 | 10.668 | 10.302 | -3.4 % | 0.0 % | yes |
| euler2d_step | tiny | 0.037 0.038 0.039 | 0.034 0.035 0.034 | 0.038 | 0.034 | -10.9 % | 5.5 % | yes |
| euler2d_step | bench_1e5 | 0.421 0.421 0.421 | 0.305 0.305 0.305 | 0.421 | 0.305 | -27.6 % | 0.1 % | yes |
| euler2d_step | bench_1e6 | 3.650 3.654 3.652 | 2.587 2.588 2.587 | 3.652 | 2.587 | -29.1 % | 0.1 % | yes |
| euler2d_step | bench_16e6 | 60.055 60.043 60.057 | 42.202 42.195 42.199 | 60.055 | 42.199 | -29.7 % | 0.0 % | yes |
| ufv_gather_step | tiny | 0.011 0.011 0.012 | 0.012 0.012 0.011 | 0.011 | 0.012 | +8.5 % | 8.8 % | yes |
| ufv_gather_step | bench_1e5 | 0.066 0.066 0.066 | 0.066 0.066 0.066 | 0.066 | 0.066 | -0.0 % | 0.0 % | yes |
| ufv_gather_step | bench_1e6 | 1.483 1.483 1.484 | 1.483 1.485 1.484 | 1.483 | 1.484 | +0.1 % | 0.1 % | yes |
| ufv_gather_step | bench_4e6 | 10.979 10.978 10.979 | 10.984 10.986 10.981 | 10.979 | 10.984 | +0.0 % | 0.0 % | yes |
| mlp3 | tiny | 0.035 0.034 0.033 | 0.035 0.036 0.036 | 0.034 | 0.036 | +6.0 % | 6.0 % | yes |
| mlp3 | bench_3M | 0.265 0.266 0.265 | 0.265 0.265 0.266 | 0.265 | 0.265 | +0.0 % | 0.4 % | yes |
| mlp3 | bench_13M | 0.609 0.611 0.610 | 0.610 0.610 0.611 | 0.610 | 0.610 | +0.0 % | 0.3 % | yes |
| mlp3 | bench_79M | 3.039 3.040 3.038 | 3.040 3.043 3.040 | 3.039 | 3.040 | +0.0 % | 0.1 % | yes |
| mlp3_push | tiny | 0.081 0.082 0.084 | 0.082 0.082 0.080 | 0.082 | 0.082 | +0.0 % | 3.8 % | yes |
| mlp3_push | bench_3M | 0.185 0.186 0.187 | 0.186 0.187 0.185 | 0.186 | 0.186 | +0.0 % | 1.1 % | yes |
| mlp3_push | bench_13M | 0.572 0.573 0.573 | 0.574 0.573 0.571 | 0.573 | 0.573 | -0.0 % | 0.2 % | yes |
| mlp3_push | bench_79M | 3.100 3.100 3.100 | 3.099 3.099 3.099 | 3.100 | 3.099 | -0.0 % | 0.0 % | yes |
| mpnn_gather | tiny | 0.048 0.047 0.050 | 0.048 0.046 0.047 | 0.048 | 0.047 | -2.1 % | 6.6 % | yes |
| mpnn_gather | bench_1e5 | 4.034 4.034 4.034 | 4.015 4.018 4.015 | 4.034 | 4.015 | -0.5 % | 0.0 % | yes |
| mpnn_gather | bench_1e6 | 39.577 39.575 39.572 | 39.556 39.549 39.549 | 39.575 | 39.549 | -0.1 % | 0.0 % | yes |
| transformer_block | tiny | 0.146 0.153 0.151 | 0.154 0.148 0.150 | 0.151 | 0.150 | -0.7 % | 4.2 % | yes |
| transformer_block | bench_d512_T512 | 3.697 3.666 3.668 | 3.660 3.666 3.696 | 3.668 | 3.666 | -0.1 % | 0.8 % | yes |
| transformer_block | bench_d1024_T1024 | 22.168 22.163 22.153 | 22.143 22.147 22.127 | 22.163 | 22.143 | -0.1 % | 0.1 % | yes |
| transformer_block | bench_d2048_T1024 | 66.791 66.789 66.789 | 66.738 66.737 66.730 | 66.789 | 66.737 | -0.1 % | 0.0 % | yes |
| unet2 | tiny | 0.393 0.391 0.392 | 0.391 0.390 0.388 | 0.392 | 0.390 | -0.5 % | 0.5 % | yes |
| unet2 | bench_c16_n256 | 13.147 13.151 13.150 | 13.145 13.139 13.145 | 13.150 | 13.145 | -0.0 % | 0.0 % | yes |
| unet2 | bench_c32_n128 | 7.887 7.888 7.889 | 7.890 7.890 7.886 | 7.888 | 7.890 | +0.0 % | 0.0 % | yes |
| unet2 | bench_c64_n64 | 6.548 6.549 6.551 | 6.546 6.533 6.545 | 6.549 | 6.545 | -0.1 % | 0.0 % | yes |
| unet2 | bench_c128_n32 | 7.031 7.033 7.034 | 7.016 7.016 7.016 | 7.033 | 7.016 | -0.3 % | 0.0 % | yes |

## Run-summing helper, Part B (jobs RS1 and GATE2)

The helper `__stade_wa_runs!` (v0.6.36, `cgen_warp_runs_helper_exprs`) sums each run of adjacent lanes that add to one address and lets the last lane of the run do one atomic add. It is a segmented inclusive scan with `shfl_up` (five steps) and one ballot. It needs no `match_any_sync`. It is exact for any pattern of addresses, and it acts only when all 32 lanes are active (the same test for all lanes of the warp, so every shuffle is reached by all 32). A warp in which no run has two lanes does the plain atomic add. The gate (`cgen_warp_gate`) now sorts the calls three ways: index class `:D` to the plain atomic, `:U` (the same in every lane) to the full-warp helper `__stade_wa_add!`, and `:S` or `:X` to `__stade_wa_runs!`. The backend field `warp_runs = false` sends `:S` and `:X` back to the full-warp helper. Since v0.6.39 the keyword `warp_runs = false` of `stade_gpu` and `stade_cuda` sets this field.

The algorithm was first checked on a model of one warp (32 lanes as vectors, 20000 random address patterns of five kinds, sums equal to the direct sums to 1e-12). Then it ran on the device as a patched script (`tools/patch_runscan.py`).

**Job RS1** (Slurm 2101750, one V100, 3558 s): the scripts of GATE1 (G = gate only) against R (G plus the run helper for every call) and X (G plus the half-warp helper of ATOM1), 6 kernels, G, R, X, G, R, X per kernel. All 36 runs exit 0. Tables: `RS1_table_G_R.md`, `RS1_table_G_X.md`; data: `RS1_time.json`.

**Job GATE2** (Slurm 2101796, one V100, 4071 s): C = the scripts before the gate (commit before v0.6.33), G = the gate of v0.6.33, N = the scripts of v0.6.36 generated by `gen_stade_scripts.sh`, order C, G, N, C, G, N for each of 7 kernels, and one run of S (see below). Table: `GATE2_table_C_N.md`; data: `GATE2_time.json`, `plan_gate2.tsv`, `versions_gate2.json`.

Adjoint time, median of the call, ms (spread of C 0.0 to 1.5 %):

| kernel | size | C | N | change |
|---|---|---:|---:|---:|
| mpnn_gather | 1e5 | 4.031 | 2.730 | -32.3 % |
| mpnn_gather | 1e6 | 39.573 | 26.965 | -31.9 % |
| euler2d_step | 1e6 | 3.648 | 2.585 | -29.1 % |
| euler2d_step | 16e6 | 60.046 | 42.204 | -29.7 % |
| ufv_gather_step | 1e6 | 1.483 | 1.418 | -4.4 % |
| ufv_gather_step | 4e6 | 10.978 | 10.607 | -3.4 % |
| ufv_gather_step | 1e5 | 0.065 | 0.067 | +2.4 % |
| mlp3 | 3M | 0.266 | 0.271 | +1.9 % |
| mlp3 | 13M | 0.611 | 0.617 | +0.9 % |
| transformer_block | all sizes | | | -0.2 to +0.4 % |
| unet2 | all sizes | | | -0.1 to +0.1 % |
| mlp3_push, mlp3 79M | | | | within 0.3 % |

Reading:
- mpnn_gather at 1e6 takes 26.97 ms. The variant without any atomic add (N of ATOM2, not correct, timing only) took 26.31 ms, and the variant of this experiment is 0.66 ms above it. So the cost of the atomic adds in this kernel was the serialization of lanes that add to one address, and the run sum removes almost all of it. The 22.6 ms that remain when the helper checks are also removed (Z of ATOM3) are not touched.
- The half-warp helper of ATOM1 (X) gives -24 % on the same scripts (G to X, 39.55 to 29.92 ms). The run helper gives -32 %. X makes unet2 up to 4.4 % slower (c16_n256), the run helper does not.
- The adjoint of mpnn_gather at 1e6 is now 26.97 ms, against 35.9 ms of JAX in the profile of PROF3. The ratio to JAX goes from 1.10 to 0.75.
- mlp3 is 0.005 and 0.006 ms slower at 3M and 13M (+1.9 % and +0.9 %), and ufv_gather_step 1e5 0.002 ms slower (+2.4 %, with a spread of C of 1.5 %). mlp3 has 6 sites of class `:U` that keep the full-warp helper and 6 sites of class `:S` or `:X` on the run helper (counted in the script of N: 9 lines with `__stade_wa_add!(` and 9 with `__stade_wa_runs!(`, of which 3 of each are the definitions). The criterion set before the run (no case slower than the spread of the baseline) is not met by these small cases. The absolute cost is a few microseconds.
- Correctness: all 43 runs end with exit 0, the gradient error against the CPU is below 1e-9 in every run of C, G and N (column "same output" of the tables), and the number of elements written equals the one of the baseline.
- **Sabotage of the check (run S of GATE2):** the script of N for mpnn_gather with the scan condition `lane - d >= start` changed to `lane - d > start` (the scan then loses one term per run). The runner reports a relative gradient error of 0.81 at the size tiny and 2.0 at 1e5 (the CPU check), against 1e-15 for the correct helper. The run still exits 0: the check is in the table column, not in the exit code of the job.

Tests (no GPU): `test/validate_warp_atomics.jl` has the three-way checks (a call of `__stade_wa_add!` has a uniform index, a call of `__stade_wa_runs!` has any other, none has a lane-distinct index), 18 hand-written index shapes with the counts of the three kinds, the same shapes with `warp_runs = false`, and the definitions that go with each kind of call. `validate_atomic_guarantee`, `validate_atomic_hoist`, `validate_coarsen`, `validate_jacc_warp`, `validate_offload` and `validate_shared_private` pass. `validate_gpu_truth.jl` (its cut of the helper text now also starts at `__stade_wa_up`) passes on the host. The GPU corpus (`validate_corpus_gpu.jl`) ran on the V100 afterwards: see `../corpus_2026-11-27/`.
