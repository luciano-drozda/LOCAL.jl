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
