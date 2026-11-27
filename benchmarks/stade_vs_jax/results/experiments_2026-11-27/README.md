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
