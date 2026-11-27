# STADE against JAX on a V100: discussion of the full benchmark

This text interprets the results in `REPORT.md`. The numbers come from two jobs. The time job FULLT1 (Slurm 2100092) and the memory job FULLM1 (Slurm 2100097) ran on one V100 (16 GB) in float64 with batch size one. The report lists all 14 kernels and 42 cases. Each gap is STADE divided by JAX. A gap below 1 means STADE is faster or uses less memory.

A statement marked **Hypothesis** was not tested. The data fit the statement, and the data do not prove it. Section 7 lists a test for each one.

## 1. Summary

1. **Memory.** STADE never uses more memory than JAX in this set. The exact gap is 0.47 for the primal (42 cases) and 0.59 for the adjoint (41 cases). STADE uses less in 38 of 42 primal cases and 37 of 41 adjoint cases. The other cases tie within 1 %.
2. **Primal runtime.** STADE is faster in 35 of 41 cases. The geometric-mean gap is 0.53. The gap is largest in the NN kernels (0.92 on average) and smallest in the flow solvers (0.28).
3. **Adjoint runtime, default configuration (C).** STADE is faster in 22 of 41 cases and slower in 19. The geometric-mean gap is 0.84. STADE is more than 5 % slower in 19 cases and more than 5 % faster in 21.
4. **Adjoint runtime, best of C and A.** The gap is 0.64. STADE is more than 5 % slower in 9 cases and more than 5 % faster in 29. Configuration A is never slower than C, and it saves more than 10 % in 31 of 42 cases.
5. **Where STADE loses.** The NN kernels (mlp3, mlp3_push, transformer_block and unet2 at c16_n256) lose in the adjoint with C. Stencil kernels at large size and ufv_gather_step also lose with C. Configuration A removes the loss for most stencil cases. It does not remove the loss for the transformer, ufv_gather_step at 1e6, mlp3 at 13M and unet2 at c16_n256.
6. **One case JAX cannot run.** JAX runs out of memory on euler2d_step at 16e6 (it asks for 8 GiB in one allocation). STADE runs this case and uses 2.0 GB for the adjoint. No gap exists for it.
7. **Which option gives the gain of A.** Job ABL1 (section 4.3) shows that `primal_outputs = false` gives most of it (28 cases: 0.80 of the C time, against 0.94 for `fresh_gradient` alone and 0.76 for both). `fresh_gradient` is the gain for mlp3_push and adds to P for mlp3 and wave2d_step. Neither option changes unet2.
8. **Profile.** The `nsys` profile of four cases (section 4.5) shows that the remaining losses are device time and not call overhead. Three of the cases that lose are the transformer adjoint (generated kernels, 5 ms), the mpnn_gather adjoint (three kernels with atomic adds hold 92 % of the time, but removing most of the adds saves 14 % and removing all of them 34 %) and the mlp3 adjoint (an extra pass over the gradient array with C, and lower bandwidth with A). `ncu` gave no counter.

## 2. How to read the numbers

- **Headline timer.** The runtime tables use the host clock with a device synchronization around one call. The event timer gives the same conclusions: the gap is 0.49 for the primal, 0.79 for the adjoint with C and 0.61 for the best of C and A.
- **Adjoint definition.** JAX measures `jit(vjp)`, which includes the forward pass. STADE C measures the generated adjoint with its default options. STADE A drops the primal outputs and starts from a fresh gradient (`primal_outputs = false`, `fresh_gradient = true`). "Best" means the lower time of C and A for each case.
- **Memory definition.** The exact reading is the memory that arrays and calls need, with no allocator rounding. For STADE it is the live arrays plus the bytes that one call allocates. For JAX it is the XLA peak of live bytes. The driver reading is what `nvidia-smi` shows. Allocators round it up, and it holds one-time library memory. The driver gap exists only for cases where both readings are at least 34 MB (23 primal cases, 27 adjoint cases). The exact gap is the headline (decision D7 in `PROCEDURE.md`).
- **Noise.** One job gives one run per case. A second run of the whole job (FULLT2, Slurm 2100137, same node, same plan) agrees with FULLT1. The median change of one case is 0.11 % for the STADE adjoint (C), 0.13 % for the STADE adjoint (A), 0.49 % for the STADE primal and 0.2 % for the JAX primal and adjoint. The largest change is 3.0 % for the JAX adjoint, 2.1 % for the STADE adjoint and 6.3 % for the STADE primal (transformer_block d512_T512). The headline gaps are the same: 0.533 and 0.533 (primal), 0.836 and 0.835 (adjoint C), 0.643 and 0.642 (best). No case changes class in the adjoint gap C. Both jobs ran on the same node, so a difference between nodes is not covered. A difference below 5 % stays a tie, as a conservative rule.
- **Averages.** A geometric mean over cases treats each case equally. The case list holds 3 sizes for most kernels, so the mean depends on the case list. Use the per-case tables for any claim about one kernel.

## 3. Runtime of the primal

| Kernel family | Cases | Geometric-mean gap | Range |
|---|---:|---:|---|
| Stencil (heat3d_step, heat3d_multi, heat1d_multi, wave2d_step, burgers2d_step) | 15 | 0.45 | 0.18 to 0.93 |
| Flow (swe2d_step, euler2d_step) | 5 | 0.28 | 0.22 to 0.38 |
| Unstructured (ufv_scatter_step, ufv_gather_step, mpnn_gather) | 8 | 0.47 | 0.23 to 0.97 |
| NN (mlp3, mlp3_push, transformer_block, unet2) | 13 | 0.92 | 0.51 to 1.44 |

- STADE is slower in 6 cases. Four are unet2 (1.44, 1.41 and 1.27 at c16_n256, c32_n128 and c64_n64, and 1.02 at c128_n32). Two are transformer_block (1.03 at d1024 and 1.02 at d2048). The cases at 1.02 and 1.03 are ties.
- The primal gap is smallest at small sizes (0.28 for the smallest size of each non-NN kernel) and grows with size (0.54 for the largest size). JAX times below 0.15 ms form a group of 12 cases with a gap of 0.34. The other 29 cases have a gap of 0.64. Many JAX times in the small cases lie between 0.09 and 0.12 ms.
- **Hypothesis.** JAX has a fixed cost per call for host dispatch. This cost explains part of the small-case advantage of STADE. It would not exist on a larger case. The data fit the hypothesis because the advantage shrinks with size, but they do not separate dispatch cost from kernel quality. Do not quote the small-case gaps as a kernel-quality result (see section 7, test 1).
- Both frameworks run the NN primal at nearly the same speed in the large cases (transformer_block at d1024 and d2048, mlp3 79M). The data do not show the reason. The data do not show which device library each framework calls for the matrix products.

## 4. Runtime of the adjoint

| Kernel family | Gap C | Gap best |
|---|---:|---:|
| Stencil | 0.81 | 0.52 |
| Flow | 0.39 | 0.34 |
| Unstructured | 0.85 | 0.68 |
| NN | 1.14 | 1.01 |
| All 41 cases | 0.84 | 0.64 |

### 4.1 Where STADE is faster

- **Flow solvers.** swe2d_step runs at 0.15 to 0.42 of the JAX time (0.23 at 16e6). euler2d_step at 1e6 runs at 0.61 (0.55 with A). These are the largest wins at scale.
- **Small stencil and scatter cases.** The gap is 0.2 to 0.6 at 1e5 for burgers2d_step, heat3d_step, wave2d_step, ufv_scatter_step and ufv_gather_step. Section 3 gives a possible part of the reason (dispatch cost).
- **ufv_scatter_step at 4e6.** The gap is 0.61 with C and 0.39 with A.
- **JAX swe2d_step at 1e6.** The JAX adjoint takes 5.5 ms, which is 8.1 times its primal. At 16e6 the JAX ratio is 5.4. The 1e6 ratio is out of line with the other sizes. The data give no cause. Check this case again before it appears in a text (section 7, test 5).

### 4.2 Where STADE is slower (configuration C)

The table lists 13 of the 19 slower cases. The other six have gaps of 1.07 to 1.19 (see `REPORT.md`).

| Case | Gap C | Gap A or best |
|---|---:|---:|
| heat1d_multi 1e6_t32 | 1.64 | 0.85 |
| heat3d_step 8e6 | 1.66 | 0.82 |
| heat1d_multi 1e6_t8 | 1.53 | 0.70 |
| mlp3 13M | 1.52 | 1.29 |
| mlp3_push 79M | 1.48 | 1.14 |
| mlp3 79M | 1.46 | 1.05 |
| heat3d_multi 1e6 | 1.46 | 0.91 |
| ufv_gather_step 1e6 | 1.43 | 1.26 |
| mlp3_push 13M | 1.41 | 0.94 |
| transformer_block d1024_T1024 | 1.28 | 1.21 |
| transformer_block d2048_T1024 | 1.27 | 1.17 |
| wave2d_step 16e6 | 1.21 | 0.85 |
| unet2 c16_n256 | 1.17 | 1.17 |

Three groups appear.

1. **Stencil and MLP cases that A repairs.** The gap drops below or near 1 with A (for example heat3d_step 8e6 from 1.66 to 0.82). C pays something that A does not pay. The ablation of section 4.3 shows which option removes the cost: `primal_outputs = false` for the stencil, ufv_gather_step and transformer cases, `fresh_gradient` for mlp3_push, and both options for mlp3 and wave2d_step.
2. **Cases that A does not repair.** These are the transformer at d1024 and d2048 (1.17 to 1.21 with A), ufv_gather_step at 1e6 (1.26), mlp3 at 13M (1.29), mlp3_push at 79M (1.14) and unet2 at c16_n256 (1.17). The options are not the cause of these gaps, and no option removes them. The data show no cause. **Hypothesis.** The generated adjoint code or its device kernels are slower than the XLA code for these shapes.
3. **Small effects.** unet2 at c32_n128, c64_n64 and c128_n32 is 9 to 14 % faster than JAX. mpnn_gather at 1e5 is a tie.

### 4.3 Ablation of the two options (job ABL1)

Job ABL1 (Slurm 2100116) ran four configs of the adjoint on 9 kernels (28 cases): C (defaults), P (`primal_outputs = false`), G (`fresh_gradient = true`) and A (P and G). The report is `../ablation_2026-11-26/ABLATION.md`.

| Family | Cases | P / C | G / C | A / C |
|---|---:|---:|---:|---:|
| Stencil | 12 | 0.63 | 0.97 | 0.62 |
| Unstructured (ufv_gather_step) | 3 | 0.85 | 1.00 | 0.85 |
| NN | 13 | 0.97 | 0.91 | 0.88 |
| All | 28 | 0.80 | 0.94 | 0.76 |

The gap to JAX over these 28 cases is 1.01 for C, 0.81 for P, 0.95 for G and 0.77 for A. STADE is slower than JAX in 17 cases with C, 11 with P, 16 with G and 9 with A.

- **P gives most of the gain.** In 14 cases P gives the gain and G gives less than 3 %. These are heat3d_multi, heat1d_multi, heat3d_step 1e5, wave2d_step 1e5, ufv_gather_step (11 to 22 %) and the transformer (5 to 8 %). P alone gives 35 to 54 % for the heat kernels at 1e6 and above.
- **G gives the gain for mlp3_push (13 to 33 %).** P has no effect there, because the generated code is the same as C. G adds a part for mlp3 (7 to 20 %) and wave2d_step at 1e6 and 16e6 (14 %), where P gives its own part. G gives 3 to 5 % for heat3d_step at 1e6 and 8e6.
- **Neither option changes unet2.** The four unet2 sizes show 0 % for both options.
- **The two gains add up or overlap.** For mlp3 they add up (79M: 8 % and 20 % give 28 %). For wave2d_step and heat3d_step they overlap a little (wave2d_step 16e6: 25 % and 14 % give 30 %).
- **Same code, same time.** For 5 of the 9 kernels, at least one pair of configs generates the same code. These 32 pairs differ by a median of 0.1 %. The largest difference is 6 % (heat3d_multi 1e5, a 0.05 ms case). The runs also reproduce FULLT1 (see section 2).
- **What the options change in the contract.** With P, the adjoint call does not write the primal outputs of the kernel. A user who needs the primal values must call the primal. With G, the gradient arrays must be zero on entry, and each reset allows one adjoint call. A loop that accumulates gradients over several calls cannot use G. The JAX adjoint of this benchmark returns the gradient only, so both options fit the comparison. They change what a user can do with one call.

The source of STADE explains the effect of G. A gradient update `g[k] = g[k] + e` becomes the store `g[k] = e` when the gradient is zero on entry, which saves one read pass over the array. G changes the code only when the analysis proves its conditions for a store. For heat1d_multi, heat3d_multi, the transformer and unet2 the pass left the code unchanged, so no gradient store met the conditions. The data do not show which condition failed.

### 4.4 Ratio of adjoint time to primal time

The geometric-mean ratio is 3.27 for STADE (C) and 2.09 for JAX. Two groups explain most of the difference.

- **mpnn_gather.** The ratio is 11.2 (1e5) and 11.4 (1e6) for STADE, against 2.6 for JAX. The STADE primal is four times faster than JAX, and the STADE adjoint is as slow as the JAX adjoint (gap 1.00 and 1.08). The adjoint of a gather is a scatter-accumulate. The STADE adjoint compiles an accumulation with a data-dependent index to an atomic add. **Hypothesis.** Atomic adds with many conflicts limit the STADE adjoint on this kernel. A device profile would show it (section 7, test 3).
- **NN kernels.** The ratio for mlp3 and mlp3_push is 2.9 to 5.0 (C), against 1.9 to 2.5 for JAX. The A configuration lowers the mlp3 13M ratio from 4.97 to 4.20. The options explain part of the difference and not all of it.

For unet2 the JAX ratio is higher than the STADE ratio (6.5, 5.5, 4.0 against 5.3, 3.5, 2.7 at three sizes). The JAX adjoint of unet2 is slow relative to its primal. This is why STADE is faster in its adjoint at three of four sizes while it is slower in its primal.

### 4.5 Profile of four cases (jobs PROF3 and PROF8)

The profile used `nsys` for the device time of each kernel (job PROF3, `results/profile_2026-11-27/`). Four cases were profiled in the primal and in the adjoint, for STADE (C and A) and for JAX: mpnn_gather 1e6, transformer_block d1024_T1024, mlp3 13M and swe2d_step 1e6. A value is the mean of 3 calls. "Kernel ms" is the sum of the device times of all kernels of one call. "Bench ms" is the time of one call in FULLT1 (host clock). The tool `ncu` gave no usable counter for the kernels that matter (see the end of this section).

| case (adjoint) | STADE C kernel ms (launches) | STADE A kernel ms (launches) | JAX kernel ms (launches) | STADE C / A / JAX bench ms |
|---|---:|---:|---:|---:|
| mpnn_gather 1e6 | 40.1 (5) | 39.4 (5) | 35.9 (36) | 39.7 / 39.6 / 36.8 |
| transformer_block d1024_T1024 | 23.9 (41) | 21.9 (39) | 16.6 (59) | 22.3 / 21.0 / 17.4 |
| mlp3 13M | 0.60 (5) | 0.49 (5) | 0.35 (12) | 0.72 / 0.61 / 0.47 |
| swe2d_step 1e6 | 0.73 (1) | - | 5.41 (9) | 0.80 / - / 5.52 |

In the primal, STADE uses 1 to 15 launches and JAX uses 6 to 25. The kernel times are 3.5 against 14.7 ms for mpnn_gather, 0.14 against 0.62 ms for swe2d_step, 0.133 against 0.132 ms for mlp3 and 6.7 against 6.0 ms for the transformer.

**Time outside the kernels (test 1, in part).** The bench time of a call minus its kernel time is small for both frameworks. Over the 8 primal and 8 adjoint (C) comparisons, it is between -1.7 and +0.12 ms for STADE and between -0.6 and +0.9 ms for JAX. For the calls shorter than 1 ms the difference is 0.01 to 0.12 ms for both. For the longer calls it is below 5 % of the call. (A negative value means that the kernels took longer in the profile run than in the benchmark.) At these sizes both frameworks are limited by device work, and the call overhead of STADE is not larger than the call overhead of JAX. This does not test the small cases, where the whole call lasts 0.02 to 0.1 ms. The host time during profiling is not usable for JAX: the trace of CUDA graphs makes it 1.1 to 35 times the bench time.

**mpnn_gather (test 3).** The STADE adjoint is limited by device work and not by launches or host time: the kernels take 40.1 ms of a call of 39.7 ms. Three kernels (`b_3_p`, `b_4_p` and `b_5_p`) take 13.5, 15.1 and 8.5 ms, which is 92 % of the device time (92 % also for A). Each of them accumulates into arrays with atomic adds (the weight gradients go through shared memory first). A count of the global atomic adds in their source, for nn = 1e6, nm = nf = 16 and a mean in-degree of 6 (6e6 edges), gives:

| kernel | global atomic adds per call (estimate) | ms | ms per 1e8 adds |
|---|---:|---:|---:|
| `b_3_p` | 5.1e8 | 13.5 | 2.6 |
| `b_4_p` | 4.8e8 | 15.1 | 3.1 |
| `b_5_p` | 2.6e8 | 8.5 | 3.3 |

The time of the three kernels grows with the number of atomic adds (3.0 to 3.8e10 adds per second). In `b_3_p` and `b_5_p` all global adds, and in `b_4_p` four adds of five, go from 16 consecutive threads to the same address in the same step. A warp holds two such groups, so the warp-level pre-reduction of STADE (`__stade_wa_add!`, which needs all 32 lanes on one address) does not apply. The other two kernels (`b_1`, `b_2`) have no atomic in their source, and `ncu` shows 0 atomic instructions for them. These numbers fit the hypothesis that atomic adds limit the adjoint of mpnn_gather, but the counts come from the source and not from a counter. A test then showed that the hypothesis is only a small part of the answer (next paragraph). **Test of the hypothesis (job ATOM1, `results/experiments_2026-11-27/`).** In the generated script, the warp-level add reduced over the half-warp (16 lanes) when the 16 lanes add to one address. This removes most of the global atomic adds (about 1.1e9 of 1.25e9 if all half-warps are active). The adjoint went from 39.68 to 34.02 ms at 1e6 and from 4.14 to 3.57 ms at 1e5, which is 14 % less (two runs of each, same values). The gradient at 1e5 agrees with the CPU to 3e-12, like the unchanged code. So the collisions of the atomic adds cost about 5.7 ms of 39.7 ms. They are not the main limit. The other 34 ms are not explained. The kernels are not limited by DRAM bandwidth either by a rough count (about 2 GB of traffic for `b_4_p`, which is 2 ms), and not by arithmetic (about 0.6 ms). A latency limit of the dependent loads in the loops over edges is possible and was not tested. A second test (job ATOM2) replaced all 44 atomic adds by plain adds. The result is wrong, and the time is only a lower bound. The adjoint took 26.3 ms at 1e6 (0.66 of C). So the atomic adds cost at most 13.4 ms (34 %), the half-warp reduction gets 5.7 ms of that (43 %), and the rest is in adds that it does not change (shared-memory adds and the random scatter into `asrcb`). Without any atomic add the three kernels still take 26.3 ms, and the cause of that time is not known.

JAX uses 15.7 ms in cuBLAS kernels and 20.2 ms in 13 other kernels (scatter fusions), so its adjoint does the same accumulations in a different way and is 10 % faster in device time.

**transformer_block (test 4).** The excess of STADE is device time, and launches do not cause it. STADE has fewer launches than JAX (41 against 59) and 1.44 times the kernel time with C (1.32 with A). Two parts make the difference:

| adjoint | GEMM ms (launches) | other kernels ms (launches) |
|---|---:|---:|
| STADE C | 16.4 (13) | 7.5 (28) |
| STADE A | 14.6 (12) | 7.3 (27) |
| JAX | 14.3 (18) | 2.4 (41) |

Both frameworks call the same cuBLAS routines (`volta_dgemm_*`). STADE spends 2.1 ms (C) or 0.3 ms (A) more in GEMMs than JAX. The larger part is in the kernels that STADE generates: 5.1 ms (C) and 4.9 ms (A) more than the other kernels of JAX. Eight of them are generated tiled batched matrix products of the attention (64 x 64 tiles, 16 heads, 1024 x 1024 scores, inner length 64). Each is a product of about 2.15 GFLOP. They take 540 to 730 us each, 4.86 ms together, which is 3.5 TFLOP/s on average and about half of the FP64 peak of the V100 (7 TFLOP/s). Another generated kernel (`b_19_a`) is elementwise over the 16.8e6 score values. It takes 0.65 ms for four streams of 134 MB, which is the bandwidth floor. In the JAX adjoint, the cuBLAS kernels `volta_dgemm_64x64_nn` and `_nt` take about 400 us per launch, which would be 5.4 TFLOP/s if they are the attention products (this was not checked). So the likely cause is that STADE does the attention products with its own tiled kernel at about half of the speed of cuBLAS. The option A lowers the generated-kernel part only by 0.2 ms. The checkpoint option was not tested.

**swe2d_step 1e6 (test 5).** The JAX adjoint is slow because of device work. Its 9 kernels take 5.41 ms, and two fused kernels take 5.03 ms of them (2.84 and 2.20 ms). The bench time (5.52 ms) is almost the same, so the cost is not host time or launch count. The XLA kernels of the primal take 0.62 ms in total, so the adjoint costs 8.7 times the primal in JAX. STADE does the adjoint in one kernel of 0.73 ms (5.3 times the primal of 0.14 ms). The profile does not say why XLA builds these two kernels. The HLO of the program was not read.

**mlp3 13M.** The STADE adjoint has 5 kernels and uses 0.60 ms (C) and 0.49 ms (A). JAX has 12 kernels and uses 0.35 ms. A weight matrix has 2048 x 2048 float64 values (33.5 MB), and one pass over it takes about 37 us at 900 GB/s. The source of the three layer kernels (`b_3`, `b_4` and `b_5`) shows how many passes they must make. With C, `b_4` and `b_5` read the weights, read the gradient array of the weights and write it again (3 passes), and `b_3` also recomputes the output (4 passes). With A, each kernel needs 2 passes. The floors are 112 us (3 passes) and 75 us (2 passes). The kernels take 157 us with C (1.4 times the floor, 640 GB/s) and 133 us with A (1.8 times the floor, 500 GB/s). JAX does one layer in two kernels of 41 to 44 us, one pass each (about 800 GB/s). So the loss with C comes mostly from the extra read of the gradient array. The option `fresh_gradient` removes it. With A, the kernels still run at lower bandwidth than the JAX kernels. The cause of this (the reduction into the vector of the layer input, for example) was not measured, and `ncu` could not measure it.

**What `ncu` could not do.** `ncu` failed on the STADE kernels in all attempts (jobs PROF3 to PROF9). Kernel replay ends with `LaunchFailed` for the second kernel of the range. Application replay records only the first 1 or 2 kernels of the range. The cause is not known. The details are in `results/profile_2026-11-27/README.md`.

## 5. Memory

### 5.1 Results

| Kernel family | Primal exact gap | Adjoint exact gap |
|---|---:|---:|
| Stencil | 0.53 | 0.65 |
| Flow | 0.29 | 0.23 |
| Unstructured | 0.54 | 0.66 |
| NN | 0.46 | 0.72 |
| All cases | 0.47 | 0.59 |

- **No case with more memory.** STADE ties JAX in 4 primal cases and 4 adjoint cases. These are the MLP cases at 3M and 13M, where the weights dominate and both frameworks hold only the weights and the gradient.
- **Largest savings.** unet2 primal (0.11 to 0.25), euler2d_step primal (0.15 to 0.32), mpnn_gather primal (0.19 to 0.24), swe2d_step (0.25 to 0.33 in the adjoint).
- **Smallest savings.** The transformer adjoint (0.77 to 0.93), ufv_gather_step (0.78 to 0.87 in the primal) and the large MLP cases (0.84).
- **Stacks.** The adjoint stacks are zero for 12 of 14 kernels. They hold 38 and 384 MB for mpnn_gather (18 % of its adjoint memory) and 7 to 55 MB for unet2. Configuration C uses `keep_push_pop = false` and `fuse_ii_loops = true`.
- **Configuration A.** A saves 0 to 1 % of memory. Only the transformer shows a change (about 1 %). A is a runtime option, not a memory option.
- **euler2d_step at 16e6.** STADE uses 1.0 GB for the primal and 2.0 GB for the adjoint. JAX asks for one 8 GiB buffer and fails on the 16 GB device.

### 5.2 Why STADE uses less

The memory gap is 1.00 for the MLP cases at 3M and 13M, where the inputs are almost the whole problem. So the saving elsewhere comes from the intermediate arrays that JAX keeps. The generated code of STADE writes into the output arrays of the kernel. JAX allocates intermediate buffers for each stage and keeps some for the backward pass. **Hypothesis.** The XLA buffer assignment does not reuse the intermediates of the stencil and flow kernels as well as the hand-written loop structure does. For the adjoint, JAX stores the residuals of the forward pass. STADE recomputes or snapshots only what it needs (the stacks above). The data show the size of the effect. They do not show the XLA mechanism.

### 5.3 Reading the driver gaps

The driver gap is 0.42 for the primal (23 cases) and 0.45 for the adjoint with C (27 cases). The cases differ from the exact set, so the means do not compare directly. On one case the driver gap is smaller than the exact gap because JAX allocates in large regions. For euler2d_step 16e6 at the primal, the JAX driver reading is 4297 MB and its exact peak is 3250 MB. The driver gap is 0.24 and the exact gap is 0.32. Use the driver reading to answer "will it fit on the device". Use the exact reading to compare the two frameworks.

## 6. Validity and limits

- **Checks.** All checks G1 to G9 pass in both jobs. There are 8 notes. Six notes are G6 notes: the JAX driver reading for inputs of 1.6 MB is 0, which is allocator granularity and not an error. Two notes are the JAX out-of-memory for euler2d_step at 16e6.
- **Numerical parity.** The tiny verification case agrees within 1e-10 for all kernels (G1). Every STADE run changed array elements (G2).
- **Same source.** The two jobs ran the same STADE source and the same kit (hash 5a6d98c18367b4a4). The two recorded commits differ only in the archived results.
- **Library versions.** STADE ran on Julia 1.11.9 with CUDA.jl 6.3.1. JAX ran as version 0.12.0 with jaxlib 0.12.0 in Python 3.12.7. The driver is 550.90.12 (CUDA 12.4 runtime for STADE, CUDA 12.9 for XLA). A different version of either framework can change the result.
- **Not covered.**
  - One GPU model, one precision (float64), batch size one. Other GPUs and float32 can change the order of the results.
  - JAX appears only as `jit`. No hand-tuned JAX version exists, and no custom call.
  - STADE has the checkpoint option off. It can cut memory at a cost in time.
  - The two timers differ. The caches are not flushed, and STADE copies its inputs between calls.
  - Each case ran once per job. Two jobs on the same node agree (section 2). No confidence interval exists, and no other node was tested.
  - Cases under 0.1 ms are mostly launch cost (section 10, item 9 of `PROCEDURE.md`).
  - The kernels are written in the STADE house style, and the JAX versions were written to match. A reviewer can find a faster JAX version for a case.

## 7. Tests that would settle the open questions

| # | Question | Test | Cost |
|---|---|---|---|
| 1 | Does call overhead explain the small-case gaps? | Time each case with a loop of 20 calls inside one `jit` (JAX) and one launch sequence (STADE), or measure the launch overhead of an empty kernel in both. | Partly done (section 4.5). At the large sizes of the profile, the time outside the kernels is at most 0.12 ms for both. The small cases were not profiled. |
| 2 | Which option makes C slower than A? | Done (job ABL1, section 4.3). `primal_outputs = false` gives most of the gain. `fresh_gradient` gives the rest for mlp3, mlp3_push and wave2d_step. | Done |
| 3 | Do atomic adds limit mpnn_gather? | Profile the adjoint kernels with `nsys` or `ncu` (kernel time, atomic throughput). | Done by experiment (section 4.5, job ATOM1). Three kernels with atomic adds take 92 % of the device time. A half-warp reduction that removes most of the adds saves 14 % of the adjoint time. With no atomic adds at all (a wrong result, timing only) the adjoint is 34 % faster. So the atomic adds are at most a third of the time, and the main limit is not known. `ncu` gave no counter. |
| 4 | Why is the transformer adjoint 17 to 28 % slower? (The range covers A and C.) | Profile both adjoints and compare the time per kernel. Test the checkpoint option. | Profile done (section 4.5). The excess is device time: 5 ms in the kernels that STADE generates and 0.3 to 2 ms in GEMMs. The checkpoint option is not tested. |
| 5 | Why is the JAX swe2d_step 1e6 adjoint slow relative to its primal? | The repeat FULLT2 reproduced it (5.517 and 5.516 ms), so it is not noise. Profile the XLA kernels of this case. | Profile done (section 4.5). Two fused XLA kernels take 5.0 of 5.4 ms. The reason that XLA builds them was not read from the HLO. |
| 6 | How large is the noise? | Done (job FULLT2, section 2). | Done |

Tests 2 and 6 are done. Tests 3, 4 and 5 are profiled and locate the cost (section 4.5). Test 1 is done only for the large sizes. The tests that remain: the split of the remaining atomic cost between the shared-memory adds and the scatter into `asrcb`, the cause of the 26 ms that remain without atomic adds, `ncu` counters for these kernels (the tool failed, section 4.5), the attention products of the transformer through cuBLAS, the HLO of the JAX swe2d_step adjoint, a profile of the small cases and the checkpoint option for the transformer. A decision about the default options of STADE belongs to the owner (section 8, item 7). A run on a second node or a second GPU would test the limit that FULLT2 leaves open.

## 8. Conclusions

1. On this V100 and this case list, the STADE adjoint uses less memory than the JAX adjoint in every case (gap 0.59). The primal does the same (gap 0.47). This result does not depend on the timer or on the noise.
2. STADE is faster than JAX in the primal in 35 of 41 cases (gap 0.53). With the best of C and A, STADE is more than 5 % faster in the adjoint in 29 of 41 cases (gap 0.64).
3. With the default options, the adjoint is slower than JAX for the NN kernels (gap 1.14) and for several large stencil cases. Configuration A removes most of the loss for stencils. For the transformer (17 to 21 %) and for ufv_gather_step at 1e6 (26 %) a loss remains.
4. The gap grows with size. The advantage at small sizes is partly a launch-cost effect and does not carry to large problems.
5. The ratio of adjoint time to primal time is worse for STADE (3.27 against 2.09). The reason is mpnn_gather and the MLP cases. These are the first places to improve.
6. The ablation (section 4.3) shows that `primal_outputs = false` gives most of the gain of config A. The gap to JAX on the 9 ablated kernels is 1.01 with the defaults and 0.81 with this option alone. `fresh_gradient` adds a gain for mlp3, mlp3_push and wave2d_step. Neither option changes unet2. The options change the transformer by at most 8 % and ufv_gather_step at 1e6 and 4e6 by at most 13 %.
7. The options change the contract of the adjoint call (section 4.3). Whether they should be the default is a decision for the owner. The data show a gain of 24 % on average (geometric mean over 28 cases) if both are on.
8. The repeat FULLT2 gives the same results as FULLT1 (section 2), so none of the conclusions above depends on noise in this node.
9. The profile (section 4.5) locates the losses that remain after A, and it shows that device work causes them and not call overhead. In the transformer adjoint, STADE spends 5 ms more than JAX in the kernels that it generates (mostly its own tiled attention products at about half of the FP64 peak), and 0.3 to 2 ms more in GEMMs. In the mpnn_gather adjoint, three kernels with atomic adds hold 92 % of the device time. A half-warp reduction that removes most of their atomic adds saves 14 % of the adjoint time, and a version without any atomic add (wrong result, timing only) saves 34 %. So the atomic adds cost at most a third of the time. In the mlp3 13M adjoint, the loss with C comes mostly from an extra read of the gradient array (3 passes over the weights instead of 2), and the A kernels run at about 500 GB/s against about 800 GB/s for the JAX kernels.
10. The profile gives no proof of a mechanism, and the one test of a mechanism (the atomic adds of mpnn_gather) gave a gain of 14 % (34 % at most) and not the larger gain that the profile suggested. `ncu` failed on the STADE kernels, so the atomic counts come from the source. The HLO of the JAX swe2d_step adjoint was not read, and the small cases were not profiled. Section 7 lists these tests.
