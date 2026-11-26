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

## 2. How to read the numbers

- **Headline timer.** The runtime tables use the host clock with a device synchronization around one call. The event timer gives the same conclusions: the gap is 0.49 for the primal, 0.79 for the adjoint with C and 0.61 for the best of C and A.
- **Adjoint definition.** JAX measures `jit(vjp)`, which includes the forward pass. STADE C measures the generated adjoint with its default options. STADE A drops the primal outputs and starts from a fresh gradient (`primal_outputs = false`, `fresh_gradient = true`). "Best" means the lower time of C and A for each case.
- **Memory definition.** The exact reading is the memory that arrays and calls need, with no allocator rounding. For STADE it is the live arrays plus the bytes that one call allocates. For JAX it is the XLA peak of live bytes. The driver reading is what `nvidia-smi` shows. Allocators round it up, and it holds one-time library memory. The driver gap exists only for cases where both readings are at least 34 MB (23 primal cases, 27 adjoint cases). The exact gap is the headline (decision D7 in `PROCEDURE.md`).
- **Noise.** One job gives one run per case. The earlier smoke jobs showed a cross-job spread up to 3.3 % for STADE and up to 11 % for JAX on one 1e5 adjoint case. Treat a difference below 5 % as a tie.
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

1. **Stencil and MLP cases that A repairs.** The gap drops below or near 1 with A (for example heat3d_step 8e6 from 1.66 to 0.82). C pays something that A does not pay. A changes two options at the same time, so the data do not show which option costs the time. The extra time with C is about 0.4 ms for heat3d_step 8e6 (128 MB arrays) and about 0.9 ms for mlp3 79M (630 MB of inputs). These values are the order of magnitude of one more pass over arrays of the size of the gradient at V100 bandwidth. **Hypothesis.** The extra cost with C is one or two passes over gradient-sized arrays (writing the primal outputs, setting the gradient to zero, or both). A test with each option alone would show it (section 7, test 2).
2. **Cases that A does not repair.** These are the transformer at d1024 and d2048 (1.17 to 1.21 with A), ufv_gather_step at 1e6 (1.26), mlp3 at 13M (1.29), mlp3_push at 79M (1.14) and unet2 at c16_n256 (1.17). A is not the cause of these gaps. The data show no cause. **Hypothesis.** The generated adjoint code or its device kernels are slower than the XLA code for these shapes.
3. **Small effects.** unet2 at c32_n128, c64_n64 and c128_n32 is 9 to 14 % faster than JAX. mpnn_gather at 1e5 is a tie.

### 4.3 Ratio of adjoint time to primal time

The geometric-mean ratio is 3.27 for STADE (C) and 2.09 for JAX. Two groups explain most of the difference.

- **mpnn_gather.** The ratio is 11.2 (1e5) and 11.4 (1e6) for STADE, against 2.6 for JAX. The STADE primal is four times faster than JAX, and the STADE adjoint is as slow as the JAX adjoint (gap 1.00 and 1.08). The adjoint of a gather is a scatter-accumulate. The STADE adjoint compiles an accumulation with a data-dependent index to an atomic add. **Hypothesis.** Atomic adds with many conflicts limit the STADE adjoint on this kernel. A device profile would show it (section 7, test 3).
- **NN kernels.** The ratio for mlp3 and mlp3_push is 2.9 to 5.0 (C), against 1.9 to 2.5 for JAX. The A configuration lowers the mlp3 13M ratio from 4.97 to 4.20. A explains part of the difference and not all of it.

For unet2 the JAX ratio is higher than the STADE ratio (6.5, 5.5, 4.0 against 5.3, 3.5, 2.7 at three sizes). The JAX adjoint of unet2 is slow relative to its primal. This is why STADE is faster in its adjoint at three of four sizes while it is slower in its primal.

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
  - Each case ran once per job. No confidence interval exists.
  - Cases under 0.1 ms are mostly launch cost (section 10, item 9 of `PROCEDURE.md`).
  - The kernels are written in the STADE house style, and the JAX versions were written to match. A reviewer can find a faster JAX version for a case.

## 7. Tests that would settle the open questions

| # | Question | Test | Cost |
|---|---|---|---|
| 1 | Does call overhead explain the small-case gaps? | Time each case with a loop of 20 calls inside one `jit` (JAX) and one launch sequence (STADE), or measure the launch overhead of an empty kernel in both. | One short job |
| 2 | Which option makes C slower than A? | Run STADE with `primal_outputs = false` alone and `fresh_gradient = true` alone, on the 13 cases of section 4.2. | One time job on 13 cases (about 10 min) |
| 3 | Do atomic adds limit mpnn_gather? | Profile the adjoint kernels with `nsys` or `ncu` (kernel time, atomic throughput). | One profiling job |
| 4 | Why is the transformer adjoint 17 to 28 % slower? (The range covers A and C.) | Profile both adjoints and compare the time per kernel. Test the checkpoint option. | One profiling job |
| 5 | Is the JAX swe2d_step 1e6 adjoint an outlier? | Repeat the case five times in separate jobs. | One short job |
| 6 | How large is the noise? | Repeat the time job FULLT1 once. Compare each case. | One job (45 min) |

Tests 2 and 6 are the most useful. Test 2 may give a default that removes the main loss of STADE in the stencil family. Test 6 gives the noise level for every number in this text.

## 8. Conclusions

1. On this V100 and this case list, the STADE adjoint uses less memory than the JAX adjoint in every case (gap 0.59). The primal does the same (gap 0.47). This result does not depend on the timer or on the noise.
2. STADE is faster than JAX in the primal in 35 of 41 cases (gap 0.53). With the best of C and A, STADE is more than 5 % faster in the adjoint in 29 of 41 cases (gap 0.64).
3. With the default options, the adjoint is slower than JAX for the NN kernels (gap 1.14) and for several large stencil cases. Configuration A removes most of the loss for stencils. For the transformer (17 to 21 %) and for ufv_gather_step at 1e6 (26 %) a loss remains.
4. The gap grows with size. The advantage at small sizes is partly a launch-cost effect and does not carry to large problems.
5. The ratio of adjoint time to primal time is worse for STADE (3.27 against 2.09). The reason is mpnn_gather and the MLP cases. These are the first places to improve.
6. The text does not claim a cause for any loss. Section 7 lists the tests that would give one.
