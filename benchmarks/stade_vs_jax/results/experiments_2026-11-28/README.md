# Experiments of 2026-11-28

Three V100 jobs (krakengpu1, float64) follow the changes of the warp-level atomic adds (`../full_2026-11-27/DISCUSSION.md`, section 4.6). SCAT1 asks whether the small slowdowns of FULLT3 are real. MPNN1 and MPNN2 ask where the time of the `mpnn_gather` adjoint goes now. All used STADE v0.6.39 code (the kit commit is in `versions_*.json`). The adjoint time is the median of the host clock with device synchronization.

## Do the small slowdowns of FULLT3 hold? (job SCAT1, Slurm 2102209, 1334 s)

Three kernels in three versions of the code, each run three times in the order O, D, F for each kernel, repeated three times (27 runs, all with exit 0, `plan_scat1.tsv`):
- O = the code of FULLT1 (STADE before v0.6.33, commit 823fa703e: the helper on every atomic add).
- D = the default code of v0.6.39 (gate and run-summing helper).
- F = v0.6.39 with `warp_runs = false` (gate only). The config F is new in `cluster/gen_stade_scripts.sh`, through `STADE_WARPRUNS=0` in `gpu/make_scripts.jl`.

Tables: `SCAT1_table_O_D.md` (D against O), `SCAT1_table_O_F.md` and `SCAT1_table_F_D.md`, made with `tools/gate_report.py`. Adjoint, median of three runs, bench sizes:

| kernel | size | O (ms) | D (ms) | D against O | F against O | spread of the three runs of O |
|---|---|---:|---:|---:|---:|---:|
| mlp3 | 3M | 0.265 | 0.270 | +1.9 % | 0.0 % | 0.4 % |
| mlp3 | 13M | 0.609 | 0.616 | +1.2 % | 0.0 % | 0.2 % |
| mlp3 | 79M | 3.039 | 3.040 | 0.0 % | +0.1 % | 0.0 % |
| ufv_scatter_step | 1e5 | 0.084 | 0.079 | -6.1 % | 0.0 % | 1.2 % |
| ufv_scatter_step | 1e6 | 1.330 | 1.279 | -3.8 % | -0.1 % | 0.3 % |
| ufv_scatter_step | 4e6 | 9.546 | 9.591 | +0.5 % | -0.1 % | 0.1 % |
| ufv_gather_step | 1e5 | 0.066 | 0.067 | +1.6 % | -1.5 % | 0.0 % |
| ufv_gather_step | 1e6 | 1.483 | 1.418 | -4.4 % | +0.7 % | 0.1 % |
| ufv_gather_step | 4e6 | 10.976 | 10.609 | -3.3 % | +0.1 % | 0.0 % |

Reading:
- **mlp3 is slower with D, and the run helper causes it.** F equals O (the gate changes no site of mlp3), and D differs from both by +1.9 % at 3M and +1.2 % at 13M. The three runs of each version agree to 0.4 %, so the difference is real. It is 5 to 7 microseconds. The sites of the run helper in mlp3 are in the kernels with 2-D blocks (32 x 16). Their index runs over `1 + (threadIdx().y - 1):16:n`, and every lane of a warp has the same `threadIdx().y`. So the index is the same in all 32 lanes, and the full-warp helper does the job at a lower cost than the run helper. The analysis called the index unknown, because it treated every launch intrinsic other than `__tid` as unknown. This is not the earlier hypothesis (the scan of the run helper at sites without a run); the earlier hypothesis was not tested. STADE v0.6.46 classes the block and grid fields and `threadIdx().y` and `.z` as uniform (`cgen_warp_thread_uniform_field`).
- **ufv_scatter_step at 4e6 is +0.5 %, not +3.1 %.** The +3.1 % of FULLT3 was noise (or an effect of that run). The three runs of D (9.589 to 9.592) are above the three of O (9.540 to 9.547) by 0.5 %, which is real and small (0.05 ms).
- **ufv_gather_step at 1e5 is +1.6 %**, which is one microsecond of 0.066 ms. The timer step is 1 microsecond. D is faster than O at 1e6 and 4e6 (-4.4 % and -3.3 %), as in FULLT3.
- The gains of D against O for ufv_scatter_step and ufv_gather_step at 1e5 to 1e6 (3.3 to 6.1 %) are the gains of FULLT3 (3.6 to 7.1 %), now with three runs of each.

## The fix for mlp3: block and grid fields are warp-uniform (STADE v0.6.46, job SCAT2, Slurm 2102388, 2939 s)

The change of `cgen_warp_class` (v0.6.46): a launch field that every lane of a warp reads alike (`blockIdx`, `blockDim`, `gridDim` and `threadIdx().y`, `.z`) has class :U. `threadIdx().x` stays :X. The sites with such an index keep the full-warp helper, and a sum of such a field and the thread index is :D (plain atomic add). Tests: `test/validate_warp_atomics.jl` (nine new hand-written cases, two new sabotage cases: all fields uniform, no field uniform; 1127 calls in 216 kernel plans, all checks pass).

Job SCAT2: D = the default code of v0.6.39 (the scripts of FULLT3), E = the default code of v0.6.46, O = the code of FULLT1 (for mlp3 only). Three runs of each in the order D, E (and O) for each kernel, repeated three times (27 runs, all with exit 0, `plan_scat2.tsv`). Helper calls in the generated scripts, D to E: mlp3 12 (6 add, 6 run) to 12 (all add), mlp3_push 12 run to 3 add and 9 run, transformer_block 48 to 38 (ten calls became plain atomic adds), unet2 88 to 86. All 12 runs of E (every kernel, every size) have the same count of elements written as D and a gradient error below 1e-9.

| kernel | size | D (ms) | E (ms) | E against D | E against O | spread of the three runs of D |
|---|---|---:|---:|---:|---:|---:|
| mlp3 | 3M | 0.271 | 0.265 | -2.3 % | 0.0 % | 0.4 % |
| mlp3 | 13M | 0.617 | 0.609 | -1.3 % | -0.2 % | 0.3 % |
| mlp3 | 79M | 3.038 | 3.038 | 0.0 % | 0.0 % | 0.1 % |
| mlp3_push | 3M, 13M, 79M | 0.185, 0.573, 3.099 | 0.185, 0.571, 3.099 | 0.0, -0.3, 0.0 % | | 0.0 to 0.6 % |
| transformer_block | d512, d1024, d2048 | 3.661, 22.146, 66.735 | 3.661, 22.132, 66.734 | 0.0, -0.1, 0.0 % | | 0.0 to 0.1 % |
| unet2 | c16, c32, c64, c128 | 13.159, 7.895, 6.549, 7.028 | 13.162, 7.896, 6.549, 7.027 | 0.0 % in all | | 0.0 to 0.1 % |

Reading: the mlp3 slowdown is gone (E equals O within 0.2 %, the spread of the runs). The other three kernels do not change, also not transformer_block with its ten sites that left the run helper. The tiny sizes (0.03 to 0.5 ms) show the timer noise (up to 6 %) and are not a result.

## Where the time of the mpnn_gather adjoint goes (jobs MPNN1 and MPNN2)

The adjoint of `mpnn_gather` at 1e6 takes 27.0 ms (config C of FULLT3 and of these jobs: 26.97 ms). The generated code has five kernels: `b_1` and `b_2` recompute the forward values (the node projections, the edge loop), `b_4_p` is the reverse edge loop, `b_3_p` and `b_5_p` are the reverse node update and the reverse node projection. Question: what is the cost of the 22.5 ms that remained in the earlier experiments (job ATOM3, `../experiments_2026-11-27/`)?

### Job MPNN1 (Slurm 2102207, 13 runs)

Variants of the generated script of config C, made with `tools/patch_mpnn_split.py` (counts of the sites in the script in brackets). Results of G0, W0 and Z0 are WRONG (races), and the times only show the cost of what was removed.
- G0: every global add through the warp helper becomes a plain add (29 sites).
- W0: every atomic add into an array of shared memory becomes a plain add (10 sites).
- Z0: G0 and W0 together. The block-final adds into the global arrays stay.
- X16: the shared-memory adds of `b_3_p` and `b_5_p` (5 sites) first add the value of the lane l + 16 (`shfl_xor`), and the lanes 0 to 15 do the atomic add. This is exact for nm = nf = 16 (lane l and lane l + 16 add to one address). The result agrees with the CPU (4.1e-12 at 1e5).

Times of the adjoint (ms), `plan_mpnn1.tsv`, the file `MPNN1_time_nsys.json`:


| variant | tiny | bench_1e5 | bench_1e6 | 1e6 against C |
|---|---:|---:|---:|---:|
| C | 0.047 | 2.731 | 26.969 | +0.00 ms (+0.0 %) |
| X16 | 0.047 | 2.668 | 26.371 | -0.60 ms (-2.2 %) |
| G0 | 0.045 | 2.358 | 23.132 | -3.84 ms (-14.2 %) |
| W0 | 0.044 | 2.444 | 24.222 | -2.75 ms (-10.2 %) |
| Z0 | 0.047 | 2.224 | 22.474 | -4.50 ms (-16.7 %) |

Per-kernel device times of one call at 1e6 (ms, the mean of 3 recorded calls, `nsys`):

| variant | b_1 | b_2 | b_3_p | b_4_p | b_5_p | sum of kernels |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.60 | 2.44 | 7.93 | 11.14 | 4.98 | 27.08 |
| X16 | 0.60 | 2.44 | 7.68 | 11.15 | 4.68 | 26.54 |
| G0 | 0.60 | 2.44 | 5.04 | 10.51 | 4.63 | 23.21 |
| W0 | 0.59 | 2.44 | 6.30 | 10.97 | 3.98 | 24.28 |

Reading:
- All atomic adds together now cost 4.5 ms (16.7 %). The earlier 17.1 ms (ATOM3) is mostly gone. The global adds cost 3.8 ms and the shared-memory adds 2.7 ms, and they overlap (together 4.5 ms).
- The global adds cost most in `b_3_p` (7.93 to 5.04 ms with G0): its two sites (`aggb` and `hb`) run in loops of 16 steps, and each call of the run helper is a segmented scan. They cost 0.6 ms in `b_4_p`.
- X16 gives 0.6 ms (-2.2 %) for the five sites it changes, with an exact result. It needs the knowledge that nm = nf = 16, so it is not a general change.
- The kernels `b_1` and `b_2` take 3.0 ms. A count of the bytes of `b_1` (about 0.38 GB: h, asrc, adst) gives 0.4 ms at 900 GB/s, and for `b_2` (about 1.5 GB) 1.7 ms. So they run at about 70 % of the DRAM bandwidth (my estimate, not a counter).

### Job MPNN2 (Slurm 2102386, 907 s, 11 runs)

Variants of the edge-loop kernel `b_4_p`, made with `tools/patch_mpnn_b4.py` on C (variant CA) and on Z0 (the others). Results of the Z variants are WRONG, timing only.
- CA: the adjoint `adstb[idx]` of the thread is kept in a register in the edge loop (exact, 2.6e-12 against the CPU at 1e5).
- ZB: no scatter into `asrcb`. ZC: no adds into `efb`. ZD: the gather `asrc[(j - 1) * nm + k]` reads `asrc[idx]` (this also removes the read of `src[e]`, which is then unused). ZE: the edge loop keeps only the read of `src[e]`. ZF: the block-final adds of all kernels become plain adds.

| variant | base | bench_1e5 (ms) | bench_1e6 runs (ms) | bench_1e6 median | against base |
|---|---|---:|---|---:|---:|
| C | - | 2.732 | 26.970 26.966 | 26.968 | - |
| CA | C | 2.705 | 26.727 26.727 | 26.727 | -0.24 ms (-0.9 %) |
| Z0 | C | 2.223 | 22.459 22.467 | 22.463 | -4.51 ms (-16.7 %) |
| ZB | Z0 | 1.809 | 17.876 | 17.876 | -4.59 ms (-20.4 %) |
| ZC | Z0 | 1.897 | 19.271 | 19.271 | -3.19 ms (-14.2 %) |
| ZD | Z0 | 2.031 | 20.279 | 20.279 | -2.18 ms (-9.7 %) |
| ZE | Z0 | 1.306 | 12.881 | 12.881 | -9.58 ms (-42.7 %) |
| ZF | Z0 | 2.222 | 22.369 | 22.369 | -0.09 ms (-0.4 %) |

Reading:
- The body of the edge loop costs 9.6 ms in Z0 (ZE: 12.9 ms against 22.5 ms). The scatter into `asrcb` costs 4.6 ms, the adds into `efb` 3.2 ms and the gather of `asrc` 2.2 ms (together 10.0 ms, so the three parts are close to additive).
- The part of the 22.5 ms that is not the body of the edge loop is: `b_3_p` 4.9 ms, `b_5_p` 3.8 ms, `b_2` 2.4 ms, `b_1` 0.6 ms and the rest of `b_4_p` (the reads of the trip counts and `aggb`, the end of the kernel) about 1.1 ms (the sum of ZE, 12.9 ms, minus the other four kernels).
- The block-final adds cost 0.09 ms. They are not a factor.
- CA gives 0.24 ms (-0.9 %). A change of STADE that keeps an array element that belongs to the thread in a register in a loop would give this gain here. It is small.
- **Conclusion.** The 22.5 ms have no single cause that a change of the atomic adds removes. About 9.6 ms are the random gathers and scatters of the edge loop (my traffic estimate for the kernel, 3.5 GB, is 3.9 ms at 900 GB/s, and the kernel takes 10.7 ms without atomics, which is about 36 % of the bandwidth), 8.7 ms are the dense weight-gradient sums of `b_3_p` and `b_5_p`, and 3.0 ms are the recomputation of the forward values. The estimates of bytes are from the source, not from a counter (`ncu` failed on these kernels, see `../profile_2026-11-27/README.md`). A change of the generated code that removes work (recompute less, or turn a scatter with 16 lanes on one address into a pull form) would be needed, and these are larger changes than the helpers.
