# STADE handoff: checkpointing (v0.5.47 to v0.5.55)

Date: 2026-11-25. For the next session or person who continues the checkpointing work. The plan is `plan_stade_checkpoint.md` (decisions D1 to D5, all agreed by the owner). Nothing is tagged or released. A tag needs the owner's approval.

## 1. State

| release | commit | content |
|---|---|---|
| v0.5.47 | 7ee85ee | `stade_adjoint(checkpoint = true)`: strip-mined sequential loops that hold tape (CPU, unrolled path) |
| v0.5.48 | 4eda8b6 | checkpoint set (inputs a later statement rewrites, every written float scalar), refusals for what cannot be sized, forced run on the corpora |
| v0.5.49 | ab2e299 | regions that hold or follow solve loops (fixed-point reverse kept) |
| v0.5.50 | 9e94d4b | scratch arrays are not saved by a segment |
| v0.5.51 | 08fd0c8 | the checkpointed adjoint runs on the CUDA and JACC stand-ins |
| v0.5.52 | 84c9bdd | V100 run of the adjoint; serial kernels carry a scalar through an enclosing loop |
| v0.5.53 | 1c38ca7 | `stade_hvp(...; checkpoint = true)` |
| v0.5.54 | 2374a00 | V100 run of the HVP; flag-off output byte-identical to v0.5.46 |
| v0.5.55 | this commit | documentation, full regression, this note |

Remote: `origin` = `https://github.com/luciano-drozda/local.jl`, push with `git push -q origin HEAD:main`. The owner allowed commits and pushes without a request each time. Commit text starts with `[v0.5.N] `.

## 2. Design in six lines

1. The flag is `checkpoint::Bool = false` on `stade_adjoint`, `stade_adjoint_file`, `stade_hvp`, `stade_hvp_file` (D1).
2. A region is a top-level sequential loop that holds tape and has a literal trip count above 8. The kernel is strip-mined: `for t = lo:hi` becomes a segment loop `__ck_sK` around an inner loop `__ck_tlK` with the guard `if t <= hi`.
3. `ck`, the segment length, is the last argument of the kernel, of the adjoint and of `initstacks_NAME_b`. In the HVP `ck` sits before the tangent arguments. `ck < 1` raises an error (D4).
4. The work is a text post-pass on the ordinary adjoint or HVP of the strip-mined kernel. The forward loop saves a checkpoint at each segment start and runs without tape. The reverse loop restores the checkpoint, runs the segment again with tape in slots, and reverses it.
5. The checkpoint set is large and safe (D2): every array and float scalar the region writes, integer scalars that hold a value on entry, and what the region reads and a later statement writes. Scratch arrays (overwritten before they are read) are not saved.
6. The HVP saves the tangents beside the state (`<ud>_ck_stack`) and allocates the shadow tape inside the HVP (D5).

Where the code is (all in `src/STADE.jl`): `agen_ckpt_*` (region choice, rewrite, sets, adjoint, HVP sets, `agen_ckpt_collect_hdefs!`); the guard line `guard = :(if ck < 1; error(...) end)`; `cgen_guard_cond_ok` and `cgen_guard_prepend` (GPU stage keeps the `ck` guard); `cgen_serial_plan` (serial track).

## 3. Tests added

| file | what it checks |
|---|---|
| `validate_checkpoint.jl` | 21 kernel shapes, 0 to 10 steps, `ck` from 1 to above `nt`, bit for bit against the adjoint without checkpoints; sabotage cases |
| `validate_checkpoint_sets.jl` | the checkpoint set and the scratch rule |
| `validate_checkpoint_fp.jl` | regions with solve loops, 144 checks |
| `validate_checkpoint_corpus.jl` | forced region on every sequential loop of every corpus (139 kernels); `--device`, `--hvp`, `--zero` |
| `validate_checkpoint_hvp.jl` | HVP, 340 checks, three seedings, 6 sabotage cases |
| `validate_checkpoint_gpu_cpu.jl` | CUDA and JACC stand-ins against the CPU; `--hvp`, `--sabotage` |
| `measure_checkpoint_device.jl` | stack size and time on a device |
| `dump_generated_text.jl` | text of every corpus adjoint and HVP, for the byte-identity check |
| `validate_serial_loops.jl` | regression for the serial-track fix |

## 4. Results

- CPU, forced corpus: 139 kernels, none differs. 17 kernels cannot be sized or rewritten (ragged tape); they keep the adjoint without checkpoints and say why.
- HVP forced corpus: 139 of 140 checked (the unet baseline is unusable in HVP corpus mode).
- V100 (Slurm, partition `gpu`): adjoint 248 of 248 on CUDA and JACC; HVP 248 of 248.
- Memory and time on a V100, `phc_heat_steps`, 10^6 points, 100 steps: stacks 1526 MiB without checkpoints, 244 MiB at `ck` = 10; 1.9 times the time. The cost is one more forward run of every step.
- Flag off: the generated adjoint and HVP are byte-identical to v0.5.46 for every corpus kernel (`dump_generated_text.jl`).
- HVP memory on a V100 (v0.5.56, `measure_checkpoint_device.jl`, `phc_heat_steps`, 10^6 points, 100 steps; stack MiB from `initstacks` + MiB allocated inside the HVP for the shadow tape; CUDA and JACC agree): no checkpoints 1526 + 1533 (3.0 GiB), time 2.9 s; `ck` = 5: 412 + 76; `ck` = 10: 336 + 153 (0.5 GiB), time 4.0 s (1.41 times); `ck` = 20: 412 + 305; `ck` = 50: 824 + 763. 2500 steps: `ck` = 50: 1556 + 763 MiB, `ck` = 100: 1938 + 1526 MiB (the plain tape would not fit). Results bit for bit equal to the HVP without checkpoints (err 0.0).
- v0.5.57: the GPU split gate keeps a loop sequential when it writes through a lone `mod` index (item 5.1 below, now fixed). `ck_index` is back in the V100 validator: adjoint and HVP 8 of 8 on CUDA and JACC (err 1e-16).
- Full CPU regression on the rome partition (RALL55): see section 6.

## 5. Known limits and findings

1. **GPU stage and non-injective indices (pre-existing, fixed in v0.5.57 for the `mod` case).** The GPU stage offloaded a loop whose written index is derived from the loop variable by `mod` alone (`ck_index`: `u[mod(3t, n) + 1]`); the plain adjoint raced on the device too. `cgen_writes_collide_by_mod` (with `cgen_nested_scalar_defs`, which reads definitions inside branches such as the segment guard) now keeps such a loop sequential unless its array is accumulate-only. `validate_mod_collide.jl` (20 checks, sabotage) holds it; `validate_offload.jl` is unchanged (998 of 998), so no corpus loop lost its kernel. Cost: a periodic roll `y[mod(i, n) + 1] = x[i]` stays sequential. Still open: an index that is a different non-one-to-one function (for example `div(i, 2)`) is covered only by the existing `div`/`mod` rule for atomic rewrites, and a gather from an array is trusted (the scatter contract).
2. **Serial-track bug (fixed in v0.5.52).** `cgen_serial_plan` lost the value of an assigned scalar that is live on entry to a loop inside a repeating loop (`gat_reduce_scalar`, `so_inner`). The scalar is now live-out (copied back from its box). `validate_serial_loops.jl` holds the regression (a parallel inner loop is needed, or the outer loop turns serial and the test is vacuous).
3. **Accumulators.** A variable that the reverse never restores (`loss[1] = loss[1] + ...`) keeps its forward value, as without checkpoints. An array that the reverse restores at some statements and not at others ends at the state after the first segment. The shadows are exact.
4. **Not covered:** keep_push_pop = true, `fuse_ii_loops = true`, regions that are not at the top level of the kernel, nested time loops.
5. **Knowns from the GPU run:** the ingest drops `if cond; error("String") end`; the guard is put back only when its condition is arithmetic or comparison over scalar parameters.

## 6. Regression

Run on the rome partition with `tools/cluster/mk_all3.py` (114 suites, one process each, 64 at a time). The results and the failures are in the section "Regression result" below.

## 7. House rules to keep

No modules, structs or global consts. Stage-prefixed names. No corpus-specific code in `STADE.jl`. `snap_*` and `agen_*` duplicates change together. Sabotage-test every guard. Adversarial kernels first. Never tag or release without the owner's approval.

## 8. Next steps (suggestions)

1. Ask the owner about a release tag.
2. Prove one-to-one indices of the kind `mod(i, n)` over `1:n` (the periodic roll) so that they offload again.
3. Checkpointing for nested time loops and for regions below the top level.

## Regression result

Two full sweeps of the CPU suites on the rome partition, each one process per suite, 64 at a time.

- RALL55 (tree e3531b7, Slurm 2098635, 4275 s): 112 of 113 passed. The failure was `validate_seed_stable` (a bare `randn(` in the new `measure_checkpoint_device.jl`), fixed in v0.5.55 (22 of 22).
- RALL58 (tree 6c2ee0b = v0.5.57, Slurm 2099068, krakenepyc1, Julia 1.11.9, 3515 s, `--mem=100G`): 113 of 114 passed. The failure was `validate_colblock`, sabotage "the accumulated array may be used elsewhere": NOT CAUGHT. Cause: the unit input of that case reads the accumulated array beside its accumulation (`g[idx] = z[j]; z[j] = z[j] + ...`), a kernel that races on a device. The v0.5.57 gate (lone `mod` index, array not accumulate-only) now refuses that loop, so the column-block guard was no longer reached by any source. This is the new guard working, not a defect. Fixed in v0.5.58: `cb_mutants` holds a hand-edited kernel with that read, so the guard is tested directly; the sabotage is caught again (and the source case stays, wanting 0 twins).
- The rome template `run_job_cpu_parallel_rome.sbatch` now asks for 100 GB instead of 240 GB (backup `.bak-20261126` beside it). Measured peak memory of eight heavy suites on a single process: 0.8 to 1.2 GiB each, so 64 at once need about 65 to 80 GiB. Slurm confirmed ReqMem 100G; no suite was killed for memory in RALL58.

Not in the sweeps (they need a device or are skipped by `mk_all3.py`): `validate_corpus_gpu`, `validate_cublas`, `validate_gpu_truth`, `validate_corpus_flags`, `validate_own_kit`, `validate_dead_stack_real`, `validate_contraction_real`, `validate_split_real`, `validate_battle_random`, and the contraction slices (`mk_slices.py`). The V100 runs cover the device side (adjoint and HVP 248 of 248, `ck_index` 8 of 8 each). The skipped CPU suites are the open item: they were submitted once on the pre-guard tree and cancelled while queued.
