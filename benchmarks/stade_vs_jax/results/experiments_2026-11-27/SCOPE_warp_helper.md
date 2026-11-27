# Scope: a proper `STADE.jl` change for the CUDA warp-atomic helper

Status: Part A is done (v0.6.33) and measured in job GATE1, see `README.md` of this folder: euler2d_step -28 to -30 %, swe2d_step -3.5 %, mpnn_gather -0.1 %, no case slower. The expected gain for mpnn_gather in section 3 (up to 3.7 ms) did not appear: the check cost is in the sites of class S and X, which Part B treats. Part B is not started. Below is the scope as it was written, before Part A. This document says what to change, where, how to test it and what gain to expect. The numbers come from the experiments ATOM1, ATOM2 and ATOM3 in `README.md` of this folder (V100, float64, mpnn_gather adjoint).

## 1. Why

The CUDA helper `__stade_wa_add!(arr, idx, val)` (`cgen_warp_helper_exprs`, `src/STADE.jl` near line 12885) is emitted for every atomic add of a named array. It does three things in each call:

1. It reads `active_mask()` and tests that all 32 lanes are active.
2. It does two shuffles and a vote to test that all 32 lanes have the same index.
3. If both tests pass, it sums the 32 values with shuffles and the first lane does one atomic. Otherwise it does the plain atomic.

The helper costs time in two cases, both measured on the mpnn_gather adjoint (1e6 nodes, 39.69 ms):

- **The address differs in adjacent lanes.** The tests can never pass, so the mask test, the shuffles and the vote are pure cost. The helper check alone costs about 3.7 ms in the unchanged kernel.
- **The address is the same in runs of lanes, not in all 32.** The full-warp test fails and the plain atomic runs, so the lanes that agree still serialize. A reduction over a smaller group of lanes (variant X, half-warp) takes the kernel from 39.69 ms to 34.02 ms (-5.7 ms, -14 %).

The ceiling is the kernel with every atomic plain (variant N, not correct, timing only): 26.31 ms. Atomics plus helper checks are about 17.1 ms (43 %) of the unchanged kernel. About 22.6 ms remain when all atomics are removed. This change cannot reach them.

## 2. Evidence: where the helper sites are

Counts of helper call sites by address class, from `/tmp/classify.jl` run on the generated CUDA scripts of the benchmark kernels. `D` = address differs in adjacent lanes, `S` = constant on runs of lanes, `U` = same in every lane, `X` = anything else.

| Kernel | Global-memory sites | Shared-memory (`__sh_*`) sites |
|---|---|---|
| mpnn_gather | D10 S6 X10 | D10 |
| transformer_block | D14 S21 | D13 |
| euler2d_step | D36 | - |
| swe2d_step | D15 | - |
| mlp3_push | D6 S3 U9 | D6 |
| unet2 | D9 S48 X14 | D1 X10 |

Reading: many sites are `D`. Only the `U` sites (9 in mlp3_push) are the case the helper was built for. Nothing in the data says that a full-warp reduction is wrong for `U`. It says that most sites are not `U`.

## 3. The change, in two independent parts

### Part A. A gate: do not call the helper where the address is `D`

Result-neutral. A `D` address never makes all 32 lanes agree, so the helper would take the plain-atomic branch anyway. The output of every kernel is bit-identical to the current output. The gain is the check time only.

**What to add** (all names with the stage prefix `cgen_`; no `module`, `struct` or `const`):

- `cgen_warp_classes(k)` and `cgen_warp_class(e, cls)`: the analysis of `jgen_warp_classes` / `jgen_warp_class` / `jgen_warp_add_class` / `jgen_warp_mul_class` / `jgen_warp_join` (`src/STADE.jl` lines 16867 to 16952), seeded with `__tid => :D`. Both analyses are pure functions of one kernel `Expr`. Two options for the code:
  - (a) Make the `jgen_` functions take the seed symbols as an argument and have both stages call them. One copy, one set of tests. The name prefix rule says a cross-stage call is allowed only for pure helpers, so the shared functions should move to a neutral prefix (for example `gpu_warp_*`). The prefix list in skill-stade-dev has no `gpu_`, so this needs a decision.
  - (b) Copy them as `cgen_*`. This follows the existing pattern of duplicated pairs (hard rule 11 asks that duplicated pairs are changed identically), but it creates a second copy to keep equal.
  - Recommendation: (b), with a test that runs both on the same kernels with the seed renamed and compares the classes. This keeps stage ownership clean.
- `cgen_warp_gate(k)`: walks a finished kernel `function`, and rewrites each `__stade_wa_add!(A, i, v)` whose index class is `:D` back to `CUDA.@atomic A[i] += v`. The conservative default (class `:X` for anything not understood) keeps the helper.
- Wiring in `stade_gpu` (line 21525 onward): run `cgen_warp_gate` after the last pass that creates or rewrites kernels (after the `glue_on` block, before `cgen_inbounds_wrap`). Reason: `cgen_private_kernel`, the coarsened `_c!` twins, `cgen_split_apply`, `cgen_serial_apply` and `cgen_glue_apply` all create kernels that carry calls. A gate that ran at `cgen_emit` would miss them.
- Recompute `helpers` after the gate. Today `helpers = plan.helpers` is fixed inside `cgen_emit` (line 13150), so a kernel set with no call left would still carry the definitions. The rule "definitions are present exactly when a call is" (check 2 of `validate_warp_atomics.jl`) must keep holding.
- The Float32 return path (`helpers = plan.helpers`, last lines of `stade_gpu`) must use the recomputed `helpers` too. It uses `plan.helpers` today. Check whether this is an existing defect before changing it.

**Where the class analysis must be careful**

- The thread variable is `__tid` (set by `backend.tid_rhs`). The jam kernel (`jb = div(__tid - 1, 32) * span + mod(__tid - 1, 32)`, line 12069) has a mod part that is `D` and a div part that is `S`. The join gives `D`, as wanted.
- `mod(__tid - 1, U)` with a uniform `U` that is 1 at run time is constant, but the class says `D`. The gate then uses the plain atomic. This costs speed on that input, never correctness (the same limit applies to the JACC pass).
- A `D` index inside a `if __tid <= n` guard: the mask test would fail in the last warp anyway. No change.
- Private-memory kernels (`_p` variants, `__sh_*` accumulators): the sites that add to shared memory are `D` in 10 of 10 cases for mpnn_gather. Whether shared-memory atomics should go through the helper at all is a separate question (section 6).

**Expected gain.** Up to the 3.7 ms of check time in mpnn_gather (9 % of 39.69 ms). The gate also avoids the regression of the kind seen on the JACC path (euler2d_step adjoint +33 to +59 percent with the helper, 36 scatter sites). The CUDA corpus run must show the same or better time on every kernel. No kernel can be slower than today, because the gate only removes work.

### Part B. A group reduction for `S` and `X` sites

Changes the summation order, so results differ from the current default in the last bits. Needs a flag and a decision on the default after measurement.

**What to add**

- A second helper `__stade_wa_add_g!(arr, idx, val)`: variant X of `tools/patch_halfwarp.py`. Reduces within a 16-lane group (`xor 8, 4, 2, 1`), one atomic per group when the 16 lanes agree on the index, the plain atomic otherwise. The group mask is `gm`, built from the lane id (`__stade_wa_lane_g`). It assumes the 1-D blocks `cgen_launch_expr` emits, as the current helper does.
- A hard rule for the new helper: **no fall through from the group operation to a full-warp operation.** If lanes of one group take the plain-atomic branch and another group takes the shuffle branch, a full-mask `shfl_sync` inside the divergent code deadlocks. Every shuffle and vote in the group helper must use the group mask and only that mask.
- `cgen_warp_gate` gets a three-way result: `:D` goes to plain atomic, `:U` stays on the full-warp helper, `:S` and `:X` go to the group helper. (`:U` sites: a full-warp reduction is 1 atomic per 32 lanes, a group reduction is 2. Keep the full-warp helper for `:U`.)
- A new keyword of `stade_gpu` / `stade_cuda` / `stade_cuda_file` (and the `stade_gpu_file` pass-through): `warp_group::Integer = 0` (0 = off, 16 = half-warp). The default is decided after the measurements in section 5. Rollback to the default of today is `warp_group = 0`.

**Expected gain.** X on all sites: -5.7 ms (-14 %) on mpnn_gather. The effect is not additive with Part A. Part A removes check time from `D` sites. Part B changes `S` and `X` sites. In X the `D` sites still paid for the checks, so a combined figure of about -6 to -8 ms is plausible. The experiments do not measure it. Do not quote a combined number before the run.

**Alternative for Part B to check first.** A reduction over exact runs of equal indices (CUDA `match_any_sync`, supported from sm_70, so on the V100) would reduce every run, not only 16-lane groups that agree. It is more exact for `S` sites. Whether CUDA.jl exposes `match_any_sync` for 64-bit keys in the version on the cluster is not verified. Check this first, in a small kernel, before choosing.

## 4. Tests and validation

The skill-stade-dev contract applies: adversarial kernels first, corpus after every change, sabotage-test each new guard.

**Part A**

1. Extend `test/validate_warp_atomics.jl` (76 lines today):
   - Check 1 (equivalence): the substitution `__stade_wa_add!` to `CUDA.@atomic` must still give exactly the output of `warp_atomics = false`. This stays true with the gate, because a gated site is already the plain atomic.
   - New check: no remaining call has a class `:D` index, and every plain atomic left in a kernel generated with `warp_atomics = true` has a `:D` index. This mirrors the "gate_ok" check of `validate_jacc_warp.jl`.
   - New check: `helpers` are present iff a call is left (after the gate), also for the Float32 path.
   - New `gate_case` kernels, copied from the JACC test (`gate_case` list in `validate_jacc_warp.jl`): fixed address, thread index, scalar chain, loop variable, gather index, `div` of the thread index, `mod` of it, flattened 2-D position with a neighbour offset, a scalar assigned twice. Add CUDA-only shapes: the jam-kernel `jb` form, a private-kernel accumulator, a coarsened `_c!` twin.
2. Sabotage: (i) make `cgen_warp_class` return `:U` for everything: the "no `:D` call left" check must fail; (ii) make it return `:D` for everything: the helper must disappear from `mlp3_push` and the check "calls exist for the `U` sites" must fail; (iii) the existing swap of index and value must still fail check 1.
3. GPU corpus: `validate_corpus_gpu.jl`, adjoint and HVP, with the flags used for generation. The change must be bit-identical on every kernel. Count elements changed, not elements compared.
4. `validate_offload.jl` and `validate_shared_private.jl`: no change expected. Run them to prove it.
5. Zero-trip run (`validate_zerotrip.jl`): no analysis in this change reasons about trip counts, but the run is required by the contract when touching the codegen stage.

**Part B**

1. Equivalence test as for Part A, with the group helper as the substitution target. The group helper with `warp_group = 0` must equal today's output byte for byte.
2. GPU corpus with a tolerance for the last bits (summation order). The exact tangent-versus-adjoint dot-product identity must still hold to about 1e-15 relative in the validator.
3. A kernel with divergent warps: threads of one warp take different branches around the atomic, some groups agree on the index and some do not. Run it on a device with `warp_group = 16` and compare against the plain atomics. This is the deadlock test. Add a timeout in the test run.
4. A kernel whose block size is not a multiple of 16 or 32 (the last warp partly active), to cover the group mask with inactive lanes.
5. Sabotage: (i) use the full mask inside the group helper: the divergent-warp test must hang or fail; (ii) use the wrong group mask: the equivalence check must fail.

## 5. Measurement plan

Use the existing time jobs of the benchmark kit (`cluster/go.sh`, V100, 2 h limit). Run each on the baseline commit and on the changed commit in the same job, so both see the same node.

| Case | Why |
|---|---|
| mpnn_gather adjoint, 1e5 and 1e6 | main target (ATOM1 to ATOM3 numbers) |
| euler2d_step, swe2d_step | 36 and 15 `D` sites: Part A must not make them slower, and should make them faster |
| unet2 | S48 X14: the case where Part B matters most |
| transformer_block | `U`-poor, `S` 21: checks the full-warp helper is still used where it pays |
| mlp3_push | the `U` sites: Part A and Part B must not move them |

Acceptance criteria (set before the run, so the run cannot move them):

- Part A: no kernel slower than baseline by more than the run-to-run spread of the baseline (the repeat jobs of this benchmark give that spread). mpnn_gather faster.
- Part B: ship the default `warp_group = 16` only if mpnn_gather improves by at least 10 % and no other case is slower by more than the spread. If not, ship with the default 0.

## 6. Risks and open questions

- **Result changes (Part B only).** Summation order changes the last bits. Any user who needs bit reproducibility keeps `warp_group = 0`. skill-stade already says a fused reduction agrees "to within rounding rather than bit for bit".
- **1-D block assumption.** Both helpers take the lane as `(threadIdx().x - 1) & 31`. A 2-D launch would break them. `cgen_launch_expr` emits 1-D blocks today. Add a guard test that fails if a kernel with a call is launched with a 2-D block.
- **Deadlock** in the group helper if a full-mask operation is reached from a divergent branch (section 3, Part B). This is the main reason for the divergent-warp test.
- **Shared-memory atomics.** The `__sh_*` accumulators sit in shared memory, where the plain atomic is cheap. Whether the helper should be applied to them at all is not measured. ATOM3 made the shared-memory adds plain (timing only, not correct) and saved about 6.7 ms in mpnn_gather. That saving overlaps with the one of the reduced global adds (7.9 ms), so the two do not add up. Not part of this change.
- **Cause of the other 22.6 ms.** Unknown. The change does not touch it. `ncu` does not work on these kernels (see `../profile_2026-11-27/README.md`), so a counter-based answer is not available now.
- **JACC, AMDGPU, Metal.** Not touched by this change. JACC already has the gate. AMDGPU and Metal output have no helper.
- **Rule 9 of skill-stade-dev** (no corpus-specific code): the gate and the helper name no kernel. The kernel names in this document are for the measurement only.

## 7. Work order and size

1. Part A: classes + gate + wiring + helper recomputation, about 120 lines in `STADE.jl`. Tests in `validate_warp_atomics.jl`. One commit.
2. GPU corpus and time jobs for Part A. Decide.
3. Part B, after the `match_any_sync` check: group helper + three-way gate + keyword + tests. About 100 lines. One commit.
4. GPU corpus and time jobs for Part B. Decide the default.
5. Update `DISCUSSION.md` section 4.5 and the README.

Nothing is started. Part A can begin on your word; Part B waits for the `match_any_sync` check and for the result of Part A.
