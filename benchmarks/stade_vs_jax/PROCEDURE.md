# Full benchmark procedure: STADE against JAX on a V100

Status: the procedure is defined. The benchmark is **not launched**. The owner must approve the launch.

This document defines one full benchmark. It measures four quantities for every case:

1. GPU runtime of the primal.
2. GPU runtime of the adjoint.
3. GPU memory of the primal.
4. GPU memory of the adjoint.

It covers 14 kernels and 42 cases. It compares STADE (the source file `src/STADE.jl` of this repository) with JAX.

Sections 1 to 4 define what to measure and how. Section 5 lists the changes that the harness of this folder still needs. Sections 6 to 10 give the steps, the checks, the time budget and the report. Section 11 lists the decisions for the owner. Appendix A lists the files of this folder.

## 0. Terms

| Term | Meaning |
|---|---|
| Kernel | One Julia function in `kernels/`. JAX has an equivalent function in `reference/jax_ref.py`. |
| Case | One kernel at one size. Example: `heat3d_step` at `1e6`. |
| Primal | The function itself. It computes the outputs from the inputs. |
| Adjoint | The function and its vector-Jacobian product. It gives the outputs and the gradients. |
| Config | One way to produce the code that is measured. See section 1.2. |
| Phase | One group of runs with one purpose: `time`, `mem_primal` or `mem_adjoint`. |
| Run | One operating-system process that measures one or more cases. |
| MB | 10^6 bytes. |
| Median, IQR | The middle value and the interquartile range (q1 to q3) of the timed calls. |
| Gap | STADE value divided by JAX value. A gap below 1 means STADE is smaller or faster. |

## 1. Scope

### 1.1 Cases

The case list is `SZ` in `gpu/specs.py`. The size definitions are in `tools/inputs.jl`.

| Kernel | What it computes | Size label (parameters) |
|---|---|---|
| `heat3d_step` | One step of the 3D heat equation, 7-point stencil, periodic | `1e5` (n=46), `1e6` (n=100), `8e6` (n=200) |
| `heat3d_multi` | The same stencil, time loop inside the kernel, 2 pairs of steps | `1e5`, `1e6` |
| `heat1d_multi` | 1D diffusion, time loop inside the kernel, 2·p steps | `1e5_t8`, `1e6_t2`, `1e6_t8`, `1e6_t32` (p is the number after `t`) |
| `wave2d_step` | One leapfrog step of the 2D wave equation | `1e5` (n=316), `1e6` (n=1000), `16e6` (n=4000) |
| `burgers2d_step` | One step of the 2D viscous Burgers equation | `1e5`, `1e6`, `16e6` (same n) |
| `swe2d_step` | One finite-volume step of the 2D shallow-water equations | `1e5`, `1e6`, `16e6` (same n) |
| `euler2d_step` | One finite-volume step of the 2D Euler equations | `1e5`, `1e6`, `16e6` (same n) |
| `ufv_scatter_step` | Advection-diffusion on an unstructured mesh, loop over faces | `1e5` (nx=316), `1e6` (nx=1000), `4e6` (nx=2000) |
| `ufv_gather_step` | The same update, loop over cells | `1e5`, `1e6`, `4e6` (same nx) |
| `mlp3` | Three-layer perceptron, batch size 1, pull form | `3M` (width 1024), `13M` (2048), `79M` (5120) |
| `mlp3_push` | The same network, push form | `3M`, `13M`, `79M` |
| `mpnn_gather` | One message-passing layer on a graph | `1e5` and `1e6` nodes (16 node features, 16 message features) |
| `transformer_block` | One pre-norm transformer encoder block, batch size 1 | `d512_T512`, `d1024_T1024`, `d2048_T1024` (d = model width, T = tokens) |
| `unet2` | Small U-Net forward pass, batch size 1 | `c16_n256`, `c32_n128`, `c64_n64`, `c128_n32` (c = channels, n = image side) |

The total is 14 kernels and 42 cases. Every case runs in float64 with batch size one.

The case `euler2d_step` at `16e6` can run out of memory in JAX, because the card has 16 GB. Run it. Record the result as it comes. An out-of-memory result is a valid result. Do not retry it.

### 1.2 Configs

| Config | Framework | Definition |
|---|---|---|
| `C` | STADE | `keep_push_pop = false`, `fuse_ii_loops = true`, all other options at their defaults. CUDA backend. |
| `A` | STADE | Config `C` with `primal_outputs = false` and `fresh_gradient = true`. This is the best opt-in setting. |
| `J` | JAX | `jax.jit` of the kernel function. The adjoint is `jax.jit` of `jax.vjp`. |

Config `A` changes the adjoint only. The primal code of `A` and `C` is the same. Measure the primal of STADE with config `C` only.

### 1.3 What the primal and the adjoint contain

The two frameworks must do the same work. The rules below define the work.

| | STADE | JAX |
|---|---|---|
| Primal call | One call of the generated function `NAME_cuda`. It launches every primal kernel. | One call of the jitted function. It returns every output array in the roles file. |
| Adjoint call | One call of the generated function `NAME_b_cuda`. It runs the forward sweep that fills the stacks, the reverse sweep, and the gradient update. | One call of the jitted `vjp` function. It runs the forward pass and the backward pass and returns the gradients. |
| Outputs | The arrays in `deps` of `roles/NAME.small/meta.json`. | The same arrays. |
| Gradients | The arrays in `indeps` of the same file. | The same arrays. |
| Not in the call | `initstacks_NAME_b_cuda` (it runs once), the reset of the inputs, the reset of the gradient buffers, any host transfer. | The upload of the inputs and of the cotangents, any host transfer. |

## 2. Fixed conditions

Apply every condition to every run.

1. Use one node of the `gpu` partition with one Tesla V100-PCIE-16GB. The node has no other job.
2. Use the driver and CUDA libraries of the node. Record the driver version.
3. Use JAX from the virtualenv `$WORK/stade_fw_venv`. Build it with `cluster/setup_jax_venv.sh`. The script installs `jax[cuda12]`. Pin the version that the owner names, and record it (change H6). Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1`. Unset `LD_LIBRARY_PATH` for Python. The script `cluster/fw_run.sh` does this.
4. Enable float64 in JAX (`jax_enable_x64`). The file `reference/jax_ref.py` does this. Do not use TF32.
5. Use the STADE source file of the repository that holds this folder. Record the commit.
6. Use the same input bytes in both frameworks. The file `tools/dump_inputs.jl` writes the inputs. JAX reads them. STADE builds the same inputs with `bench_inputs`.
7. Use the role files from `tools/dump_case.jl` at the size `small`. They name the outputs and the differentiated arrays for both frameworks.
8. Run one kernel at a time. Never run two measurements on the GPU at the same time.
9. Do not change the order of the phases during one benchmark.

## 3. Runtime protocol

### 3.1 Timers

| Framework | Timer | What it counts |
|---|---|---|
| STADE | CUDA events (`CUDA.@elapsed`) around one call | GPU time from the first event to the last event. Gaps between launches inside the call count. The wait before the call does not count. |
| JAX | Host clock (`time.perf_counter`) around one call and `jax.block_until_ready` | The host launch time and the GPU time. |

The two timers are not equal. The JAX timer includes the host launch time of the call. The STADE timer does not include the wait before the first launch. For small cases the difference can be large. The harness of this folder uses these two timers.

This procedure adds a second STADE timer (change H3 in section 5). It uses the host clock and a device synchronization, like the JAX timer. The report shows both STADE readings. The like-for-like gap uses the host clock for both frameworks. The decision about the headline timer is in section 11.

### 3.2 Steps of one timing

Do these steps for the primal and then for the adjoint. Use one case in one run.

1. Reset the inputs of the call. For STADE, copy the original inputs back to the device. For the adjoint, also set the gradient buffers to their start values. JAX needs no reset, because its functions do not change their inputs.
2. Synchronize the device. This ends the reset before the timer starts.
3. Make the first call. Record its duration as `first_call_s`. This call includes the compile time. Do not use it in the statistics.
4. Make the warm-up calls. STADE makes 4 calls if the first call took less than 1 s, 2 calls if it took 1 to 5 s, and 1 call if it took more than 5 s. JAX always makes 5 calls.
5. Make the timed calls. STADE makes 30, 10 or 5 calls, in the same three classes as step 4. JAX always makes 30 calls. Reset and synchronize before each STADE call.
6. Stop the STADE timed loop after 90 s of wall time if it has not finished. Record the number of calls made (`n`).
7. Compute the median, q1, q3, minimum and maximum of the timed calls. The median is the result.

The runs do not flush the GPU caches. STADE copies the inputs back between calls. This copy changes the cache state. JAX does not copy. The report states this limit (section 10).

### 3.3 Statistics

1. Report the median and the IQR for every cell.
2. Compute the gap per case from the medians.
3. Summarize the gaps with the geometric mean. Also report the count of cases where STADE is slower than JAX (gap above 1).
4. Treat a change below 5 % between two runs of the same code as noise. This limit is a working value. To check it, repeat one phase and compare the medians.
5. Report the ratio adjoint time over primal time for each case and each framework.

### 3.4 Run isolation

1. Run STADE with one process for each kernel and each config. The process measures every size of that kernel in order of size.
2. Run JAX with one process for each case.
3. Before each run, ask `nvidia-smi` for the compute processes on the GPU. The list must be empty. If it is not empty, stop and report.
4. Before and after each run, record the SM clock, the memory clock, the temperature and the clock throttle reasons (change H4).

## 4. Memory protocol

### 4.1 What the reading is

The reading is the device memory in use for the whole GPU. The driver reports it. The reading minus the baseline is the memory that the case uses.

| Framework | Instrument |
|---|---|
| STADE | `CUDA.total_memory() - CUDA.free_memory()`. This calls `cuMemGetInfo`. |
| JAX | `nvidia-smi --query-gpu=memory.used`. The value is in MiB. Multiply it by 1.048576 to get MB. |

The baseline `mem0` is the reading before the case data exists. For STADE it comes after the context start and a memory settle (step 1 below). For JAX it comes after a first small device operation, which starts the runtime.

The reading counts all device memory of the process. It includes the memory that the allocator keeps after a call. The allocator of each framework keeps memory in a pool. The pool does not give memory back at once. The reading after the calls is therefore the high-water mark of the process.

### 4.2 Quantities

| Quantity | Definition |
|---|---|
| `mem_context_mb` | The baseline `mem0`. |
| `mem_inputs_mb` | The reading after the inputs are on the device, minus `mem0`. |
| `mem_peak_primal_mb` | The reading after the primal calls, minus `mem0`. The process holds the inputs of the primal only. |
| `mem_peak_adjoint_mb` | The reading after the adjoint calls, minus `mem0`. The process holds the data of the adjoint (see 4.3). |
| `stack_mb`, `grad_mb` | STADE only. The size of the stacks and of the gradient buffers. |
| `peak_mb` | JAX only. The XLA live-bytes peak (`peak_bytes_in_use`). It has no region rounding. |

Also compute the working memory: the peak minus `mem_inputs_mb`. It shows what the call adds to the inputs.

### 4.3 Data held in each phase

The STADE runner of this folder allocates the gradient buffers and the stacks before it measures the primal. The same arrays are on the device during the primal calls and during the adjoint calls. The primal reading then equals the adjoint reading. The primal reading is not a valid primal memory. The JAX script has the same flaw in a smaller form, because it uploads the cotangents before the primal calls.

This procedure measures each peak in its own phase and its own run.

| Phase | STADE data on the device | JAX data on the device |
|---|---|---|
| `mem_primal` | The inputs. Nothing else. | The inputs. Nothing else. |
| `mem_adjoint` | The inputs, the gradient buffers and the stacks. | The inputs and the cotangents. |

The adjoint of each framework needs its own extra arrays. They belong to the adjoint peak. They do not belong to the primal peak.

### 4.4 Steps of one memory measurement

Use one phase and one case in one run. Do not time anything in this run.

1. STADE only: settle the memory. Call `GC.gc()` and `CUDA.reclaim()`, twice each. Read `mem0`. For JAX, read `mem0` after a first small device operation.
2. Build the inputs on the device. Settle. Read `mem_inputs_mb`.
3. Phase `mem_adjoint` only: build the gradient buffers and the stacks (STADE) or the cotangents (JAX). Settle. Read the setup memory.
4. Make the first call. This call compiles the code.
5. STADE only: settle the memory.
6. Make 3 calls. Reset the inputs before each STADE call.
7. Synchronize the device. Read the memory. The result is the peak minus `mem0`.
8. JAX only: read `peak_bytes_in_use` and `pool_bytes` from `jax.devices()[0].memory_stats()`.
9. STADE only: measure the bytes that one call allocates (`CUDA.@allocated`). Record the value.

Settle the memory between the sizes of a STADE run. A STADE run in a memory phase may measure all sizes of a kernel.

### 4.5 Limits of the reading

1. The CUDA.jl pool has a minimum chunk of about 34 MB. A STADE reading below about 34 MB does not show the true use. Do not compute gaps from these readings. The report marks them.
2. `nvidia-smi` reports whole MiB. The resolution is 1 MiB.
3. A JAX pool never shrinks. Step 5 has no JAX equivalent. The JAX reading can include transient memory of the first call.
4. Both readings include small fixed amounts, such as loaded GPU modules. The baseline `mem0` does not include them for the kernels of the case.
5. The two instruments read the same driver quantity.

## 5. Changes that the harness of this folder still needs

The folder holds the harness without these changes. It runs and gives runtime and adjoint memory. It does not give a valid primal memory, a STADE host-clock timer or a GPU state record. Make the changes before the launch. Each change is small. Do not launch before the owner has seen the result of changes H1 to H6 on a local test.

| Id | File | Change | Reason |
|---|---|---|---|
| H1 | `gpu/runner_body.jl` | Add the environment variable `PHASE` with the values `time`, `mem_primal`, `mem_adjoint`. The `time` phase does not make the memory pass. The two memory phases follow section 4.3 and 4.4. | The primal memory reading holds the adjoint buffers (section 4.3). |
| H2 | `reference/time_fw.py` | Add the option `--phase time|mem_primal|mem_adjoint`, with the same meaning as H1. The `time` phase does not read the memory. | The cotangents are on the device during the primal calls. |
| H3 | `gpu/runner_body.jl` | Add a second timer for each call. It uses `time_ns` and a device synchronization. Store it as `primal_wall` and `adjoint_wall`. | The timers differ (section 3.1). |
| H4 | New script | Before and after each run, record the GPU compute processes, the clocks, the temperature and the throttle reasons. Use `nvidia-smi`. | Check that the GPU is free and not throttled. |
| H5 | New script in `cluster/` | Pack the runs of all phases and both frameworks into one payload. Each run has its own process and its own time limit. | One job submission (section 8). |
| H6 | Both runners | Add to each result: the Julia, CUDA.jl, JAX, `jaxlib` and Python versions, the driver version, the STADE commit and the hash of the files of this folder. | Make the results reproducible. |

Until H5 exists, the folder builds two payloads: one for STADE and one for JAX. The steps of section 6 then use two submissions. The script `cluster/report.py` builds the report parts 1, 2, 3, 4 (adjoint memory), 5 and 6 of section 9. The report part for the primal memory needs the result of H1 and H2.

## 6. Procedure

All paths are relative to this folder. Julia 1.10 or later, Python 3 and the repository source file `src/STADE.jl` are needed on the sandbox. The cluster needs the JAX virtualenv and the relay of the cluster skill.

### 6.1 Preparation (on the sandbox, no GPU)

1. Freeze the STADE tree. Record `git log -1` and check that `git status` shows no change.
2. Make the changes H1 to H6 (section 5).
3. Set the path of the STADE source file: `export STADE_SRC=$(git rev-parse --show-toplevel)/src/STADE.jl`.
4. Make the role files: `bash cluster/make_roles.sh work`. The script runs `tools/dump_case.jl` at the size `small` for each kernel.
5. Build the JAX job: `python3 cluster/build_jax_job.py work work/jax 3600`. It writes `work/jax/jax_job.json`. The bundle holds `tools/inputs.jl`, `tools/dump_inputs.jl`, `kernels/`, `reference/` and the role files.
6. Build the STADE job: `bash cluster/build_stade_job.sh work/stade 1800`. It generates the CUDA runner scripts for the configs `C` and `A`, with `gpu/make_scripts.jl`. Config `A` uses `STADE_PO=0` and `STADE_FRESHKW=1`. The size `tiny!` of each kernel is the verify size. The script writes `work/stade/stade_job.json`.
7. The kernel `kernels/nn/mpnn_gather.jl` uses the variable name `sdst`. The current STADE refuses the name `sd`. Keep this name.

### 6.2 Local gates (before the job)

1. Run the STADE test suites that the owner names for the commit. At least run `test/validate_offload.jl`, `test/validate_mod_collide.jl` and `test/validate_checkpoint_gpu_cpu.jl`.
2. Check the JAX kernels against the STADE kernels on the CPU: `bash tools/parity_all.sh`. It needs Python with JAX. Every kernel must agree.
3. Run each phase on the sizes `tiny` and `small` on the CPU stand-ins, where this is possible. Check that every result field of section 9 is present.

### 6.3 The job

1. Check the time limit of the V100 `sbatch` template. The owner reported 2 hours. Check the file.
2. Check that the `gpu` partition has a free node. Use `sinfo -p gpu`.
3. Submit each payload with `RELAY_ENV=FILE RES_DIR=work/results bash cluster/go.sh NAME payload.json TIMEOUT`. `FILE` defines the relay variables (see the header of `cluster/go.sh`). Set `TIMEOUT` to the template limit plus a margin.
4. If the wait in `go.sh` ends before the job ends, fetch the result with `poll_result.sh NAME 30 3` from the relay folder. Do not submit the job again.
5. Tell the owner the job name and the Slurm job id.

The job runs the phases in this order:

| Order | Phase | Framework | Runs |
|---|---|---|---|
| 1 | `time` | STADE, configs `C` and `A` | 28 (14 kernels, 2 configs) |
| 2 | `time` | JAX | 42 (one per case) |
| 3 | `mem_primal` | STADE, config `C` | 14 |
| 4 | `mem_primal` | JAX | 42 |
| 5 | `mem_adjoint` | STADE, configs `C` and `A` | 28 |
| 6 | `mem_adjoint` | JAX | 42 |

Before the JAX runs, the job writes the inputs of each case once with `dump_inputs.jl` and keeps them for the three JAX phases. It removes them at the end of the job.

### 6.4 After the job

1. Fetch the result files. Copy them to the archive folder (section 9).
2. Check every acceptance rule of section 7.
3. Build the report: `python3 cluster/report.py --stade work/results/NAME.json --jax work/results/NAME2.json --out OUTDIR`.
4. If the owner gives an earlier report, compare the medians with it. Report every change above 5 %.
5. Give the report to the owner. State every case that failed a check.

## 7. Acceptance checks

A case is valid only if it passes every check that applies to it.

| Id | Check | Limit |
|---|---|---|
| G1 | STADE device result against the CPU result, on `tiny!` and `small!`, for the primal and for the gradients | Relative error at most 1e-10. |
| G2 | Elements changed by the STADE run (`elements_changed`) | Greater than 0. Equal for configs `A` and `C`. |
| G3 | JAX gradients are finite (`grad_finite`) | True. |
| G4 | No other compute process on the GPU before the run | The list is empty. |
| G5 | Number of timed calls (`n`) | 30, 10 or 5 as in section 3.2, or fewer if the 90 s limit stopped the loop. Record the reason. |
| G6 | `mem_inputs_mb` against the size of the inputs | Not below the input size minus 1 MB. |
| G7 | STADE adjoint setup memory against its parts | Inputs plus gradient buffers plus stacks, within 5 % or 34 MB. |
| G8 | JAX primal peak against JAX adjoint peak, same case | The adjoint peak is not below the primal peak minus 1 MB. |
| G9 | Clock throttle reasons during the run | None, except the idle reason. If one occurs, mark the case. |

Handle an invalid case as follows.

1. Mark the case in the report. Do not delete it.
2. Run the case again once, only if the cause is outside the code. Examples are a node fault and a killed process.
3. If the case fails again, report it as failed with the reason.

## 8. Time budget

The owner reported that the V100 template allows 2 hours. The benchmark fits in one job only if the total stays below the limit. The values below are estimates.

| Block | Estimate | Basis |
|---|---|---|
| STADE `time` phase | About 41 min | 28 runs. Estimate. |
| JAX `time` phase | About 11 min | 42 cases. Estimate. |
| STADE memory phases | About 30 min | 42 runs of about 45 s (start, compile, 3 calls per size). Not measured. |
| JAX memory phases | About 15 min | 84 runs, no timing loop. The inputs come from the saved dump. Not measured. |
| Total | About 97 min | |

The estimate is below 120 min. The margin is small. A job cannot change its own plan after it starts. Use this rule before the launch.

1. Measure the time of one STADE memory run and one JAX memory run on the V100 first. A short job of one kernel is enough.
2. If the new estimate is above 105 min, split the benchmark in two jobs: the `time` phases, then the memory phases.

## 9. Outputs

Keep every output in one archive folder, for example `results/full_YYYY-MM-DD/` beside this file.

| Item | Content |
|---|---|
| `raw/` | The raw result file of each job. |
| `scripts/` | The generator scripts, the packers and the report script, as used. |
| `versions.json` | The commit, the file hash and all versions of section 5, H6. |
| `rows.json` | One record per case, with every quantity of sections 3 and 4. |
| `REPORT.md` | The report. |

Each result record holds these fields.

| Field | Meaning |
|---|---|
| `kernel`, `size`, `config` | The case and the config. |
| `primal`, `adjoint` | Median, q1, q3, min, max, n and `first_call_s`. For STADE, also `primal_wall` and `adjoint_wall` (change H3). |
| `mem_context_mb`, `mem_inputs_mb` | Section 4.2. |
| `mem_peak_primal_mb`, `mem_peak_adjoint_mb` | Section 4.2. |
| `stack_mb`, `grad_mb`, `alloc_primal_call_mb`, `alloc_adjoint_call_mb` | STADE only. |
| `peak_mb`, `pool_mb` | JAX only. |
| `primal_relerr`, `grad_relerr`, `elements_changed` | STADE checks G1 and G2. |
| `grad_finite` | JAX check G3. |
| `ok`, `error` | The status. An out-of-memory error has `ok` equal to false and a short error text. |

The report has these parts.

1. Runtime of the primal: one table, STADE `C`, JAX, gap.
2. Runtime of the adjoint: one table, STADE `C`, STADE `A`, JAX, both gaps.
3. The ratio adjoint over primal, for each framework.
4. Memory of the primal: STADE `C`, JAX, gap. This part needs the phases `mem_primal` (changes H1 and H2).
5. Memory of the adjoint: STADE `C`, STADE `A`, JAX, gap. Include the stacks and the working memory.
6. A summary: the geometric mean of each gap, and the count of cases where STADE is slower or larger.
7. The list of failed checks and invalid cases.
8. The limits of section 10.

## 10. Known limits

State these limits in the report.

1. Only one GPU model (V100, 16 GB), one precision (float64) and batch size one.
2. JAX appears only as `jit`. The benchmark does not include other JAX styles, such as `lax.scan` or `jax.checkpoint`. It does not include PyTorch.
3. The two runtime timers differ (section 3.1). The caches are not flushed. The STADE copies between calls change the cache state.
4. JAX reads the memory from `nvidia-smi` and cannot give pool memory back. The reading can include transient memory of the first call.
5. STADE memory readings below about 34 MB show the pool minimum, not the true use.
6. The JAX run of `euler2d_step` at `16e6` can run out of memory. The 16 GB card limits it.
7. The result of one job comes from one run. The noise limit of 5 % is a working value.
8. The STADE checkpointing option (`checkpoint = true`) is not part of this benchmark. It is off.

## 11. Decisions for the owner

Each decision has a recommendation. The owner decides.

| Id | Decision | Recommendation | Reason |
|---|---|---|---|
| D1 | Headline timer | Use the host clock with a device synchronization for both frameworks. Report the STADE CUDA-event time as a second column. | The host clock measures the same thing in both frameworks. The JAX timer includes the host launch time. The STADE event timer does not. The second column shows how much of a small case is launch time. This needs change H3. |
| D2 | One job or two | Use two jobs. Run the `time` phases first (about 52 min). Run the memory phases second (about 45 min). | The memory phases are not measured. The estimate for one job is 97 min, and the limit is 120 min. A time-out in one job loses the timing data too. Each run is its own process, so a split does not change what is measured. The cost is a second queue wait. Use one job only if the smoke job (D6) shows that the total stays well below 105 min. |
| D3 | Phase split | Accept it. | The primal memory reading is not valid without it (section 4.3). The cost is about 45 min of job time. |
| D4 | STADE config `A` for the adjoint memory | Measure `A` and `C`. Measure the primal memory with `C` only. | Config `A` is the setting that targets memory. The comparison with JAX is incomplete without it. The cost is 14 more STADE runs. Config `A` does not change the primal code. |
| D5 | Scope of JAX styles | Keep `jit` only in this benchmark. | STADE runs with `checkpoint = false`. A `jax.checkpoint` or `lax.scan` variant is fair only against STADE with `checkpoint = true`. That is a separate study with its own cases. Do it as a follow-up if a reviewer asks. |
| D6 | Launch approval | Approve in two steps. First, run a smoke job on the V100: one kernel (for example `heat3d_step`), all phases, both frameworks, and one repeated `time` phase. Then approve the full launch. | The smoke job checks G1 to G9. It measures the real duration of a memory run. It confirms the primal-memory fix. The repeated phase checks the 5 % noise limit, which is a working value. |

Do not launch the full benchmark until the owner approves it.

## Appendix A. Files of this folder

| Purpose | Path |
|---|---|
| Case list | `gpu/specs.py` |
| Size definitions and input builders | `tools/inputs.jl` |
| Inputs for JAX | `tools/dump_inputs.jl` |
| Role files and parity cases | `tools/dump_case.jl`, `tools/check_kernel.jl` |
| Parity check, STADE against JAX | `tools/parity_all.sh`, `reference/parity_jax.py`, `reference/parity_common.py` |
| Kernels (STADE source form) | `kernels/solvers/*.jl`, `kernels/nn/*.jl` |
| STADE runner (timing and memory) | `gpu/runner_body.jl` |
| STADE script generator | `gpu/make_scripts.jl` |
| STADE job packer | `gpu/pack_job.py` |
| JAX kernels | `reference/jax_ref.py` |
| JAX timing and memory | `reference/time_fw.py` |
| JAX run script and packer | `cluster/fw_run.sh`, `cluster/pack_py.py` |
| JAX virtualenv | `cluster/setup_jax_venv.sh` |
| Role files, jobs, submission, report | `cluster/make_roles.sh`, `cluster/build_stade_job.sh`, `cluster/build_jax_job.py`, `cluster/go.sh`, `cluster/report.py` |
| The STADE source | `../../src/STADE.jl` |

The relay variables file (`RELAY_ENV`) is not in this folder. It holds secrets. Never commit it.
