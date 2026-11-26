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
3. Use JAX from the virtualenv `$WORK/stade_fw_venv`. Build it with `cluster/setup_jax_venv.sh`. The script installs `jax[cuda12]`. Pin the version that the owner names, and record it (change H6). Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1`. Unset `LD_LIBRARY_PATH` for Python. The script `cluster/job_run.sh` does this.
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

The two timers are not equal. The JAX timer includes the host launch time of the call. The STADE event timer does not include the wait before the first launch. For small cases the difference can be large.

The STADE runner therefore measures a second timer (change H3, section 5). It uses the host clock and a device synchronization, like the JAX timer. The two STADE timers alternate: one call with the event timer, then one call with the host-clock timer. Both see the same machine state. The report shows both STADE readings. The headline gap uses the host clock for both frameworks (`report.py --headline wall`, the default). The option `--headline events` uses the event timer. The decision is D1 in section 11.

### 3.2 Steps of one timing

Do these steps for the primal and then for the adjoint. Use one case in one run.

1. Reset the inputs of the call. For STADE, copy the original inputs back to the device. For the adjoint, also set the gradient buffers to their start values. JAX needs no reset, because its functions do not change their inputs.
2. Synchronize the device. This ends the reset before the timer starts.
3. Make the first call. Record its duration as `first_call_s`. This call includes the compile time. Do not use it in the statistics.
4. Make the warm-up calls. STADE makes 4 calls if the first call took less than 1 s, 2 calls if it took 1 to 5 s, and 1 call if it took more than 5 s. JAX always makes 5 calls.
5. Make the timed calls. STADE makes 30, 10 or 5 calls with each timer, in the same three classes as step 4. JAX always makes 30 calls. Reset and synchronize before each STADE call.
6. Stop the STADE timed loop after 180 s of wall time (90 s for each of the two timers) if it has not finished. Record the number of calls made (`n`).
7. Compute the median, q1, q3, minimum and maximum of the timed calls. The median is the result.

The runs do not flush the GPU caches. STADE copies the inputs back between calls. This copy changes the cache state. JAX does not copy. The report states this limit (section 10).

### 3.3 Statistics

1. Report the median and the IQR for every cell.
2. Compute the gap per case from the medians.
3. Summarize the gaps with the geometric mean. Also report the count of cases where STADE is slower than JAX (gap above 1).
4. Treat a change below 5 % between two runs of the same code as noise. This limit is a working value. To check it, repeat one phase and compare the medians.
5. Report the ratio adjoint time over primal time for each case and each framework.

### 3.4 Run isolation

1. Run STADE with one process for each kernel, each config and each phase. The process measures every size of that kernel in order of size.
2. Run JAX with one process for each case and each phase.
3. Before each run, `cluster/gpu_state.sh snap` asks `nvidia-smi` for the compute processes and the memory in use. The GPU is busy if a process is listed or if more than 64 MiB are in use. The job waits up to 60 s for a busy GPU. If it is still busy, the job does not start the run. It records the status `gpu_busy` and goes on (check G4).
4. During each run, `cluster/gpu_state.sh start` and `stop` sample the SM clock, the memory clock, the temperature, the power and the clock throttle reasons once each second (change H4). The result of each run holds the summary (check G9).
5. No new run starts after the time budget of the job. The remaining runs get the status `skipped_budget`. This makes the job print its result before the Slurm limit ends it.

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

## 5. Harness changes H1 to H6

The changes H1 to H6 are implemented in this folder. They were tested on the CPU, with a stand-in for the GPU and a fake `nvidia-smi` (section 6.2). They were not run on a GPU. The smoke job (D6) is the first GPU run.

| Id | File | Change | Reason |
|---|---|---|---|
| H1 | `gpu/runner_body.jl` | The environment variable `STADE_PHASE` selects `time`, `mem_primal` or `mem_adjoint`. The `time` phase makes no memory reading. The memory phases follow sections 4.3 and 4.4 and skip the verify sizes. | The primal memory reading held the adjoint buffers (section 4.3). |
| H2 | `reference/time_fw.py` | The option `--phase time|mem_primal|mem_adjoint`, with the same meaning as H1. The `time` phase reads no memory. The cotangents are not on the device in `mem_primal`. | The cotangents were on the device during the primal calls. |
| H3 | `gpu/runner_body.jl` | A second timer for each call: `time_ns` with a device synchronization. The results are `primal_wall` and `adjoint_wall`. | The two frameworks used different timers (section 3.1). |
| H4 | `cluster/gpu_state.sh` | Records the compute processes, the memory in use, the clocks, the temperature, the power and the throttle reasons. It is called by `cluster/job_run.sh` for every run. | Check that the GPU is free and not throttled. |
| H5 | `cluster/build_job.py`, `cluster/job_run.sh`, `cluster/job_driver.jl` | One packer for both frameworks and any set of phases. One payload holds the STADE scripts, the JAX files and the plan. Each run has its own process and its own time limit. | One job submission for each job. |
| H6 | Both runners, `cluster/build_job.py` | Each run records the Julia, CUDA.jl, driver and runtime versions (STADE) or the JAX, `jaxlib`, Python, NumPy and driver versions (JAX). The job records the STADE commit, a dirty flag and a hash of the kit files. | Make the results reproducible. |

## 6. Procedure

All paths are relative to this folder. The sandbox needs Julia 1.10 or later, Python 3 and the repository source file `src/STADE.jl`. The cluster needs the JAX virtualenv and the relay of the cluster skill.

### 6.1 Preparation (on the sandbox, no GPU)

1. Freeze the STADE tree. Record `git log -1`. Check that `git status` shows no change. The job records a dirty flag.
2. Set the path of the STADE source file: `export STADE_SRC=$(git rev-parse --show-toplevel)/src/STADE.jl`.
3. Make the role files: `bash cluster/make_roles.sh work`. The script runs `tools/dump_case.jl` at the size `small` for each kernel.
4. Generate the STADE runner scripts: `bash cluster/gen_stade_scripts.sh work/stade`. It writes `work/stade/all/KERNEL_C.jl` and `KERNEL_A.jl` for config `C` and config `A`. Config `A` uses `STADE_PO=0` and `STADE_FRESHKW=1`. The size `tiny!` of each kernel is the verify size.
5. Build the jobs with `cluster/build_job.py`:
   - Smoke job (D6): `MAX_SIZES=2 bash cluster/gen_stade_scripts.sh work/stade_smoke heat3d_step`, then `python3 cluster/build_job.py work/smoke --stade-dir work/stade_smoke --roles work --kernels heat3d_step --max-sizes 2`. It runs all three phases.
   - Time job: `python3 cluster/build_job.py work/job_time --stade-dir work/stade --roles work --phases time`.
   - Memory job: `python3 cluster/build_job.py work/job_mem --stade-dir work/stade --roles work --phases mem_primal,mem_adjoint`.
   - The packer writes `job.json` (the payload), `plan.tsv` and `versions.json`. The option `--limit-min` sets the time limit of the V100 template (default 120). The job starts no new run during the last 15 min.
6. The kernel `kernels/nn/mpnn_gather.jl` uses the variable name `sdst`. The current STADE refuses the name `sd`. Keep this name.

### 6.2 Local gates (before any job)

1. Run the STADE test suites that the owner names for the commit. At least run `test/validate_offload.jl`, `test/validate_mod_collide.jl` and `test/validate_checkpoint_gpu_cpu.jl`.
2. Check the JAX kernels against the STADE kernels on the CPU: `bash tools/parity_all.sh`. It needs Python with JAX. Every kernel must agree.
3. Test the phase split of the STADE runner on the CPU:
   - `MAX_SIZES=1 bash cluster/gen_stade_scripts.sh work/local heat3d_multi`
   - `python3 tools/cpu_standin.py work/local/all/heat3d_multi_C.jl work/local/standin.jl`
   - `bash tools/test_runner_phases.sh work/local/standin.jl`
   The test runs the three phases on a stand-in for the GPU. It checks that the primal memory phase holds only the inputs and that the adjoint phase holds the inputs, the gradient buffers and the stacks. Then it breaks the runner on purpose. The check must fail. If it passes, the guard is wrong.
4. Test the whole job chain on the CPU: `bash tools/test_job_chain.sh work/chain`. It takes about 10 min. The script builds a job from stand-in scripts and runs the driver of the payload with a fake `nvidia-smi` and the JAX of the sandbox. It builds the report. Then it runs four failure tests: GPU busy, time budget, run time-out and throttle reason. Each test must give its check or status. In the main report, G6 must fail for the JAX memory runs. The fake `nvidia-smi` reports a constant memory, and G6 must catch that.

### 6.3 The jobs

1. Check the time limit of the V100 `sbatch` template. The owner reported 2 hours. Check the file.
2. Check that the `gpu` partition has a free node. Use `sinfo -p gpu`.
3. Run the smoke job first. Submit it with `RELAY_ENV=FILE RES_DIR=work/results bash cluster/go.sh smoke work/smoke/job.json TIMEOUT`. `FILE` defines the relay variables (see the header of `cluster/go.sh`). Set `TIMEOUT` to the template limit plus a margin.
4. Read the smoke report (section 6.4). Check G1 to G9. Check the duration of each phase. Repeat the `time` phase once to check the 5 % noise limit. Then ask the owner to approve the full launch.
5. Submit the time job and the memory job one after the other. Wait for the first to end.
6. If the wait in `go.sh` ends before the job ends, fetch the result with `poll_result.sh NAME 30 3` from the relay folder. Do not submit the job again.
7. Tell the owner the job name and the Slurm job id.

Order of the runs inside a job: every STADE kernel (config `C`, then config `A`, with the phases in the order given), then every JAX case (the phases in the order given). Config `A` skips `mem_primal`.

| Job | Phases | STADE runs | JAX runs |
|---|---|---|---|
| Time job | `time` | 28 | 42 |
| Memory job | `mem_primal`, `mem_adjoint` | 42 (14 + 28) | 84 |
| One job for all | all three | 70 | 126 |

The job writes the inputs of one JAX case with `dump_inputs.jl`, runs the phases of that case, and removes the inputs.

### 6.4 After the job

1. Fetch the result files. Copy them to the archive folder (section 9).
2. Build the report: `python3 cluster/report.py --results work/results/time.json --results work/results/mem.json --out OUTDIR`. The report evaluates the checks of section 7.
3. If the owner gives an earlier report, compare the medians with it. Report every change above 5 %.
4. Give the report to the owner. State every case that failed a check.

## 7. Acceptance checks

`cluster/report.py` evaluates every check. A check has the status FAIL or NOTE. The report lists each one in section 7 of the report.

| Id | Check | Limit | Status if not met |
|---|---|---|---|
| G1 | STADE device result against the CPU result on the size `tiny`, for the primal and for the gradients | Relative error at most 1e-10. | FAIL |
| G2 | Elements changed by the STADE run (`elements_changed`) | Greater than 0. Equal for config `A` and config `C`. | FAIL |
| G3 | JAX gradients are finite (`grad_finite`) | True. | FAIL |
| G4 | The GPU is free before each run | No compute process, and at most 64 MiB in use. | FAIL |
| G5 | Number of timed calls (`n`) | 30, 10 or 5 for STADE, 30 for JAX. A smaller value means that the time limit of the loop stopped it. | NOTE |
| G6 | `mem_inputs_mb` against the size of the inputs | Not below the input size minus 1 MB. | FAIL |
| G7 | STADE adjoint setup memory (`mem_setup_mb`) against its parts | Inputs plus gradient buffers plus stacks, within the larger of 5 % and 34 MB. | FAIL |
| G8 | JAX adjoint peak against JAX primal peak, same case | The adjoint peak is not below the primal peak minus 1 MB. | FAIL |
| G9 | Clock throttle reasons during the run | No power cap, slowdown, thermal reason or power brake (bits 0x4, 0x8, 0x20, 0x40, 0x80). The idle bit and the application-clocks bit do not count. | FAIL |

A run that has a status other than `ok` (`gpu_busy`, `skipped_budget`, `dump_failed`), a non-zero exit or no result is a FAIL of the kind `RUN`. A JAX out-of-memory error is a NOTE.

Each check was tested by breaking its input on purpose. Each broken input made the check fail. A fake `nvidia-smi` that reports a constant memory made G6 fail, as it must.

Handle an invalid case as follows.

1. Keep the case in the report. Do not delete it.
2. Run the case again once, only if the cause is outside the code. Examples are a node fault and a killed process.
3. If the case fails again, report it as failed with the reason.

## 8. Time budget

The owner reported that the V100 template allows 2 hours. All values below are estimates. The smoke job (D6) gives the real values.

| Block | Estimate | Basis |
|---|---|---|
| STADE `time` phase | About 60 min | 28 runs. The second timer adds up to 50 % to the timed part. |
| JAX `time` phase | About 11 min | 42 cases. |
| STADE memory phases | About 30 min | 42 runs of about 45 s (start, compile, 3 calls per size). Not measured. |
| JAX memory phases | About 15 min | 84 runs, no timing loop. Not measured. |
| Input dumps for JAX | About 5 min for each job | Not measured. |
| Total, one job | About 125 min | Above the limit of 120 min. |
| Time job | About 75 min | |
| Memory job | About 50 min | |

One job does not fit the limit. Use two jobs (D2). The job also protects itself: it starts no new run during its last 15 min, and it prints its result before the Slurm limit. If the real values from the smoke job are larger than these estimates, split the memory job by kernel with the option `--kernels`.

## 9. Outputs

Keep every output in one archive folder, for example `results/full_YYYY-MM-DD/` beside this file.

| Item | Content |
|---|---|
| `raw/` | The raw result file of each job. |
| `scripts/` | The generated STADE scripts, `plan.tsv` and the scripts of this folder, as used. |
| `versions.json` | The commit, the dirty flag and the kit hash (H6). The raw results hold the framework versions. |
| `rows.json` | One record per case, with every quantity of sections 3 and 4. |
| `REPORT.md` | The report. |

Each job returns one JSON line. It holds the host, the Slurm job id, the versions and one record for each run. A run record holds `fw`, `config`, `kernel`, `size`, `phase`, `status`, `exit`, `seconds`, `gpu_before` (check G4), `gpu` (check G9), `result` and `stderr_tail`. The `result` of a STADE run holds one record for each size. The `result` of a JAX run is one record. The fields are:

| Field | Meaning |
|---|---|
| `primal`, `adjoint` | Median, q1, q3, min, max, n and `first_call_s`. For STADE: the CUDA-event timer. For JAX: the host clock. |
| `primal_wall`, `adjoint_wall` | STADE only: the host-clock timer (change H3). |
| `mem_context_mb`, `mem_inputs_mb`, `mem_setup_mb` | Section 4.2. Memory phases only. |
| `mem_peak_primal_mb`, `mem_peak_adjoint_mb` | Section 4.2. Each one comes from its own phase. |
| `input_mb`, `stack_mb`, `grad_mb`, `device_mb_estimate` | The sizes of the arrays. |
| `alloc_primal_call_mb`, `alloc_adjoint_call_mb` | STADE only. Bytes that one call allocates. |
| `peak_mb`, `pool_mb` | JAX only. The XLA live-bytes peak and the pool. Memory phases only. |
| `primal_relerr`, `grad_relerr`, `elements_changed` | STADE checks G1 and G2. |
| `grad_finite` | JAX check G3. |
| `ok`, `error` | JAX only. An out-of-memory error has `ok` equal to false and a short error text. |

The report has these parts.

1. Runtime of the primal: STADE (headline timer and event timer), JAX, gap.
2. Runtime of the adjoint: STADE `C` (both timers), STADE best, JAX, both gaps.
3. The ratio adjoint over primal, for each framework.
4. Memory of the primal: inputs, STADE `C`, JAX, JAX own peak, gap.
5. Memory of the adjoint: stacks, STADE `C`, STADE `A`, JAX, JAX own peak, both gaps.
6. A summary: the geometric mean of each gap, and the count of cases where STADE is slower.
7. The list of failed checks and notes.
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
| D2 | One job or two | Use two jobs. Run the `time` phases first (about 75 min). Run the memory phases second (about 50 min). | The estimate for one job is about 125 min, and the limit is 120 min. Each run is its own process, so a split does not change what is measured. The cost is a second queue wait. Use one job only if the smoke job (D6) shows that the total stays well below 105 min. |
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
| STADE runner (phases, timers, memory) | `gpu/runner_body.jl` |
| STADE script generator | `gpu/make_scripts.jl`, `cluster/gen_stade_scripts.sh` |
| JAX kernels | `reference/jax_ref.py` |
| JAX timing and memory | `reference/time_fw.py` |
| JAX virtualenv | `cluster/setup_jax_venv.sh` |
| Role files | `cluster/make_roles.sh` |
| Job packer, driver and run script | `cluster/build_job.py`, `cluster/job_driver.jl`, `cluster/job_run.sh` |
| GPU state record | `cluster/gpu_state.sh` |
| Submission | `cluster/go.sh` |
| Report and checks | `cluster/report.py` |
| CPU tests of the runner and of the job chain | `tools/cpu_standin.jl`, `tools/cpu_standin.py`, `tools/check_phase_results.py`, `tools/test_runner_phases.sh`, `tools/test_job_chain.sh`, `tools/fake_nvidia_smi` |
| The STADE source | `../../src/STADE.jl` |

The relay variables file (`RELAY_ENV`) is not in this folder. It holds secrets. Never commit it.
