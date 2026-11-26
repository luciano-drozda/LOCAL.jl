# STADE against JAX: GPU benchmark

This folder holds everything that the benchmark needs, except the STADE source (`../../src/STADE.jl`) and the secrets of the cluster relay.

The full procedure is in [`PROCEDURE.md`](PROCEDURE.md). Read it first. It defines the protocols for GPU runtime and GPU memory, the acceptance checks and the report.

## Status

The benchmark is defined and the harness is complete. The benchmark is not launched. The changes H1 to H6 of section 5 of `PROCEDURE.md` are implemented and tested on the CPU. They have not run on a GPU. The first GPU run is the smoke job (decision D6 of `PROCEDURE.md`).

## Layout

| Folder | Content |
|---|---|
| `kernels/` | The 14 kernels in STADE source form (`solvers/`, `nn/`). |
| `reference/` | The JAX kernels (`jax_ref.py`), the JAX timing and memory script (`time_fw.py`) and the parity check. |
| `tools/` | Input builders, role-file maker, local kernel check, parity driver, CPU stand-in tests of the runner. |
| `gpu/` | Case list (`specs.py`), STADE runner (`runner_body.jl`), script generator (`make_scripts.jl`). |
| `cluster/` | Role files, script generator, job packer and driver, GPU state record, submission (`go.sh`), JAX virtualenv, report. |

## Quick start (no GPU needed)

```bash
export PATH=/path/to/julia-1.10/bin:$PATH
export STADE_SRC=$(git rev-parse --show-toplevel)/src/STADE.jl
cd benchmarks/stade_vs_jax
bash cluster/make_roles.sh work                        # role files, all kernels
bash cluster/gen_stade_scripts.sh work/stade           # STADE runner scripts, configs C and A
python3 cluster/build_job.py work/job_time --stade-dir work/stade --roles work --phases time
python3 cluster/build_job.py work/job_mem  --stade-dir work/stade --roles work --phases mem_primal,mem_adjoint
```

Each `build_job.py` call writes `job.json`, the payload for the cluster. For a smoke job (one kernel, all phases) see section 6.1 of `PROCEDURE.md`.

## Submission

`cluster/go.sh` sends a payload through the GitHub relay of the Slurm pipeline. It reads the relay variables from the file in `RELAY_ENV`. The file must define `GIT_PAT`, `GIT_REPO`, `GIT_BRANCH` and `SK` (the folder of the relay scripts). Never commit this file.

```bash
RELAY_ENV=/path/to/relay_env.sh RES_DIR=work/results bash cluster/go.sh NAME work/job_time/job.json TIMEOUT
```

The V100 job needs the JAX virtualenv. Build it once on the cluster with `cluster/setup_jax_venv.sh`.

## Report

```bash
python3 cluster/report.py --results work/results/time.json --results work/results/mem.json --out OUTDIR
```

## Option ablation

`CONFIGS="C A P G" bash cluster/gen_stade_scripts.sh OUTDIR "KERNELS"` also generates the configs P (`primal_outputs = false` alone) and G (`fresh_gradient = true` alone). Build a STADE-only time job with `build_job.py ... --no-jax --phases time --configs C,P,G,A`. Build the report with `python3 cluster/ablation_report.py --results RESULT.json --stade-dir OUTDIR/all --ref FULLT1.json --out OUTDIR`. `tools/test_ablation.sh` tests the chain on the CPU.
