# STADE against JAX: GPU benchmark

This folder holds everything that the benchmark needs, except the STADE source (`../../src/STADE.jl`) and the secrets of the cluster relay.

The full procedure is in [`PROCEDURE.md`](PROCEDURE.md). Read it first. It defines the protocols for GPU runtime and GPU memory, the acceptance checks and the report.

## Status

The benchmark is defined. It is not launched. The harness runs and gives runtime and adjoint memory. The changes H1 to H6 of section 5 of `PROCEDURE.md` are not implemented yet. They add the separate memory phases, the host-clock timer for STADE, the GPU state record, one packer for all phases and the version record.

## Layout

| Folder | Content |
|---|---|
| `kernels/` | The 14 kernels in STADE source form (`solvers/`, `nn/`). |
| `reference/` | The JAX kernels (`jax_ref.py`), the JAX timing and memory script (`time_fw.py`) and the parity check. |
| `tools/` | Input builders, role-file maker, local kernel check, parity driver. |
| `gpu/` | Case list (`specs.py`), STADE runner (`runner_body.jl`), script generator (`make_scripts.jl`), job packer. |
| `cluster/` | Role files, job builders, submission (`go.sh`), JAX virtualenv, report. |

## Quick start (no GPU needed)

```bash
export PATH=/path/to/julia-1.10/bin:$PATH
export STADE_SRC=$(git rev-parse --show-toplevel)/src/STADE.jl
cd benchmarks/stade_vs_jax
bash cluster/make_roles.sh work                 # role files, all kernels
python3 cluster/build_jax_job.py work work/jax 3600
bash cluster/build_stade_job.sh work/stade 1800
```

The last two commands write `work/jax/jax_job.json` and `work/stade/stade_job.json`. These are the payloads for the cluster.

## Submission

`cluster/go.sh` sends a payload through the GitHub relay of the Slurm pipeline. It reads the relay variables from the file in `RELAY_ENV`. The file must define `GIT_PAT`, `GIT_REPO`, `GIT_BRANCH` and `SK` (the folder of the relay scripts). Never commit this file.

```bash
RELAY_ENV=/path/to/relay_env.sh RES_DIR=work/results bash cluster/go.sh NAME work/stade/stade_job.json TIMEOUT
```

The V100 job needs the JAX virtualenv. Build it once on the cluster with `cluster/setup_jax_venv.sh`.

## Report

```bash
python3 cluster/report.py --stade work/results/NAME.json --jax work/results/NAME2.json --out OUTDIR
```
