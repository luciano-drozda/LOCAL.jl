"""Builds the JAX job: one V100 payload that runs every benchmark case with jax.jit, one process for each case.
Usage: python3 build_jax_job.py ROLES_OUTDIR OUTDIR [TIMEOUT_S] [KERNEL ...]
ROLES_OUTDIR is the OUTDIR of make_roles.sh. Result: OUTDIR/jax_job.json. Submit it with go.sh.
The job needs the virtualenv of setup_jax_venv.sh on the cluster."""
import os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PKG, "gpu"))
from specs import SZ
roles, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
tmo = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
kernels = sys.argv[4:] or list(SZ)
os.makedirs(out, exist_ok=True)
B = out + "/bundle"; shutil.rmtree(B, ignore_errors=True)
os.makedirs(B + "/tools"); os.makedirs(B + "/roles")
for f in ("inputs.jl", "dump_inputs.jl"):
    shutil.copy(f"{PKG}/tools/{f}", f"{B}/tools/{f}")
shutil.copytree(f"{PKG}/kernels", f"{B}/kernels")
shutil.copytree(f"{PKG}/reference", f"{B}/reference", ignore=shutil.ignore_patterns("__pycache__"))
for k in kernels:
    os.makedirs(f"{B}/roles/{k}.small")
    shutil.copy(f"{roles}/roles/{k}.small/meta.json", f"{B}/roles/{k}.small/meta.json")
subprocess.check_call(["tar", "czf", out + "/bundle.tgz", "-C", B, "."])
cases = " ".join(f"{k}:{s}" for k in kernels for s in SZ[k])
run = f"{out}/run_jax.sh"
open(run, "w").write(f'export FWNAME=jax VARIANT=jit CASES="{cases}"\n' + open(f"{HERE}/fw_run.sh").read())
subprocess.check_call([sys.executable, f"{HERE}/pack_py.py", f"{out}/jax_job.json", str(tmo), out + "/bundle.tgz", run])
print(len(cases.split()), "cases ->", out + "/jax_job.json")
