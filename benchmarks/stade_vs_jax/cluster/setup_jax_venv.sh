#!/usr/bin/env bash
# Builds the JAX virtualenv on the cluster. Run it once, on a node that has the V100 driver.
# The module name and the path are the values of the cluster of the first runs. Change them for another site.
set -u
for f in /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do [ -f $f ] && source $f && break; done
module load python/anaconda3.12.7
V=${WORK:?set WORK}/stade_fw_venv
[ -x $V/bin/python ] || python3 -m venv $V
source $V/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet "jax[cuda12]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
env -u LD_LIBRARY_PATH python - <<'PY'
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print("jax", jax.__version__, jax.devices())
a = jnp.ones((1000, 1000), dtype=jnp.float64); print("float64 matmul", float((a @ a).block_until_ready()[0, 0]))
PY
