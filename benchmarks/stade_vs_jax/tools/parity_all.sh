#!/usr/bin/env bash
# Cross-framework parity: STADE kernels against JAX, primal and gradient, at the size "small".
# Usage: STADE_SRC=/path/to/STADE.jl/src/STADE.jl bash parity_all.sh [WORKDIR]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${1:-$HERE/parity_work}"; mkdir -p "$W/cases"
export CK_WORK="$W/generated"; mkdir -p "$CK_WORK"
KERNELS="heat3d_step heat3d_multi wave2d_step burgers2d_step swe2d_step euler2d_step ufv_scatter_step ufv_gather_step mlp3 mlp3_push mpnn_gather transformer_block unet2"
for k in $KERNELS; do
  julia "$HERE/dump_case.jl" "$k" small "$W/cases/$k.small" > "$W/$k.dump.log" 2>&1 || echo "DUMP FAILED $k (see $W/$k.dump.log)"
done
python3 "$HERE/../reference/parity_jax.py" "$W"/cases/*.small
