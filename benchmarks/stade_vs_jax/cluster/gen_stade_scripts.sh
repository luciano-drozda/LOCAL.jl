#!/usr/bin/env bash
# Generates the STADE runner scripts: one script for each kernel and each config, in OUTDIR/all/KERNEL_C.jl and KERNEL_A.jl.
# Config C = defaults. Config A = primal_outputs = false and fresh_gradient = true (the best opt-in setting).
# Ablation configs: P = primal_outputs = false alone, G = fresh_gradient = true alone, F = warp_runs = false alone (STADE v0.6.39 and later).
# CONFIGS="C A P G" (environment): the configs to generate. The default is "C A".
# Usage: STADE_SRC=/path/to/src/STADE.jl bash gen_stade_scripts.sh OUTDIR [KERNELS]
# The scripts name the sizes: the verify size tiny! and every benchmark size of gpu/specs.py. Julia only, no GPU.
# MAX_SIZES=N (environment): only the first N benchmark sizes of each kernel (a smoke job or a local test).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STADE_SRC:?set STADE_SRC to the path of src/STADE.jl}"
OUT="$1"; KERNELS="${2:-}"
mkdir -p "$OUT"
SPEC=$(python3 - "$HERE" "$KERNELS" <<'PY'
import sys; sys.path.insert(0, sys.argv[1] + "/../gpu")
from specs import SZ
import os
ks = sys.argv[2].split() or list(SZ)
n = int(os.environ.get("MAX_SIZES", "0")) or None
print(" ".join(f"{k}:cuda:1:" + "+".join(["tiny!"] + SZ[k][:n]) for k in ks))
PY
)
CONFIGS="${CONFIGS:-C A}"
declare -A CENV=([C]="" [A]="STADE_PO=0 STADE_FRESHKW=1" [P]="STADE_PO=0" [G]="STADE_FRESHKW=1" [F]="STADE_WARPRUNS=0")
for v in $CONFIGS; do [ -n "${CENV[$v]+x}" ] || { echo "unknown config $v" >&2; exit 2; }; done
rm -rf "$OUT/all"; for v in A C P G F; do rm -rf "$OUT/$v"; done; mkdir -p "$OUT/all"
# the configs run side by side, the generator is single-threaded
for v in $CONFIGS; do
  (env ${CENV[$v]} julia --startup-file=no "$HERE/../gpu/make_scripts.jl" "$OUT/$v" "$SPEC" > "$OUT/gen$v.log" 2>&1) &
done
wait
for v in $CONFIGS; do
  for f in "$OUT/$v"/*.cuda.f1.jl; do
    k=$(basename "$f" .cuda.f1.jl); cp "$f" "$OUT/all/${k}_$v.jl"
  done
done
ls "$OUT/all" | wc -l | xargs echo "scripts:"
