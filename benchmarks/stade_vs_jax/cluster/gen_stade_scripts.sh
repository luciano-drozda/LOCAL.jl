#!/usr/bin/env bash
# Generates the STADE runner scripts: one script for each kernel and each config, in OUTDIR/all/KERNEL_C.jl and KERNEL_A.jl.
# Config C = defaults. Config A = primal_outputs = false and fresh_gradient = true (the best opt-in setting).
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
rm -rf "$OUT/A" "$OUT/C" "$OUT/all"; mkdir -p "$OUT/all"
# the two configs run side by side, the generator is single-threaded
(STADE_PO=0 STADE_FRESHKW=1 julia --startup-file=no "$HERE/../gpu/make_scripts.jl" "$OUT/A" "$SPEC" > "$OUT/genA.log" 2>&1) &
(julia --startup-file=no "$HERE/../gpu/make_scripts.jl" "$OUT/C" "$SPEC" > "$OUT/genC.log" 2>&1) &
wait
for v in A C; do
  for f in "$OUT/$v"/*.cuda.f1.jl; do
    k=$(basename "$f" .cuda.f1.jl); cp "$f" "$OUT/all/${k}_$v.jl"
  done
done
ls "$OUT/all" | wc -l | xargs echo "scripts:"
