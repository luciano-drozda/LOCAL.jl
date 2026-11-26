#!/usr/bin/env bash
# Writes the role files (outputs and differentiated arrays) of every benchmark kernel, at the size "small".
# Usage: STADE_SRC=/path/to/src/STADE.jl bash make_roles.sh OUTDIR [KERNELS]
# Result: OUTDIR/roles/KERNEL.small/meta.json. It needs Julia only, no GPU.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STADE_SRC:?set STADE_SRC to the path of src/STADE.jl}"
OUT="$1"; mkdir -p "$OUT/roles"
KERNELS="${2:-$(python3 -c "import sys; sys.path.insert(0, '$HERE/../gpu'); from specs import SZ; print(' '.join(SZ))")}"
for k in $KERNELS; do
  julia --startup-file=no "$HERE/../tools/dump_case.jl" "$k" small "$OUT/roles/$k.small" > "$OUT/roles/$k.log" 2>&1 \
    && echo "ok $k" || { echo "FAILED $k (see $OUT/roles/$k.log)"; exit 1; }
done
