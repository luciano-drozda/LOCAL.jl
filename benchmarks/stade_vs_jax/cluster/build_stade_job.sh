#!/usr/bin/env bash
# Builds the STADE job: one runner script for each kernel and each config, packed into one V100 payload.
# Config C = defaults. Config A = primal_outputs = false and fresh_gradient = true (the best opt-in setting).
# Usage: STADE_SRC=/path/to/src/STADE.jl bash build_stade_job.sh OUTDIR [TIMEOUT_PER_RUN_S] [KERNELS]
# Result: OUTDIR/stade_job.json. Submit it with go.sh.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STADE_SRC:?set STADE_SRC to the path of src/STADE.jl}"
OUT="$1"; TMO="${2:-1800}"; KERNELS="${3:-}"
mkdir -p "$OUT"
SPEC=$(python3 - "$HERE" "$KERNELS" <<'PY'
import sys; sys.path.insert(0, sys.argv[1] + "/../gpu")
from specs import SZ
ks = sys.argv[2].split() or list(SZ)
print(" ".join(f"{k}:cuda:1:" + "+".join(["tiny!"] + SZ[k]) for k in ks))
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
python3 "$HERE/../gpu/pack_job.py" "$OUT/stade_job.json" "$TMO" "$OUT"/all/*.jl
