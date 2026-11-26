#!/usr/bin/env bash
# Local test of the phase split of the STADE runner, with the CPU stand-in. No GPU.
# Usage: bash test_runner_phases.sh STANDIN_SCRIPT.jl [WORKDIR]
# 1. Run the three phases and check them (check_phase_results.py).
# 2. Sabotage: put the adjoint setup into the mem_primal phase. The check must FAIL. A guard that never failed is not a guard.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$1"; W="${2:-$(mktemp -d)}"; mkdir -p "$W"
run3() { for ph in time mem_primal mem_adjoint; do STADE_PHASE=$ph BENCH_CONFIG=T julia --startup-file=no "$1" > "$W/$2.$ph.json" 2> "$W/$2.$ph.err" || { echo "run failed: $ph"; tail -3 "$W/$2.$ph.err"; return 2; }; done
        python3 "$HERE/check_phase_results.py" "$W/$2.time.json" "$W/$2.mem_primal.json" "$W/$2.mem_adjoint.json"; }
echo "plain: $(run3 "$S" plain)"
python3 - "$S" "$W/sab.jl" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = "    reset_p!() = (for n in DEPS; copyto!(dev[n], orig[n]); end; nothing)\n"
assert s.count(a) == 1
b = '        reset_p!(); _sync(); res["alloc_primal_call_mb"] = _alloc_mb(() -> fp(pargs...))\n'
assert s.count(b) == 1
# the later use keeps the arrays alive past the memory reading (Julia frees a local after its last use)
s = s.replace(a, a + "    _sab = _adj_setup(fb, fi, dev, _zdev)\n").replace(b, b + '        res["sab"] = length(_sab[1])\n')
open(sys.argv[2], "w").write(s)
PY
echo "sabotage (adjoint data in the primal phase): $(run3 "$W/sab.jl" sab)"
