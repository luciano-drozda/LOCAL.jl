#!/usr/bin/env bash
# Local test of the ablation chain on the CPU. No GPU, no cluster.
# Usage: STADE_SRC=/path/to/src/STADE.jl bash test_ablation.sh WORKDIR [KERNELS]     (default kernels: heat3d_multi mlp3)
# It generates the STADE scripts of the configs C, A, P and G (first benchmark size only), turns them into CPU stand-in scripts, builds a
# time-only STADE job, runs the driver of the payload with a fake nvidia-smi and builds the ablation report. Then it checks four things:
# every config ran, the four configs changed the same number of elements, the generated code of P and G differs from C, and the sabotage
# (one config changes fewer elements) produces a failed G2 check.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$HERE/.." && pwd)"
: "${STADE_SRC:?set STADE_SRC}"
W="$(mkdir -p "$1" && cd "$1" && pwd)"; KS="${2:-heat3d_multi mlp3}"
mkdir -p "$W/bin" "$W/home"; cp "$HERE/fake_nvidia_smi" "$W/bin/nvidia-smi"; export PATH="$W/bin:$PATH"
CONFIGS="C A P G" MAX_SIZES=1 bash "$PKG/cluster/gen_stade_scripts.sh" "$W/gen" "$KS"
mkdir -p "$W/standin/all"
for f in "$W"/gen/all/*.jl; do python3 "$HERE/cpu_standin.py" "$f" "$W/standin/all/$(basename "$f")"; done
for k in $KS; do for c in P G A; do
  cmp -s "$W/gen/all/${k}_C.jl" "$W/gen/all/${k}_$c.jl" && echo "NOTE: $k config $c has the same code as C" || echo "$k config $c differs from C"
done; done
python3 "$PKG/cluster/build_job.py" "$W/job" --stade-dir "$W/standin" --roles "$W/gen" --kernels $KS --max-sizes 1 --no-jax --phases time --configs C,P,G,A > "$W/build.log" 2>&1
python3 -c "import json,sys; open(sys.argv[2],'w').write(json.load(open(sys.argv[1]))['input']['_script'])" "$W/job/job.json" "$W/job/driver.jl"
(cd /tmp && env WORK="$W/home" GPU_WAIT_S=5 FAKE_USED=0 julia --startup-file=no "$W/job/driver.jl" > "$W/job/result.json" 2> "$W/job/driver.err")
python3 "$PKG/cluster/ablation_report.py" --results "$W/job/result.json" --out "$W/job/report" | tail -1
grep -c "^| " "$W/job/report/ABLATION.md" | xargs echo "table rows:"
# sabotage: edit one result so that config G changed fewer elements. G2 must fail.
python3 - "$W/job/result.json" "$W/job/sab.json" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]); d = d.get("result") or d
d = json.loads(d) if isinstance(d, str) else d
n = 0
for r in d["runs"]:
    if r["config"] == "G" and r.get("result"):
        for x in r["result"]["results"]:
            if x.get("elements_changed"): x["elements_changed"] -= 1; n += 1
assert n > 0
open(sys.argv[2], "w").write(json.dumps(d) + "\n")
PY
python3 "$PKG/cluster/ablation_report.py" --results "$W/job/sab.json" --out "$W/job/sab_report" | tail -1
grep -q "FAIL G2" "$W/job/sab_report/ABLATION.md" && echo "sabotage: G2 fails, as it must" || { echo "SABOTAGE NOT DETECTED"; exit 1; }
