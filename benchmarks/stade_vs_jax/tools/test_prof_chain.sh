#!/usr/bin/env bash
# Local test of the profile chain on the CPU (phases prof_primal, prof_adjoint, ncu_adjoint). No GPU, no cluster, fake nsys and ncu.
# Usage: STADE_SRC=/path/to/src/STADE.jl bash test_prof_chain.sh WORKDIR [KERNEL]     (default kernel: heat3d_multi)
# It builds a job from a plan file (one STADE config and one JAX case), runs the driver of the payload, and checks the result:
# every run exits 0, the nsys runs give a kernel table and an API table, the ncu run gives a counter table, the program result is inside,
# and the fallback works (a capture range with no kernel makes prof_cmd.sh run again without it).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$HERE/.." && pwd)"
: "${STADE_SRC:?set STADE_SRC}"
W="$(mkdir -p "$1" && cd "$1" && pwd)"; K="${2:-heat3d_multi}"
mkdir -p "$W/bin" "$W/home"; cp "$HERE/fake_nvidia_smi" "$W/bin/nvidia-smi"; cp "$HERE/fake_prof/nsys" "$HERE/fake_prof/ncu" "$W/bin/"
export PATH="$W/bin:$PATH"
MAX_SIZES=1 bash "$PKG/cluster/gen_stade_scripts.sh" "$W/gen" "$K"
bash "$PKG/cluster/make_roles.sh" "$W/gen" "$K"
mkdir -p "$W/standin/all"
for f in "$W"/gen/all/*.jl; do python3 "$HERE/cpu_standin.py" "$f" "$W/standin/all/$(basename "$f")"; done
[ -x "$W/home/stade_fw_venv/bin/python" ] || python3 -m venv --system-site-packages "$W/home/stade_fw_venv"
SZ=$(python3 -c "import sys; sys.path.insert(0,'$PKG/gpu'); from specs import SZ; print(SZ['$K'][0])")
printf 'stade\tC\t%s\t%s\tprof_primal,prof_adjoint,ncu_adjoint\njax\tJ\t%s\t%s\tprof_primal,prof_adjoint\n' "$K" "$SZ" "$K" "$SZ" > "$W/plan.tsv"
run_job() {   # name, environment assignments...
  local n=$1; shift
  python3 "$PKG/cluster/build_job.py" "$W/$n" --stade-dir "$W/standin" --roles "$W/gen" --plan-file "$W/plan.tsv" > "$W/$n.build.log" 2>&1
  python3 -c "import json,sys; open(sys.argv[2],'w').write(json.load(open(sys.argv[1]))['input']['_script'])" "$W/$n/job.json" "$W/$n/driver.jl"
  (cd /tmp && env WORK="$W/home" GPU_WAIT_S=5 FAKE_USED=0 "$@" julia --startup-file=no "$W/$n/driver.jl" > "$W/$n/result.json" 2> "$W/$n/driver.err")
}
check() {   # name, expected capture of the nsys runs
python3 - "$W/$1/result.json" "$2" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]); want = sys.argv[2]; bad = 0
for r in d["runs"]:
    x = r["result"] or {}
    prog = x.get("prog") or {}
    ok = r["exit"] == 0 and prog.get("phase") and x.get("kern_csv" if r["phase"] != "ncu_adjoint" else "ncu_csv")
    if r["phase"] != "ncu_adjoint": ok = ok and x.get("api_csv") and x.get("capture") == want
    calls = [q for q in (prog.get("results") or [prog]) if q.get("prof_calls_s")]
    calls = calls[0]["prof_calls_s"] if calls else []
    ok = ok and len(calls) == 3
    print(("ok  " if ok else "BAD ") + f"{r['fw']:5s} {r['phase']:13s} exit={r['exit']} mode={x.get('mode')} capture={x.get('capture')} prog.phase={prog.get('phase')} calls={calls}")
    bad += not ok
sys.exit(1 if bad else 0)
PY
}
echo "== main job"; run_job main; check main range
echo "== fallback job (the capture range gives no kernel)"; run_job fb FAKE_NSYS_NO_RANGE=1; check fb none || true
python3 - "$W/fb/result.json" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
n = [r for r in d["runs"] if r["phase"].startswith("prof_") and (r["result"] or {}).get("capture") == "none" and (r["result"] or {}).get("kern_csv")]
print("fallback runs with a kernel table:", len(n)); sys.exit(0 if len(n) == 4 else 1)
PY
