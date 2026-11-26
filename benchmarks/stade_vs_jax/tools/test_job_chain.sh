#!/usr/bin/env bash
# Local test of the whole job chain on the CPU. No GPU, no cluster.
# Usage: STADE_SRC=/path/to/src/STADE.jl bash test_job_chain.sh WORKDIR [KERNELS]     (default kernels: heat3d_multi mlp3 mpnn_gather)
# It generates the STADE scripts (first benchmark size only), turns them into CPU stand-in scripts, builds a job, runs the driver of the
# payload with a fake nvidia-smi and the JAX of this machine, and builds the report. Then it runs four failure tests:
# GPU busy, time budget, run time-out, throttle reason. Each must produce its check or status.
# Expected in the report: G6 fails for every JAX memory run, because the fake nvidia-smi reports a constant memory. That proves G6 works.
# Needs Julia, Python 3 with JAX and NumPy, and `python3 -m venv`. It takes about 10 min.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$HERE/.." && pwd)"
: "${STADE_SRC:?set STADE_SRC}"
W="$(mkdir -p "$1" && cd "$1" && pwd)"; KS="${2:-heat3d_multi mlp3 mpnn_gather}"
mkdir -p "$W/bin" "$W/home"; cp "$HERE/fake_nvidia_smi" "$W/bin/nvidia-smi"
export PATH="$W/bin:$PATH"
MAX_SIZES=1 bash "$PKG/cluster/gen_stade_scripts.sh" "$W/gen" "$KS"
bash "$PKG/cluster/make_roles.sh" "$W/gen" "$KS"
mkdir -p "$W/standin/all"
for f in "$W"/gen/all/*.jl; do python3 "$HERE/cpu_standin.py" "$f" "$W/standin/all/$(basename "$f")"; done
[ -x "$W/home/stade_fw_venv/bin/python" ] || python3 -m venv --system-site-packages "$W/home/stade_fw_venv"
run_job() {   # name, build options (one string), environment assignments...
  local n=$1 ba=$2; shift 2
  python3 "$PKG/cluster/build_job.py" "$W/$n" --stade-dir "$W/standin" --roles "$W/gen" --kernels $KS --max-sizes 1 $ba > "$W/$n.build.log" 2>&1
  python3 -c "import json,sys; open(sys.argv[2],'w').write(json.load(open(sys.argv[1]))['input']['_script'])" "$W/$n/job.json" "$W/$n/driver.jl"
  (cd /tmp && env WORK="$W/home" GPU_WAIT_S=5 "$@" julia --startup-file=no "$W/$n/driver.jl" > "$W/$n/result.json" 2> "$W/$n/driver.err")
  python3 "$PKG/cluster/report.py" --results "$W/$n/result.json" --out "$W/$n/report" | tail -1
}
echo "== main job (3 phases, STADE C and A, JAX)"; run_job main "" FAKE_USED=0
grep -E "^- (FAIL|NOTE)" "$W/main/report/REPORT.md" | cut -c1-160 || true
K1="${KS%% *}"
for t in "busy|--phases mem_primal --no-jax|FAKE_APPS=123,julia,500" "budget|--phases mem_primal --no-jax --limit-min 15|FAKE_USED=0" \
         "timeout|--phases mem_primal --no-jax --stade-tmo 1|FAKE_USED=0" "throttle|--phases mem_primal --no-jax|FAKE_THR=0x0000000000000005"; do
  IFS='|' read -r n ba ev <<< "$t"
  echo "== failure test: $n"; KS_SAVE="$KS"; KS="$K1"; run_job "f_$n" "$ba" "$ev" > /dev/null; KS="$KS_SAVE"
  grep -E "^- FAIL" "$W/f_$n/report/REPORT.md" | cut -c1-160 | head -2
done
