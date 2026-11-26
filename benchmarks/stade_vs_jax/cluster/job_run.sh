#!/usr/bin/env bash
# Runs the plan of one job on the GPU node. Started by the Julia driver (job_driver.jl) with these variables:
#   BUNDLE         the unpacked bundle (plan.tsv, stade/, tools/, kernels/, reference/, roles/, cluster/)
#   JULIA_BIN, JPROJ   the Julia binary and the project of the driver
#   JOB_BUDGET_S   no new run starts after this many seconds. Later runs get status skipped_budget.
#   STADE_RUN_TMO, JAX_RUN_TMO   time limit of one run, in seconds
#   STADE_COMMIT, KIT_HASH       recorded in the results
# plan.tsv, tab separated: fw config kernel size phases. For STADE the size is "-". phases is a comma list.
# Each run is one process. Each run writes OUT/NNNN/{info.txt,out.json,err.txt,gpu_before.json,gpu.json}.
set -u
OUT=$BUNDLE/out; mkdir -p "$OUT"
T0=$(date +%s); DEADLINE=$((T0 + ${JOB_BUDGET_S:-6300}))
GS="bash $BUNDLE/cluster/gpu_state.sh"
export STADE_COMMIT KIT_HASH

# run_one FW CONFIG KERNEL SIZE PHASE TIMEOUT COMMAND...
run_one() {
  local fw=$1 cfg=$2 k=$3 sz=$4 ph=$5 tmo=$6; shift 6
  local d; d=$OUT/$(printf %04d $(( $(ls "$OUT" | wc -l) + 1 ))); mkdir -p "$d"
  printf 'fw=%s\nconfig=%s\nkernel=%s\nsize=%s\nphase=%s\n' "$fw" "$cfg" "$k" "$sz" "$ph" > "$d/info.txt"
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "status=skipped_budget" >> "$d/info.txt"; return 0; fi
  # the GPU must be free: wait up to GPU_WAIT_S seconds (default 60) for the previous process to release it
  local i
  for i in $(seq 1 $(( ${GPU_WAIT_S:-60} / 5 + 1 ))); do
    $GS snap > "$d/gpu_before.json"; grep -q '"busy":false' "$d/gpu_before.json" && break; sleep 5
  done
  if ! grep -q '"busy":false' "$d/gpu_before.json"; then echo "status=gpu_busy" >> "$d/info.txt"; return 0; fi
  $GS start "$d/mon.csv"
  local t1; t1=$(date +%s)
  timeout -k 30 "$tmo" "$@" > "$d/stdout.txt" 2> "$d/err.txt" < /dev/null
  local ec=$?
  echo "exit=$ec" >> "$d/info.txt"; echo "seconds=$(( $(date +%s) - t1 ))" >> "$d/info.txt"
  $GS stop "$d/mon.csv" > "$d/gpu.json"
  grep '^{' "$d/stdout.txt" | tail -n 1 > "$d/out.json"
  tail -c 1500 "$d/err.txt" > "$d/err_tail.txt"
  rm -f "$d/stdout.txt" "$d/err.txt" "$d/mon.csv"
  return 0
}

# STADE: one Julia process for each kernel, config and phase
while IFS=$'\t' read -r fw cfg k sz phases; do
  [ "$fw" = stade ] || continue
  for ph in ${phases//,/ }; do
    run_one stade "$cfg" "$k" "" "$ph" "${STADE_RUN_TMO:-1800}" \
      env STADE_PHASE="$ph" BENCH_CONFIG="$cfg" "$JULIA_BIN" --startup-file=no --project="$JPROJ" "$BUNDLE/stade/${k}_${cfg}.jl"
  done
done < "$BUNDLE/plan.tsv"

# JAX: one Python process for each case and phase. The Python module and the virtualenv stay inside this subshell.
if grep -q '^jax' "$BUNDLE/plan.tsv"; then
( for f in /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do [ -f "$f" ] && { source "$f"; break; }; done
  module load python/anaconda3.12.7
  export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
  source "${WORK:?set WORK}/stade_fw_venv/bin/activate"
  cd "$BUNDLE"
  while IFS=$'\t' read -r fw cfg k sz phases; do
    [ "$fw" = jax ] || continue
    D=$BUNDLE/case_${k}_${sz}
    if [ "$(date +%s)" -lt "$DEADLINE" ] && ! "$JULIA_BIN" --startup-file=no tools/dump_inputs.jl "$k" "$sz" "$D" > "$D.dump.log" 2>&1 < /dev/null; then
      for ph in ${phases//,/ }; do
        d=$OUT/$(printf %04d $(( $(ls "$OUT" | wc -l) + 1 ))); mkdir -p "$d"
        printf 'fw=jax\nconfig=J\nkernel=%s\nsize=%s\nphase=%s\nstatus=dump_failed\n' "$k" "$sz" "$ph" > "$d/info.txt"
      done
      rm -rf "$D"; continue
    fi
    for ph in ${phases//,/ }; do
      run_one jax J "$k" "$sz" "$ph" "${JAX_RUN_TMO:-500}" \
        env -u LD_LIBRARY_PATH python reference/time_fw.py --case "$D" --roles "roles/$k.small" --phase "$ph"
    done
    rm -rf "$D"
  done < "$BUNDLE/plan.tsv"
)
fi
echo "# done $(( $(date +%s) - T0 )) s"
