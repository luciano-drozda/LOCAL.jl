#!/usr/bin/env bash
# Records the state of the GPU around one run. Prints one JSON line. It needs bash and nvidia-smi only.
#   gpu_state.sh snap              state now: compute processes, memory in use, clocks, temperature, throttle reasons
#   gpu_state.sh start FILE        sample the GPU once each second into FILE, in the background
#   gpu_state.sh stop FILE         stop the sampling and summarize it
# snap reports busy=true if a compute process is listed or the memory in use is above GPU_IDLE_MIB (default 64).
# The summary reports throttle_bad=true if a sample shows a power cap, a slowdown, a thermal reason or a power brake
# (bits 0x4 0x8 0x20 0x40 0x80). The idle bit (0x1) and the application-clocks bit (0x2) do not count.
set -u
SMI=nvidia-smi
fld_thr() {   # the name of the throttle field depends on the driver
  if $SMI --help-query-gpu 2>/dev/null | grep -q '"clocks_throttle_reasons.active"'; then echo clocks_throttle_reasons.active
  else echo clocks_event_reasons.active; fi
}
q() { $SMI --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '; }
num() { case "$1" in ''|*[!0-9.]*) echo null;; *) echo "$1";; esac; }
hexv() { local h="${1:-0x0}"; case "$h" in 0x*|0X*) echo $((h));; *) echo 0;; esac; }
bad_mask=$((0x4 | 0x8 | 0x20 | 0x40 | 0x80))

case "${1:-snap}" in
  snap)
    THR=$(fld_thr)
    IFS=, read -r drv name used sm mem temp thr <<< "$($SMI --query-gpu=driver_version,name,memory.used,clocks.sm,clocks.mem,temperature.gpu,$THR --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed 's/, */,/g')"
    apps=$($SMI --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d')
    napps=$(printf '%s' "$apps" | grep -c . || true)
    lim=${GPU_IDLE_MIB:-64}; busy=false
    [ "$napps" -gt 0 ] && busy=true
    case "${used:-}" in ''|*[!0-9]*) ;; *) [ "$used" -gt "$lim" ] && busy=true;; esac
    esc=$(printf '%s' "$apps" | tr '\n"\\' ';  ')
    printf '{"driver":"%s","gpu":"%s","mem_used_mib":%s,"compute_apps":%s,"apps":"%s","busy":%s,"sm_clock":%s,"mem_clock":%s,"temp":%s,"throttle":"%s"}\n' \
      "${drv:-}" "${name:-}" "$(num "${used:-}")" "$napps" "$esc" "$busy" "$(num "${sm:-}")" "$(num "${mem:-}")" "$(num "${temp:-}")" "${thr:-}"
    ;;
  start)
    F="$2"; THR=$(fld_thr)
    $SMI --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,$THR --format=csv,noheader,nounits -lms 1000 > "$F" 2>/dev/null &
    echo $! > "$F.pid"
    ;;
  stop)
    F="$2"
    [ -f "$F.pid" ] && { kill "$(cat "$F.pid")" 2>/dev/null; wait "$(cat "$F.pid")" 2>/dev/null; rm -f "$F.pid"; }
    S=$(sed 's/, */,/g' "$F" 2>/dev/null | grep -E '^[0-9]' || true)
    n=$(printf '%s' "$S" | grep -c . || true)
    if [ "$n" -eq 0 ]; then echo '{"samples":0,"throttle_bad":false}'; exit 0; fi
    col() { printf '%s\n' "$S" | cut -d, -f"$1" | grep -E '^[0-9.]+$' | sort -n; }
    med() { col "$1" | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "null"}'; }
    mn() { v=$(col "$1" | head -n 1); echo "${v:-null}"; }
    mx() { v=$(col "$1" | tail -n 1); echo "${v:-null}"; }
    or=0
    for h in $(printf '%s\n' "$S" | cut -d, -f6 | sort -u); do or=$((or | $(hexv "$h"))); done
    bad=false; [ $((or & bad_mask)) -ne 0 ] && bad=true
    printf '{"samples":%s,"sm_clock_min":%s,"sm_clock_median":%s,"sm_clock_max":%s,"mem_clock_median":%s,"temp_max":%s,"power_max":%s,"mem_used_max_mib":%s,"throttle_or":"0x%x","throttle_bad":%s}\n' \
      "$n" "$(mn 1)" "$(med 1)" "$(mx 1)" "$(med 2)" "$(mx 3)" "$(mx 4)" "$(mx 5)" "$or" "$bad"
    ;;
  *) echo "usage: gpu_state.sh snap | start FILE | stop FILE" >&2; exit 2;;
esac
