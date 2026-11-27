#!/usr/bin/env bash
# Runs one command under nsys or ncu, with the capture range = profiler API (the command calls cuProfilerStart and cuProfilerStop around the
# calls to record). Prints ONE JSON line (see prof_pack.jl). The standard output of the command is kept in the file prog.out.
# Usage: prof_cmd.sh nsys|ncu COMMAND...        Variables: BUNDLE (a folder for the files), JULIA_BIN
# The exit code is the exit code of the profiled command (of the last run, if nsys ran twice).
#   nsys  --cuda-graph-trace=node lists the kernels inside CUDA graphs (XLA runs repeated calls as graphs). The kernel summary (cuda_gpu_kern_sum) and the CUDA API summary (cuda_api_sum) of the recorded calls.
#         If the capture range gives no kernel, the command runs again without a capture range (capture = none in the result).
#   ncu   a counter set for each kernel of the recorded calls. NCU_LAUNCH_COUNT (default 300) is the largest number of kernels.
set -u
# nsys and ncu are on the PATH of a login shell (the module of the HPC SDK) and not on the PATH of the job: find them through a login shell.
for tool in nsys ncu; do
  p=$(command -v $tool 2>/dev/null) || p=$(bash -lc "command -v $tool" 2>/dev/null | tail -n 1)
  [ -n "$p" ] && PATH="$(dirname "$p"):$PATH"
done
export PATH
MODE=$1; shift
D=$(mktemp -d "${BUNDLE:?set BUNDLE}/prof_XXXXXX")
M=gpu__time_duration.sum,smsp__inst_executed_op_global_red.sum,smsp__inst_executed_op_global_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_atom.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
CAPTURE=range; PEC=0
case $MODE in
  nsys)
    nsys_run() {
      rm -f "$D"/prof.*
      nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $1 -o "$D/prof" --force-overwrite true "${@:2}" > "$D/prog.out" 2> "$D/prog.err" < /dev/null
      PEC=$?
      nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$D/prof.nsys-rep" > "$D/kern.csv" 2> "$D/stats.err"
    }
    nsys_run "--capture-range=cudaProfilerApi --capture-range-end=stop" "$@"
    if ! grep -q '^[0-9]' "$D/kern.csv"; then CAPTURE=none; nsys_run "" "$@"; fi
    nsys stats --force-export=true --report cuda_api_sum --format csv "$D/prof.nsys-rep" > "$D/api.csv" 2>> "$D/stats.err"
    ;;
  ncu)
    ncu --target-processes all --profile-from-start off --launch-count "${NCU_LAUNCH_COUNT:-300}" --metrics "$M" --csv --page raw \
        --log-file "$D/ncu.csv" "$@" > "$D/prog.out" 2> "$D/prog.err" < /dev/null
    PEC=$?
    ;;
  *) echo "usage: prof_cmd.sh nsys|ncu COMMAND..." >&2; exit 2;;
esac
"${JULIA_BIN:-julia}" --startup-file=no "$(dirname "${BASH_SOURCE[0]}")/prof_pack.jl" "$MODE" "$D" "$CAPTURE" "$PEC"
rm -rf "$D"
exit "$PEC"
