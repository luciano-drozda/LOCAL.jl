set -u
# env: BUNDLE (unpacked bundle), FWNAME (jax|torch), VARIANT, CASES ("kernel:size kernel:size ...")
for f in /usr/share/Modules/init/bash; do source $f; done
module load python/anaconda3.12.7
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
case "$FWNAME" in
  jax)   VENV=$WORK/stade_fw_venv ;;
  torch) VENV=$WORK/stade_fw_venv_torch126 ;;
esac
source $VENV/bin/activate
cd $BUNDLE
echo "node: $(hostname)  fw: $FWNAME  variant: $VARIANT  smi: $(command -v nvidia-smi) $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 | head -1)"
for cs in $CASES; do
  k=${cs%%:*}; s=${cs##*:}
  D=$BUNDLE/case_${k}_${s}
  t0=$(date +%s)
  julia tools/dump_inputs.jl $k $s $D > $D.dump.log 2>&1 || { echo "{\"kernel\":\"$k\",\"size\":\"$s\",\"ok\":false,\"error\":\"dump failed\"}"; rm -rf $D; continue; }
  t1=$(date +%s)
  env -u LD_LIBRARY_PATH timeout 500 python reference/time_fw.py --fw $FWNAME --variant $VARIANT --case $D --roles roles/$k.small 2> $D.err || echo "{\"kernel\":\"$k\",\"size\":\"$s\",\"ok\":false,\"error\":\"timeout or crash: $(tail -c 200 $D.err | tr '\n"' '  ')\"}"
  echo "# $k $s dump $((t1-t0))s total $(( $(date +%s)-t0 ))s" >&2
  rm -rf $D
done
echo "# done"
