#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = trinity-2-23
ROOT=/tmp/shihangz/photo-tone-transfer
DEST=/data3/shihangz/photo-tone-transfer
PY="$ROOT/Runtime/venv/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/Logs/bulk60000"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$memory" -lt 1000 || { echo "GPU $gpu occupied; stop"; exit 1; }
done
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/Research/data_collection/trinity_generate.py" --root "$ROOT" --manifest "$ROOT/Dataset/Photo_Pairs_60000_v1/pairs.json" --worker "$gpu" --workers 8 --run bulk60000 > "$ROOT/Logs/bulk60000/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
echo "$failed" > "$ROOT/Logs/bulk60000/worker_exit_status"
"$PY" "$ROOT/Research/data_collection/trinity_finalize_triplets.py" "$ROOT" || failed=1
# One deliberate archive after generation, including partial results on failure.
mkdir -p "$DEST"
rsync -a -- "$ROOT/Dataset" "$ROOT/Research" "$ROOT/Logs" "$DEST/"
printf '%s\n' "$failed" > "$DEST/Logs/bulk60000/final_exit_status"
exit "$failed"
