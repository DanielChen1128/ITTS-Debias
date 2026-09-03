#!/usr/bin/env bash
# Sync and classify completed repaired VoxInstruct batches from both GPU hosts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/md531/anaconda3/envs/gender-classifier/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-RTX5090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/r13942135/workspace/projects/ITTS_Debias/binding_effect}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/r13942135/workspace/miniconda3_new/envs/voxinstruct/bin/python}"
MODEL_CACHE="${BINDING_GENDER_DETECTOR_DIR:-models/gender_detector}"
CUDA_LIB_ROOT="${CUDA_LIB_ROOT:-/home/md531/anaconda3/envs/rl_finetune/lib/python3.10/site-packages/nvidia}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RESULT_ROOT="results/voxinstruct"
ANALYSIS_ROOT="runs/analysis/voxinstruct"

cuda_libs=(cublas cuda_runtime cudnn curand cufft cusparse nvjitlink)
for library in "${cuda_libs[@]}"; do
  LD_LIBRARY_PATH="$CUDA_LIB_ROOT/$library/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
export BINDING_GENDER_DETECTOR_DIR="$MODEL_CACHE"
export BINDING_GENDER_DEVICE=cuda

stages=(stage1 stage1 stage1 stage2 stage2 stage1 stage1 stage1 stage2 stage2)
conditions=(original original original original original ar6-nar2 ar6-nar2 ar6-nar2 ar6-nar2 ar6-nar2)
axes=(status career persona two-axis three-axis status career persona two-axis three-axis)
expected_counts=(200 2700 4000 3200 3200 200 2700 4000 3200 3200)
inputs=(
  descriptions/descriptions_status_bias.json
  descriptions/description_career_bias.json
  descriptions/descriptions_persona_bias.json
  data/parler-large/stage2/descriptions_two_axis.json
  data/parler-large/stage2/descriptions_multi_axis.json
  descriptions/descriptions_status_bias.json
  descriptions/description_career_bias.json
  descriptions/descriptions_persona_bias.json
  data/parler-large/stage2/descriptions_two_axis.json
  data/parler-large/stage2/descriptions_multi_axis.json
)

csv_is_complete() {
  local csv="$1"
  local expected="$2"
  [[ -f "$csv" ]] && "$PYTHON_BIN" -c \
    'import pandas,sys; raise SystemExit(0 if len(pandas.read_csv(sys.argv[1])) == int(sys.argv[2]) else 1)' \
    "$csv" "$expected"
}

local_batch_is_complete() {
  local path="$1"
  local expected="$2"
  "$PYTHON_BIN" -c \
    'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/"generation_manifest.json").read_text()); raise SystemExit(0 if m.get("status")=="complete" and m.get("failed_count")==0 and len(list(p.glob("*.wav")))==int(sys.argv[2]) else 1)' \
    "$path" "$expected" >/dev/null 2>&1
}

remote_batch_is_complete() {
  local relative="$1"
  local expected="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
    "$REMOTE_PYTHON -c \"import json,pathlib; p=pathlib.Path('$REMOTE_ROOT/$RESULT_ROOT/$relative'); m=json.loads((p/'generation_manifest.json').read_text()); raise SystemExit(0 if m.get('status') == 'complete' and m.get('failed_count') == 0 and len(list(p.glob('*.wav'))) == $expected else 1)\"" \
    >/dev/null 2>&1
}

pending=1
while [[ "$pending" -gt 0 ]]; do
  pending=0
  progressed=0
  for index in "${!axes[@]}"; do
    stage="${stages[$index]}"
    condition="${conditions[$index]}"
    axis="${axes[$index]}"
    expected="${expected_counts[$index]}"
    relative="$condition/$stage/$axis"
    wav_dir="$RESULT_ROOT/$relative"
    output="$ANALYSIS_ROOT/$relative"
    result_csv="$output/detection_results.csv"

    if csv_is_complete "$result_csv" "$expected"; then
      continue
    fi
    pending=$((pending + 1))

    if [[ "$condition" == "original" ]]; then
      if ! local_batch_is_complete "$wav_dir" "$expected"; then
        continue
      fi
    else
      if ! remote_batch_is_complete "$relative" "$expected"; then
        continue
      fi
      echo "[INFO] Syncing completed RTX5090 batch: $relative"
      mkdir -p "$wav_dir"
      rsync -a "$REMOTE_HOST:$REMOTE_ROOT/$RESULT_ROOT/$relative/" "$wav_dir/"
    fi

    echo "[INFO] Classifying $expected WAVs: $relative"
    "$PYTHON_BIN" analyze_gender.py \
      --wav_path "$wav_dir" \
      --json "${inputs[$index]}" \
      --model-name "voxinstruct-v2-$condition" \
      --output "$output"
    progressed=1
  done

  if [[ "$pending" -gt 0 && "$progressed" -eq 0 ]]; then
    echo "[INFO] Waiting for $pending VoxInstruct v2 batches ($(date '+%F %T'))"
    sleep "$POLL_SECONDS"
  fi
done

echo "[INFO] All VoxInstruct v2 gender classifications complete"
