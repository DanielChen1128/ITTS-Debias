#!/usr/bin/env bash
# Wait for both local generation stages and run the full steering analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/md531/anaconda3/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
METHOD="constant-steering-male-strength2"
WAV_ROOT="results/parler-large/constant-steering-2x"
ANALYSIS_ROOT="runs/analysis/parler-large"
INTERACTION_ROOT="$ANALYSIS_ROOT/interactions"
PROMPTS="data/parler-large/stage2"

count_local() {
  "$PYTHON_BIN" -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(sum(1 for _ in p.rglob("*.wav")) if p.exists() else 0)' "$1"
}

while true; do
  local_count="$(count_local "$WAV_ROOT/stage1")"
  stage2_count="$(count_local "$WAV_ROOT/stage2")"
  echo "[INFO] Stage 1: $local_count/6900; Stage 2: $stage2_count/6400"
  if [[ "$local_count" -eq 6900 && "$stage2_count" -eq 6400 ]]; then
    break
  fi
  if [[ "$local_count" -lt 6900 ]]; then
    stage1_pid_file="runs/generation/parler_large_steering_2x_stage1_4090.pid"
    if [[ ! -f "$stage1_pid_file" ]] || ! kill -0 "$(<"$stage1_pid_file")" 2>/dev/null; then
      echo "[ERROR] Local generation stopped at $local_count/6900" >&2
      exit 1
    fi
  fi
  if [[ "$stage2_count" -lt 6400 ]]; then
    stage2_pid_file="runs/generation/parler_large_steering_2x_stage2_4090.pid"
    if [[ ! -f "$stage2_pid_file" ]] || ! kill -0 "$(<"$stage2_pid_file")" 2>/dev/null; then
      echo "[ERROR] Local Stage 2 generation stopped at $stage2_count/6400" >&2
      exit 1
    fi
  fi
  sleep "$POLL_SECONDS"
done

verify_count() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(count_local "$path")"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "[ERROR] $path has $actual WAVs; expected $expected" >&2
    exit 1
  fi
}

verify_count "$WAV_ROOT/stage1/status" 200
verify_count "$WAV_ROOT/stage1/career" 2700
verify_count "$WAV_ROOT/stage1/persona" 4000
verify_count "$WAV_ROOT/stage2/two-axis" 3200
verify_count "$WAV_ROOT/stage2/three-axis" 3200

inputs=(
  "descriptions_status_bias.json"
  "description_career_bias.json"
  "descriptions_persona_bias.json"
  "descriptions_two_axis.json"
  "descriptions_multi_axis.json"
)
stages=("stage1" "stage1" "stage1" "stage2" "stage2")
axes=("status" "career" "persona" "two-axis" "three-axis")
expected_counts=(200 2700 4000 3200 3200)

for index in "${!inputs[@]}"; do
  stage="${stages[$index]}"
  axis="${axes[$index]}"
  output="$ANALYSIS_ROOT/constant-steering-2x/$stage/$axis"
  result_csv="$output/detection_results.csv"
  if [[ -f "$result_csv" ]]; then
    result_count="$($PYTHON_BIN -c 'import pandas,sys; print(len(pandas.read_csv(sys.argv[1])))' "$result_csv")"
    if [[ "$result_count" -eq "${expected_counts[$index]}" ]]; then
      echo "[INFO] Reusing complete gender analysis: $result_csv"
      continue
    fi
  fi
  "$PYTHON_BIN" analyze_gender.py \
    --wav_path "$WAV_ROOT/$stage/$axis" \
    --json "$PROMPTS/${inputs[$index]}" \
    --model-name "parler-large-$METHOD" \
    --output "$output"
done

"$PYTHON_BIN" build_small_context_interaction_specs.py \
  --stage1-root "$ANALYSIS_ROOT" \
  --stage1-method-root "original=$ANALYSIS_ROOT/original/stage1" \
  --stage1-method-root "$METHOD=$ANALYSIS_ROOT/constant-steering-2x/stage1" \
  --stage2-root "$ANALYSIS_ROOT" \
  --stage2-method-root "original=$ANALYSIS_ROOT/original/stage2" \
  --stage2-method-root "$METHOD=$ANALYSIS_ROOT/constant-steering-2x/stage2" \
  --output-root "$INTERACTION_ROOT" \
  --method original \
  --method "$METHOD" \
  --matched-generation

for method in original "$METHOD"; do
  "$PYTHON_BIN" analyze_interactions.py \
    --spec "$INTERACTION_ROOT/$method/interaction_spec.json" \
    --output "$INTERACTION_ROOT/$method/interactions_10000.csv" \
    --iterations 10000 \
    --seed 20260818
done

"$PYTHON_BIN" analyze_small_context_diagnostics.py \
  --stage1-root "$ANALYSIS_ROOT" \
  --stage1-method-root "original=$ANALYSIS_ROOT/original/stage1" \
  --stage1-method-root "$METHOD=$ANALYSIS_ROOT/constant-steering-2x/stage1" \
  --interaction-root "$INTERACTION_ROOT" \
  --output-dir "$ANALYSIS_ROOT/diagnostics-steering-2x" \
  --model parler-large \
  --method original \
  --method "$METHOD" \
  --matched-generation

echo "[INFO] Full Parler Large 2x steering analysis complete"
