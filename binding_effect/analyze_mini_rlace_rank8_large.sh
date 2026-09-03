#!/usr/bin/env bash
# Classify Mini RLACE rank-8 outputs and run matched diagnostics.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV="${CONDA_ENV:-base}"
MODEL_CACHE="${BINDING_GENDER_DETECTOR_DIR:-models/gender_detector}"
METHODS=("original" "rlace-rank-8")
PROMPTS="data/parler-mini/stage2"
WAVS="results/parler-mini"
ANALYSIS="runs/analysis/parler-mini"
INTERACTIONS="$ANALYSIS/large-v1/interactions"
SEED="${SEED:-20260818}"

run_python() {
  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" python "$@"
}

classify() {
  local wav_path="$1"
  local prompts="$2"
  local output="$3"
  local model_name="$4"
  BINDING_GENDER_DETECTOR_DIR="$MODEL_CACHE" run_python analyze_gender.py \
    --wav_path "$wav_path" \
    --json "$prompts" \
    --model-name "$model_name" \
    --output "$output"
}

classify "$WAVS/original/stage1/status" \
  "descriptions/descriptions_status_bias.json" "$ANALYSIS/original/stage1/status" \
  "parler-mini-original"
classify "$WAVS/original/stage1/career" \
  "descriptions/description_career_bias.json" "$ANALYSIS/original/stage1/career" \
  "parler-mini-original"
classify "$WAVS/original/stage1/persona" \
  "descriptions/descriptions_persona_bias.json" "$ANALYSIS/original/stage1/persona" \
  "parler-mini-original"
classify "$WAVS/rlace-rank-8/stage1/status" \
  "descriptions/descriptions_status_bias.json" "$ANALYSIS/rlace-rank-8/stage1/status" \
  "parler-mini-rlace-rank-8"
classify "$WAVS/rlace-rank-8/stage1/career" \
  "descriptions/description_career_bias.json" "$ANALYSIS/rlace-rank-8/stage1/career" \
  "parler-mini-rlace-rank-8"
classify "$WAVS/rlace-rank-8/stage1/persona" \
  "descriptions/descriptions_persona_bias.json" "$ANALYSIS/rlace-rank-8/stage1/persona" \
  "parler-mini-rlace-rank-8"
classify "$WAVS/rlace-rank-8/stage2/two-axis" \
  "$PROMPTS/descriptions_two_axis.json" "$ANALYSIS/rlace-rank-8/stage2/two-axis" \
  "parler-mini-rlace-rank-8"
classify "$WAVS/rlace-rank-8/stage2/three-axis" \
  "$PROMPTS/descriptions_multi_axis.json" "$ANALYSIS/rlace-rank-8/stage2/three-axis" \
  "parler-mini-rlace-rank-8"

# Normalize reused detections created before the binary female/male protocol.
run_python recompute_binary_gender.py \
  "$ANALYSIS/original" \
  "$ANALYSIS/rlace-rank-8"

run_python build_small_context_interaction_specs.py \
  --stage1-root "$ANALYSIS" \
  --stage1-method-root "original=$ANALYSIS/original/stage1" \
  --stage1-method-root "rlace-rank-8=$ANALYSIS/rlace-rank-8/stage1" \
  --stage2-root "$ANALYSIS" \
  --stage2-method-root "original=$ANALYSIS/original/stage2" \
  --stage2-method-root "rlace-rank-8=$ANALYSIS/rlace-rank-8/stage2" \
  --output-root "$INTERACTIONS" \
  --matched-generation \
  --method original \
  --method rlace-rank-8

for method in "${METHODS[@]}"; do
  run_python analyze_interactions.py \
    --spec "$INTERACTIONS/$method/interaction_spec.json" \
    --output "$INTERACTIONS/$method/interactions_10000.csv" \
    --iterations 10000 \
    --seed "$SEED"
done

run_python analyze_small_context_diagnostics.py \
  --stage1-root "$ANALYSIS" \
  --stage1-method-root "original=$ANALYSIS/original/stage1" \
  --stage1-method-root "rlace-rank-8=$ANALYSIS/rlace-rank-8/stage1" \
  --interaction-root "$INTERACTIONS" \
  --output-dir "$ANALYSIS/large-v1/diagnostics" \
  --model parler-mini \
  --matched-generation \
  --method original \
  --method rlace-rank-8

echo "[INFO] Mini RLACE rank-8 analysis complete: $ANALYSIS/large-v1"
