#!/usr/bin/env bash
# Run gender, interaction, diagnostics, and paired-quality analysis for one condition.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:?Set MODEL to parler-mini, parler-large, or voxinstruct}"
CONDITION="${CONDITION:?Set the candidate condition directory name}"
PYTHON_BIN="${PYTHON_BIN:-/home/md531/anaconda3/bin/python}"
GENDER_PYTHON="${GENDER_PYTHON:-/home/md531/anaconda3/envs/gender-classifier/bin/python}"
QUALITY_PYTHON="${QUALITY_PYTHON:-$PYTHON_BIN}"
MODEL_CACHE="${BINDING_GENDER_DETECTOR_DIR:-models/gender_detector}"
SEED="${SEED:-20260818}"
WAV_ROOT="results/$MODEL"
ANALYSIS_ROOT="runs/analysis/$MODEL"
COMPARISON_ROOT="$ANALYSIS_ROOT/comparison/$CONDITION"
INTERACTION_ROOT="$COMPARISON_ROOT/interactions"

case "$MODEL" in
  parler-mini)
    stage2_prompts="data/parler-mini/stage2"
    quality_model="mini"
    ;;
  parler-large)
    stage2_prompts="data/parler-large/stage2"
    quality_model="large"
    ;;
  voxinstruct)
    stage2_prompts="data/parler-large/stage2"
    quality_model="voxinstruct"
    ;;
  *)
    echo "[ERROR] unsupported MODEL: $MODEL" >&2
    exit 2
    ;;
esac

stages=(stage1 stage1 stage1 stage2 stage2)
axes=(status career persona two-axis three-axis)
expected=(200 2700 4000 3200 3200)
inputs=(
  descriptions/descriptions_status_bias.json
  descriptions/description_career_bias.json
  descriptions/descriptions_persona_bias.json
  "$stage2_prompts/descriptions_two_axis.json"
  "$stage2_prompts/descriptions_multi_axis.json"
)

for method in original "$CONDITION"; do
  for index in "${!axes[@]}"; do
    stage="${stages[$index]}"
    axis="${axes[$index]}"
    wav_dir="$WAV_ROOT/$method/$stage/$axis"
    output="$ANALYSIS_ROOT/$method/$stage/$axis"
    csv="$output/detection_results.csv"
    wav_count="$($PYTHON_BIN -c 'import pathlib,sys; print(len(list(pathlib.Path(sys.argv[1]).glob("*.wav"))))' "$wav_dir")"
    if [[ "$wav_count" -ne "${expected[$index]}" ]]; then
      echo "[ERROR] $wav_dir has $wav_count WAVs; expected ${expected[$index]}" >&2
      exit 1
    fi
    if [[ -f "$csv" ]] && [[ "$($PYTHON_BIN -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' "$csv")" -eq "${expected[$index]}" ]]; then
      continue
    fi
    BINDING_GENDER_DETECTOR_DIR="$MODEL_CACHE" BINDING_GENDER_DEVICE=cuda \
      "$GENDER_PYTHON" analyze_gender.py \
      --wav_path "$wav_dir" \
      --json "${inputs[$index]}" \
      --model-name "$MODEL-$method" \
      --output "$output"
  done
done

"$PYTHON_BIN" recompute_binary_gender.py \
  "$ANALYSIS_ROOT/original" "$ANALYSIS_ROOT/$CONDITION"

"$PYTHON_BIN" build_small_context_interaction_specs.py \
  --stage1-root "$ANALYSIS_ROOT" \
  --stage1-method-root "original=$ANALYSIS_ROOT/original/stage1" \
  --stage1-method-root "$CONDITION=$ANALYSIS_ROOT/$CONDITION/stage1" \
  --stage2-root "$ANALYSIS_ROOT" \
  --stage2-method-root "original=$ANALYSIS_ROOT/original/stage2" \
  --stage2-method-root "$CONDITION=$ANALYSIS_ROOT/$CONDITION/stage2" \
  --output-root "$INTERACTION_ROOT" \
  --matched-generation \
  --method original \
  --method "$CONDITION"

for method in original "$CONDITION"; do
  "$PYTHON_BIN" analyze_interactions.py \
    --spec "$INTERACTION_ROOT/$method/interaction_spec.json" \
    --output "$INTERACTION_ROOT/$method/interactions_10000.csv" \
    --iterations 10000 \
    --seed "$SEED"
done

"$PYTHON_BIN" analyze_small_context_diagnostics.py \
  --stage1-root "$ANALYSIS_ROOT" \
  --stage1-method-root "original=$ANALYSIS_ROOT/original/stage1" \
  --stage1-method-root "$CONDITION=$ANALYSIS_ROOT/$CONDITION/stage1" \
  --interaction-root "$INTERACTION_ROOT" \
  --output-dir "$COMPARISON_ROOT/diagnostics" \
  --model "$MODEL" \
  --matched-generation \
  --method original \
  --method "$CONDITION"

quality_manifest="data/$MODEL/${CONDITION}_quality_v1.json"
quality_report="$COMPARISON_ROOT/quality/quality_v1.json"
"$PYTHON_BIN" build_full_quality_manifest.py \
  --model "$quality_model" \
  --candidate "$CONDITION" \
  --pairs-per-stratum 100 \
  --seed unified-constant-steering-quality-v1 \
  --output "$quality_manifest"
"$QUALITY_PYTHON" evaluate_screen_quality.py \
  --prompts "$quality_manifest" \
  --root . \
  --candidate "$CONDITION" \
  --output "$quality_report"

echo "[INFO] Full analysis complete: $COMPARISON_ROOT"
