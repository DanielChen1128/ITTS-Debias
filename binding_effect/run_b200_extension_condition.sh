#!/usr/bin/env bash
# Fit or generate one resumable B200 extension condition.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:?Set MODEL to parler-mini, parler-large, or voxinstruct}"
ACTION="${ACTION:?Set ACTION to fit, screen, or full}"
PAIRS="${PAIRS:?Set PAIRS to the anchor-pair count}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PAIR_SEED="${PAIR_SEED:-20260909}"
PARLER_MINI_MODEL_ID="${PARLER_MINI_MODEL_ID:-parler-tts/parler-tts-mini-v1}"
PARLER_MINI_REVISION="${PARLER_MINI_REVISION:-0392b9451a601e528fd863bbb0598431fee810d9}"
PARLER_LARGE_MODEL_ID="${PARLER_LARGE_MODEL_ID:-parler-tts/parler-tts-large-v1}"
PARLER_LARGE_REVISION="${PARLER_LARGE_REVISION:-50cb4b874c83902f930d7c2e753224c15654f11e}"
VOXINSTRUCT_MODEL_ID="${VOXINSTRUCT_MODEL_ID:-models/voxinstruct/pretrained}"
VOXINSTRUCT_BACKEND_PATH="${VOXINSTRUCT_BACKEND_PATH:-backends/voxinstruct}"

slug() { printf '%s' "$1" | tr '.' 'p'; }

prepare_parler_artifact() {
  local model_id="$1" revision="$2" anchors="$3" pool_cache="$4" subset_dir="$5" artifact="$6"
  if [[ ! -f "$pool_cache" ]]; then
    "$PYTHON_BIN" fit_rlace.py --json "$anchors" --model-id "$model_id" \
      --model-revision "$revision" --activations-cache "$pool_cache" --extract-only
  fi
  "$PYTHON_BIN" build_anchor_pair_subsets.py --anchors "$anchors" --output-dir "$subset_dir" \
    --pair-count "$PAIRS" --seed "$PAIR_SEED" --activation-cache "pooled=$pool_cache"
  if [[ ! -f "$artifact" ]]; then
    "$PYTHON_BIN" fit_constant_steering.py --anchors "$subset_dir/pairs-$PAIRS.json" \
      --activations-cache "$subset_dir/pairs-$PAIRS-pooled-activations.pt" --output "$artifact"
  fi
}

prepare_vox_artifacts() {
  local anchors="data/voxinstruct/voxinstruct-training-anchors-v1.json"
  local base="data/extension-ablation/voxinstruct"
  local ar_pool="$base/full-ar-activations.pt" nar_pool="$base/full-nar-activations.pt"
  local subset="$base/subsets"
  mkdir -p "$base/artifacts"
  if [[ ! -f "$ar_pool" || ! -f "$nar_pool" ]]; then
    "$PYTHON_BIN" extract_voxinstruct_activations.py --json "$anchors" \
      --model-id "$VOXINSTRUCT_MODEL_ID" --backend-path "$VOXINSTRUCT_BACKEND_PATH" \
      --ar-output "$ar_pool" --nar-output "$nar_pool" --prompt-text-field prompt_text
  fi
  "$PYTHON_BIN" build_anchor_pair_subsets.py --anchors "$anchors" --output-dir "$subset" \
    --pair-count "$PAIRS" --seed "$PAIR_SEED" \
    --activation-cache "ar=$ar_pool" --activation-cache "nar=$nar_pool"
  for stage in ar nar; do
    local artifact="$base/artifacts/pairs-$PAIRS-$stage-female-to-male.pt"
    if [[ ! -f "$artifact" ]]; then
      "$PYTHON_BIN" fit_constant_steering.py --anchors "$subset/pairs-$PAIRS.json" \
        --activations-cache "$subset/pairs-$PAIRS-$stage-activations.pt" --output "$artifact"
    fi
  done
}

case "$MODEL" in
  parler-mini)
    [[ "$PAIRS" == "270" ]] || { echo "[ERROR] Parler Mini has exactly 270 available pairs" >&2; exit 2; }
    STRENGTH="${STRENGTH:?Set STRENGTH for Parler}"
    base="data/extension-ablation/parler-mini"
    artifact="$base/artifacts/pairs-270-female-to-male.pt"
    mkdir -p "$base/artifacts"
    prepare_parler_artifact "$PARLER_MINI_MODEL_ID" "$PARLER_MINI_REVISION" \
      "data/parler-mini/training_anchors.json" "$base/full-pooled-activations.pt" "$base/subsets" "$artifact"
    condition="anchor-pairs-270-strength-$(slug "$STRENGTH")"
    ;;
  parler-large)
    STRENGTH="${STRENGTH:?Set STRENGTH for Parler}"
    base="data/extension-ablation/parler-large"
    artifact="$base/artifacts/pairs-$PAIRS-female-to-male.pt"
    mkdir -p "$base/artifacts"
    prepare_parler_artifact "$PARLER_LARGE_MODEL_ID" "$PARLER_LARGE_REVISION" \
      "data/parler-large/training_anchors.json" "$base/full-pooled-activations.pt" "$base/subsets" "$artifact"
    condition="anchor-pairs-$PAIRS-strength-$(slug "$STRENGTH")"
    ;;
  voxinstruct)
    AR_STRENGTH="${AR_STRENGTH:?Set AR_STRENGTH for VoxInstruct}"
    NAR_STRENGTH="${NAR_STRENGTH:?Set NAR_STRENGTH for VoxInstruct}"
    prepare_vox_artifacts
    ar_artifact="data/extension-ablation/voxinstruct/artifacts/pairs-$PAIRS-ar-female-to-male.pt"
    nar_artifact="data/extension-ablation/voxinstruct/artifacts/pairs-$PAIRS-nar-female-to-male.pt"
    condition="anchor-pairs-$PAIRS-ar-$(slug "$AR_STRENGTH")-nar-$(slug "$NAR_STRENGTH")"
    ;;
  *) echo "[ERROR] unsupported MODEL: $MODEL" >&2; exit 2 ;;
esac

if [[ "$ACTION" == "fit" ]]; then
  echo "[INFO] Artifact ready for $MODEL with $PAIRS pairs"
  exit 0
fi
if [[ "$ACTION" != "screen" && "$ACTION" != "full" ]]; then
  echo "[ERROR] unsupported ACTION: $ACTION" >&2
  exit 2
fi

if [[ "$ACTION" == "screen" ]]; then
  [[ -f "data/extension-screen-v1/$MODEL/manifest.json" ]] || \
    "$PYTHON_BIN" build_extension_screen.py --model "$MODEL"
  screen_root="data/extension-screen-v1/$MODEL"
  inputs=("$screen_root/status.json" "$screen_root/career.json" "$screen_root/persona.json" "$screen_root/two-axis.json" "$screen_root/three-axis.json")
else
  stage2_model="parler-large"
  [[ "$MODEL" == "parler-mini" ]] && stage2_model="parler-mini"
  inputs=(descriptions/descriptions_status_bias.json descriptions/description_career_bias.json descriptions/descriptions_persona_bias.json "data/$stage2_model/stage2/descriptions_two_axis.json" "data/$stage2_model/stage2/descriptions_multi_axis.json")
fi
stages=(stage1 stage1 stage1 stage2 stage2)
axes=(status career persona two-axis three-axis)
output_root="results/extension-ablation/$MODEL/$condition/$ACTION"

for index in "${!inputs[@]}"; do
  common=(--model "$MODEL" --json "${inputs[$index]}" --output "$output_root/${stages[$index]}/${axes[$index]}" --skip-existing --intervention-mode pooled-shift)
  case "$MODEL" in
    parler-mini)
      "$PYTHON_BIN" generate_wav.py "${common[@]}" --model-id "$PARLER_MINI_MODEL_ID" \
        --model-revision "$PARLER_MINI_REVISION" --batch-size "${BATCH_SIZE:-8}" \
        --steering-artifact "$artifact" --intervention-strength "$STRENGTH"
      ;;
    parler-large)
      "$PYTHON_BIN" generate_wav.py "${common[@]}" --model-id "$PARLER_LARGE_MODEL_ID" \
        --model-revision "$PARLER_LARGE_REVISION" --batch-size "${BATCH_SIZE:-8}" \
        --steering-artifact "$artifact" --intervention-strength "$STRENGTH"
      ;;
    voxinstruct)
      "$PYTHON_BIN" generate_wav.py "${common[@]}" --model-id "$VOXINSTRUCT_MODEL_ID" \
        --backend-path "$VOXINSTRUCT_BACKEND_PATH" \
        --voxinstruct-ar-steering-artifact "$ar_artifact" \
        --voxinstruct-nar-steering-artifact "$nar_artifact" \
        --voxinstruct-ar-strength "$AR_STRENGTH" --voxinstruct-nar-strength "$NAR_STRENGTH"
      ;;
  esac
done

echo "[INFO] Complete: $output_root"
