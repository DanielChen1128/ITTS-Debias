#!/usr/bin/env bash
# Generate one repaired VoxInstruct condition over the frozen 13,300 prompts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${VOXINSTRUCT_MODEL_ID:-models/voxinstruct/pretrained}"
BACKEND_PATH="${VOXINSTRUCT_BACKEND_PATH:-backends/voxinstruct}"
CONDITION="${CONDITION:?Set CONDITION to original, constant-steering-2x, or ar6-nar2}"
PROMPTS="data/parler-large/stage2"
AR_ARTIFACT="data/voxinstruct/artifacts/voxinstruct-ar-constant-steering-female-to-male-v2.pt"
NAR_ARTIFACT="data/voxinstruct/artifacts/voxinstruct-nar-constant-steering-female-to-male-v2.pt"
AR_SHA256="262d8f2757c99114c725821decec645267004944d58c180bad3d1f01faf42325"
NAR_SHA256="03aa25d0c9eef5fd2d71b03996dc805b3dcee03298e712a175aed3c73ea71932"
OUTPUT="results/voxinstruct"

if [[ "$CONDITION" != "original" && "$CONDITION" != "constant-steering-2x" && "$CONDITION" != "ar6-nar2" ]]; then
  echo "[ERROR] unsupported CONDITION: $CONDITION" >&2
  exit 2
fi

steering_args=()
if [[ "$CONDITION" != "original" ]]; then
  if [[ "$(sha256sum "$AR_ARTIFACT" | cut -d' ' -f1)" != "$AR_SHA256" ]]; then
    echo "[ERROR] AR steering artifact hash mismatch" >&2
    exit 1
  fi
  if [[ "$(sha256sum "$NAR_ARTIFACT" | cut -d' ' -f1)" != "$NAR_SHA256" ]]; then
    echo "[ERROR] NAR steering artifact hash mismatch" >&2
    exit 1
  fi
  ar_strength=2
  if [[ "$CONDITION" == "ar6-nar2" ]]; then
    ar_strength=6
  fi
  steering_args=(
    --voxinstruct-ar-steering-artifact "$AR_ARTIFACT"
    --voxinstruct-nar-steering-artifact "$NAR_ARTIFACT"
    --voxinstruct-ar-strength "$ar_strength"
    --voxinstruct-nar-strength 2
    --intervention-mode pooled-shift
  )
fi

inputs=(
  descriptions/descriptions_status_bias.json
  descriptions/description_career_bias.json
  descriptions/descriptions_persona_bias.json
  "$PROMPTS/descriptions_two_axis.json"
  "$PROMPTS/descriptions_multi_axis.json"
)
stages=(stage1 stage1 stage1 stage2 stage2)
axes=(status career persona two-axis three-axis)

for index in "${!inputs[@]}"; do
  "$PYTHON_BIN" generate_wav.py \
    --model voxinstruct \
    --model-id "$MODEL_ID" \
    --backend-path "$BACKEND_PATH" \
    --json "${inputs[$index]}" \
    --output "$OUTPUT/$CONDITION/${stages[$index]}/${axes[$index]}" \
    --skip-existing \
    "${steering_args[@]}"
done

echo "[INFO] VoxInstruct v2 $CONDITION generation complete: 13,300 prompts"
