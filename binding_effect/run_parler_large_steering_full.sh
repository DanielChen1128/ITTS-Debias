#!/usr/bin/env bash
# Generate one disjoint stage of the frozen Parler Large 2x steering run.
set -euo pipefail

ROOT="$(dirname "$0")"
cd "$ROOT"

STAGE="${STAGE:?Set STAGE to stage1 or stage2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="parler-tts/parler-tts-large-v1"
REVISION="50cb4b874c83902f930d7c2e753224c15654f11e"
PROMPTS="data/parler-large/stage2"
ARTIFACT="data/parler-large/artifacts/constant-steering-female-to-male.pt"
OUTPUT="results/parler-large/constant-steering-2x"

case "$STAGE" in
  stage1)
    inputs=(
      "descriptions_status_bias.json"
      "description_career_bias.json"
      "descriptions_persona_bias.json"
    )
    names=("status" "career" "persona")
    ;;
  stage2)
    inputs=("descriptions_two_axis.json" "descriptions_multi_axis.json")
    names=("two-axis" "three-axis")
    ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    exit 2
    ;;
esac

for index in "${!inputs[@]}"; do
  "$PYTHON_BIN" generate_wav.py \
    --model parler-large \
    --model-id "$MODEL_ID" \
    --model-revision "$REVISION" \
    --json "$PROMPTS/${inputs[$index]}" \
    --output "$OUTPUT/$STAGE/${names[$index]}" \
    --batch-size 8 \
    --steering-artifact "$ARTIFACT" \
    --intervention-strength 2 \
    --intervention-mode pooled-shift \
    --skip-existing
done
