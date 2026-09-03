#!/usr/bin/env bash
# Fit and generate one stage of the frozen Parler Mini 2x steering run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

STAGE="${STAGE:?Set STAGE to fit, stage1, or stage2}"
PYTHON_BIN="${PYTHON_BIN:-/home/md531/anaconda3/bin/python}"
MODEL_ID="parler-tts/parler-tts-mini-v1"
REVISION="0392b9451a601e528fd863bbb0598431fee810d9"
ANCHORS="data/parler-mini/training_anchors.json"
CACHE="data/parler-mini/artifacts/parler-mini-training-activations.pt"
ARTIFACT="data/parler-mini/artifacts/constant-steering-female-to-male.pt"
PROMPTS="data/parler-mini/stage2"
OUTPUT="results/parler-mini/constant-steering-2x"

fit_artifact() {
  if [[ ! -f "$CACHE" ]]; then
    "$PYTHON_BIN" fit_rlace.py \
      --json "$ANCHORS" \
      --model-id "$MODEL_ID" \
      --model-revision "$REVISION" \
      --activations-cache "$CACHE" \
      --extract-only
  fi
  if [[ ! -f "$ARTIFACT" ]]; then
    "$PYTHON_BIN" fit_constant_steering.py \
      --anchors "$ANCHORS" \
      --activations-cache "$CACHE" \
      --output "$ARTIFACT"
  fi
  "$PYTHON_BIN" - "$ARTIFACT" "$ANCHORS" "$MODEL_ID" "$REVISION" <<'PY'
import hashlib
import sys
import torch

artifact, anchors, model_id, revision = sys.argv[1:]
state = torch.load(artifact, map_location="cpu", weights_only=True)
metadata = state.get("metadata", {})
expected_anchor = hashlib.sha256(open(anchors, "rb").read()).hexdigest()
if state.get("format") != "constant-steering-v1":
    raise SystemExit("[ERROR] unsupported Mini steering artifact")
if metadata.get("anchor_sha256") != expected_anchor:
    raise SystemExit("[ERROR] Mini steering anchor hash mismatch")
if metadata.get("model_id") != model_id or metadata.get("model_revision") != revision:
    raise SystemExit("[ERROR] Mini steering model provenance mismatch")
print(f"[OK] Mini steering artifact: {hashlib.sha256(open(artifact, 'rb').read()).hexdigest()}")
PY
}

fit_artifact
if [[ "$STAGE" == "fit" ]]; then
  exit 0
fi

case "$STAGE" in
  stage1)
    inputs=(
      "descriptions/descriptions_status_bias.json"
      "descriptions/description_career_bias.json"
      "descriptions/descriptions_persona_bias.json"
    )
    names=(status career persona)
    ;;
  stage2)
    inputs=("$PROMPTS/descriptions_two_axis.json" "$PROMPTS/descriptions_multi_axis.json")
    names=(two-axis three-axis)
    ;;
  *)
    echo "[ERROR] unknown STAGE: $STAGE" >&2
    exit 2
    ;;
esac

for index in "${!inputs[@]}"; do
  "$PYTHON_BIN" generate_wav.py \
    --model parler-mini \
    --model-id "$MODEL_ID" \
    --model-revision "$REVISION" \
    --json "${inputs[$index]}" \
    --output "$OUTPUT/$STAGE/${names[$index]}" \
    --batch-size 8 \
    --steering-artifact "$ARTIFACT" \
    --intervention-strength 2 \
    --intervention-mode pooled-shift \
    --skip-existing
done

echo "[INFO] Parler Mini 2x steering $STAGE generation complete"
