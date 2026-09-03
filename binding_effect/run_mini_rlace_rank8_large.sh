#!/usr/bin/env bash
# Generate the frozen 13,300-prompt Mini RLACE rank-8 evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL="parler-mini"
REVISION="0392b9451a601e528fd863bbb0598431fee810d9"
METHOD="rlace-rank-8"
CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV="${CONDA_ENV:-base}"
ARTIFACT="data/parler-mini/artifacts/parler-mini-rlace-rank-8.pt"
ARTIFACT_SHA256="b2f1e366e70a8b5455125bb5f6634bc8aee482375439ade5cc143e132ea5187a"
OUTPUT="results/parler-mini/$METHOD"
STAGE2="data/parler-mini/stage2"

actual_sha256="$(sha256sum "$ARTIFACT" | cut -d' ' -f1)"
if [[ "$actual_sha256" != "$ARTIFACT_SHA256" ]]; then
  echo "[ERROR] RLACE artifact hash mismatch: $actual_sha256" >&2
  exit 1
fi

generate() {
  local input="$1"
  local output="$2"
  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" python generate_wav.py \
    --model "$MODEL" \
    --model-revision "$REVISION" \
    --json "$input" \
    --output "$output" \
    --batch-size 8 \
    --leace-artifact "$ARTIFACT"
}

generate "descriptions/descriptions_status_bias.json" "$OUTPUT/stage1/status"
generate "descriptions/description_career_bias.json" "$OUTPUT/stage1/career"
generate "descriptions/descriptions_persona_bias.json" "$OUTPUT/stage1/persona"
generate "$STAGE2/descriptions_two_axis.json" "$OUTPUT/stage2/two-axis"
generate "$STAGE2/descriptions_multi_axis.json" "$OUTPUT/stage2/three-axis"

echo "[INFO] Mini RLACE rank-8 generation complete: 13,300 prompts"
