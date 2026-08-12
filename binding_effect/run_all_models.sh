#!/bin/bash

# Script to run all TTS models on all description files
# Usage: bash run_all_models.sh

set -e  # Exit on error

# Paths can be overridden via environment variables; defaults resolve to this repo.
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DESC_DIR="${DESC_DIR:-${SCRIPT_DIR}/descriptions}"
OUTPUT_BASE="${OUTPUT_BASE:-./ITTS_audios}"
CONFIG_PATH="${CONFIG_PATH:-}"
CONFIG_ARGS=()
if [ -n "${CONFIG_PATH}" ]; then
    CONFIG_ARGS=(--config "${CONFIG_PATH}")
fi
FAILURES=0

# Find the canonical Stage 1 JSON files in descriptions directory
JSON_FILES=$(find "${DESC_DIR}" -maxdepth 1 -name "*.json" -type f)

if [ -z "$JSON_FILES" ]; then
    echo "[ERROR] No JSON files found in ${DESC_DIR}"
    exit 1
fi

echo "=========================================="
echo "Starting TTS generation for all models"
echo "=========================================="
echo ""

# Count total files
TOTAL_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found ${TOTAL_FILES} description file(s)"
echo ""

# Run models with BindingBias environment
MODELS_COP=("parler-large" "parler-mini" "promptttspp")


for json_file in $JSON_FILES; do
    json_basename=$(basename "$json_file" .json)
    echo "=========================================="
    echo "Processing: ${json_basename}"
    echo "=========================================="
    echo ""

    # Run BindingBias models
    for model in "${MODELS_COP[@]}"; do
        echo "------------------------------------------"
        echo "Model: ${model} | File: ${json_basename}"
        echo "------------------------------------------"

        if ! conda run -n BindingBias python "${SCRIPT_DIR}/generate_wav.py" \
            --model "${model}" \
            --json "${json_file}" \
            --output "${OUTPUT_BASE}/${model}/${json_basename}" \
            "${CONFIG_ARGS[@]}"; then
            echo "[ERROR] ${model} failed on ${json_basename}"
            FAILURES=$((FAILURES + 1))
        fi

        echo ""
    done

    # Run voxinstruct with voxinstruct environment
    echo "------------------------------------------"
    echo "Model: voxinstruct | File: ${json_basename}"
    echo "------------------------------------------"

    if ! conda run -n voxinstruct python "${SCRIPT_DIR}/generate_wav.py" \
        --model voxinstruct \
        --json "${json_file}" \
        --output "${OUTPUT_BASE}/voxinstruct/${json_basename}" \
        "${CONFIG_ARGS[@]}"; then
        echo "[ERROR] voxinstruct failed on ${json_basename}"
        FAILURES=$((FAILURES + 1))
    fi

    echo ""
done

echo "=========================================="
echo "All models completed!"
echo "=========================================="
echo "Output directory: ${OUTPUT_BASE}"
if [ "${FAILURES}" -ne 0 ]; then
    echo "[ERROR] ${FAILURES} model/dataset run(s) failed"
    exit 1
fi
