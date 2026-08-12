#!/bin/bash

# Script to analyze gender for all TTS models
# Usage: bash analyze_all_models.sh

set -e  # Exit on error

# Paths can be overridden via environment variables; defaults resolve to this repo.
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DESC_DIR="${DESC_DIR:-${SCRIPT_DIR}/descriptions}"
WAV_BASE="${WAV_BASE:-./ITTS_audios}"
ANALYSIS_BASE="${ANALYSIS_BASE:-${SCRIPT_DIR}/analysis}"
FAILURES=0

# Define all models
MODELS=("parler-large" "parler-mini" "promptttspp" "voxinstruct")
# Find the canonical Stage 1 JSON files in descriptions directory
JSON_FILES=$(find "${DESC_DIR}" -maxdepth 1 -name "*.json" -type f)

if [ -z "$JSON_FILES" ]; then
    echo "[ERROR] No JSON files found in ${DESC_DIR}"
    exit 1
fi

echo "=========================================="
echo "Starting Gender Analysis for all models"
echo "=========================================="
echo ""

# Count total files
TOTAL_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found ${TOTAL_FILES} description file(s)"
echo ""

# Analyze each model and JSON combination
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Analyzing Model: ${model}"
    echo "=========================================="
    echo ""

    for json_file in $JSON_FILES; do
        json_basename=$(basename "$json_file" .json)
        wav_path="${WAV_BASE}/${model}/${json_basename}"
        output_path="${ANALYSIS_BASE}/${model}/${json_basename}"

        # Check if WAV directory exists
        if [ ! -d "${wav_path}" ]; then
            echo "[WARNING] WAV directory not found: ${wav_path}"
            echo "          Skipping..."
            FAILURES=$((FAILURES + 1))
            echo ""
            continue
        fi

        # Check if there are WAV files
        wav_count=$(find "${wav_path}" -name "*.wav" -type f | wc -l)
        if [ ${wav_count} -eq 0 ]; then
            echo "[WARNING] No WAV files found in: ${wav_path}"
            echo "          Skipping..."
            FAILURES=$((FAILURES + 1))
            echo ""
            continue
        fi

        echo "------------------------------------------"
        echo "Model: ${model} | Dataset: ${json_basename}"
        echo "WAV Path: ${wav_path}"
        echo "WAV Count: ${wav_count}"
        echo "------------------------------------------"

        # Run gender analysis
        if ! conda run -n BindingBias python "${SCRIPT_DIR}/analyze_gender.py" \
            --wav_path "${wav_path}" \
            --json "${json_file}" \
            --model-name "${model}" \
            --output "${output_path}"; then
            echo "[ERROR] Gender analysis failed for ${model}/${json_basename}"
            FAILURES=$((FAILURES + 1))
        fi

        echo ""
    done

    echo ""
done

echo "=========================================="
echo "All gender analysis completed!"
echo "=========================================="
echo "Results saved to: ${ANALYSIS_BASE}"
echo ""
echo "To view results:"
echo "  - Overall stats: ${ANALYSIS_BASE}/<model>/<dataset>/overall_gender_distribution.csv"
echo "  - By trait: ${ANALYSIS_BASE}/<model>/<dataset>/gender_by_trait.csv"
echo "  - By keyword: ${ANALYSIS_BASE}/<model>/<dataset>/gender_by_keyword.csv"
echo "  - Full results: ${ANALYSIS_BASE}/<model>/<dataset>/detection_results.csv"
echo ""
if [ "${FAILURES}" -ne 0 ]; then
    echo "[ERROR] ${FAILURES} model/dataset analysis run(s) were incomplete"
    exit 1
fi
