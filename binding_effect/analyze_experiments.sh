#!/bin/bash
set -e

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "[INFO] Model-specific Stage 2 prompts must be built from Stage 1 classifier results."
echo "[INFO] Analyzing canonical Stage 1 JSON files under ${DESC_DIR:-${SCRIPT_DIR}/descriptions}."
exec bash "${SCRIPT_DIR}/analyze_all_models.sh"
