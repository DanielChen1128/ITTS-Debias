#!/usr/bin/env bash
# Main experiment generation on the frozen legacy-5900-v1 test input.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/md531/anaconda3/envs/paraspeechcaps/bin/python
export HF_HUB_OFFLINE=1
JSON=datasets/legacy-5900-v1/data/main.json
ROOT=${ROOT:-results/rerun-main}
COMMON="--model parler-mini --json $JSON --skip-existing"

echo "[$(date)] original"
$PY generate_wav.py $COMMON --output $ROOT/original

echo "[$(date)] random"
$PY generate_wav.py $COMMON --leace-artifact artifacts/random.pt --output $ROOT/random

echo "[$(date)] gender_direction"
$PY generate_wav.py $COMMON --leace-artifact artifacts/gender-direction.pt --output $ROOT/gender_direction

echo "[$(date)] leace_bypass"
$PY generate_wav.py $COMMON --leace-artifact artifacts/leace.pt --output $ROOT/leace_bypass

echo "[$(date)] leace_nobypass"
$PY generate_wav.py $COMMON --leace-artifact artifacts/leace.pt --no-bypass --output $ROOT/leace_nobypass

echo "[$(date)] DONE"
