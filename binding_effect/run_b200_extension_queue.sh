#!/usr/bin/env bash
# Run the fixed extension screening or pair-scaling queue on one B200 GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PHASE="${1:?Usage: bash run_b200_extension_queue.sh screen|scaling-full}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="runs/b200-extension/logs"
mkdir -p "$LOG_DIR"

run_condition() {
  local model="$1" action="$2" pairs="$3" first="$4" second="${5:-}"
  local name
  if [[ "$model" == "voxinstruct" ]]; then
    name="$model-pairs-$pairs-ar-$first-nar-$second-$action"
    env MODEL="$model" ACTION="$action" PAIRS="$pairs" AR_STRENGTH="$first" NAR_STRENGTH="$second" \
      PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh > "$LOG_DIR/$name.log" 2>&1
  else
    name="$model-pairs-$pairs-strength-$first-$action"
    env MODEL="$model" ACTION="$action" PAIRS="$pairs" STRENGTH="$first" \
      PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh > "$LOG_DIR/$name.log" 2>&1
  fi
  date -Is > "$LOG_DIR/$name.done"
  echo "[DONE] $name"
}

case "$PHASE" in
  screen)
    for model in parler-mini parler-large voxinstruct; do
      "$PYTHON_BIN" build_extension_screen.py --model "$model"
    done
    for strength in 0.5 1 1.5 2 2.5; do run_condition parler-mini screen 270 "$strength"; done
    for pairs in 300 500 1000 1500; do run_condition parler-large screen "$pairs" 2; done
    for strength in 0.5 1 1.5 2.5 3; do run_condition parler-large screen 1500 "$strength"; done
    for pairs in 300 500 1000 1500; do run_condition voxinstruct screen "$pairs" 2 2; done
    for ar in 2 4 6; do
      for nar in 1 2 3; do
        [[ "$ar/$nar" == "2/2" ]] || run_condition voxinstruct screen 1500 "$ar" "$nar"
      done
    done
    ;;
  scaling-full)
    run_condition parler-mini full 270 2
    for pairs in 300 500 1000 1500; do run_condition parler-large full "$pairs" 2; done
    for pairs in 300 500 1000 1500; do run_condition voxinstruct full "$pairs" 2 2; done
    ;;
  *) echo "[ERROR] phase must be screen or scaling-full" >&2; exit 2 ;;
esac
