#!/usr/bin/env bash
# Complete the unified three-model constant-steering study across both GPU hosts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

REMOTE_HOST="${REMOTE_HOST:-RTX5090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/r13942135/workspace/projects/ITTS_Debias/binding_effect}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/r13942135/workspace/miniconda3_new/envs/voxinstruct/bin/python}"
LOCAL_PYTHON="${LOCAL_PYTHON:-/home/md531/anaconda3/bin/python}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-/home/md531/anaconda3/envs/gender-classifier/bin/python}"
QUALITY_PYTHON="${QUALITY_PYTHON:-/home/md531/anaconda3/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STATE_DIR="runs/unified-constant-steering"
LOCK_DIR="$STATE_DIR/orchestrator.lock"
LOG_DIR="$STATE_DIR/logs"
mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[ERROR] unified constant-steering orchestrator is already active" >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
printf '%s\n' "$$" > "$STATE_DIR/orchestrator.pid"

mark_done() {
  local step="$1"
  date -Is > "$STATE_DIR/$step.done"
}

is_done() {
  [[ -f "$STATE_DIR/$1.done" ]]
}

wait_for_local_command() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    echo "[INFO] Waiting for local job: $pattern"
    sleep "$POLL_SECONDS"
  done
}

wait_for_remote_command() {
  local pattern="$1"
  while ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
    "pgrep -f '$pattern' >/dev/null" 2>/dev/null; do
    echo "[INFO] Waiting for $REMOTE_HOST job: $pattern"
    sleep "$POLL_SECONDS"
  done
}

remote_tree_complete() {
  local condition="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
    "$REMOTE_PYTHON -c \"import json,pathlib; r=pathlib.Path('$REMOTE_ROOT/results/voxinstruct/$condition'); specs=[('stage1/status',200),('stage1/career',2700),('stage1/persona',4000),('stage2/two-axis',3200),('stage2/three-axis',3200)]; ok=all((p:=r/rel).joinpath('generation_manifest.json').is_file() and json.loads((p/'generation_manifest.json').read_text()).get('status')=='complete' and len(list(p.glob('*.wav')))==n for rel,n in specs); raise SystemExit(0 if ok else 1)\"" \
    >/dev/null 2>&1
}

wait_for_remote_tree() {
  local condition="$1"
  until remote_tree_complete "$condition"; do
    echo "[INFO] Waiting for complete remote Vox tree: $condition"
    sleep "$POLL_SECONDS"
  done
}

sync_remote_condition() {
  local condition="$1"
  mkdir -p "results/voxinstruct/$condition"
  rsync -a --partial \
    "$REMOTE_HOST:$REMOTE_ROOT/results/voxinstruct/$condition/" \
    "results/voxinstruct/$condition/"
}

if ! is_done vox_ar6_generated; then
  wait_for_remote_tree ar6-nar2
  sync_remote_condition ar6-nar2
  mark_done vox_ar6_generated
fi

if ! is_done vox_2x_started; then
  wait_for_remote_command "run_voxinstruct_v2_full.sh"
  rsync -a run_voxinstruct_v2_full.sh generate_wav.py \
    "$REMOTE_HOST:$REMOTE_ROOT/"
  rsync -a debias/ "$REMOTE_HOST:$REMOTE_ROOT/debias/"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_ROOT/$LOG_DIR' && cd '$REMOTE_ROOT' && nohup env PYTHONUNBUFFERED=1 CONDITION=constant-steering-2x PYTHON_BIN='$REMOTE_PYTHON' bash run_voxinstruct_v2_full.sh > '$LOG_DIR/voxinstruct-2x-5090.log' 2>&1 < /dev/null &"
  mark_done vox_2x_started
fi

if ! is_done mini_2x_generated; then
  wait_for_local_command "run_voxinstruct_v2_full.sh"
  wait_for_local_command "watch_voxinstruct_v2_gender.sh"
  env STAGE=fit PYTHON_BIN="$LOCAL_PYTHON" bash run_parler_mini_steering_full.sh \
    > "$LOG_DIR/mini-2x-fit.log" 2>&1
  env STAGE=stage1 PYTHON_BIN="$LOCAL_PYTHON" bash run_parler_mini_steering_full.sh \
    > "$LOG_DIR/mini-2x-stage1.log" 2>&1
  env STAGE=stage2 PYTHON_BIN="$LOCAL_PYTHON" bash run_parler_mini_steering_full.sh \
    > "$LOG_DIR/mini-2x-stage2.log" 2>&1
  mark_done mini_2x_generated
fi

if ! is_done mini_2x_analyzed; then
  env MODEL=parler-mini CONDITION=constant-steering-2x \
    PYTHON_BIN="$LOCAL_PYTHON" GENDER_PYTHON="$ANALYSIS_PYTHON" QUALITY_PYTHON="$QUALITY_PYTHON" \
    bash analyze_constant_steering_full.sh > "$LOG_DIR/mini-2x-analysis.log" 2>&1
  mark_done mini_2x_analyzed
fi

if ! is_done vox_2x_generated; then
  wait_for_remote_tree constant-steering-2x
  sync_remote_condition constant-steering-2x
  mark_done vox_2x_generated
fi

if ! is_done vox_ar6_analyzed; then
  env MODEL=voxinstruct CONDITION=ar6-nar2 \
    PYTHON_BIN="$LOCAL_PYTHON" GENDER_PYTHON="$ANALYSIS_PYTHON" QUALITY_PYTHON="$QUALITY_PYTHON" \
    bash analyze_constant_steering_full.sh > "$LOG_DIR/voxinstruct-ar6-nar2-analysis.log" 2>&1
  mark_done vox_ar6_analyzed
fi

if ! is_done vox_2x_analyzed; then
  env MODEL=voxinstruct CONDITION=constant-steering-2x \
    PYTHON_BIN="$LOCAL_PYTHON" GENDER_PYTHON="$ANALYSIS_PYTHON" QUALITY_PYTHON="$QUALITY_PYTHON" \
    bash analyze_constant_steering_full.sh > "$LOG_DIR/voxinstruct-2x-analysis.log" 2>&1
  mark_done vox_2x_analyzed
fi

if ! is_done large_2x_analyzed; then
  env MODEL=parler-large CONDITION=constant-steering-2x \
    PYTHON_BIN="$LOCAL_PYTHON" GENDER_PYTHON="$ANALYSIS_PYTHON" QUALITY_PYTHON="$QUALITY_PYTHON" \
    bash analyze_constant_steering_full.sh > "$LOG_DIR/parler-large-2x-analysis.log" 2>&1
  mark_done large_2x_analyzed
fi

"$LOCAL_PYTHON" summarize_unified_constant_steering.py \
  --output "$STATE_DIR/summary.json"

mark_done complete
echo "[INFO] Unified constant-steering study complete"
