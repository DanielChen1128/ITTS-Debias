#!/usr/bin/env bash
# Summarize unified constant-steering orchestration and output counts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
STATE_DIR="runs/unified-constant-steering"

if [[ -f "$STATE_DIR/orchestrator.pid" ]] && kill -0 "$(<"$STATE_DIR/orchestrator.pid")" 2>/dev/null; then
  echo "orchestrator: running (PID $(<"$STATE_DIR/orchestrator.pid"))"
else
  echo "orchestrator: stopped"
fi

for step in \
  vox_ar6_generated vox_2x_started mini_2x_generated mini_2x_analyzed \
  vox_2x_generated vox_ar6_analyzed vox_2x_analyzed large_2x_analyzed complete; do
  if [[ -f "$STATE_DIR/$step.done" ]]; then
    echo "$step: done ($(cat "$STATE_DIR/$step.done"))"
  else
    echo "$step: pending"
  fi
done

python - <<'PY'
from pathlib import Path

for model, condition in (
    ("parler-mini", "constant-steering-2x"),
    ("parler-large", "constant-steering-2x"),
    ("voxinstruct", "ar6-nar2"),
    ("voxinstruct", "constant-steering-2x"),
):
    root = Path("results") / model / condition
    print(f"{model}/{condition}: {len(list(root.glob('*/*/*.wav'))):,}/13,300 WAV")
PY
