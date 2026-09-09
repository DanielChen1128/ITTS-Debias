# B200 Extension Experiments

This plan extends the ICASSP study on a machine that already has Parler-TTS
Mini/Large and VoxInstruct. PromptTTS++ is intentionally excluded because its
backend and pretrained weights are not installed on that machine.

## Design

The study is staged to avoid a full factorial over 13,300 prompts.

| Phase | Conditions | Outputs per condition | Total WAVs | Purpose |
|---|---:|---:|---:|---|
| Screen | 35 | 500 | 17,500 | Select useful strengths and reject saturation |
| Pair scaling | 18 | 13,300 | 239,400 | Main nested-pair ablation |
| Selected full | at most 6 | 13,300 | at most 79,800 | Confirm up to two screened settings per model |

The fixed screening matrix is:

| Model | Pair counts | Hyperparameters | Conditions |
|---|---|---|---:|
| Parler Mini scaling | `300, 500, 1000, 1500, 2000, 2500` | strength `2` | 6 |
| Parler Mini strength | 2500 | `.5, 1, 1.5, 2, 2.5` | 5 total, 4 additional |
| Parler Large scaling | `300, 500, 1000, 1500, 2000, 2500` | strength `2` | 6 |
| Parler Large strength | 2500 | `.5, 1, 1.5, 2, 2.5, 3` | 6 total, 5 additional |
| VoxInstruct scaling | `300, 500, 1000, 1500, 2000, 2500` | AR/NAR `2/2` | 6 |
| VoxInstruct grid | 2500 | AR `2,4,6` x NAR `1,2,3` | 9 total, 8 additional |

This is 35 unique screening conditions. Each model's 500-prompt screen contains
100 deterministically selected prompts from status, career, persona, two-axis,
and three-axis strata. Mini retains its model-specific Stage-2 prompt pool;
Large and Vox use the Large Stage-2 protocol, matching the existing evaluation.
The screen must only be used for candidate selection; final numbers come from
the frozen 13,300-prompt benchmark.

The pair-scaling subsets are deterministic and nested. A 300-pair subset is
contained in the 500-, 1000-, 1500-, 2000-, and 2500-pair subsets. All three
models use the same 2,520-pair description pool, encoded independently through
each model's frozen conditioning path. This replaces Mini's earlier 270-pair
pool for the extension study. The 2500-pair setting is a near-replication of the
successful historical Large 2520-pair setting, but the two must remain labeled
separately because 20 pairs are excluded by the fixed nested sampling order.

## Clone And Environment

```bash
git clone https://github.com/DanielChen1128/ITTS-Debias.git
cd ITTS-Debias/binding_effect
```

Use the existing environment that can already run the three installed models:

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export VOXINSTRUCT_BACKEND_PATH=/path/to/VoxInstruct
export VOXINSTRUCT_MODEL_ID=/path/to/voxinstruct/assets
export HF_HOME=/path/to/shared/huggingface/cache
```

Parler defaults to the immutable Hugging Face revisions used in the paper.
Override `PARLER_MINI_MODEL_ID`, `PARLER_LARGE_MODEL_ID`, or their revision
variables only if the local installation requires an explicit path. The
VoxInstruct model directory must contain `voxinstruct-sft-checkpoint/` and
`google-mt5-base-checkpoint/`.

Quick preflight:

```bash
$PYTHON_BIN -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())"
$PYTHON_BIN -c "import parler_tts, transformers; print('Parler OK')"
test -f "$VOXINSTRUCT_BACKEND_PATH/configs/train_ar.yaml"
test -f "$VOXINSTRUCT_MODEL_ID/voxinstruct-sft-checkpoint/ar_1800k.pyt"
```

No PromptTTS++ installation is required for this queue.

## Schedule

Run all commands from `binding_effect/`. Every generator uses `--skip-existing`,
so an interrupted condition can be restarted with the same command.

### Day 0: Fit and smoke test

Fit one artifact for each backend, then generate one 500-prompt condition per
model. Fitting first extracts the full aligned activation cache; later pair
counts reuse it and are cheap.

```bash
MODEL=parler-mini ACTION=fit PAIRS=2500 STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
MODEL=parler-large ACTION=fit PAIRS=2500 STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
MODEL=voxinstruct ACTION=fit PAIRS=2500 AR_STRENGTH=2 NAR_STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh

MODEL=parler-mini ACTION=screen PAIRS=2500 STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
MODEL=parler-large ACTION=screen PAIRS=2500 STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
MODEL=voxinstruct ACTION=screen PAIRS=2500 AR_STRENGTH=2 NAR_STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
```

Check the three logs/manifests before starting the queue. Do not launch multiple
VoxInstruct processes on one GPU; a single process already loads AR and NAR.

### Day 1: Screening queue

```bash
nohup env PYTHON_BIN="$PYTHON_BIN" \
  VOXINSTRUCT_BACKEND_PATH="$VOXINSTRUCT_BACKEND_PATH" \
  VOXINSTRUCT_MODEL_ID="$VOXINSTRUCT_MODEL_ID" \
  bash run_b200_extension_queue.sh screen \
  > runs/b200-extension/screen-queue.log 2>&1 < /dev/null &
```

The queue runs sequentially on one GPU. The `.done` files under
`runs/b200-extension/logs/` identify completed conditions. Generated audio is
under `results/extension-ablation/`.

After screening, classify the 17,500 WAVs and rank settings by:

1. global calibration error `abs(female_rate - 0.5)`;
2. mean and worst-stratum calibration error;
3. rejection of all-one-class saturation;
4. paired UTMOS/WER on the best settings if compute permits.

Select no more than two non-scaling hyperparameter settings per model for full
evaluation. Selection rules and all screened values must be retained, including
failed conditions.

The existing `analyze_gender.py` can write each condition's
`detection_results.csv`. Mirror the generated directory under an analysis root,
then rank completed conditions with:

```bash
$PYTHON_BIN summarize_extension_screen.py \
  --analysis-root runs/analysis/extension-ablation/parler-large \
  --output runs/analysis/extension-ablation/parler-large/screen-ranking.json
```

Use the same command for `parler-mini` and `voxinstruct`. The summarizer expects
the layout `<condition>/screen/<stage>/<stratum>/detection_results.csv` and
reports global, mean-stratum, and worst-stratum calibration errors. Gender
classification can run on the B200 if its classifier environment is installed,
or after the WAV trees are copied back to the current analysis machine.

### Day 2 onward: Full nested-pair scaling

```bash
nohup env PYTHON_BIN="$PYTHON_BIN" \
  VOXINSTRUCT_BACKEND_PATH="$VOXINSTRUCT_BACKEND_PATH" \
  VOXINSTRUCT_MODEL_ID="$VOXINSTRUCT_MODEL_ID" \
  bash run_b200_extension_queue.sh scaling-full \
  > runs/b200-extension/scaling-full-queue.log 2>&1 < /dev/null &
```

Then run each selected strength/grid condition explicitly with `ACTION=full`.
For example:

```bash
MODEL=parler-large ACTION=full PAIRS=2500 STRENGTH=1.5 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
MODEL=voxinstruct ACTION=full PAIRS=2500 AR_STRENGTH=4 NAR_STRENGTH=2 PYTHON_BIN="$PYTHON_BIN" bash run_b200_extension_condition.sh
```

Do not choose these example values before inspecting the screen.

## Runtime Estimate

Hardware, codec decoding, and output duration dominate runtime, so the first
500-prompt smoke run is the timing calibration. If that run takes `T` hours:

- complete screen: approximately `35T` GPU-hours;
- 18-condition pair scaling: approximately `18 x 26.6T = 478.8T` GPU-hours;
- each selected full condition: approximately `26.6T` GPU-hours.

For example, if 500 prompts take 20 minutes, screening is about 11.7 GPU-hours,
pair scaling about 159.6 GPU-hours, and each selected full condition about 8.9
GPU-hours. These are throughput estimates, not guaranteed wall-clock times.
Use the measured B200 smoke time in place of the example.

## Outputs To Return

Retain the following paths when copying results back:

```text
binding_effect/results/extension-ablation/
binding_effect/data/extension-ablation/
binding_effect/data/extension-screen-v1/
binding_effect/runs/b200-extension/
```

The `.pt`, WAV, log, and `runs/` contents are intentionally ignored by Git.
Use `rsync`, shared storage, or an archive for results; do not commit generated
audio or model artifacts.

## Paper Tables

The main pair-count figure should plot pair count against global calibration
error, mean stratum error, worst-stratum error, and mean interaction magnitude.
Plot all three models on the common six-point pair-count axis. Show the existing
Large 2520-pair result as a separately marked historical reference rather than
merging it with the new 2500-pair condition. Put all 35 screening settings in
supplementary material and only
the selected full settings in the main cross-model table. Utility evaluation
remains paired against Original with the existing 500-prompt holdout and NI
margins.
