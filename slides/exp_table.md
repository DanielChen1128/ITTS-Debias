# ICASSP Experiment Completion Matrix

This file separates traceable results from placeholders used in `main.tex`.
Pending or historical values must not become paper claims without matching
artifacts, generation manifests, and analysis outputs.

## Current Evidence

| Model / condition | 13,300 WAV | Gender table | 64 interactions | 500-pair quality | Paper status |
|---|:---:|:---:|:---:|:---:|---|
| Parler-Large Original / 2x | Yes | Yes | Yes | Yes | Main completed result |
| Parler-Mini Original / RLACE-8 | Yes | Yes | Yes | Yes | Completed ablation |
| Parler-Mini Original / 2x | No maintained tree | Slide only | Slide only | Slide only | Historical, rerun required |
| VoxInstruct Original / AR6+NAR2 | Stage 1 only | Stage 1 only | No | No | Preliminary |
| VoxInstruct Original / AR2+NAR2 | No | No | No | No | Pending |
| PromptTTS++ | No code/weights | No | No | No | Blocked |

## Values Already Inserted

| Condition | Global female | Mean absolute interaction | UTMOS delta | WER delta |
|---|---:|---:|---:|---:|
| Parler-Large Original | 74.73% | 3.2581 | reference | reference |
| Parler-Large 2x | 51.37% | 1.3546 | -0.1163 [-0.1916, -0.0397] | +0.0586 [+0.0026, +0.1452] |
| Parler-Mini RLACE-8 | 59.77% | 0.7689 | -0.0128 [-0.0628, +0.0370] | -0.0190 [-0.0369, -0.0013] |
| Parler-Mini 2x (historical only) | 56.26% | 0.7431 | -0.0321 [-0.0777, +0.0142] | -0.0002 [-0.0185, +0.0184] |
| Vox AR6+NAR2 Stage 1 | 58.46% | pending | pending | pending |

## Required Runs Before Submission

1. Regenerate and analyze Parler-Mini strict 2x with full provenance.
2. Complete VoxInstruct Original, AR2+NAR2, and AR6+NAR2 Stage 2.
3. Add Vox AR-only and NAR-only ablations to identify stage contribution.
4. Obtain PromptTTS++ code/weights, verify the conditioning tensor, fit a model-local direction, and run the same protocol.
5. Recompute interaction terms with a neutral intercept as sensitivity analysis.
6. Freeze a development-only steering-strength selection rule.
7. Run explicit-control tests for every model and condition.
8. Run 500-pair quality evaluation and a stratified blinded listening test.
9. Audit the acoustic gender classifier on a stratified sample.

## Source of Truth

- Large: `../binding_effect/data/parler-large/PARLER_LARGE_STEERING_2X_RESULTS.md`
- Mini RLACE: `../binding_effect/data/parler-mini/MINI_RLACE_RANK8_LARGE_V1_RESULTS.md`
- Mini 2x history: `../slides/lab_meeting_2026-08-31_zh-TW.md`
- Vox Stage 1: `../binding_effect/runs/analysis/voxinstruct/`
- Execution pause: `../binding_effect/runs/unified-constant-steering/PAUSED.md`
