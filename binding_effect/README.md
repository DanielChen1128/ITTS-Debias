# The Binding Effect: Debiasing Instruction TTS

Code and prompt sets for measuring and reducing multi-dimensional gender bias
in instruction TTS. The maintained debiasing configurations are:

- Parler Mini: rank-8 RLACE at revision
  `0392b9451a601e528fd863bbb0598431fee810d9`.
- Parler Large: `2x` female-to-male constant steering at revision
  `50cb4b874c83902f930d7c2e753224c15654f11e`.
- VoxInstruct: model-local AR/NAR female-to-male constant centroid steering,
  currently evaluated at AR strength `6` and NAR strength `2` after restoring
  upstream classifier-free guidance and NAR initialization behavior.

Both configurations use the frozen 13,300-prompt protocol: 6,900 Stage 1
univariate prompts and 6,400 Stage 2 compositional prompts. Explicit gender
requests use regex bypass. Acoustic gender is the higher of the model's female
and male scores; the model's child score is retained only as raw metadata and
does not participate in classification or gender-rate denominators.

## Active Workflows

### Parler Mini RLACE

```bash
bash run_mini_rlace_rank8_large.sh
bash analyze_mini_rlace_rank8_large.sh
```

The runner verifies the artifact SHA-256 before generation and fixes the
checkpoint revision and batch size (`8`). The analyzer classifies all five axes
and runs 10,000-iteration matched interaction diagnostics.

### Parler Large Constant Steering

Run the disjoint stages separately so two machines never write to the same
output directory:

```bash
STAGE=stage1 bash run_parler_large_steering_full.sh
STAGE=stage2 bash run_parler_large_steering_full.sh
bash finish_parler_large_steering_full.sh
```

Stage 1 contains status (200), career (2,700), and persona (4,000). Stage 2
contains two-axis (3,200) and three-axis (3,200), for 13,300 WAVs total. The
finisher verifies every count, classifies the outputs, and runs the matched
interaction analysis.

The completed Large tables and interpretation are in
`data/parler-large/PARLER_LARGE_STEERING_2X_RESULTS.md`.
The common Mini/Large protocol and side-by-side results are in
`data/shared/DEBIAS_MODEL_COMPARISON.md`.

### VoxInstruct Constant Steering

Run Original and the selected candidate on separate GPUs:

```bash
CONDITION=original bash run_voxinstruct_v2_full.sh
CONDITION=ar6-nar2 bash run_voxinstruct_v2_full.sh
bash watch_voxinstruct_v2_gender.sh
```

VoxInstruct uses separate AR and NAR artifacts fit from paired descriptions and
pair-matched neutral transcripts. Formal outputs are isolated under
`results/voxinstruct/`.

### PromptTTS++ Smoke / Batch Runs

PromptTTS++ is wired through the shared generation entry point with the local
backend checkout and pretrained weights:

```bash
python generate_wav.py --model promptttspp --config model_config.json \
  --json descriptions/promptttspp_test.json --output results/promptttspp_test
STAGE=stage1 bash run_promptttspp_full.sh
STAGE=stage2 bash run_promptttspp_full.sh
```

PromptTTS++ uses the same prompt protocol and can reuse the shared analysis
helpers once the stage outputs are complete.

### Unified Constant Steering Queue

Complete Mini and VoxInstruct `2x`, retain the calibrated VoxInstruct
`AR6 + NAR2` condition, and refresh all three model analyses automatically:

```bash
nohup bash run_unified_constant_steering.sh > unified-constant-steering.log 2>&1 &
bash status_unified_constant_steering.sh
```

The queue waits for existing VoxInstruct work, assigns Mini to the local GPU and
VoxInstruct `AR2 + NAR2` to the RTX 5090, verifies manifests and WAV counts, and
then runs matched gender, interaction, and 500-pair quality reports.

### B200 Extension Ablations

The staged pair-count and hyperparameter ablations for Parler Mini/Large and
VoxInstruct are specified in [`B200_EXTENSION_EXPERIMENTS.md`](B200_EXTENSION_EXPERIMENTS.md).
The B200 queue deliberately excludes PromptTTS++ because that host does not have
its backend or weights. Start with the 500-prompt screen before any additional
full 13,300-prompt conditions:

```bash
bash run_b200_extension_queue.sh screen
bash run_b200_extension_queue.sh scaling-full
```

## Evaluation

`analyze_gender.py` writes acoustic gender predictions and scores.
`build_small_context_interaction_specs.py` creates matched condition specs,
`analyze_interactions.py` performs the seeded constrained-null analysis, and
`analyze_small_context_diagnostics.py` summarizes global balance, career gaps,
descriptor groups, and interaction families.

For paired speech-quality evaluation:

```bash
python materialize_quality_subset.py --help
python evaluate_screen_quality.py --help
```

The configured quality gates are UTMOS delta `>= -0.10` and WER delta
`<= +0.03`. The explicit-prompt safety check requires accuracy no more than
five percentage points below Original.

## Artifact Fitting

Fit Mini RLACE from cached or newly extracted pooled encoder states:

```bash
python fit_rlace.py \
  --json data/parler-mini/training_anchors.json \
  --training-manifest data/parler-mini/training_manifest.json \
  --model-revision 0392b9451a601e528fd863bbb0598431fee810d9 \
  --rank 8 \
  --output rlace-rank-8.pt
```

Fit the Large directed centroid offset from a provenance-matched cache:

```bash
python fit_constant_steering.py \
  --anchors ANCHORS.json \
  --activations-cache ACTIVATIONS.pt \
  --output constant-steering-female-to-male.pt
```

Production artifacts are stored under each model's `data/<model>/artifacts/`
directory. Formal WAVs follow the same model-first convention under `results/`.

## Maintained Code

```text
generate_wav.py                          TTS generation and interventions
fit_rlace.py                             Mini RLACE fitting
fit_constant_steering.py                 Large steering fitting
analyze_gender.py                        Acoustic gender classification
build_small_context_interaction_specs.py Matched interaction specifications
analyze_interactions.py                  Interaction significance analysis
analyze_small_context_diagnostics.py     Aggregate diagnostics
evaluate_screen_quality.py               Paired UTMOS and WER evaluation
debias/leace.py                          RLACE-compatible affine artifact
debias/rlace.py                         RLACE optimization
debias/steering.py                       Constant steering artifact
debias/parler.py                         Parler intervention integration
debias/voxinstruct.py                    VoxInstruct intervention integration
```

Superseded pilots, screening WAVs, calibration audio, and archived execution
code are intentionally not retained in this working tree.

## Tests

```bash
python -m unittest discover -v
```

## Citation

```bibtex
@inproceedings{chen2026binding,
  title     = {The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS},
  author    = {Chen, Kuan-Yu and Lin, Yi-Cheng and Hsieh, Po-Chung and Chou, Huang-Cheng and Hsu, Chih-Fan and Li, Jeng-Lin and Lee, Hung-yi and Ding, Jian-Jiun},
  booktitle = {Interspeech},
  year      = {2026},
  note      = {arXiv:2603.20743}
}
```
