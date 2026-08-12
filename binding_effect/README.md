<div align="center">

# The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Official code and available prompt sets for **"The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS"** (Interspeech 2026).

**Authors:** Kuan-Yu Chen, Yi-Cheng Lin, Po-Chung Hsieh, Huang-Cheng Chou, Chih-Fan Hsu, Jeng-Lin Li, Hung-yi Lee, and Jian-Jiun Ding

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2603.20743-b31b1b.svg)](https://arxiv.org/abs/2603.20743)
[![Interspeech 2026](https://img.shields.io/badge/Interspeech-2026-blue.svg)](https://arxiv.org/abs/2603.20743)

</div>

---

## 📌 Overview

Bias evaluations in Instruction TTS (ITTS) usually test one attribute at a time, although real prompts combine social cues. This project studies how three theoretically grounded axes interact to shape the perceived gender of synthesized speech:

* **Social Status**: Weberian stratification represented with high/low Social Dominance Orientation descriptors.
* **Career**: Socially structured occupational roles with female-, mixed-, or male-leaning priors.
* **Persona**: Big Five traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism.

> [!NOTE]
> **The Binding Effect:**
> The **Binding Effect** is the non-additive interaction of these cues. The paper-primary probability is `female / (female + male + child)`, matching the event `D(y) = Female` over successful classifier outcomes. Unknown/failed outcomes have no `D(y)` result and are excluded but reported. Adult-only `female / (female + male)` is also reported separately.

---

## 💡 Method

The paper uses two stages:
1. **Stage 1** estimates the empirical female probability for each isolated descriptor.
2. **Stage 2** evaluates bi- and tri-dimensional combinations against those baselines.

For female probability `P(x)`, `L(x) = ln(P(x) / (1 - P(x)))`. Pairwise interaction is `I(w1,w2) = L(x_bi) - L(x_uni1) - L(x_uni2)`. Three-way interaction subtracts all univariate and pairwise effects from `L(x_multi)` as in paper equations 5 and 6. At an all-female or all-non-female boundary, the implementation applies the binomial Haldane-Anscombe correction `(female + 0.5) / (non-female + 0.5)`; interior proportions are unchanged.

The paper states 10,000 iterations with random-label shuffling but does not publish the exact algorithm. A pooled shuffle tests equality of condition probabilities, not the required additive-logit null `I = 0`. `analyze_interactions.py` therefore fits independent binomial condition probabilities constrained to `I = 0`, then performs 10,000 seeded random binary-label draws at the original condition sizes. It uses a two-sided statistic and the `(extreme + 1) / (iterations + 1)` Monte Carlo correction. This choice is explicitly a constrained-null randomization approximation to the unavailable paper implementation. Reports retain the paper's moderate (`p < 0.05`, `|I| > 1.0`) and strong (`p < 0.01`, `|I| > 2.8`) categories.

Optional `semantic_bias.py` implements equation 7, the female-minus-male anchor cosine bias `Delta`, and configured group comparisons with Cohen's d. It fails with an installation instruction when `sentence-transformers` is unavailable.

---

## 📊 Repository Status

> [!IMPORTANT]
> **Prompt Availability & Reproduction Gap:**
> The prompt builder, generation, classifier, interaction-statistics, semantic-analysis, configuration preflight, and prompt-audit paths are runnable. Model weights, external TTS implementations, generated audio, paper outputs, and the full human-verified prompt collection are not bundled.
>
> The paper specifies 13,300 prompts per model: 6,900 Stage 1 univariate prompts and 6,400 model-specific Stage 2 compositional prompts. The canonical `descriptions/` directory now contains the complete 6,900-prompt Stage 1 reconstruction. Stage 2 must be built separately for each model after ranking that model's Stage 1 classifier results.
>
> `build_prompts.py` provides a clearly labeled, deterministic **paper-aligned reconstruction**. The 40-persona subset, ten transcripts, ten-template expansion, seeds, and provenance are explicit. The paper does not publish its literal templates, full descriptor list, human-verification records, ordering, or seeds, so structural alignment is reproducible but identity with the original stimuli is not claimed.
>
> The original incomplete Stage 2 files are retained only under `examples/legacy_stage2/`. Batch scripts intentionally process root-level canonical Stage 1 files and do not treat those examples as experimental inputs.

### Debiasing extension

This checkout also contains an independent Parler-TTS Mini study that removes a
linear gender direction from projected description-encoder states. The completed
pilot and main experiments used the earlier 5,900-record repository snapshot.
Those inputs are frozen under `datasets/legacy-5900-v1/`; they are not silently
upgraded to the canonical 6,900 Stage 1 reconstruction. Trackable evaluation
records and hashes for excluded WAV/model/artifact binaries are stored under
`experiment_records/`.

The historical runtime did not record an immutable Parler revision. The local
cache currently resolves to commit `0392b9451a601e528fd863bbb0598431fee810d9`,
but this is observational and is not claimed as the verified historical
checkpoint. The original screening code also evaluated all grounded careers
despite labeling the report `spillover_split=dev`; the frozen report preserves
that fact, while current screening code performs the requested split filter.

---

## 📁 Structure

```text
.
├── generate_wav.py             # Unified generation and model preflight
├── build_prompts.py             # Deterministic two-stage protocol reconstruction
├── analyze_gender.py            # wav2vec 2.0 classifier and descriptive summaries
├── analyze_interactions.py      # Equations 5/6 and permutation significance
├── binding_stats.py            # Dependency-free statistical core
├── semantic_bias.py            # Optional embedding Delta and Cohen's d
├── fit_leace.py                # Fit LEACE and rank-matched comparator artifacts
├── debias/                     # Encoder intervention and evaluation utilities
├── prompt_audit.py             # Prompt schema/count/hash audit
├── prompt_manifest.json        # Audit of the canonical Stage 1 prompts
├── model_config.example.json    # External model/source configuration example
│
├── descriptions/               # Canonical 6,900-prompt Stage 1 reconstruction
├── datasets/legacy-5900-v1/   # Frozen inputs for completed debias experiments
├── experiment_records/        # Versioned metrics and binary hash manifests
├── examples/legacy_stage2/     # Original incomplete Stage 2 examples
├── run_all_models.sh            # Generate every available JSON with all four models
├── analyze_all_models.sh        # Analyze every configured model and available JSON
└── tests/                      # Offline synthetic statistics and schema tests

```

Canonical prompt items have `id`, `description`, `trait`, `keywords`, and gender-neutral `prompt_text`, plus `axis`, `descriptor_id`, `template_id`, `transcript_id`, `cell_id`, `seed`, and `provenance` reconstruction metadata.

---

## 🛠️ Setup

See [`setup.txt`](setup.txt). A minimal analysis setup is:

```bash
conda create -n BindingBias python=3.9 -y
conda activate BindingBias
pip install torch==2.3.0 torchaudio==2.3.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt

```

> [!TIP]
> Heavy TTS imports are lazy, so `--help`, prompt audit, statistics tests, and generation `--check` do not load model code.

---

## 📦 External Assets

* **Parler-TTS:** Defaults to the public IDs `parler-tts/parler-tts-mini-v1` and `parler-tts/parler-tts-large-v1`. Install [Parler-TTS](https://github.com/huggingface/parler-tts); downloads follow Hugging Face cache settings.
* **PromptTTS++ & VoxInstruct:** Require separately obtained official source checkouts and checkpoint assets. They are supported through `--backend-path` plus `--model-id`, environment variables shown by preflight errors, or [`model_config.example.json`](model_config.example.json). No weights are redistributed here.
* **Classifier:** The acoustic classifier is AudEERING's [wav2vec2 age/gender model](https://zenodo.org/record/7761387) and downloads on first real analysis run.

---

## 🚀 Usage

### 1. Validate Canonical Stage 1

The tracked `descriptions/` files contain the paper's `69 × 10 × 10 = 6,900` Stage 1 design:

```bash
python prompt_audit.py \
  --descriptions descriptions \
  --output prompt_manifest.json \
  --strict-stage1
```

This validates 200 status, 2,700 career, and 4,000 persona prompts, including their schema, Cartesian structure, metadata, and canonical content hashes. The full 13,300 count remains intentionally incomplete until a model-specific Stage 2 is built.

### 2. Validate Model Preflight

Validate without loading or downloading a model:

```bash
python generate_wav.py --model parler-mini --check
python generate_wav.py --model promptttspp --config model_config.json --check

```

### 3. Generate and Classify Stage 1

```bash
python generate_wav.py --model parler-mini \
  --model-revision COMMIT_SHA \
  --json descriptions/descriptions_persona_bias.json \
  --output ITTS_audios/parler-mini/stage1-persona
python analyze_gender.py --wav_path ITTS_audios/parler-mini/stage1-persona \
  --json descriptions/descriptions_persona_bias.json \
  --model-name parler-mini \
  --output analysis/parler-mini/stage1-persona

```

Repeat generation and classification for status and career. Outputs preserve all protocol metadata, include `detection_results.csv`, and report trait/keyword summaries. Generation applies each row's deterministic seed and writes `generation_manifest.json`; pin Parler to a commit revision and use immutable local checkpoint directories. Backend and hardware differences can still prevent bitwise-identical audio. Existing valid WAVs are skipped unless `--no-skip` is passed.

### 4. Build Model-Specific Stage 2

Stage 2 selects the two lowest and two highest career and persona female probabilities from that model's Stage 1 results, then builds all 32 bi-axis and 32 tri-axis cells:

```bash
python build_prompts.py stage2 \
  --stage1-dir descriptions \
  --detections \
    analysis/parler-mini/stage1-status/detection_results.csv \
    analysis/parler-mini/stage1-career/detection_results.csv \
    analysis/parler-mini/stage1-persona/detection_results.csv \
  --model-name parler-mini \
  --output-dir reconstructed_prompts/parler-mini
python prompt_audit.py \
  --descriptions reconstructed_prompts/parler-mini \
  --output reconstructed_prompts/parler-mini_manifest.json \
  --strict-reconstruction
```

The complete output contains 13,300 structurally aligned prompts plus `_protocol/stage2_selection.json`. `--strict-paper` still fails by design because no published hashes can establish original content identity.

### 5. Run the encoder intervention

Fit artifacts only from matched explicit-gender anchors. Career stereotype
labels are analysis covariates and are never LEACE fit labels.

```bash
python fit_leace.py \
  --json datasets/legacy-5900-v1/data/anchors_fit.json \
  --output artifacts/leace.pt
python generate_wav.py --model parler-mini \
  --json datasets/legacy-5900-v1/data/main.json \
  --output results/new-main/leace-bypass \
  --leace-artifact artifacts/leace.pt
```

Generation records the input and artifact hashes, deterministic seed strategy,
resolved model configuration, and bypass policy in `generation_manifest.json`.
Use a new output directory: completed legacy WAV directories predate these
manifests and are retained only as local historical outputs.

### 6. Analyze Interactions

For interactions, provide a JSON spec whose order-2 entries contain exactly three conditions in `[joint, uni1, uni2]` order, or whose order-3 entries contain seven in `[triple, pair12, pair13, pair23, uni1, uni2, uni3]` order. Every condition has `name` and a `csv` path to classifier output:

```bash
python analyze_interactions.py --spec interactions.json --output analysis/interactions.csv
python semantic_bias.py --spec semantic.json --output analysis/semantic.json

```

### 7. Batch Execution

Batch generation and analysis use all four configured models:

```bash
CONFIG_PATH=./model_config.json OUTPUT_BASE=./ITTS_audios bash run_all_models.sh
WAV_BASE=./ITTS_audios ANALYSIS_BASE=./analysis bash analyze_all_models.sh

```

Generation exits nonzero if any item fails. Batch scripts continue to report all incomplete model/dataset combinations, then exit nonzero if any generation, missing WAV set, or analysis failed.

---

## 🔬 Reproducibility and Paper Alignment

Run the prompt audit and offline tests with no model or network:

```bash
python prompt_audit.py --output prompt_manifest.json
python prompt_audit.py --strict-stage1
python prompt_audit.py --strict-paper  # always fails: original content identity is unverifiable
python build_splits.py --profile canonical-stage1-6900-v1 \
  --output data/canonical-stage1-6900-v1-splits.json
python verify_provenance.py --require-binaries
python -m unittest discover -v

```

The audit distinguishes schema validity, paper count/Cartesian alignment, and original-content identity. The implementation aligns with paper equations 5-7, uses 10,000 constrained-null randomizations as documented above, validates complete named interaction condition sets, includes `child` as a non-female paper-primary classifier outcome, and reports adult-only and unknown outcomes separately. Reproducing paper tables additionally requires the missing human-verified prompts, original model/checkpoint versions, generated waveforms, classifier audit data, and the authors' exact permutation implementation; these remain unresolved external blockers.

---

## 📖 Citation

```bibtex
@inproceedings{chen2026binding,
  title     = {The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS},
  author    = {Chen, Kuan-Yu and Lin, Yi-Cheng and Hsieh, Po-Chung and Chou, Huang-Cheng and Hsu, Chih-Fan and Li, Jeng-Lin and Lee, Hung-yi and Ding, Jian-Jiun},
  booktitle = {Interspeech},
  year      = {2026},
  note      = {arXiv:2603.20743}
}

```

---

## 📜 Licenses and Acknowledgements

The arXiv paper is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No source-code license file is currently included, so reuse of this repository's code is not granted beyond applicable law until the authors add one. External models, checkpoints, datasets, and backends retain their own licenses and terms.

This work is intended for bias diagnosis and mitigation research. Binary acoustic classification is a measurement simplification, not a statement about gender identity. Acknowledgements and funding details are provided in the paper.
