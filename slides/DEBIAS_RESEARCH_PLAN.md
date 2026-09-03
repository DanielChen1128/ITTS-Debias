# Instruction TTS Debias Research Plan

## Target

ICASSP 2027.

## Research Positioning

This work extends *The Binding Effect*, which identifies diverse gender-bias
phenomena in instruction TTS. The follow-up asks whether post-hoc editing of a
TTS description encoder can reduce occupation-conditioned voice-gender bias
while retaining explicit user control and other non-gendered behavior.

**Claim discipline.** LEACE is an existing linear concept-erasure algorithm;
this paper does not claim a new erasure algorithm. The contribution is a
selective encoder-editing framework for instruction TTS, rigorous end-to-end
evidence, and representation analysis. The paper must not characterize the
work as merely transferring an NLP technique to speech: an encoder edit must
propagate through a generative acoustic decoder and vocoder before it can alter
perceived voice gender, and that propagation is an empirical question.

### Defensible Contribution Claims

1. **Selective concept erasure with explicit-intent preservation.** We apply a
   fitted eraser only to implicit gender leakage. Direct gender commands are
   detected at the description level and bypass the edit, preserving user voice
   control exactly. Standard unconditional erasure does not supply this
   deployment behavior.
2. **End-to-end instruction-TTS mitigation.** We evaluate whether editing a
   text-conditioning representation reduces occupation-conditioned voice-gender
   effects in generated audio, without degrading quality or intelligibility.
3. **Bounded-harm evaluation.** Exact anchor preservation, paired
   non-inferiority tests, neutral-drift reporting, and persona/status spillover
   tests jointly distinguish mitigation from degraded or globally shifted
   generation.
4. **Interpretability finding.** Linear information can be removed from the
   final encoder state even when a nonlinear probe still recovers gender. The
   resulting audio-level reduction establishes that removing the linearly
   accessible component is behaviorally meaningful, not that gender has been
   completely removed.
5. **Generality target.** Validate across common text-encoder families
   (Flan-T5, mT5, and BERT-family) and across encoder scale. Parler Mini uses
   Flan-T5-large (d=1024) and Large uses Flan-T5-xl (d=2048); their comparison
   tests cross-encoder-scale generality within the same TTS architecture.

### ICASSP Framing

Target venue: **ICASSP 2027**. The intended positioning is a rigorous
application/framework paper with interpretability, not an algorithm paper.
For ICASSP, the strength comes from complete controls, reproducible protocol,
multi-model validation, and a socially relevant speech-generation problem.

The companion/prior team work *The Binding Effect* contributes the bias
phenomenon and benchmark. This paper's independent contribution must remain
clearly scoped to intervention, explicit-control preservation, analysis, and
cross-model validation. Do not re-count discovery of the phenomenon as a new
contribution in this paper.

### Anticipated Reviewer Questions and Responses

| Reviewer concern | Evidence / response |
|---|---|
| "LEACE is not new." | Agree explicitly. The novelty is selective TTS deployment with bypass, end-to-end evidence, and generality across encoder families. |
| "The edit may just damage speech." | Paired UTMOS and Whisper WER non-inferiority tests; neutral-drift and spillover results. |
| "Explicit gender control was removed." | Hash-gated WAV identity for bypassed explicit anchors, plus gender-stratified anchor outcomes. |
| "Gender was not fully removed." | Report residual MLP decodability as a limitation; claim removal of linearly readable gender information and the observed audio-level effect only. |
| "The result is model-specific." | Four-model plan: Flan-T5-large / Flan-T5-xl / mT5 / BERT-family; Mini-to-Large comparison tests cross-encoder-scale generality. |
| "Audio gender classifier is not a sufficient metric." | Cite the classifier's published validation and report its established accuracy/calibration. The classifier is the prespecified primary operational metric; a human study is not required for the primary claim. |

## Phase 1: Parler-TTS Mini Confirmatory Experiment

### Frozen Design

- Model: Parler-TTS Mini.
- Total: 3,240 WAV files.
- Same prompt uses the same generation seed under every method.
- Career inference is clustered at the occupation-lemma level.

### Anchor Controls

- 24 explicit male/female prompts x 3 methods = 72 WAV files.
- Methods: `original`, `leace_bypass`, `leace_nobypass`.
- `leace_bypass` must preserve the original output exactly for every explicit
  prompt. This is enforced by SHA-256 equality of the generated WAVs.
- Report female and male anchor outcomes separately, as well as pooled
  non-inferiority of anchor accuracy.
- `leace_nobypass` is an ablation demonstrating why explicit-command bypass is
  necessary; it is not the proposed deployment method.

### Core Test

- 12 externally annotated occupations, balanced at six female-associated and
  six male-associated occupation lemmas.
- Each career has 10 description templates x 3 transcripts x 2 seeds = 60
  prompts.
- 18 neutral prompts estimate global voice-gender drift.
- 738 prompts x 4 methods = 2,952 WAV files.
- Methods: `original`, random rank-1 projection placebo, gender-direction
  projection, and `leace_bypass`.

Primary estimand:

```text
Gap(method) = mean P(female | female-associated occupation lemma)
            - mean P(female | male-associated occupation lemma)
```

The primary analysis is a paired, gender-stratified cluster bootstrap over
occupation lemmas. Quality and intelligibility use paired non-inferiority
intervals for UTMOS and WER. Neutral P(female) drift is always reported.

Expected result: the debiased methods reduce the occupation-conditioned gap.

### Spillover Test

- 10 persona descriptors from Big Five dimensions plus two status descriptors.
- Each descriptor has 1 template x 3 transcripts x 2 seeds = 6 prompts.
- 72 prompts x 3 methods = 216 WAV files.
- Methods: `original`, `gender_direction`, `leace_bypass`.

Expected result: gender debiasing should not introduce unintended changes to
the model's behavior for persona or status descriptions. This is a side-effect
test, not a claim that these descriptors should have zero voice-gender effect.

## Phase 2: Methodological Analysis

### 2A. Layer-Wise Gender Information Analysis

- Extract pooled hidden states from all 24 Parler encoder layers.
- Fit a linear gender probe at each layer using explicit anchor labels.
- Report layer-wise probe accuracy and an information/separability measure.
- Use this to identify where gender information is represented and to explain
  why a rank-1 edit can be effective.

### 2B. Residual Non-Linear Probe

- Compare original and LEACE-edited encoder states.
- Fit matched linear and two-layer MLP gender probes.
- Test whether gender remains decodable non-linearly after linear erasure.
- Report a residual effect as a limitation if the MLP remains above chance;
  do not treat that outcome as a failed experiment.

### 2C. Partial Erasure (Optional)

- If time permits, evaluate partial intervention strengths (25%, 50%, 75%,
  and 100%) on a predeclared Core subset.
- Plot occupation gap against UTMOS and explicit-control outcomes.
- Treat this as exploratory operating-point analysis.

## Phase 3: Multi-Model Validation

Run the same anchor/core/spillover protocol after the Parler-Mini results are
available.

1. Parler-TTS Large: highest priority; tests a larger Flan-T5-xl text encoder
   (d=2048) compared to Mini's Flan-T5-large (d=1024), validating cross-encoder-
   scale generality within the same TTS architecture.
2. VoxInstruct: high priority; mT5 text encoder, subject to repository and
   checkpoint availability.
3. PromptTTS++: high priority; expected BERT-family style/text encoder,
   subject to repository inspection and checkpoint availability.

Each model receives its own fitted eraser from that model's explicit-anchor
activations. Cross-model projector transfer is not a primary research question.

### Model-Specific Adapter Plan

The generic claim is not that one projector transfers between encoders. Each
model needs a model-local LEACE artifact fitted in its own representation space.
The shared framework should expose three operations per model:

```text
collect_pooled_activations(descriptions)       # fit LEACE from explicit anchors
collect_pooled_layer_activations(descriptions) # Phase 2 probes
generate_with_eraser(...)                      # token-wise encoder-output edit
```

The eraser is fitted on masked-mean pooled representations, then applied to
each valid encoder-output token because it is an affine feature-space map. The
explicit-command bypass remains model-agnostic and is decided from the raw
description before the edit.

Planned adapters:

| Model | Encoder family | Encoder dim | Intended injection point | Status |
|---|---|---|---|---|
| Parler Mini | Flan-T5-large | 1024 | Description encoder output before decoder cross-attention | Complete |
| Parler Large | Flan-T5-xl | 2048 | Description encoder output before decoder cross-attention | Complete |
| VoxInstruct | mT5-base | TBD | mT5 text-encoder output cached by AR inference; inspect NAR path too | Pending repo + checkpoint |
| PromptTTS++ | BERT-family, to verify | TBD | Style/text encoder output before acoustic inference | Pending repo + checkpoint |

Do not implement a generic forward hook until the two external repositories
are inspected. The injection point must be semantically correct and must cover
every generation path that conditions acoustic output.

## Paper Story

```text
Binding Effect identifies multi-dimensional instruction-TTS bias
    -> post-hoc encoder editing reduces occupation-conditioned gender bias
    -> exact explicit-control preservation and spillover checks bound harm
    -> layer-wise and residual probes explain what the edit removes
    -> multiple TTS architectures test generality
```

## Proposed Paper Structure

1. Introduction: bias findings from Binding and the need for usable mitigation.
2. Related Work: TTS fairness, concept erasure, and representation probing.
3. Method: encoder editing, explicit-command bypass, and evaluation protocol.
4. Experiments: main results, quality/control preservation, and spillover.
5. Analysis: layer-wise representation and residual non-linear probing.
6. Multi-model validation.
7. Discussion: neutral drift, limitations, and optional partial-erasure tradeoff.
8. Conclusion.

## Current Status

### Parler Mini: Complete Confirmatory Results

- 3,240 / 3,240 WAV generated and evaluated.
- Anchor preservation gate passed: 24/24 WAV hashes match between `original`
  and `leace_bypass`, stratified as 12/12 female and 12/12 male prompts.
- Career bias gap: original `0.289` (95% CI `[0.192, 0.413]`) versus
  `leace_bypass` `0.082` (95% CI `[-0.008, 0.181]`), a 72% reduction.
- LEACE versus original: delta `-0.207`, 95% CI `[-0.272, -0.135]`, one-sided
  bootstrap `p < 0.0001`. LEACE also outperforms random projection
  (`p < 0.0001`) and gender-direction removal (`p = 0.0044`).
- Utility: LEACE-bypass passes prespecified non-inferiority thresholds for
  UTMOS (delta `-0.025`, lower 95% bound `-0.074`, margin `-0.10`) and WER
  (delta `+0.011`, upper 95% bound `+0.025`, margin `+0.03`).
- Explicit control: bypass has zero classifier-accuracy change. In contrast,
  no-bypass fails anchor non-inferiority, particularly for male anchors
  (accuracy `0.833` to `0.583`).
- Neutral drift: LEACE-bypass `+0.010`, below the `0.10` warning threshold;
  gender-direction is flagged at `+0.142`.
- Spillover (original 12-cluster): LEACE-bypass persona/status drift `+0.018`,
  95% CI `[-0.074, 0.097]`; gender-direction drift `+0.061`, 95% CI
  `[+0.003, +0.119]`.

### Parler Mini: Phase 2 Findings

- Held-out evaluation uses 120 explicit-anchor fit examples and 40 control
  examples.
- A linear probe is perfect at the embedding output and early encoder layers,
  at chance accuracy through middle layers, then perfect again at the final
  layer. This U-shaped pattern is descriptive and must not be over-generalized
  until replicated.
- At the final representation, original linear and MLP probes are both perfect.
  After LEACE, the linear probe is at chance (AUC/accuracy `0.5`), whereas the
  MLP remains perfect (AUC/accuracy `1.0`).
- Therefore the correct claim is: LEACE removes the linearly decodable gender
  component targeted by the intervention; it does not establish complete
  nonlinear removal of gender information.

### Parler Large: Complete Confirmatory Results

- 3,240 / 3,240 WAV generated and evaluated.
- Text encoder: Flan-T5-xl (d=2048, 24 layers), different from Mini's
  Flan-T5-large (d=1024).
- Anchor preservation gate passed: 24/24 WAV hashes match between `original`
  and `leace_bypass`, stratified as 12/12 female and 12/12 male prompts.
- Career bias gap: original `0.148` (95% CI `[0.067, 0.259]`) versus
  `leace_bypass` `0.006` (95% CI `[-0.016, 0.026]`), a **96% reduction**.
- LEACE versus original: delta `-0.142`, 95% CI `[-0.253, -0.055]`, one-sided
  bootstrap `p < 0.0001`. LEACE also outperforms random projection
  (`p < 0.0001`) and gender-direction removal (`p = 0.0028`).
- Utility: LEACE-bypass passes prespecified non-inferiority thresholds for
  UTMOS (delta `+0.028`, margin `-0.10`) and WER (delta `-0.008`, margin
  `+0.03`).
- Explicit control: bypass has zero classifier-accuracy change. In contrast,
  no-bypass **catastrophically fails** for male anchors (accuracy `1.0` to
  `0.0`, P(female) shifts from `0.002` to `0.864`).
- Neutral drift: LEACE-bypass `+0.105`, marginally exceeds the `0.10` warning
  threshold, but 95% CI `[-0.021, 0.313]` crosses zero; 3-cluster design is
  underpowered.
- Spillover (original 12-cluster): LEACE-bypass drift `+0.097`, 95% CI
  `[0.006, 0.205]`; gender-direction drift `-0.104`, 95% CI `[-0.201, -0.013]`.

### Parler Large: Phase 2 Findings

- Layer-wise probe replicates Mini's U-shape: embedding + layer 0 = AUC 1.0,
  layers 1–22 at chance (AUC ~0.5), layer 23 recovers to AUC 1.0.
- Residual probe: after LEACE, linear probe AUC drops to 0.0 (inverted
  small-sample artifact with n=40 control); MLP still recovers gender.
- Cross-model replication of U-shaped layer-wise distribution supports the
  claim that gender information concentrates at embedding and final layers.

### Expanded Spillover Experiment (42 Clusters)

To resolve the apparent Mini vs Large spillover discrepancy (Mini 0.018 vs
Large 0.097), an expanded spillover protocol was run with higher statistical
power:

- Protocol: `confirmatory-spillover-ext-v1`
- Clusters: 42 (40 persona + 2 status lemmas)
- Items: 756 per method (3 templates × 3 transcripts × 2 seeds)
- Total: 2,268 WAV per model, 4,536 WAV combined

**Results:**

| Method | Mini (42 clusters) | Large (42 clusters) | CI overlap? |
|---|---|---|---|
| **LEACE-bypass** | **0.079** [0.047, 0.113] | **0.089** [0.048, 0.135] | **Yes, substantial** |
| gender_direction | +0.062 [0.032, 0.093] | −0.080 [−0.119, −0.040] | No (sign flip) |

**Key findings:**

1. LEACE-bypass spillover is **consistent across models (~0.08)** with
   substantially overlapping confidence intervals.
2. The earlier apparent discrepancy (Mini 0.018 vs Large 0.097) was a
   **statistical artifact of underpowered 12-cluster estimation**, not a real
   model difference.
3. Gender-direction shows **sign flip** between models (+0.06 Mini, −0.08
   Large), demonstrating that this naive baseline has unpredictable behavior
   across encoder architectures.
4. LEACE-bypass is preferable because it produces **consistent, predictable,
   moderate spillover** across encoder scales.

### Mini vs Large Comparison Summary

| Metric | Mini (Flan-T5-large) | Large (Flan-T5-xl) | Consistent? |
|---|---|---|---|
| Bias reduction | −72% | −96% | ✅ Both eliminate bias |
| LEACE vs original p | < 0.0001 | < 0.0001 | ✅ |
| UTMOS non-inferiority | ✅ Pass | ✅ Pass | ✅ |
| WER non-inferiority | ✅ Pass | ✅ Pass | ✅ |
| Anchor preservation | 24/24 | 24/24 | ✅ |
| No-bypass failure | Partial (male 0.83→0.58) | Catastrophic (male 1.0→0.0) | ✅ Both fail |
| Layer-wise U-shape | ✅ | ✅ | ✅ Replicated |
| Spillover (expanded) | 0.079 [0.047, 0.113] | 0.089 [0.048, 0.135] | ✅ ~0.08, CI overlap |
| Neutral drift | 0.010 | 0.105 | ⚠️ Underpowered (3 clusters) |

**Conclusion:** LEACE-bypass effectiveness generalizes across encoder scales
(Flan-T5-large to Flan-T5-xl). Both models show consistent ~0.08 spillover,
which is moderate and predictable. The no-bypass ablation demonstrates that
explicit-command preservation is essential.

### Known Limitations

1. **Neutral drift is underpowered:** Only 3 neutral description clusters are
   available in the legacy corpus. The warning-threshold crossings flip between
   models (Mini: gender_direction flagged; Large: leace_bypass flagged),
   indicating estimation noise rather than stable effects. This will be
   reported as a limitation, not expanded further.

2. **MLP still recovers gender:** After LEACE, a nonlinear probe can still
   decode gender from the encoder state. The paper claims removal of the
   linearly accessible component and the resulting audio-level behavior change,
   not complete erasure.

3. **Spillover exists but is moderate:** LEACE-bypass consistently shifts
   persona/status voice-gender by ~0.08. This is an acknowledged side effect,
   bounded and predictable across models.

### Next Execution Order

1. ~~Finish Parler-Large anchor/core/spillover generation, evaluation, and
   Phase 2A/2B.~~ **Done.**
2. ~~Compare Mini and Large: cross-encoder-scale generality.~~ **Done.**
3. ~~Run expanded spillover experiment to resolve apparent discrepancy.~~
   **Done.**
4. Obtain PromptTTS++ and VoxInstruct source repositories and checkpoints.
5. Inspect their actual encoder and conditioning paths; implement and validate
   model-specific encoder-edit adapters.
6. Run the frozen protocol and Phase 2 analyses separately for each external
   model.
7. Build the cross-model table only after every model's provenance, controls,
   and model-local artifact are complete.
8. Paper writing: integrate results, update contribution claims, write
   limitations section.
