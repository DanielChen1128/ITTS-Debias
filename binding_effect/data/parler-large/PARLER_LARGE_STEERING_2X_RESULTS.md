# Parler Large Constant Steering 2x Results

## Scope

The frozen evaluation contains 13,300 prompts per method: 6,900 single-axis
prompts and 6,400 frozen multi-axis prompts. The comparison is Original versus
female-to-male constant steering with intervention strength 2. Both methods use
Parler-TTS Large revision `50cb4b874c83902f930d7c2e753224c15654f11e`, batch
size 8, matched prompt IDs, and matched seeds.

Female rate below uses a binary decision between the acoustic model's female
and male scores. The model's child score is retained as raw metadata but does
not participate in the argmax or denominator. One degenerate two-axis WAV per
method could not be classified and is excluded from binary rates.

## Gender Balance

| Prompt set | N | Binary N | Original female | 2x female | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Status | 200 | 200 | 58.00% | 30.00% | -28.00 pp |
| Career | 2,700 | 2,700 | 88.37% | 72.48% | -15.89 pp |
| Persona | 4,000 | 4,000 | 85.58% | 61.30% | -24.28 pp |
| Stage 1 pooled | 6,900 | 6,900 | 85.87% | 64.77% | -21.10 pp |
| Two-axis | 3,200 | 3,199 | 64.05% | 40.23% | -23.82 pp |
| Three-axis | 3,200 | 3,200 | 61.41% | 33.62% | -27.78 pp |
| Stage 2 pooled | 6,400 | 6,399 | 62.73% | 36.93% | -25.80 pp |
| Global | 13,300 | 13,299 | 74.73% | 51.37% | -23.36 pp |

The global result is close to the prespecified 50% target under both summaries:

| Measure | Original | 2x steering | Absolute error from 0.5 |
| --- | ---: | ---: | ---: |
| Binary female argmax rate | 0.7473 | 0.5137 | 0.0137 |
| Mean conditional female score | 0.7232 | 0.5067 | 0.0067 |

This global calibration is not uniform across axes. Career prompts remain
female-skewed, whereas status and compositional prompts overshoot toward male.
The global value therefore should not be interpreted as subgroup parity.

## Generalization Groups

| Group | Original female | 2x female | Change |
| --- | ---: | ---: | ---: |
| Status excluded from training | 58.00% | 30.00% | -28.00 pp |
| Career training-seen | 81.00% | 62.50% | -18.50 pp |
| Career unseen non-neighbor | 90.00% | 76.14% | -13.86 pp |
| Career mechanician lexical neighbor | 82.00% | 32.00% | -50.00 pp |
| Persona training-seen | 79.63% | 55.25% | -24.38 pp |
| Persona unseen | 87.06% | 62.81% | -24.25 pp |

Descriptor heterogeneity decreases for status but increases for career and
persona:

| Axis | Original range | 2x range | Range change | Original SD | 2x SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Status | 0.58 | 0.52 | -0.06 | 0.2900 | 0.2600 |
| Career | 0.70 | 0.89 | +0.19 | 0.1655 | 0.2951 |
| Persona | 0.58 | 0.62 | +0.04 | 0.1082 | 0.1583 |

## Career Separation

Binary hard-label rates are equally weighted over the six female-associated and
six male-associated careers in the frozen protocol.

| Method | Female-associated | Male-associated | Midpoint offset | Gap | Mean absolute distance from 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.9817 | 0.7983 | +0.3900 | 0.1833 | 0.3900 |
| 2x steering | 0.9633 | 0.4333 | +0.1983 | 0.5300 | 0.2883 |

Paired descriptor bootstrap changes use 10,000 iterations and seed 20260818:

| Measure | 2x minus Original | 95% CI |
| --- | ---: | ---: |
| Female-associated rate | -0.0183 | [-0.0300, -0.0067] |
| Male-associated rate | -0.3650 | [-0.4083, -0.3150] |
| Midpoint offset | -0.1917 | [-0.2150, -0.1658] |
| Career gap | +0.3467 | [+0.2950, +0.3933] |
| Mean absolute distance | -0.1017 | [-0.1550, -0.0308] |

Steering substantially reduces the global career midpoint skew and the mean
distance from 0.5, but it does so mainly by changing male-associated careers.
Consequently, conditional career separation increases rather than decreases.

## Safety Screen

The matched screen contains 100 implicit prompts and all 16 explicit controls.
Conditional female score is computed over the implicit subset.

| Measure | Original | 2x steering | Gate | Result |
| --- | ---: | ---: | --- | :---: |
| Binary female rate | 85.00% | 66.00% | Closer to 50% | Pass |
| Conditional female score | 0.8414 | 0.6245 | Closer to 0.5 | Pass |
| Calibration error | 0.3414 | 0.1245 | Lower than Original | Pass |
| Career gap | 0.0500 | 0.1833 | Descriptive | Worsened |
| Classification failures | 0 | 0 | Descriptive | Unchanged |
| Explicit-control accuracy | 93.75% | 100.00% | At least Original - 5 pp | Pass |

## Binding Interactions

| Interaction family | Count | Original mean absolute | 2x mean absolute | Reduction | Strong interactions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Career + persona | 16 | 3.5665 | 1.3568 | 62.0% | 10 to 1 |
| Status + career | 8 | 2.3916 | 1.6084 | 32.7% | 4 to 1 |
| Status + career + persona | 32 | 3.5393 | 1.4063 | 60.3% | 20 to 1 |
| Status + persona | 8 | 2.3830 | 0.8895 | 62.7% | 3 to 0 |
| Count-weighted total | 64 | 3.2581 | 1.3546 | 58.4% | 37 to 3 |

Moderate interactions increase from 13 to 20 while strong interactions fall
from 37 to 3. Thus, the intervention attenuates rather than eliminates binding
effects.

## Audio Quality

The paired quality evaluation uses 100 implicit prompts and all 16 explicit
controls selected by the pre-existing deterministic v1 rule. Candidate-minus-
Original intervals use 10,000 paired bootstrap samples.

| Metric | Original mean | 2x mean | Delta [95% CI] | Margin | NI result |
| --- | ---: | ---: | ---: | ---: | :---: |
| UTMOS | 3.8596 | 3.8089 | -0.0506 [-0.1868, +0.0843] | -0.10 | Fail |
| WER | 0.1238 | 0.1338 | +0.0101 [-0.0417, +0.0630] | +0.03 | Fail |

Both point estimates are inside the non-inferiority margins, but the UTMOS
lower confidence bound and WER upper confidence bound cross their margins.
The 116-pair evaluation therefore does not establish quality or intelligibility
non-inferiority.

## Interpretation

Parler Large 2x steering succeeds at the primary global calibration objective:
the binary female rate moves from 74.73% to 51.37%, the mean conditional female
score moves from 0.7232 to 0.5067, and strong binding interactions fall by
91.9%. The explicit-control safety gate passes.

The supported claim is limited to global gender calibration and interaction
attenuation. The intervention does not produce uniform axis-level parity,
increases conditional career separation, and does not establish paired UTMOS
or WER non-inferiority at the current sample size.

## Result Sources

- Full detections: `runs/analysis/parler-large/`
- Diagnostics: `runs/analysis/parler-large/comparison/diagnostics-steering-2x/`
- Interactions: `runs/analysis/parler-large/comparison/interactions/`
- Quality report: `runs/analysis/parler-large/screening/constant-steering-male-strength2-matched/quality-v1.json`
- Quality manifest: `data/parler-large/parler_large_constant_steering_male_strength2_matched_quality_v1.json`
