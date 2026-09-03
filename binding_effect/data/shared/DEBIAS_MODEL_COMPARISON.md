# Parler Mini and Large Debiasing Comparison

## Common Evaluation Protocol

Both maintained configurations use the same frozen 13,300-prompt structure:
6,900 single-axis prompts and 6,400 compositional prompts. Generation uses
batch size 8, fixed model revisions, matched prompt IDs, and matched seeds.
Explicit gender requests bypass the intervention through the same regex rule.

Acoustic gender is evaluated identically for both models:

```text
female_score >= male_score -> female
female_score <  male_score -> male
```

The detector's `child_score` is retained as raw metadata only. It does not
participate in classification, rates, career metrics, or interactions. Invalid
female/male scores are labeled unknown and excluded from binary denominators.

## Debiasing Configuration

| Item | Parler Mini | Parler Large |
| --- | --- | --- |
| Model revision | `0392b9451a601e528fd863bbb0598431fee810d9` | `50cb4b874c83902f930d7c2e753224c15654f11e` |
| Selected method | RLACE rank 8 | Female-to-male constant centroid steering |
| Intervention target | Pooled projected final encoder representation | Pooled projected final encoder representation |
| Operation | Remove an adversarially learned rank-8 gender-predictive subspace | Add `2x` the male-centroid minus female-centroid direction |
| Training anchors | 540 balanced anchors, 45 contexts | 5,040 balanced anchors, 180 contexts |
| Status used for fitting | No | No |
| Explicit gender handling | Regex bypass | Regex bypass |
| Main interpretation | Representation projection/attenuation | Directed output calibration |

Mini's initial Small-context LEACE candidate remains in the 39,900-WAV
three-method analysis, but rank-8 RLACE is the maintained Mini configuration.
Large RLACE variants were screened, but `2x` constant steering was selected for
the full Large evaluation because it provided the better output calibration.

## Overall Gender Results

Female percentages below use the common binary rule.

| Prompt set | N/model/method | Mini Original | Mini RLACE-8 | Large Original | Large 2x steering |
| --- | ---: | ---: | ---: | ---: | ---: |
| Status | 200 | 80.50% | 67.50% | 58.00% | 30.00% |
| Career | 2,700 | 79.56% | 61.96% | 88.37% | 72.48% |
| Persona | 4,000 | 77.05% | 57.15% | 85.58% | 61.30% |
| Stage 1 pooled | 6,900 | 78.13% | 59.33% | 85.87% | 64.77% |
| Two-axis | 3,200 | 78.94% | 60.12% | 64.05% | 40.23% |
| Three-axis | 3,200 | 77.72% | 60.34% | 61.41% | 33.62% |
| Stage 2 pooled | 6,400 | 78.33% | 60.23% | 62.73% | 36.93% |
| Global | 13,300 | 78.23% | 59.77% | 74.73% | 51.37% |

| Global measure | Mini Original | Mini RLACE-8 | Large Original | Large 2x steering |
| --- | ---: | ---: | ---: | ---: |
| Binary female rate | 0.7823 | 0.5977 | 0.7473 | 0.5137 |
| Error from 0.5 | 0.2823 | 0.0977 | 0.2473 | 0.0137 |
| Mean conditional female score | 0.7576 | 0.5877 | 0.7232 | 0.5067 |
| Valid binary N | 13,300 | 13,300 | 13,299 | 13,299 |
| Change in female rate | - | -18.46 pp | - | -23.36 pp |

Mini improves every prompt set but remains female-skewed globally. Large is
closer to 50% globally, but this is an aggregate cancellation: career remains
female-skewed while status and compositional prompts overshoot toward male.

## Career Separation

Rates are equally weighted over six female-associated and six male-associated
career descriptors.

| Model/method | Female-associated | Male-associated | Midpoint offset | Gap | Mean absolute distance from 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mini Original | 0.9500 | 0.6400 | +0.2950 | 0.3100 | 0.2950 |
| Mini RLACE-8 | 0.7633 | 0.4283 | +0.0958 | 0.3350 | 0.1675 |
| Large Original | 0.9817 | 0.7983 | +0.3900 | 0.1833 | 0.3900 |
| Large 2x steering | 0.9633 | 0.4333 | +0.1983 | 0.5300 | 0.2883 |

Both methods reduce the overall female midpoint offset. Neither reduces the
female-associated versus male-associated career gap: Mini changes from 0.3100
to 0.3350 without a statistically established increase, while Large increases
from 0.1833 to 0.5300.

## Binding Interactions

Interaction statistics use 64 matched two- and three-axis cells and 10,000
constrained-null randomizations with seed `20260818`.

| Model | Original mean absolute | Debiased mean absolute | Reduction | Original strong | Debiased strong | Debiased moderate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mini | 1.7151 | 0.7689 | 55.2% | 11 | 0 | 16 |
| Large | 3.2581 | 1.3546 | 58.4% | 37 | 3 | 20 |

Both interventions strongly attenuate binding interactions, but neither result
should be described as eliminating all interaction structure.

## Audio Quality

Non-inferiority margins are UTMOS delta `>= -0.10` and absolute WER delta
`<= +0.03`.

| Model/evaluation | Pairs | UTMOS delta [95% CI] | UTMOS NI | WER delta [95% CI] | WER NI |
| --- | ---: | ---: | :---: | ---: | :---: |
| Mini RLACE-8 v1 | 100 | -0.1570 [-0.2584, -0.0592] | Fail | +0.0014 [-0.0417, +0.0454] | Fail |
| Mini RLACE-8 v2 sensitivity | 500 | -0.0128 [-0.0628, +0.0370] | Pass | -0.0190 [-0.0369, -0.0013] | Pass |
| Large 2x steering | 116 | -0.0506 [-0.1868, +0.0843] | Fail | +0.0101 [-0.0417, +0.0630] | Fail |

The larger Mini sensitivity evaluation supports quality and intelligibility
non-inferiority, while its original smaller screen must still be disclosed.
Large point estimates meet both margins, but its confidence intervals do not;
Large quality non-inferiority is therefore not established.

## Supported Conclusions

| Question | Parler Mini | Parler Large |
| --- | --- | --- |
| Reduced global female skew? | Yes, 78.23% to 59.77% | Yes, 74.73% to 51.37% |
| Reached uniform 50% parity? | No | No; global balance masks axis overshoot |
| Weakened binding interactions? | Yes, 55.2% lower mean magnitude | Yes, 58.4% lower mean magnitude |
| Reduced career separation? | No evidence | No; separation increased |
| Preserved explicit requests? | Regex bypass by design; no comparable full safety table | Yes, 100% explicit-control accuracy in the matched screen |
| Established audio NI? | Supported by 500-pair sensitivity, not by initial 100-pair screen | No at 116 pairs |
| Established representation erasure? | No; held-out linear-probe AUC was 0.9959 before and 0.9954 after RLACE | Not the objective of directed steering |

The strongest shared claim is output-level global calibration together with
binding-interaction attenuation. The evidence does not support uniform subgroup
parity or complete removal of gender information from internal representations.

## Result Sources

- Mini report: `data/parler-mini/MINI_RLACE_RANK8_LARGE_V1_RESULTS.md`
- Large report: `data/parler-large/PARLER_LARGE_STEERING_2X_RESULTS.md`
- Mini diagnostics: `runs/analysis/parler-mini/comparison/large-v1/`
- Large diagnostics: `runs/analysis/parler-large/comparison/`
- Mini manifest: `data/parler-mini/mini_rlace_rank8_large_manifest.json`
- Large quality report: `runs/analysis/parler-large/screening/constant-steering-male-strength2-matched/quality-v1.json`
