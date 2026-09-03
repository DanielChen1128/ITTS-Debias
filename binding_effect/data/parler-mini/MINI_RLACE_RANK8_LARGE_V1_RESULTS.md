# Parler Mini RLACE Rank-8 Large-v1 Results

## Scope

The frozen large-v1 evaluation contains 13,300 prompts per method: 6,900
single-axis prompts and 6,400 frozen multi-axis prompts. The compared methods
are Original, Small-context LEACE with regex bypass, and RLACE rank 8.

Female rate uses a binary decision between the acoustic model's female and male
scores. The model's child score is retained as raw metadata but does not
participate in the decision or denominator. All 13,300 outputs per method have
valid female and male scores.

## Gender Balance

| Prompt set | N | Original female | RLACE rank 8 female | Change |
| --- | ---: | ---: | ---: | ---: |
| Status | 200 | 80.50% | 67.50% | -13.00 pp |
| Career | 2,700 | 79.56% | 61.96% | -17.59 pp |
| Persona | 4,000 | 77.05% | 57.15% | -19.90 pp |
| Stage 1 pooled | 6,900 | 78.13% | 59.33% | -18.80 pp |
| Two-axis | 3,200 | 78.94% | 60.12% | -18.81 pp |
| Three-axis | 3,200 | 77.72% | 60.34% | -17.38 pp |
| Stage 2 pooled | 6,400 | 78.33% | 60.23% | -18.09 pp |
| Global | 13,300 | 78.23% | 59.77% | -18.46 pp |

The global mean conditional female score falls from 0.7576 to 0.5877. RLACE
therefore reduces female skew consistently across all five prompt sets, but
does not reach the 50% global target.

Career results require separating global offset from conditional separation.
Using binary hard-label rates and equal weighting over the six female-associated
and six male-associated descriptors:

| Method | Female-associated | Male-associated | Midpoint offset | Gap | Mean absolute distance from 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.9500 | 0.6400 | +0.2950 | 0.3100 | 0.2950 |
| Small-context LEACE | 0.9250 | 0.7667 | +0.3458 | 0.1583 | 0.3458 |
| RLACE rank 8 | 0.7633 | 0.4283 | +0.0958 | 0.3350 | 0.1675 |

For Original to RLACE, the midpoint offset changes by -0.1992 (paired
descriptor bootstrap 95% CI [-0.2267, -0.1725]), while the gap changes by
+0.0250 (95% CI [-0.0317, +0.0767]). The evidence therefore supports a large
reduction in global female skew, but not a reduction in conditional career
separation. It does not establish that the career separation worsened.

## Binding Interactions

RLACE reduces the count-weighted mean absolute interaction from 1.7151 to
0.7689, a 55.2% reduction relative to Original. Family-level reductions range
from 51.4% to 62.0%. Strong interactions fall from 11 under Original to zero
under RLACE, with 16 moderate
interactions remaining.

## Audio Quality

The frozen v1 screen used 20 prompts from each of five strata (100 pairs). It
did not establish non-inferiority: RLACE UTMOS delta was -0.1570 with 95% CI
[-0.2584, -0.0592], and WER delta was +0.0014 with 95% CI
[-0.0417, +0.0454].

Because the v1 intervals were wide, a separately frozen v2 precision
sensitivity used 100 prompts per stratum (500 pairs). Its results were:

| Candidate | UTMOS delta [95% CI] | WER delta [95% CI] | UTMOS NI | WER NI |
| --- | ---: | ---: | :---: | :---: |
| Small-context LEACE | +0.0382 [-0.0061, +0.0830] | -0.0137 [-0.0330, +0.0059] | Pass | Pass |
| RLACE rank 8 | -0.0128 [-0.0628, +0.0370] | -0.0190 [-0.0369, -0.0013] | Pass | Pass |

The non-inferiority margins are -0.10 UTMOS and +0.03 absolute WER. The v2
subset was frozen before its evaluation but was created after observing the
wide v1 interval, so both results must be reported. The v1/v2 discrepancy is
driven mainly by within-stratum sampling, especially the 20-item v1 career
stratum; pairing and WAV path checks found no evaluator error.

## Interpretation

RLACE rank 8 provides strong evidence of reduced global female skew and weaker
binding interactions. The larger quality sensitivity supports audio-quality
and intelligibility non-inferiority. The claim should remain limited to global
calibration and interaction attenuation: RLACE does not reach global parity or
demonstrate reduced conditional career separation, and the quality conclusion
should disclose the smaller v1 screen that did not pass.

## Result Sources

- Full detections: `runs/analysis/parler-mini/`
- Diagnostics: `runs/analysis/parler-mini/comparison/large-v1/diagnostics/`
- Interactions: `runs/analysis/parler-mini/comparison/large-v1/interactions/`
- The sibling `parler-mini/interactions/` tree is a superseded provisional
  three-method analysis that predates the binary female/male protocol.
- Quality manifests: `data/parler-mini/mini_rlace_rank8_quality_v1.json` and `mini_rlace_rank8_quality_v2.json`
