# Mini RLACE Rank-8 Paired Quality Protocol

The frozen `mini_rlace_rank8_quality_v1.json` subset compares Original with
`small-context-leace-regex` and `rlace-rank-8` on the same prompts. It contains 20 pairs from each of five
strata: Stage 1 status, career, and persona, and Stage 2 two-axis and
three-axis. This is 100 paired prompts and 200 WAV evaluations, a moderate
screen that keeps UTMOS plus Whisper practical while representing every major
part of the completed 13,300-prompt design.

The frozen `mini_rlace_rank8_quality_v2.json` manifest is a pre-evaluation
precision-sensitivity extension prompted by the wide v1 WER interval. It uses
the same five strata and stable-hash procedure with a distinct v2 seed, taking
100 pairs per stratum (500 paired prompts and 1,000 WAV evaluations per
comparison). It supplements rather than replaces the frozen v1 screen.

Selection is deterministic and independent of input ordering. Within each
stratum, prompts are ranked by SHA-256 of the frozen seed, stratum, and prompt
ID separated by NUL bytes; the manifest-specific number of pairs is retained.
Each manifest records the selected IDs, canonical source hashes, population
counts, and existing WAV directories. The evaluator verifies source hashes and
reads WAVs in place.

Run from the repository root without regenerating or copying audio:

```bash
python evaluate_screen_quality.py \
  --prompts data/parler-mini/mini_rlace_rank8_quality_v1.json \
  --root . \
  --baseline original \
  --candidate rlace-rank-8 \
  --asr-model openai/whisper-tiny.en \
  --output runs/analysis/parler-mini/comparison/large-v1/quality/paired_quality_v1.json
```

The existing noninferiority margins remain unchanged: candidate-minus-baseline
UTMOS must have a 95% CI lower bound at least -0.10, and WER must have a 95% CI
upper bound at most +0.03.

Run the same command with `--candidate small-context-leace-regex` and a distinct
output path for the Small-context LEACE comparison.
