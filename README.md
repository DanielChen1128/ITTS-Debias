# ITTS Debias

Research workspace for representation-level gender debiasing in
instruction-guided text-to-speech. The maintained study evaluates Parler-TTS
Mini, Parler-TTS Large, and VoxInstruct under one frozen 13,300-prompt protocol.

## Status

- Parler Mini rank-8 RLACE and Parler Large `2x` constant-steering evaluations
  are complete.
- Repaired VoxInstruct Original versus `AR6 + NAR2` constant steering is in
  progress on matched prompts and seeds.
- Acoustic gender, binding interactions, UTMOS, and WER use the common protocol
  documented under `binding_effect/data/shared/`.

## Layout

- `binding_effect/`: maintained source, canonical prompts, production artifacts,
  formal WAV results, analyses, and tests.
- `slides/`: progress report and research-question deck sources/exports.
- `papers/`: local third-party literature. PDFs are intentionally ignored.

See `binding_effect/README.md` for maintained workflows and
`binding_effect/data/shared/DEBIAS_MODEL_COMPARISON.md` for the
current paper-facing Parler results.
