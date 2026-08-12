# ITTS Debias

Research workspace for encoder-space gender debiasing in instruction-guided
text-to-speech. The current study applies LEACE and rank-matched controls to
Parler-TTS Mini description states while preserving explicit voice controls.

## Status

- Screening, smoke pilot, 700-WAV pilot, and 1,050-WAV main experiment complete.
- Main LEACE+bypass stereotype gap: `0.389 -> 0.049` (87% reduction).
- Main LEACE+bypass UTMOS: `3.834`; WER: `0.191`; anchor accuracy: `1.00`.
- Mixed-effects analysis and publication packaging remain pending.

## Layout

- `binding_effect/`: source, canonical prompts, frozen experiment inputs, tests,
  and trackable evaluation records.
- `slides/`: progress report and research-question deck sources/exports.
- `papers/`: local third-party literature. PDFs are intentionally ignored.

The canonical Binding framework and this debias study have separate provenance.
See `binding_effect/datasets/README.md` and
`binding_effect/experiment_records/manifest.json` before rebuilding data or
interpreting completed results.
