# Versioned datasets

`../descriptions/` is the canonical 6,900-record Stage 1 reconstruction from
[Binding commit `19accf5704f9aff2f66bf083aa9902098cfe08e1`](https://github.com/DanielChen1128/The-Binding-Effect-ITTS-bias-analysis/commit/19accf5704f9aff2f66bf083aa9902098cfe08e1).
Its schema, counts, and hashes are recorded in
`../prompt_manifest.json`.

`legacy-5900-v1/` is an immutable snapshot of the inputs used by the completed
Parler-TTS Mini debiasing experiments. It contains the earlier 3,900 Stage 1
records, 2,000 incomplete composite records, and the derived pilot/main input
JSON files. Do not regenerate completed experiment outputs against the
canonical corpus while retaining their old experiment IDs.

New split manifests must use a new output path, for example:

```bash
python build_splits.py \
  --profile canonical-stage1-6900-v1 \
  --output data/canonical-stage1-6900-v1-splits.json
```
