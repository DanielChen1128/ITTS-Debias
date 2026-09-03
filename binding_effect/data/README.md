# Data Layout

Data is grouped by model rather than by intervention method:

- `shared/`: cross-model protocol and comparison documentation.
- `parler-mini/`: Mini prompts, anchors, production artifact, and quality manifests.
- `parler-large/`: Large prompts, anchors, production artifact, and quality manifests.
- `voxinstruct/`: VoxInstruct paired anchors, activation caches, and AR/NAR artifacts.

Methods belong inside their model directory. Do not add method-named directories
at the top level.
