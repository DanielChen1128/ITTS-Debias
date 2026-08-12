# Legacy Stage 2 Examples

These files preserve the repository's original compositional prompt examples:

- `descriptions_two_axis.json`: 1,200 prompts across 12 of the paper's 32 cells.
- `descriptions_multi_axis.json`: 800 prompts across 8 of the paper's 32 cells.

They are incomplete examples and must not be used as the paper-aligned Stage 2 experiment. Build each model's 6,400 Stage 2 prompts with `build_prompts.py stage2` after classifying all 6,900 canonical Stage 1 prompts in `descriptions/`.
