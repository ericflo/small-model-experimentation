# Experiment Log

## 2026-06-24

- Created a fresh standalone attribution-ablation experiment directory.
- Core question: does the long-program compiler result come from structural slot expansion, from curriculum/objective tuning, or from general training budget?
- Required controls:
  - `expand_copy`: staged 8 -> 16 -> 24 compiler with copied learned slots.
  - `max24_curriculum`: max-24 compiler from the start, same staged training lengths and budget.
  - `expand_random_new_slots`: staged compiler, but newly introduced slots are randomly initialized instead of copied from the last learned slot.
  - `max24_no_curriculum`: max-24 compiler from the start, trained directly on lengths 1..24.
  - `train_to16_eval24`: max-24 compiler trained only on lengths up to 16, evaluated on length 24.
- Added held-out wording templates that are not sampled by `mixed` training mode.
- Planned process:
  - Smoke: verify all new split names and expansion modes.
  - Pilot: short controlled arms to verify the suite runs and exposes expected floor/learning behavior.
  - Main: full ablation arms with Markdown/HTML report and charts.

### Smoke

- Run: `smoke_random_new_slots`
- Configuration: Qwen/Qwen3-4B QLoRA, one update per stage, random-new-slot expansion, tiny eval.
- Outcome: completed successfully.
- Verified:
  - `random_new` expansion mode.
  - Held-out wording splits: `heldout_L*`.
  - Standard-vs-heldout paired splits: `paired_heldout_L*`.
  - Markdown/HTML report generation.

### Pilot

- Runs:
  - `pilot_expand_copy`
  - `pilot_max24_curriculum`
  - `pilot_expand_random_new_slots`
  - `pilot_max24_no_curriculum`
  - `pilot_train_to16_eval24`
- Configuration: reduced-width compiler, small datasets, short budgets, no checkpoints.
- Outcome: all five controls completed and produced comparable metrics. As expected at this tiny budget, all length-24 final program-exact metrics stayed at 0%.
- Interpretation: the pilot was a harness validation, not a result-bearing run. Proceeding to full ablation with tuned width, loss weights, batch size, evaluation size, and checkpoints.

### Main Ablation

- Runs:
  - `main_expand_copy_s750`
  - `main_max24_curriculum_s750`
  - `main_expand_random_new_slots_s750`
  - `main_max24_no_curriculum_s750`
  - `main_train_to16_eval24_s750`
- Configuration: Qwen/Qwen3-4B QLoRA, 512-wide compiler, LoRA rank 8, batch 8, strong init/argument trace supervision, full state loss, 64 examples per single-template split, 32 paired examples per paired split.
- Main length-24 executable accuracy:
  - `max24_curriculum`: standard 96.9%, heldout 46.9%, paired 89.1%, paired-heldout 78.1%.
  - `expand_copy`: standard 1.6%, heldout 51.6%, paired 29.7%, paired-heldout 25.0%.
  - `expand_random_new_slots`: standard 1.6%, heldout 6.2%, paired 14.1%, paired-heldout 1.6%.
  - `max24_no_curriculum`: standard 4.7%, heldout 18.8%, paired 9.4%, paired-heldout 7.8%.
  - `train_to16_eval24`: standard 0.0%, heldout 1.6%, paired 1.6%, paired-heldout 0.0%.
- Interpretation:
  - Copied structural expansion is not the winning explanation in this run.
  - The best explanation is a max-24 compiler trained from the start with a staged length curriculum.
  - Randomly initialized new expansion slots perform poorly at length 24.
  - Removing curriculum fails at length 24 despite strong length-8 and length-16 performance.
  - Training only through length 16 does not extrapolate to length 24.
  - Held-out wording is much harder than seen-family paraphrase, even for the best arm.

### Reports

- Markdown: `reports/structural_compiler_attribution_ablation_report.md`
- HTML: `reports/structural_compiler_attribution_ablation_report.html`
- Figures: `reports/figures/`
