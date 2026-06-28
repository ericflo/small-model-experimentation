# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory for structural latent compiler expansion.
- Motivation: prior beam/search runs were demoted from candidate method to diagnostic because the deployable learned selector repeatedly failed to close the oracle gap. This run removes beam search from the method.
- Planned run sequence:
  - Smoke: verify the real Qwen QLoRA path, expandable compiler, metrics, checkpoint routing, and report generation.
  - Pilot: short staged expansion run to check whether the objective moves held-out executable accuracy.
  - Main: longer staged expansion run with selected metrics and standalone Markdown/HTML report.

### Smoke Run

- Run: `smoke_qwen_structural_expansion`
- Configuration: Qwen/Qwen3-4B QLoRA, stages 8 -> 16 -> 24, one update per stage, tiny eval sets.
- Outcome: completed successfully. This verified the real model load, latent register collation, compiler expansion, differentiable executor, CSV/JSON logging, and Markdown/HTML report generation.
- Interpretation: accuracy is not meaningful at this size; the smoke was only an integration gate.

### Pilot Run

- Run: `pilot_qwen_structural_expansion_s90`
- Configuration: Qwen/Qwen3-4B QLoRA, stages 8 -> 16 -> 24, 30 updates per stage, weak default trace/state weights.
- Outcome: completed successfully but did not learn executable programs. Final exact program recovery was 0% on all splits and final length-24 executable accuracy remained at 0% on standard/paraphrase/paired splits.
- Interpretation: the structural path itself was not falsified; the objective was underweighted and undertrained compared with prior single-compiler recipes. The main run was adjusted to use heavier init/argument trace weights, full state loss, larger compiler width, batch 8, and a longer short-stage bootstrap.

### Main Run

- Run: `main_qwen_structural_expansion_s750`
- Configuration: Qwen/Qwen3-4B QLoRA, stage expansion 8 -> 16 -> 24, steps 300/150/300, train lengths 1..8 then 1..16 then 8..24.
- Outcome:
  - After stage 1: length-8 executable/program exact accuracy was 100.0% standard, 98.4% paraphrase, 100.0% paired.
  - After stage 2: length-16 executable/program exact accuracy was 98.4% standard, 98.4% paraphrase, 100.0% paired, while length-8 retention was 100.0%.
  - After stage 3: length-8 and length-16 retention were 100.0%; length-24 executable/program exact accuracy was 82.8% standard, 100.0% paraphrase, 93.8% paired.
  - Final length-24 state-prefix recovery was 97.9% standard, 100.0% paraphrase, 99.2% paired.
- Interpretation: this is a clean positive for direct structural expansion of an executable latent compiler without beam search, candidate reranking, or tokenized program output. The remaining weakness is the standard length-24 split, where exact recovery is materially below the paraphrase and paired splits despite very high prefix recovery.

### Reports

- Markdown: `reports/structural_latent_compiler_expansion_report.md`
- HTML: `reports/structural_latent_compiler_expansion_report.html`
- Figures: `reports/figures/`
