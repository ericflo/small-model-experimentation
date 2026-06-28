# Experiment Log

## 2026-06-26

- Created standalone experiment directory with separate large-artifact storage.
- Scope: offline hard-negative coverage DPO for Qwen3.5-4B on MBPP code generation.
- Primary gate: a DPO adapter must beat tuned-hot base sampling at matched K on held-out coverage without collapsing pass@1 proxy or functional diversity.
- Mechanism controls: shuffled-pair DPO and positive-only SFT if pair mining produces enough usable data.
- Stop condition: if pair mining cannot produce a nontrivial task-diverse pair set, record that as the gate result rather than training a degenerate adapter.
- Smoke sampling on 2 train tasks passed: records, candidate execution, and manifests were written correctly.
- First pair-mining pilot on 12 train tasks at K=8 produced only 7 pairs across 3 tasks, below the training gate. Expanding the mining pool before training.
- Expanded mining with 24 additional train tasks at K=12. Combined pool has 58 preference pairs across 20 tasks, which clears the pilot training gate.
- Trained three pilot adapters in large-artifact storage: real hard-negative DPO, shuffled-pair DPO control, and positive-only SFT control.
- Held-out K=4 evaluation:
  - base hot: 62.5% coverage, 50.0% pass@1 proxy.
  - aggressive hard-negative DPO: 0.0% coverage, 0 parse successes/task.
  - aggressive shuffled DPO: 0.0% coverage, 0 parse successes/task.
  - positive-only SFT: 54.2% coverage.
- Rescue iteration:
  - conservative hard-negative DPO (10 steps, lower LR/beta): 66.7% coverage, matching base hot K=8 at less than half the forward tokens, but pass@1 dropped to 37.5%.
  - conservative shuffled DPO: 58.3% coverage.
  - Formal gate fails due pass@1 regression, but the conservative DPO arm gives a weak coverage-efficiency signal worth only a multi-seed/pass@1-regularized follow-up.
