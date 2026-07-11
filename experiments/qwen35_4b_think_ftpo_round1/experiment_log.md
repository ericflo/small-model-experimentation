# Think-block FTPO round 1: outcome-conditioned pivot steering — Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-10 — design v1 → v2 (before any full-scale GPU spend)

- v1 centered on loop-repair FTPO (published-replication). Two forcing events:
  (1) user redirection toward outcome-conditioned pivot steering ("capability
  elicitation, leading the thinking to more fruitful places", not suppression of
  known-bad patterns); (2) the adversarial design review's blocking finding —
  the mining detector flags ~0.1% of existing greedy base completions at deployed
  budgets (independently verified: 1/1200 atoms think@1024, 0/786 episode turns).
  Loops live at 16k+, not at agentic budgets.
- v2: pivot arm primary (prefix-tree divergence mining over n=8 verifier-scored
  rollouts), pivot-shuffled label-permutation control, loop arm descoped to the
  zero-GPU census artifact + queued long-context follow-up. Decision layer
  recalibrated (2+1 null protocol, 3 quick seeds, null-scaled thresholds,
  conditional medium, dose-sufficiency precondition for NEGATIVE). Full finding
  dispositions: reports/design_review.md.
- Environments bootstrapped on the fresh pod: .venv-vllm (pinned lock, smoke
  passed), .venv (torch 2.11.0+cu129 / transformers 5.13.0 / peft 0.19.1),
  pinned model downloaded.

## 2026-07-11 — round-1 execution and verdict

- Harvest: 7 slices, 2,800 prompts, 7.7h (amendment 4 sized the extension from
  measured yield). P0 PASS (25.6% eligible / 49.5% mixed); 879-row pool → 615
  training rows.
- Training: both arms 39 steps / 13 min each; padding-equivalence gate failed
  (0.30–0.44 logits, hybrid arch) → batch-of-1 mode as preregistered; both
  C49 merge gates PASS.
- Eval battery: 15/15 stages OK. P1 FAIL (−0.039/−0.076 vs +0.05 bar);
  shuffled control degrades identically → generic-regime damage; collapse and
  no-think guards clean; gym guard fail for pivot. Menagerie correctly NOT run
  (mechanism-gate rule) — zero benchmark seeds consumed.
- Verdict: training-recipe failure with a clean mechanism story (attractor
  precondition). Round-2 levers queued; full report in reports/report.md.
