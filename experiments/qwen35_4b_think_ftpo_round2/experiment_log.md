# Entropy-routed think-pivot optimization round 2 — experiment log

## Design and smoke (before scientific run)

- Routed to `agentic_breadth_installation`; closest duplicate is round 1/C52.
- Exploratory replay of the already-regularized real rows found 290/615 failed
  tokens were base argmax and 172 led their successful sibling by ≥0.5 logits.
  Entropy and varentropy expose a spiky-conflicted subset; thresholds were then
  frozen before scoring the shuffled pool or training.
- Uncapping prefix-tree nodes yielded only 885 raw nodes vs 879 previously, so
  round 2 is explicitly a low-dose controlled pilot rather than a new harvest.
- CPU invariants pass for all repository families, sandbox paths, parser, and
  both objectives. Exact-logit and vLLM GPU paths loaded successfully.
- The initial one-operator repair tasks saturated (base patched 6/6) and were
  rejected during smoke. Replaced with semantic multi-line maintenance faults;
  tiny calibration: deep base final-workspace 2/6, matched two-by-four-turn
  baseline 1/6. These adaptive smoke items are not scientific evidence.

## Frozen design checkpoint

- Design, code, smoke receipts, and preregistration were committed before the
  scientific run as `c6480dee` and pushed to `agent/think-ftpo-round2`.
- Full command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python3 \
    experiments/qwen35_4b_think_ftpo_round2/scripts/run.py --full \
    --artifact-root \
    /workspace/small-model-experimentation/large_artifacts/qwen35_4b_think_ftpo_round2
  ```

## Full run (2026-07-11)

- P0 passed: 155/615 real rows and 166/661 shuffled rows met the frozen
  confident-wrong-turn/entropy/varentropy geometry; seeded matching retained
  155 rows per arm (minimum 128).
- Training safety-stopped as registered: demote at 8/20 steps (165 s), uplift
  at 5/20 (104 s), and shuffled uplift at 5/20 (103 s).
- P1 failed for every arm. Objective hit/non-target drift were 40.0%/0.229
  logits (demote), 75.5%/0.145 (uplift), and 76.1%/0.120 (shuffled); the frozen
  bars were ≥35% and ≤0.10. Uplift was 36.6% less collateral than demotion but
  did not localize the update.
- Entropy was not monotone with safety. Uplift entropy-quartile drift was
  0.163/0.134/0.137/0.147. Varentropy was more diagnostic but ran opposite the
  naive “more conflict is better” read: Q1 drift 0.122 was cleanest, Q3 drift
  0.176 was worst. These strata are explanatory only, not a post-hoc selector.
- P2 failed. Fresh paired whitebox uplift-minus-base was +0.26pp at 1024
  (95% CI −3.57,+4.08) and −3.06pp at 2048 (−7.40,+1.28). Uplift-minus-
  shuffled changed sign (+2.55pp/−2.04pp). At 2048, natural closure improved
  +4.08pp but answer-limit contacts worsened +3.32pp, failing the termination
  guard.
- P3 failed. On 72 hidden-tested repository repairs, deep base passed 43,
  demote 34, uplift 39, and shuffled uplift 29. Paired uplift-minus-base was
  −5.56pp (95% CI −19.44,+8.33); uplift-minus-shuffled was +13.89pp with the
  interval touching zero (0.00,+27.78). The matched two-by-four-turn base
  baseline passed 22; uplift beat it by +23.61pp, but the registered north star
  also required beating the stronger deep base.
- P4 passed for all trained arms: merged C49 on/off, gym floor, collapse
  greedy/pass@8, and no-think guards were clean. Gym aggregate was 48.55%
  base, 49.29% demote, 53.27% uplift, and 47.02% shuffled.
- Final registered label: `LOW_DOSE_NULL`. P5 was ineligible; menagerie was not
  run and no blackbox seeds were consumed.

## Protocol deviation

The whitebox plan named N=400, while the frozen per-cell integer allocator
materialized 392 paired tasks (98 per cell), a 2% shortfall. No outcomes were
inspected or filtered and every materialized task was retained. The effective
N and exact paired intervals are reported rather than silently rounding to the
nominal target.

## Read and next decision

Outcome labels carried a substrate-local direction signal relative to the
shuffled control, but shared-weight collateral exceeded it. Confident-outlier
geometry is necessary but not sufficient for FTPO. Do not scale this LoRA
recipe or select higher-varentropy rows. First test a lower +0.25 uplift or a
genuinely context-gated last-layer/activation edit behind the same P1 locality
gate; only then fund a larger fresh harvest and agentic transfer run.
