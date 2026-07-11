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

## Next action

Commit and push the frozen design, then run full row scoring and obey P0.
