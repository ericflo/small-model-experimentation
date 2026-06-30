# Qwen3.5-4B Thinking-Budget Controller Report

## Summary

Following the thinking-budget sweep (which found native thinking lifts deployable MBPP greedy
pass@1 +15pp, with an overthinking optimum ~1024 and uneven per-task value), we ask whether a
**deployable controller** can allocate the thinking budget better than a fixed one. Offline, over
the sweep's stored greedy answers at every budget, we simulate visible-test escalation controllers
(draft → if the answer fails the one visible assert, escalate to more thinking) and plot deployable
full-test accuracy vs mean thinking-token cost. Result: the controller **Pareto-dominates every
fixed budget except the peak** — it matches think_256/512 accuracy (~0.88) at ¼–½ the cost
(113–317 vs 246–404 mean thinking tokens) — but it does **not** beat the best fixed budget
(think_1024, 0.91); it trades ~2pp accuracy for a large cost cut. The gap to the non-deployable
oracle (0.93) is bounded by visible-test false-passes (~8–11%), a concrete instance of C2.

## Research Program Fit

Second experiment of `test_time_reasoning_budget`. The sweep established that the thinking budget
is a deployable lever with an overthinking cost; this experiment asks the deployment question —
how to spend that budget — and grounds the program's controller line with a measured Pareto and a
clear ceiling (the oracle and the C2-bounded false-pass rate).

## Method

- **Offline / reuse:** no new generation. We copy the greedy generations (one per task per budget)
  from `qwen35_4b_thinking_budget_scaling` into `data/`; each already has its full-test pass.
- **New signal:** we re-verify the **visible** test (the first assert, shown in the prompt) for each
  stored greedy answer in a sandboxed subprocess — the only signal a deployable controller may read.
- **Strategies:** fixed budgets (points); visible-test **escalation ladders** (ascending budgets;
  commit at the first rung whose answer passes the visible test, else the last rung); a 2-tier
  `no_think → 1024` variant; cumulative-cost (re-generate each rung) vs continue-cost accounting;
  and a non-deployable **oracle ceiling** (per task, cheapest budget that full-passes).
- **Metrics:** deployable full-test accuracy; mean thinking tokens (greedy); `false-visible-commit`
  (committed on a visible pass but full-fails). n=100 MBPP test tasks.

## Results

| strategy | deployable acc | mean think tok | false-visible-commit |
| --- | ---: | ---: | ---: |
| fixed: no_think | 0.760 | 0 | 0.08 |
| fixed: think_256 | 0.870 | 246 | 0.07 |
| fixed: think_512 | 0.870 | 404 | 0.07 |
| **fixed: think_1024** | **0.910** | 507 | 0.06 |
| fixed: think_2048 | 0.860 | 596 | 0.05 |
| fixed: think_unbudgeted | 0.840 | 473 | 0.05 |
| esc[256→512→1024] | 0.890 | 317 | 0.08 |
| esc[256→512→1024] +continue | 0.890 | 277 | 0.08 |
| esc[256→512→1024→2048] | 0.890 | 359 | 0.08 |
| esc[no_think→256→512→1024] | 0.870 | 81 | 0.11 |
| **esc[no_think→1024] (2-tier)** | 0.880 | 113 | 0.10 |
| ORACLE ceiling (non-deployable) | 0.930 | 132 | – |

Figure: `analysis/pareto.png`. (Mean-thinking-token values use the greedy generations and differ
slightly from the sweep report's sampled means.)

- **Efficiency Pareto win.** The deployable frontier is {2-tier 0.88@113, esc[256→512→1024]
  0.89@317, fixed think_1024 0.91@507}. The escalation controllers dominate fixed think_256, 512,
  2048, and unbudgeted (same-or-better accuracy at lower cost). The 2-tier rule reaches 0.88 at
  113 mean thinking tokens — ~22% of fixed think_1024's cost — losing only ~3pp accuracy.
- **No peak-accuracy win.** No deployable controller exceeds fixed think_1024 (0.91). Notably,
  think_1024 alone is already close to the oracle (0.91 vs 0.93), so a fixed near-optimal budget
  captures most of the achievable accuracy; the controller's lever is cost.
- **Escalation is cheap because most tasks stop early.** Cumulative vs continue cost differs little
  (e.g. 317 vs 277), so few tasks pay for multiple rungs.

## Controls

Cumulative-cost (re-generate each escalated rung — faithful to budget forcing) vs continue-cost
(only the committed rung — an optimistic "keep thinking" bound) bracket the true deployment cost;
both leave the controllers Pareto-dominant on the cheap frontier. The oracle ceiling controls for
"how much is achievable by *any* per-task budget choice."

## Oracle Versus Deployable Evidence

Deployable strategies read only the visible assert; full-test accuracy and the oracle ceiling use
hidden asserts and are labelled non-deployable. The decisive deployable-vs-oracle quantity is
`false-visible-commit` (~8–11%): tasks the controller commits because the visible test passes but
which fail the hidden tests. This caps deployable accuracy ~2–4pp below the oracle and is the
reasoning-budget instance of the corpus's C2 bottleneck (a visible pass is not a correct answer).

## Interpretation

The thinking budget is a real *efficiency* knob: a trivial draft-then-escalate rule gets most of
thinking's deployable benefit far more cheaply than any fixed budget. But it cannot beat the best
fixed budget's accuracy, because its only deployable signal sometimes passes on wrong answers (C2),
and because a near-optimal fixed budget is already close to the oracle. So the controller's value is
cost reduction at near-iso-accuracy, not a new accuracy frontier.

## Next Experiments

- **Learned controller with richer visible signals** (token entropy/logprob, self-consistency across
  2 cheap samples) — can it close the 0.89→0.93 oracle gap that the single visible test leaves?
- A latency/wall-clock cost axis (not just token count), connecting to the on-device program.
- The same controller on a harder substrate where the optimum and the C2 false-pass rate are larger.

## Artifact Manifest

See `artifact_manifest.yaml`. Offline experiment; inputs reused from the sibling sweep (copied into
`data/`); no external model call at run time.
