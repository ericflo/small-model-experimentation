# Qwen3.5-4B Thinking-Budget Controller

**Status:** finished

## Research Program

- Program: `test_time_reasoning_budget`
- Program question: can a deployable controller allocate the thinking-token budget better than
  a fixed budget, given thinking has an overthinking cost and uneven per-task value?
- Prior anchors: [`qwen35_4b_thinking_budget_scaling`](../qwen35_4b_thinking_budget_scaling/reports/report.md)
  (the sweep that found the +15pp deployable gain, the ~1024 optimum, and the overthinking decline);
  [`qwen35_4b_adaptive_evidence_budget_policy`](../qwen35_4b_adaptive_evidence_budget_policy/) (the
  STOP/MORE analog over *evidence* probes rather than thinking tokens).

## Question

Can a visible-signal controller that decides *how much to think per task* beat fixed thinking
budgets on the deployable accuracy-vs-cost (mean thinking tokens) Pareto?

## Hypothesis

The sweep showed thinking's value is uneven (easy tasks barely benefit; hard tasks need the full
budget) and that overthinking hurts. So a controller that thinks little by default and escalates
only when a cheap visible signal says the answer is wrong should match a fixed budget's accuracy
at much lower mean cost. The natural deployable signal is whether a draft answer passes the
**visible test** (the one assert shown in the prompt).

## Setup

- Model: Qwen3.5-4B (results reused; this experiment is **offline**, no new generation).
- Dataset/task source: MBPP sanitized `test` split, 100 tasks — the same greedy generations at
  each thinking budget produced by `qwen35_4b_thinking_budget_scaling`, copied into `data/`.
- Train/eval split: none — strategies are fixed rules (no learned parameters); evaluated on all 100.
- Baselines: every fixed budget (no_think, 256, 512, 1024, 2048, unbudgeted) as Pareto points.
- Controls: cumulative-cost (re-generate at each rung) vs continue-cost accounting; random/oracle.
- Primary metric (deployable): full-test accuracy vs **mean thinking tokens** (Pareto), using only
  the visible test as the escalation signal.
- Oracle-only metric: per-task cheapest full-passing budget → a non-deployable accuracy/cost ceiling.
- Hidden-label boundary: controllers may read only the **visible** test (first assert); full-test
  pass and the oracle ceiling use hidden asserts and are reported separately. `false-visible-commit`
  = fraction of tasks committed on a visible pass that actually fail the full test (the C2 risk).

## Run

Smoke (re-verify visible tests + load):

```bash
../../.venv/bin/python scripts/run.py --smoke
```

Full (offline; ~1–2 min to re-verify 600 stored answers, then simulate):

```bash
../../.venv/bin/python scripts/run.py            # re-verifies visible tests
../../.venv/bin/python scripts/run.py --no-reverify   # reuse cached visible-test results
```

## Results

Full table in [reports/report.md](reports/report.md); figure `analysis/pareto.png`. Headline
(deployable full-test accuracy @ mean thinking tokens):

| strategy | acc | think tok |
| --- | ---: | ---: |
| fixed think_256 | 0.870 | 246 |
| fixed think_512 | 0.870 | 404 |
| fixed think_1024 (best fixed) | 0.910 | 507 |
| fixed think_2048 | 0.860 | 596 |
| **esc[no_think→1024] (2-tier)** | 0.880 | **113** |
| **esc[256→512→1024]** | 0.890 | 317 |
| oracle ceiling (non-deployable) | 0.930 | 132 |

- **Efficiency win:** the visible-test escalation controller **Pareto-dominates** every fixed
  budget except the peak — it matches think_256/512 accuracy (~0.88) at **¼–½ the thinking cost**
  (113–317 vs 246–404 tokens).
- **Not an accuracy win:** it does *not* beat the best fixed budget (think_1024, 0.91); it trades
  ~2pp accuracy for a large cost cut. If only peak accuracy matters, fixed ~1024 wins.
- **Bounded by C2:** the gap to the oracle (0.93) is set by visible-test false-passes
  (false-visible-commit ~8–11%) — the visible test is a decent but imperfect signal.

## Interpretation

The reasoning budget is a worthwhile *efficiency* knob: a trivial draft-then-escalate rule gets
most of thinking's deployable benefit far more cheaply than any fixed budget, but it cannot exceed
the best fixed budget's accuracy because the only deployable signal (the visible test) sometimes
passes on wrong answers (C2). The headroom to the oracle (0.91→0.93) is small, so the lever here is
cost, not peak accuracy; a *learned* controller's job would be to push toward the oracle by reading
richer visible signals (token entropy, self-consistency) than a single visible test.

## Knowledgebase Update

- Program evidence updated: yes (`research_programs/test_time_reasoning_budget/evidence.md`).
- Program backlog updated: yes (learned controller with richer visible signals).
- Claim ledger updated: C9 extended (controller is an efficiency win, not an accuracy win; bounded by C2).

## Artifacts

- `src/controller.py` simulation + visible-test verifier; `scripts/run.py` runner.
- `data/greedy_records.jsonl`, `data/tasks.json` (copied from the sibling sweep; self-contained),
  `data/greedy_with_visible.jsonl` (cached visible-test results).
- `runs/summary.json`; `analysis/pareto_table.md`, `analysis/pareto.png`.
- `reports/report.md`, `reports/artifact_manifest.yaml`.
