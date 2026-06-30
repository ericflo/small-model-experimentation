# Evidence

## Seed Experiments

- [`qwen_python_shaped_silent_executor`](../../experiments/qwen_python_shaped_silent_executor/reports/qwen_python_shaped_silent_executor_report.md):
  the only corpus experiment that ever enabled native thinking. Its CoT baseline reached
  62.5% on length-4 programs (746 emitted tokens) but collapsed to 0% on length-24/32 at a
  **fixed ~768-token thinking budget that was never swept**. It was framed as a foil for
  "silent latent compute" (which was itself a controlled negative). This is the contrast the
  program exists to revisit: was the collapse a reasoning limit or a *budget* limit?

## Corpus-Wide Fact (verified)

- Across all 155 experiments, native thinking is disabled (`enable_thinking=False` ×48,
  `True` ×0 besides the one seed); `<think>` blocks are stripped as boilerplate. "Budget"
  always means evidence/probe/tool/program/sample budget, never reasoning tokens. So the
  reasoning-budget axis is genuine, verified white space.

## Anchor Experiments

- [`qwen35_4b_thinking_budget_scaling`](../../experiments/qwen35_4b_thinking_budget_scaling/reports/report.md)
  (n=100 MBPP test, k=8; numbers independently recomputed from raw data and audited) — the sweep.
- [`qwen35_4b_thinking_budget_controller`](../../experiments/qwen35_4b_thinking_budget_controller/reports/report.md)
  (offline, reuses the sweep's greedy answers) — the deployable controller.
- [`qwen35_4b_thinking_separability_probe`](../../experiments/qwen35_4b_thinking_separability_probe/reports/report.md)
  (per-layer linear probes on answer-token activations) — the interpretability/internal-signal angle.
- [`qwen35_4b_thinking_content_vs_compute`](../../experiments/qwen35_4b_thinking_content_vs_compute/reports/report.md)
  (foreign-task-thinking ladder) — the decisive content control.

## Confirmed Claims

- **Native thinking is a deployable win the corpus disabled.** Greedy pass@1 0.76 → 0.91 (+15pp);
  the deployable line moves *more* than the oracle ceiling (pass@8 0.91 → 0.96) and the
  oracle−deployable gap *narrows* — the opposite of the C2-based prior. Paired-robust (17 fail→pass
  vs 2 pass→fail at think_1024 vs no_think; McNemar p≈0.001). So C2 (coverage ≫ deployable
  selection) does **not** hold for the thinking axis on MBPP.
- **More thinking is not monotonically better.** Broad optimum ~512–1024 tokens then decline;
  `unbudgeted` (greedy 0.84) is worse than a cap (0.91). Shape corroborated by greedy and pass@1.
- **A visible-test budget controller is an efficiency win, not an accuracy win.** A draft→escalate
  rule Pareto-dominates every fixed budget except the peak (matches ~0.88 at 113 mean thinking
  tokens vs fixed think_256/512 at 246–404), but cannot beat the best fixed budget (think_1024,
  0.91). Its deployable gap to the oracle ceiling (0.93) is bounded by visible-test false-passes
  (~8–11%) — the C2 effect made concrete on the thinking axis.

## Negative / Cautionary Findings

- **Much of the "thinking" benefit is not coherent reasoning.** A shuffled-thinking control
  (scramble the model's own thinking tokens, keep count/scaffold) reproduces most of the gain;
  at 2048 shuffled = real. Evidence that coherent reasoning *order* adds beyond compute + scaffold
  + token-presence is weak and budget-dependent. Needs a stronger control (substitute a different
  task's thinking) before claiming the gain is "reasoning."
- The exact optimum (1024) and the never-solved-bucket effect (3/9 tasks) rest on small n /
  single-seed; treat as suggestive, not pinned.
- **The model uses thinking as CONTENT (separability probe + foreign control).** Linear probes show
  correctness is moderately decodable from the answer-token activation (AUC 0.64–0.76). The
  foreign-task-thinking ladder is decisive: splicing a *different* task's thinking collapses accuracy to
  ~4% (the model follows it to the wrong problem) — so thinking is not a content-free compute/scaffold
  crutch. Weak deployable spinoff: the probe partially flags C2 false-passes (visible-passer AUC
  ~0.60–0.68) only under thinking.

## Correction

- **"Thinking is mostly compute/scaffold, not reasoning" was overstated.** The foreign ladder shows
  the efficient-budget behavioral gain IS coherent reasoning over relevant content: foreign (irrelevant)
  collapses to 0.04, shuffle (relevant, scrambled) ≈ no_think (0.74 vs 0.76 on sampled full-pass), real
  (coherent) 0.86. The "mostly compute" read was a greedy-metric artifact (greedy shuffle recovered ~⅓,
  sampled ~0), held mainly at high budgets (2048 shuffle ≈ real, overthinking), and at the
  representational level (separability differences small/noisy). See claim C9 (corrected).

## Current Read

Turning thinking on is a real, cheap deployable lever the corpus left unused — but it is a
*budget* to be controlled, and the controller experiment shows the budget knob is mostly an
**efficiency** lever (near-iso-accuracy at much lower cost), not a new accuracy frontier: a
near-optimal fixed budget already sits close to the oracle, and the deployable controller is
capped by C2 false-passes. The foreign control then refined the *nature* of the gain: at the
**efficient budget** the behavioral gain **is coherent reasoning over relevant content** (the model
uses thinking as content — irrelevant thinking is catastrophic), though that advantage washes out at
high budgets (overthinking) and is not clearly reflected in internal correctness-decodability. Honest
read: *use thinking, cap it, and a cheap controller buys back most of the cost; at the efficient budget
the gain is genuine reasoning (don't dismiss it as mere compute) — but it is budget-dependent and
behavioral ≠ representational.* Priority follow-ups: the **filler/pause-token arm** (isolate pure
compute, since foreign adds misleading content not contentless compute); the foreign/shuffle/real
ladder at a high budget; a learned controller with richer visible signals; harder/contamination-
controlled substrates.
