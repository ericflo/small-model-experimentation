# Test-Time Reasoning Budget

## Purpose

Study the native thinking-token (reasoning) budget as a first-class, controllable
test-time-compute axis for Qwen3.5-4B. The imported 155-experiment corpus universally
disabled this axis: `enable_thinking=False` appears 48 times in the code and `True`
zero times, except a single fixed-768-token CoT baseline that was never swept. Every
"budget" the corpus studies is an *external* one — evidence probes, tool calls, program
portfolios, sample counts. None is the model's *internal* reasoning-token budget.

This program asks what happens when the 4B is allowed to think, and treats the amount
of thinking as a dial to be measured, controlled, and ultimately learned.

## Why This Is A Program (Not A Variant)

- **Distinct compute axis.** `process_control_and_tool_use` budgets *observations of the
  world* (probes, tools); this program budgets *the model's own latent reasoning*. They
  trade off against each other but are mechanistically different and scale differently.
- **It hosts many experiments.** Scaling curves; oracle-ceiling vs deployable-line
  decomposition; a STOP/MORE controller over thinking tokens; distillation of long
  thinking into short/no-think; thinking-as-verifier vs thinking-as-generator; whether
  more thinking rescues the corpus's length-generalization failures.
- **It produces positive and negative knowledge.** A clean negative ("thinking only
  raises cost, not the deployable line") is as valuable as a positive, and directly
  pressure-tests the corpus's implicit "silent latent compute beats CoT" thesis
  (`qwen_python_shaped_silent_executor`).
- **Existing programs would hide the uncertainty.** Folded into posttraining or process
  control, the reasoning-budget question disappears; it deserves its own evidence ledger.

## Progress Signals

The line is advancing when we can answer, with controls:

- Does more thinking raise the **oracle ceiling** (pass@k), the **deployable line**
  (greedy / visible-selected pass@1), or **only cost**? (The corpus's oracle-vs-deployable
  framing, applied to thinking for the first time.)
- Does the **content** of thinking matter, or only the `<think>` scaffold + extra compute?
  (Shuffled/truncated-thinking controls.)
- Is there a **budget sweet spot**, and does it depend on task difficulty / length?
- Can a deployable **controller** allocate thinking tokens better than a fixed budget?
- Does thinking **transfer** the corpus's confirmed bottleneck (C2: coverage ≫ deployable
  selection) — i.e., does thinking widen or close the selection gap?

## Boundaries

- Not about external evidence/tool budgets (that is `process_control_and_tool_use`).
- Not about new model architectures; the substrate model is fixed at Qwen3.5-4B.
- Training-based methods (controller, distillation) belong here only insofar as they
  manage the reasoning budget; generic posttraining belongs in `posttraining_and_adaptation`.

## Anchors

- Seed/contrast: [`qwen_python_shaped_silent_executor`](../../experiments/qwen_python_shaped_silent_executor/reports/qwen_python_shaped_silent_executor_report.md)
  — the one corpus run that enabled thinking (62.5% at len-4, 0% at len-24, fixed 768-token budget, never swept).
- First experiment: [`qwen35_4b_thinking_budget_scaling`](../../experiments/qwen35_4b_thinking_budget_scaling/README.md).
