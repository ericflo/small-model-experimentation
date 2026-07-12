# Backlog

## Done

- `qwen35_4b_thinking_budget_scaling`: MBPP scaling curve; oracle-vs-deployable decomposition;
  shuffled-thinking control. (+15pp greedy, greedy overthinking optimum ~1024. NOTE: its "gain is mostly
  compute/scaffold" and "2048 shuffle ≈ real" readings were later corrected — see the content ladders.)
- `qwen35_4b_thinking_budget_controller`: fixed-rule visible-test escalation controller — an
  efficiency win (Pareto-dominates fixed budgets except the peak), bounded by C2 false-passes.
- `qwen35_4b_thinking_separability_probe`: per-layer probes on answer-token activations. Correctness
  is moderately decodable (AUC 0.64–0.76); thinking raises decodability; shuffled ≈ real in
  decodability (representational side is noisy across experiments). Weak probe signal on C2 false-passes.
- `qwen35_4b_thinking_content_vs_compute`: full content ladder (no_think / **filler** / shuffle / real /
  foreign). Complete attribution at budget 512: pure compute (filler) ≈ 0, token-presence (shuffle) ≈ 0,
  coherent content = the entire +0.122, misleading content (foreign) −0.71 (the model follows it to the
  wrong problem). **Conclusively corrected** the earlier "mostly compute/scaffold" claim — pure compute
  buys nothing; the efficient-budget gain is 100% coherent reasoning content.
- `qwen35_4b_overthinking_content_ladder`: the content ladder across budgets {512,1024,2048}. The
  coherence advantage (real − shuffle) **grows** with budget (+0.105 → +0.108 → +0.150), refuting the
  overthinking-washout hypothesis; pure compute (filler) ≈ no-think and foreign catastrophic at every
  budget. So coherent reasoning is the entire gain at ALL budgets; the scaling run's "2048 shuffle ≈ real"
  was a shuffle-protocol artifact.
- `qwen35_4b_answer_potential_trace_sft`: terminal `SCORER_NEGATIVE`. Answer gain had modest real
  within-task signal but missed its actionability bars; 99.37% cap contact and 13.2% answer parsing exposed
  a forced-close deployment-seam mismatch. No N=128 harvest or SFT ran.

## Next Experiments

- Completed cross-program qualification negative:
  `qwen35_4b_pareto_policy_integration` found that C54's short/deep tier labels
  did not become a clean procedural teacher crossover. `blend` lost both quick
  blocks; `apex` won deep capability but missed retention. The visible
  two-checkpoint tier router remains an external upper reference, not a
  one-checkpoint installation result.
- **Close/commit potential follow-up (new experiment only):** compare joint probability of autonomously
  closing and emitting the canonical answer against answer-only potential, after a frozen natural-close and
  parseability calibration gate. Do not reuse C51's observed tasks to tune thresholds.
- **Symmetric loop-control follow-up (next; new experiment):** the exact-capture verified-macro
  ladder is terminal `pass=false`, with no selected budget and no authorized K=12/semantic stage.
  Its clean 61k envelope still produced 40/48 exact loops, 8/48 unresolved contacts, and 4/48
  answer-limit contacts at 397.688 sampled tokens/s; 49k produced 38/10/6 at 491.396 tokens/s.
  Stop increasing context. Preregister one loop-control intervention symmetrically across all arms
  and matched-compute baselines, keep the unresolved-contact and answer-limit gates unchanged, and
  use fresh artifacts. Loop detection is a termination mechanism, not evidence of correctness.
- **Learned thinking-budget controller with richer visible signals** (token entropy/logprob,
  self-consistency): can it close the deployable→oracle gap (0.89→0.93)? (Queue: `thinking_budget_controller`.)
- Thinking-budget sweep on the **silent_executor substrate** (modular-program execution):
  its CoT collapsed to 0% at len-24 at a fixed 768-token budget — does a larger thinking
  budget rescue length generalization? Direct pressure-test of the silent-compute thesis.
- **Matched-compute scratchpad compression at inference time:** periodically ask the same
  Qwen3.5-4B call protocol to replace a long intermediate state with a short verified checkpoint,
  then spend the saved context on continued reasoning. Compare against ordinary sampling at equal
  model-forward tokens; no teacher, distillation, or cross-model trace source is permitted.
- **Thinking as verifier vs generator** under matched token budget: spend the budget on a
  thinking *generator* vs a thinking *verifier/selector* over no-think candidates — which
  buys more deployable accuracy? (Couples to evidence_conditioned_selection / C2.)
- Replicate the sweep + controller on a **harder substrate** (full MBPP / LiveCodeBench / math):
  does the optimum move, does the C2 false-pass rate grow?

## Required Controls

- Baseline: `no_think` (enable_thinking=False) — the corpus's universal setting.
- Mechanism-falsifying control: **shuffled-thinking** (scramble the model's own thinking
  tokens, then force the answer) and **truncated-thinking** — isolates thinking *content*
  from the `<think>` scaffold + extra compute.
- Matched-compute control: compare conditions at equal total forward tokens (the corpus's
  matched-compute methodology), so gains are not just "more tokens".
- Shift check: difficulty / length slices (does the optimal budget move with task length?).
- For trace selection, report natural-close rate and autonomous answer parse rate before scaling the pool.

## Stop Conditions

- Retire/demote if, across substrates and at matched compute, thinking never raises the
  deployable line beyond shuffled-thinking — i.e., only the scaffold + extra compute help.
- Branch a new program if a thinking-budget *controller* becomes a rich line of its own.
- Stop answer-only trace-potential scaling when the thought pool is cap-bound; changing N does not repair
  an injected close/commit seam.

- Experiment: long-context loop-FTPO (the loop-control mandate's intervention
  branch): loops are ~0.1% at deployed budgets but 56–83% of cap contacts at
  32k–61k (C52 census + verified-macro ladder). Mine loop-initiation rows at
  16k+ budgets, train with contexts ≤6k (short-to-long transfer is the open
  question), preregister the symmetric-arms rule from the model playbook.
