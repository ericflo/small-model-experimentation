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
- [`qwen35_4b_answer_potential_trace_sft`](../../experiments/qwen35_4b_answer_potential_trace_sft/reports/report.md)
  (claim C51) — answer-potential selection over sampled thinking, stopped at its scorer gate.
- [`qwen35_4b_native_thought_seam_budget_ladder`](../../experiments/qwen35_4b_native_thought_seam_budget_ladder/reports/report.md)
  (unclaimed) — a fresh natural-close selector that exhausted 256/512/1024 with
  0/48 closes and left confirmation unopened.
- [`qwen35_4b_forced_commit_jacobian_value_transport`](../../experiments/qwen35_4b_forced_commit_jacobian_value_transport/reports/report.md)
  (unclaimed) — close-only forced commit reproduced the low-parse interface wall
  and stopped before internal value.
- [`qwen35_4b_commit_slot_jacobian_value_transport`](../../experiments/qwen35_4b_commit_slot_jacobian_value_transport/reports/report.md)
  (unclaimed; terminal `COMMIT_SLOT_SEAM_FAIL`) — fixed syntax made an alias the
  unmasked next token on 41/48 long traces and exposed a +6.25pp/+8.33pp
  1,024-token hint over no-thought/shuffled controls, but only five of six
  required tasks mixed outcomes and task-level intervals crossed zero. It
  separates a repaired answer mode from still-unconfirmed semantic benefit.
- [`qwen35_4b_think_ftpo_round2`](../../experiments/qwen35_4b_think_ftpo_round2/reports/report.md)
  (claim C52) — entropy/varentropy-routed single-token thought steering, a
  low-dose capability null after exact-logit locality and agentic transfer gates.
- [`qwen35_4b_pareto_policy_integration`](../../experiments/qwen35_4b_pareto_policy_integration/reports/report.md)
  — C54's aggregate short/deep frontier did not transport into a clean
  quick/deep teacher crossover on fresh procedural states; future consolidation
  needs state-local continuation advantage rather than aggregate budget labels.
- [`qwen35_4b_verified_macro_long_context_rerun`](../../experiments/qwen35_4b_verified_macro_long_context_rerun/)
  (contamination-free procedural macro induction) — a workload-shift stress test for budget
  calibration and anti-censoring, still in progress.

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
- **Reasoning budgets do not automatically transfer across workload classes within one
  substrate.** In the
  verified-macro follow-up, a train-only plan-given calibration selected think@16,384 and a
  disjoint plan-given interface passed 16/16 records with zero unresolved caps. The fresh induction
  base at the identical budget then contacted the cap in 144/144 samples; only 13 were exact loops,
  leaving 131/144 unresolved and 60 answer-limit contacts. Doubling the allowance did not clear the
  workload: at think@32,768 all 144 samples again contacted the boundary, with 81 exact loops,
  63 unresolved contacts, and 37 answer-limit contacts. Both rungs were excluded before decoding or
  scoring. A max-seqs-64 K=4 probe at think@49,152 later force-closed all 48 samples (34 loops, 14
  unresolved, 13 answer-limit contacts) while generating 2,366,620 tokens in 4,035.356 seconds
  (586.47 tokens/s), but Amendment 12 had already made it diagnostic-only before its receipt:
  block-rounded demand was 2,433,024 tokens against 995,328 of cache. No decoded or scored content
  was inspected. The independent capacity-fit follow-up then completed a fresh K=4 probe with
  max-seqs 19; its live audit fit 963,072 tokens into a 997,888-token cache, leaving 34,816. All 48
  samples still contacted the boundary (37 exact token-ID loops, 11 unresolved, 9 answer-limit), so
  49k was rejected before decoding or scoring. It sampled 2,364,643 tokens in 5,012.451 seconds
  (471.754 tokens/s), 19.6% slower than the max-seqs-64 diagnostic's 586.471 tokens/s. The subsequent
  61k attempt was stopped before a receipt after an audit found that its implicit CUDA-graph list
  covered only through width 8 rather than max-seqs 15, leaving no reusable rows. A separate
  exact-capture follow-up then passed both live-KV and exact-graph gates at 49k: 963,072 required
  tokens fit into 996,864 live tokens, and `[1, 2, 4, 8, 16, 19]` resolved exactly. Termination still
  failed with 38/48 periodic loops, 10/48 unresolved contacts, and 6/48 answer-limit contacts. That
  probe generated 2,363,163 tokens in 4,809.081 seconds (491.396 tokens/s), descriptively 4.16%
  faster than the implicit-capture capacity-fit probe. The terminal exact-capture 61k probe passed
  the same runtime gates: 950,400 required tokens fit into 997,888 with 47,488 headroom, and FULL
  decode graphs resolved exactly at `[1, 2, 4, 8, 15]`. Termination nevertheless failed with 40/48
  periodic loops, 8/48 unresolved contacts, and 4/48 answer-limit contacts; 2,951,995 sampled tokens
  took 7,422.886 seconds (397.688 tokens/s). The selector ended `pass=false` with no selected
  budget, authorizing no K=12 arm or semantic analysis. Cache-safe concurrency and active-width
  graph coverage are both required for a clean inference envelope, but neither guarantees useful
  termination. This is setup evidence, not a task result: calibrate on the actual workload class,
  gate termination before correctness, stop increasing context once the registered ladder is
  exhausted, and never interpret a cap-bound score. No decoded or scored content informed these
  decisions.
- **Answer-conditioned trace scores can validate an artificial post-thinking state.** In C51, 99.37%
  of 2,048 thoughts hit the 512-token cap. Canonical-answer gain after an injected close contained real
  trace information, but fresh answers parsed only 13.2% and the scorer missed its actionable G0 bars.
  Natural closure and autonomous commit must be launch gates when thinking traces feed selection or SFT.
- **Natural termination is absent through the deployed 1,024 scale on the fresh
  list-composition workload.** The frozen paired selector observed 0/48 closes
  at 256, 512, and 1,024; all rows hit the largest cap and the untouched
  confirmation remained sealed. No exact 1--32-token periodicity occupied any
  final 256-token tail, so this does not join the 16k+ exact-loop line. Do not
  keep raising a natural cap. An external commit action is now a distinct
  deployable policy: valid only if the same forced interface clears parse and
  headroom gates at calibration and deployment, and always labeled
  counterfactual relative to autonomous close.
- **A close token alone does not establish answer mode.** On a new exact-depth
  workload, forced-only parse stayed 12.5%--18.8%, success was 1/48 at every cap,
  and 85%--96% of post-close outputs exhausted 16 answer tokens. Decoded only
  after the automatic failure, many rows restarted analysis. A tolerant parser
  remained <=22.9%, so formatting edge cases do not explain the stop. The next
  controller may supply a fixed answer slot but must treat that syntax as part
  of deployment and retain close-only output as a control.
- **Entropy/varentropy localize interesting forks but do not make weight edits local.** C52 round 2
  selected 155 low-entropy, non-degenerate-varentropy confident wrong turns and compared demotion,
  positive-only uplift, and shuffled uplift. Pull-up was safer and true labels separated from shuffled
  locally, but every LoRA arm exceeded the 0.10 non-target-logit-drift ceiling and held-out coding stayed
  below base (39/72 vs 43/72). Higher varentropy was not monotonically safer; the lowest quartile was
  cleanest. Treat uncertainty as a routing/diagnostic variable, not a correctness label or pressure scale.
- **The model uses thinking as CONTENT (separability probe + foreign control).** Linear probes show
  correctness is moderately decodable from the answer-token activation (AUC 0.64–0.76). The
  foreign-task-thinking ladder is decisive: splicing a *different* task's thinking collapses accuracy to
  ~4% (the model follows it to the wrong problem) — so thinking is not a content-free compute/scaffold
  crutch. Weak deployable spinoff: the probe partially flags C2 false-passes (visible-passer AUC
  ~0.60–0.68) only under thinking.

## Correction

- **"Thinking is mostly compute/scaffold, not reasoning" was wrong for the efficient budget.** The full
  content ladder (pure-compute filler arm included) gives, at budget 512: no_think 0.749, **filler 0.744**,
  shuffle 0.739, **real 0.861**, foreign 0.040. Attribution: pure compute (filler − no_think) **−0.005**;
  relevance (shuffle − filler) **−0.005**; coherent content (real − shuffle) **+0.122**; misleading content
  (foreign) **−0.709**. So pure compute buys ~0 and the efficient-budget gain is **100% coherent reasoning
  content**, which the model uses (foreign → wrong problem). A budget sweep (512/1024/2048) then showed the
  coherence advantage (real − shuffle) does not shrink but **grows** (+0.105 → +0.108 → +0.150) — so this
  holds at **every** budget, and the scaling run's "2048 shuffle ≈ real" was a shuffle-protocol artifact.
  The "mostly compute" read only appeared through a greedy-metric lens (the representational slice is
  separate and noisy). See claim C9 (corrected).

## Current Read

Turning thinking on is a real, cheap deployable lever the corpus left unused — but it is a
*budget* to be controlled, and the controller experiment shows the budget knob is mostly an
**efficiency** lever (near-iso-accuracy at much lower cost), not a new accuracy frontier: a
near-optimal fixed budget already sits close to the oracle, and the deployable controller is
capped by C2 false-passes. The foreign control then refined the *nature* of the gain: at the
behavioral gain **is coherent reasoning over relevant content** at every budget (the model uses thinking
as content — irrelevant thinking is catastrophic; pure-compute filler ≈ no-think; and the coherence
advantage *grows* with budget). It is not clearly reflected in internal correctness-decodability
(separability noisy). Honest read: *use thinking, cap it (greedy overthinking optimum ~1024), and a cheap
controller buys back most of the cost; the accuracy gain is genuine reasoning content the model uses at
all budgets — don't dismiss it as compute; behavioral ≠ representational.* The filler arm (pure compute ≈
0) and the budget sweep (coherence grows) are now done. Priority follow-ups: a learned controller with
richer visible signals to chase the C2 wall; and a **contamination-controlled / harder substrate** — does
coherent reasoning still carry the whole gain when the no-think baseline is weaker and memorization is
defeated? (This is the most load-bearing open question given MBPP is basic and likely partly contaminated.)
C51 adds that a thinking budget is not merely a token count: when almost every trace is force-closed, a
teacher-forced answer score can describe a counterfactual state rather than deployable reasoning. Calibrate
termination on the actual workload and include the close/commit event in any trace-value measurement.
C52 adds that editing one thought token in the loss is not the same as making a context-local model edit:
future steering must pass an exact-logit locality preflight before a larger harvest. This does not weaken
the separate long-context loop-control mandate, whose pathology begins only at 16k+.

## Pareto Policy Integration Qualification (2026-07-12)

The proposed short/deep policy consolidation did not reach distillation.
Across two fresh blocks, C54's `blend` checkpoint was not a better quick teacher
than `apex` (`-0.00693`, `-0.03789`; pooled `-0.02241`), while `apex` had a
replicated deep advantage (`+0.04563`) but missed retention. Thus an external
thinking-budget tier split does not by itself define a same-prefix teacher
split. A future policy-space test should estimate verified continuation
advantage at the actual state, then freeze and replicate the route before
training; the visible two-checkpoint tier router remains only an inference
upper reference.

## Replicated fixed-cap semantic commit (2026-07-12)

`qwen35_4b_commit_slot_semantic_power_replication` resolves the earlier 16-task
near miss without raising the budget. At fixed cap 1,024, 113-task qualification
and 113-task untouched confirmation independently passed every frozen semantic,
task-bootstrap, breadth, and unrestricted-interface gate. Ordered versus exact-
token-shuffle accuracy was 27.14% versus 13.57%, then 28.91% versus 13.86%;
no-thought was 9.73% then 7.08%. One-sided task lower bounds were +8.85pp and
+9.44pp. Every path still required an external cap commit, so this is evidence
for coherent thought content at a fixed counterfactual state, not autonomous
termination. Hold the budget fixed and test whether task-held-out prefix value
can route or causally improve this deployed state. That test is now negative for
the frozen shared J readout: overall AUC was 0.5021, below slot margin 0.5448 and
non-J residual features 0.5292. Midpoint prospective AUC was 0.6083 but endpoint
reversed to 0.3958, so no causal/controller stage opened. The allowed
phase-specific audit found midpoint-only J AUC 0.5375 (lower 0.4417), below
equal-width non-J 0.6000 and tied slot margin 0.5396. The apparent midpoint lead
is retired; keep the replicated semantic-content seam, not its failed scalar
readout, and do not infer a larger token budget.
