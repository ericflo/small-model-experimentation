# Qwen3.5-4B Model Playbook

How to elicit and evaluate this specific model correctly, distilled from the claim ledger
(`knowledge/claims/claim_ledger.json`). Each rule cites the claims behind it. This is the
operational companion to `knowledge/synthesis.md`: synthesis says what we learned; this says
what to DO about it when designing the next experiment. Keep it current — when a new claim
changes a rule here, update the rule in the same commit.

## Elicitation rules (getting capability out)

- **Always give chain-of-thought for induction / multi-step computation.** The forward-pass
  induction wall is a serial-compute limit, not a knowledge limit: reasoning-SFT induces
  held-out rules at 1.00 via generation but 0.01 in one forced forward pass (C44). Never
  expect a one-token answer to a multi-step problem, and never use answer-only SFT to install
  a serial skill — it forces the computation where it cannot live and catastrophically
  forgets execution (C43/C44).
- **Budget the CoT generously and check truncation.** The general hypothesize-and-verify CoT
  needs ≥400 generated tokens; a 256 cap truncated it into a false 0.00 (C45). Thinking-budget
  accuracy is non-monotonic: optimum ≈512–1024 thinking tokens, and *unbudgeted* is worse than
  a cap (C9). Record generation lengths and report the truncation rate.
- **Modality matters: natural language elicits reasoning; dict/JSON format triggers
  code-mode.** The compositional execution wall exists in formal task presentations but not in
  natural language (C37). No-think mode avoids code-mode on tasks where code is a distractor
  (C39). Present reasoning tasks in prose unless code is the point.
- **The model is an executor/retriever, not an inducer.** It executes a stated novel rule at
  0.97 but cannot induce one from examples (0.12 ≈ chance) — in-context learning retrieves
  familiar structure rather than inducing novel structure (C39, C38, C32/C36). Externalize
  structure proposal (tool search, enumeration, taught serial procedures); let the model
  execute/verify/fill values. "Tools identify, the model compiles" (C13): plan-given execution
  is ≈1.00 across substrates (C16) — never spend training on execution.
- **Read confidence from a single-token logit readout — never a sequence average, never a
  sampled self-report.** The answer token's probability predicts correctness at AUROC 0.95 on
  the toy substrate (C40); on real code the analog is P(True): show the model its own output
  with an A/B "is this correct?" judge prompt and read P(A) after `Answer: ` — one no-think
  forward pass (C46, following the C10 readout). Sequence mean-logprob dilutes the signal
  below deployable significance on programs, and verbalized confidence is a constant (C40).
- **Sample-and-confidence-select; don't majority-vote.** Confidence-select beats
  self-consistency at every budget (toy: 0.62 vs 0.48, C41; MBPP: 0.762 vs 0.717, p=0.005,
  C46) because the mode is confidently wrong on hard problems while high-confidence samples
  are right. Abstain on low max-confidence: the top third by P(True) runs ~0.95 on MBPP vs
  0.70 unfiltered (C46), AUROC ≈0.83–0.84 for solvability on both substrates (C41/C46).
- **When any executable test exists, execute it — confidence is for the verifier-free
  regime.** One visible test (0.816 selection) beats every verification-free signal (C46);
  execution-filtering recovers the coverage ceiling for free (C17).
- **Per-step confidence localizes the first error** (dips at the first slip, surviving
  de-trending; C42) — use it for targeted repair instead of full resampling.
- **To install a missing capability, bank verified solutions — and seed rungs the model
  cannot sample.** Self-training on execution-verified self-solutions banks deployable
  capability (C11, C18) but only for depths the base already samples (C21); deeper rungs need
  an external explorer (tool search) to harvest training data (C22–C24). The gain is
  data-DIVERSITY-driven (C24). To install a *skill* rather than answers, teach a GENERAL
  serial strategy across diverse families and deploy with CoT (C45).

## Evaluation-method rules (not fooling ourselves)

- **Gate on the base execute-given-rule ceiling** before interpreting any induction failure:
  an induction 0 is meaningless if the base cannot execute the stated rule either (the C39
  cipher trap; enforced as mandatory in C43).
- **Headline calibration numbers must be WITHIN-cell/within-problem AUROC against a surface
  baseline** (length/verbosity for code, external I/O features generally) — pooled AUROC is
  inflated by item difficulty, and RoPE makes last-token layer-0 a degenerate "surface"
  control, so use an external baseline (C40, C30/C31, C46).
- **Paired bootstrap over items for every selection/method delta** — few-point deltas on
  n≈250 look real and aren't; report the honest negatives alongside (C46).
- **Forced-answer evals (argmax over the answer-token set after `Answer: `)** give a fast,
  deterministic, fair comparison between base and SFT'd models — the base otherwise rambles
  without concluding (C43). Compare the SAME model in forced vs generation mode to localize
  whether reasoning is load-bearing (C44).
- **Never trust nominal composition depth** — ~40% of random depth-3 compositions are
  shallower-equivalent; verify behavioral min-depth (C13).
- **Don't conclude "thinking hurts" from an unstructured prompt** — that artifact retracted a
  finding once (C15); give a procedure before judging the thinking channel. And a
  shuffled-thinking control separates compute/scaffold effects from coherent reasoning (C9).
- **Unit-test parsers on the exact failing case before any scored run** — a quote-blind parser
  nearly fabricated a false cross-substrate law (C16).
- **Interrogate whether two levers act on the same axis before running the comparison** — the
  banking arc almost conflated "thinking is useless" with "the model was never trained to
  think in this mode" (C26 scope caveat).
- **Design-review before expensive runs** (see AGENTS.md): reviews caught a
  hidden-set-peeking search bug (C25) and redundant framings; save the review as
  `reports/design_review.md`, and note there when a workflow review died and the design was
  self-vetted instead.
