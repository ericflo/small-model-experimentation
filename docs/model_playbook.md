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
  **Substrate scoping (C47): the no-think readout only works where correctness is semantically
  readable** (docstring-style code). Where correctness must be COMPUTED (procedural
  candidates checked against I/O examples) it is within-task CHANCE (0.471) — let the model
  think before the same A/B readout (0.471 → 0.845 within-task, even 99% budget-truncated at
  512): C44's serial-compute law governs the judge seat too.
- **Sample-and-confidence-select; don't majority-vote.** Confidence-select beats
  self-consistency at every budget (toy: 0.62 vs 0.48, C41; MBPP: 0.762 vs public-output
  majority 0.721, p=0.014, C46) because the mode is often confidently wrong while
  high-confidence samples are right. On all-task HumanEval with no public probes,
  P(True)-select beats mean-logprob 0.835 vs 0.787 (C46). Abstain on low max-confidence:
  the top third by P(True) runs ~0.95 on MBPP vs 0.70 unfiltered (C46), AUROC ≈0.83–0.86
  for solvability across toy/MBPP/HumanEval substrates (C41/C46).
- **When any executable test exists, execute it — confidence is for the verifier-free
  regime.** One visible test (0.816 selection) beats every verification-free signal (C46);
  execution-filtering recovers the coverage ceiling for free (C17).
- **Per-step confidence localizes the first error** (dips at the first slip, surviving
  de-trending; C42) — use it for targeted repair instead of full resampling.
- **To install a missing capability, bank verified solutions — and seed rungs the model
  cannot sample.** Self-training on execution-verified self-solutions banks deployable
  capability (C11, C18) but only for depths the base already samples (C21); deeper rungs need
  an external explorer (tool search) to harvest training data (C22–C24). The gain is
  data-DIVERSITY-driven (C24) — and dose-limited: C18's 3× headline was a low-dose
  overestimate (matched-dose rerun flat under the strict eval; C47/C18 audit). To install a
  *skill* rather than answers, teach a GENERAL serial strategy across diverse families and
  deploy with CoT (C45) — but expect the install to be DEPTH-LOCAL: C48's procedure-SFT
  doubles structure proposal at its practiced depths (d2 0.37→0.70, zero forgetting via the
  think-channel recipe) and moves nothing one composition step deeper; nothing measured (answers
  C21, diversity C24, procedure C48) banks depth across. Use procedure-SFT to consolidate depths
  the model already touches; keep external search (C34/C35) for the frontier. Never deploy a
  trained-strategy adapter outside its training substrate — C48's c45_zero arm shows ~zero
  transfer with ACTIVE interference (list d2 0.37→0.00).
- **Confidence cannot replace the execution verifier at the TRAINING seat.** Banking
  top-think-P(True) solutions (purity 0.43, ~15× random) gains exactly as much as banking
  unfiltered data — only 100%-pure execution-verified data teaches at C18-scale dose (C47;
  untested at larger harvests, where a top-rank slice could be both pure and diverse). But the judge
  SURVIVES self-training as a ranker (within-AUROC 0.872 → 0.883 even when trained on its own
  approvals), while raw scores inflate on the model's own new mistakes (0.091 → 0.204). If a
  flywheel must run verifier-free: filter by RANK within depth strata each round (quotas by
  judge-score mass, never candidate counts — wrong candidates explode at hard depths), never
  by a fixed P(True) threshold.

## Evaluation-method rules (not fooling ourselves)

- **Gate on the base execute-given-rule ceiling** before interpreting any induction failure:
  an induction 0 is meaningless if the base cannot execute the stated rule either (the C39
  cipher trap; enforced as mandatory in C43).
- **Headline calibration numbers must be WITHIN-cell/within-problem AUROC against a surface
  baseline** (length/verbosity for code, external I/O features generally) — pooled AUROC is
  inflated by item difficulty, and RoPE makes last-token layer-0 a degenerate "surface"
  control, so use an external baseline (C40, C30/C31, C46).
- **Paired bootstrap over items for every selection/method delta** — few-point deltas on
  n≈250 look real and aren't; report the honest negatives alongside (C46). Pair on a FULLY
  QUALIFYING key — task ids are only unique within a family, and a bare-id dict pairing
  silently matched cross-family rows, inflating a published contrast +0.050 → +0.083 until an
  adversarial audit caught it (C48). Never hand-compute a published statistic: extend the
  committed analyzer so the number is regenerable.
- **Forced-answer evals (argmax over the answer-token set after `Answer: `)** give a fast,
  deterministic, fair comparison between base and SFT'd models — the base otherwise rambles
  without concluding (C43). Compare the SAME model in forced vs generation mode to localize
  whether reasoning is load-bearing (C44).
- **Never trust nominal composition depth** — ~40% of random depth-3 compositions are
  shallower-equivalent; verify behavioral min-depth (C13).
- **A post-SFT no-think coverage drop is not necessarily capability loss.** No-think SFT
  reallocates the no-think proposal prior onto the banked op-family (correct mass conserved,
  off-family tasks lose coverage) while CoT re-derivation shields think-mode sampling —
  evaluate think-mode before concluding damage, and check WHICH task families lost coverage
  against the training mix (C47).
- **Don't conclude "thinking hurts" from an unstructured prompt** — that artifact retracted a
  finding once (C15); give a procedure before judging the thinking channel. And a
  shuffled-thinking control separates compute/scaffold effects from coherent reasoning (C9).
- **Unit-test parsers on the exact failing case before any scored run** — a quote-blind parser
  nearly fabricated a false cross-substrate law (C16).
- **Behavioral structure metrics must be probe-hardened** — a program that hardcodes the visible
  examples passes a visible-only behavioral skeleton metric by construction; require the SAME
  skeleton fill to also match fresh probe inputs, and report the mimicry rate (C48). And check
  PARSE RATE before reading any prompt-scaffold null: long procedure prompts collapse output
  formatting (0.89→0.44) before they change behavior (C48).
- **Calibrate reproduction gates to the source experiment's committed artifact number**, never a
  neighboring claim's headline from memory — a 0.95 gate built from C44's shift number stopped a
  pipeline whose regenerated C45 adapter (0.920) actually EXCEEDED C45's real 0.905 (C48).
- **Interrogate whether two levers act on the same axis before running the comparison** — the
  banking arc almost conflated "thinking is useless" with "the model was never trained to
  think in this mode" (C26 scope caveat).
- **Design-review before expensive runs** (see AGENTS.md): reviews caught a
  hidden-set-peeking search bug (C25) and redundant framings; save the review as
  `reports/design_review.md`, and note there when a workflow review died and the design was
  self-vetted instead.
