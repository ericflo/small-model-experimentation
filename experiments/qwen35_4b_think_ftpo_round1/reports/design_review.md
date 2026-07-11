# Adversarial design review — think-block FTPO round 1

Four independent adversarial reviewers (methodology/statistics, corpus-rules compliance,
feasibility/compute, mechanism skepticism) attacked the v1 design (README +
preregistration v1) before any full-scale GPU spend. 35 findings; the material ones and
their dispositions are recorded here. The review, combined with a user redirection toward
outcome-conditioned pivot steering, produced preregistration v2.

## Blocking finding (mechanism) — upheld, redesigned around

**The v1 primary premise was empirically false at deployed budgets.** The reviewer ran
this experiment's own mining detector over existing greedy base-model completions and
found exact-repetition loops in 1/1200 gym atoms at think@1024 and 2/1600 harvest atoms
at 2048–4096 — ~0.1%, two orders of magnitude under the v1 census gate (8%). Verified
independently before redesign: **1/1200 atoms (0.08%), 0/786 episode turns (0.00%)**.
The corpus's own budget decomposition agrees: at think@16,384 only 13/144 forced-close
tails were loops; loops dominate only at 32k+. The v1 anchor stats (79/80 forced-close on
quick, etc.) prove budget exhaustion, not looping — conflating them was the v1 design's
central error (the C50 "truncation cascade" account already said the binding failure is
non-repetitive verbose non-convergence).

**Disposition:** loop-repair arm descoped to a zero-GPU census artifact (the numbers
above, now a reported result); loop-FTPO requeued as a long-context (16k+) loop-control
follow-up where the corpus knows the pathology lives. Round 1's trained arms are now
**pivot** (outcome-conditioned divergence steering — unaffected by this finding, since
its signal is verifier outcomes, not repetition) and **pivot-shuffled** (label-permuted
control).

## Methodology/statistics — accepted

- **Thresholds vs measured null:** +0.03 is ~1σ of the gauntlet's measured ~0.034
  same-seed spread; single-event quick detectability is honestly ~+0.06; medium null was
  never calibrated. → v2: 2+1 base-vs-base null calibration whose measurement REPLACES
  the prior; three quick seeds with threshold max(+0.03, 2·SD_null/√3); medium stage
  conditional on the quick gate with its own same-seed null realization; multiplicity
  note (round-2 iteration uses fresh seeds, no re-reads).
- **Census/yield gate incoherence:** 8% × 5k prompts ≈ 400 events < 800-row gate; and at
  gate minimums the POSITIVE bar was arithmetically unreachable while NEGATIVE
  overclaimed ("underdosed" ≡ "not binding"). → v2: adaptive dosing to a projected
  ≥1,200-row pool (5h harvest cap), training floor 600 rows with a power caveat, and a
  dose-sufficiency precondition (≥1,200 rows) required for any NEGATIVE label —
  otherwise UNDERDOSED → iterate.
- **Chosen-count collapse at T=0.01 renormalization** (v1 loop arm): moot after descope;
  arm P's chosen tokens are observed branch tokens; chosen-per-row is a mandatory
  reported diagnostic.
- **Config freeze was prose-only:** → configs/default.yaml now carries every constant and
  `run.py --smoke` asserts the loaded config against frozen literals.

## Compliance — accepted

- **Held-out families were in the v1 harvest mix** (brinework, spindle), destroying the
  program's held-out-family control. → v2 harvests the 10 trained families only.
- **base+rp had no decision role** → dropped entirely in v2 (its purpose was the loop
  question, which the census closed); the C5/matched-compute discipline is now served by
  the labeled non-deployable base n=8 coverage reference plus the shuffled-label control.
- **Seed hygiene:** closed ranges registered; menagerie union-check now includes the
  suite baseline seed 31337 and the gauntlet's event log.
- **Termination accounting:** full triple (exact-loop / unresolved-contact / answer-limit)
  preregistered for whitebox and gym evals, per the playbook.
- **Cross-backend readout parity concern:** moot — v2 mining uses observed tokens only;
  no logprob readout pass exists in the pipeline.
- Verified clean: one-model rule (rule-based detector, own-distribution tokens, no
  refusal classifier), benchmark firewall usage pattern, artifact-policy plan.

## Feasibility — accepted

- **Per-arm C49 gates:** every adapter arm (pivot AND pivot-shuffled) deploys as a merged
  composite checkpoint and must pass its own on-vs-off behavioral diff (vLLM runtime LoRA
  is a verified silent no-op, C49).
- **Menagerie wall-times:** medium events measured 34–49 min on this instrument (not the
  README's 300s); → medium stage made conditional on the quick gate; menagerie budgeted
  from measured event times.
- **Trainer landmines:** vocab is 248,320 — full-sequence logits are a ~3 GB/row tensor;
  → final-position-only logit computation (gather last-real hidden, lm_head on gathered),
  gradient checkpointing, expandable_segments, AcceleratorError-inclusive OOM handling,
  max-length memory smoke: all preregistered requirements.
- **Hybrid-architecture padding hazard:** q/k/v/o exist on only 8/32 layers (the rest are
  linear-attention with different projection names), and left-padded batches can
  contaminate recurrent-layer state. → RIGHT-padding with last-real-index gather
  (contiguous real prefix is safe for causal recurrent scans), plus a preregistered
  padded-vs-unpadded equivalence gate on real rows before training. Target-module
  coverage (MLP on 32/32, attention on 8/32) documented; adapter ≈ 0.34B params
  (~0.68 GB bf16).
- **Whitebox/guard sample sizes frozen:** N=500/arm/budget; collapse guard 120 tasks.

## Mechanism — accepted

- **Format-transfer confound (C48 precedent):** P1 now has a format-shifted slice
  (2 alternate scaffolds), required to hold before any NEGATIVE claim
  (else INCONCLUSIVE-transfer).
- **No-think/answer-channel interference:** a no-think forced-answer guard batch (120 gym
  L1 atoms, pivot vs base, −2pp tolerance) added.
- **Menagerie-spend gating:** the quick-conditional medium stage plus the P0/P1 gates
  ensure blackbox spend follows mechanism evidence.

## Residual risks accepted knowingly

- Round-1 wall clock (~9–13h GPU serial) exceeds the program's fast-round doctrine;
  accepted for round 1 because it front-loads reusable pipeline, null calibrations, and
  the census artifact. Subsequent rounds inherit all of it.
- n=8 Monte-Carlo pivot labels are individually noisy; the design leans on aggregate
  dose, flattening, the shuffled control, and the C29 collapse guard.
- Menagerie prompt formats are unknown by contract; the format-shifted whitebox slice is
  the only available transfer probe.
