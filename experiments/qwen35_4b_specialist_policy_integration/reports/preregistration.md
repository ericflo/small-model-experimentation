# Specialist Policy Integration Preregistration

Date locked: 2026-07-11, before any model baseline, training, or benchmark run.

## Primary Hypothesis

Four independently execution-improved policies descended from the merged C53
incumbent can provide trustworthy same-prefix supervision to a fresh student.
Correctly routed on-policy multi-teacher distillation will inherit their
individual headroom more uniformly than joint RL, off-policy SFT, or parameter
merging, and the integrated student will combine their primitives on held-out
compound tasks better than every individual teacher and matched-compute
sample-more.

## Fixed Model and Runtime

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, may be loaded.
- Every trainable and frozen checkpoint descends from one regenerated C53
  merged incumbent `S0`.
- All generative arms use the same pinned vLLM runner. Transformers is limited
  to training and frozen logit prefills where internals are the measurement.
- Every LoRA is explicitly merged and must pass a nonzero behavioral on/off
  gate before result-bearing evaluation.

## Frozen Domains and Splits

The exact lists and seed namespaces live in `configs/default.yaml`.

- `discover`: `glyphgate`, `loomfix`.
- `control`: `kilnrite`, `burrowmaze`.
- `tools`: `ferrier`, with `foundry_ledger` atom replay only.
- `compose`: `cipherkiln`, `mazeferry`.
- no-new-exposure primitive transfer: `patchwheel`, `spindle`, `gatepost`.
- fully held-out compound transfer: `patchferry`, `tripleforge`, plus held-out
  action order reversals.

No transfer-family DAgger row, reward trajectory, MOPD prompt, or retention row
may enter training. `S0` has historical exposure to the primitive transfer
families; that is disclosed and is why the claim is incremental transfer, not
never-seen content.

## Compound Necessity Gate

Every family must be deterministic and JSON-safe, remain inside prompt and
observation limits, and satisfy:

- exact oracle mean `>=0.95` at every level;
- generic random-policy mean `<=0.15`;
- each named primitive-removal policy has full-success rate `<=0.20`; and
- on disjoint calibration seeds, `S0` compound macro is `<0.60` before the
  confirmatory level distribution is frozen.

Individual confirmatory items may never be accepted or rejected based on any
model output.

## Stage 1: Specialist Production

Train `T_discover`, `T_control`, `T_tools`, and `T_compose` independently from
`S0`. Each receives a state-aware DAgger warm start and then grouped sequence
optimization with exact terminal reward, success-only efficiency tie-break,
zero gradient for constant-outcome groups, an `S0` KL anchor, and 20% permitted
retention replay. Expert state is used to label, never appended to inputs.

For each domain, compare:

1. `S0` greedy and execution-filtered best-of-8;
2. DAgger-only;
3. compute-overmatched additional SFT;
4. shuffled group rewards; and
5. real-reward specialist.

A specialist qualifies only if it clears every registered gate:

- own-domain macro `>= S0 + 0.10`;
- `>= DAgger + 0.05`;
- `>= additional-SFT + 0.05`;
- `>= shuffled-reward + 0.03`;
- greedy/pass@1 beats `S0` execution-filtered best-of-8 under the inference
  token ledger;
- no retention family loses more than `0.05`; and
- parse, natural close, entropy, and correction markers remain in band.

Proceed only if all four specialists qualify. The decision record's “at least
three primitive specialists plus composition” is equivalent here because
there are exactly three primitive specialists.

## Stage 2: Same-Prefix Teacher Audit

On frozen `S0` trajectories, branch from exact visited prefixes. Compare `S0`,
the correct specialist, and a wrong specialist selected within initial-KL bins.

Required gates per teacher:

- continuation reward `>= S0 + 0.08`, with paired-bootstrap 95% lower bound
  above zero;
- continuation reward `>= wrong route + 0.05`;
- for the union of teacher/student top-four alternatives, four `S0`
  continuations per forced token: correct-pressure top-vs-bottom quartile lift
  `>=0.08` and `>= wrong-pressure lift + 0.05`; and
- after exactly five miniature MOPD updates, correct-teacher divergence falls,
  batch-one per-row median non-target exact-logit drift is `<=0.10`, and both
  entropy and explicit correction-marker rate fall by no more than 10%.

Failure stops that teacher and all integration spend.

## Stage 3: Integration

Initialize a fresh rank-matched student from `S0`. Each student trajectory is
generated on-policy, tagged with the exact student checkpoint digest, routed
to the matching frozen teacher using trainer-only metadata, consumed by at
most one update, and discarded. Policy lag must be at most one update.

The primary loss is MOPD's bias-corrected top-50 reverse KL with corrected tail
mass. A synthetic-vocabulary unit test must match the full-vocabulary objective
before training. All trainable integration arms receive the same 20% retention
dose. Rank is chosen once from a memory-only smoke and cannot be selected from
task performance.

Integration controls:

- end-to-end matched joint DAgger+GRPO;
- off-policy SFT on successful specialist trajectories;
- convex/task-vector parameter merge; and
- initial-KL-binned wrong-route MOPD.

No objective may be swapped after outcomes are visible. Entropy-aware,
purified, position-weighted, and RL-auxiliary variants require new experiments.

## Compute Matching

Report two ledgers:

- conditional integration: student rollout, teacher prefill, optimizer, and
  target tokens after specialists exist;
- end-to-end: specialist production plus integration. Joint RL receives this
  entire forward-token and optimizer-token budget.

Wall time is descriptive, never the matching variable. Every inference arm has
identical backend, prompts, parser, stopping, temperature, and token budgets.

## Confirmatory Evaluation

- At least 128 paired episodes per family/level, increased if a frozen power
  simulation cannot detect an absolute 0.08 effect at 80% power.
- Three independent end-to-end seeds for MOPD and joint RL.
- Episode bootstrap stratified by family and level; turns are not independent.
- Report pass@1, pass@8, unique valid coverage, execution-filtered coverage,
  action validity, natural close, length, entropy, and correction markers.

For domain `d`, define

`I_d = (score(integrated,d)-score(S0,d))/(score(T_d,d)-score(S0,d))`.

Integration succeeds only if mean `I_d>=0.75`, every `I_d>=0.50`, none is
negative, normalized integration is `>= joint-RL + 0.10` and
`>= off-policy-SFT + 0.05`, and correct routing is `>= wrong routing + 0.05`.

Held-out composition succeeds only if the integrated student is:

- `>= S0 + 0.15` family macro;
- `>= best specialist (including T_compose) + 0.10`;
- `>= joint RL and off-policy SFT + 0.05`;
- above `S0` execution-filtered best-of-8 at no greater inference-token cost;
  and
- positive by paired-bootstrap lower bound in every compound family.

Primitive transfer must be nonnegative in all three families with at least two
at `>=+0.05`.

## Benchmark Firewall and Final Gate

The benchmark directory remains read-forbidden. Only after every whitebox gate
may one frozen MOPD checkpoint be passed to the public CLI. It must achieve
medium macro `>= C53 + 0.05`, quick delta `>=0`, and slow delta `>=0`.
Benchmark output cannot revise any training choice.

## Interpretation

- Specialist failure: capability production failed; distillation is unlicensed.
- Specialist success but prefix/locality failure: endpoint advantage is not a
  safe dense signal.
- Individual inheritance without compound gain: union, not composition.
- Compound gain without primitive/blackbox transfer: substrate-local result.
- All gates: evidence for stronger composable capability from same-origin
  specialist RL plus on-policy policy integration.
