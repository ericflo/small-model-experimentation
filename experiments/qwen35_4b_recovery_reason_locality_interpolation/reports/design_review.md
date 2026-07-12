# Adversarial design review

## Verdict

Proceed after the listed mitigations. This is a bounded, high-information reuse
of already-paid training rather than another SFT dose search. Its strongest
feature is the order of observation: fixed ladder → locality screen → frozen
calibration selection → independent locality confirmation → untouched transfer.

## 1. Is this merely tuning on the failed locality set?

**Attack.** The ladder was chosen using the parent's observed endpoint drifts,
and the same 48 contexts screen the candidates. Selecting the largest passing
scale could overfit one small locality instrument.

**Mitigation.** The ladder is fixed before any scaled merge, locality is only a
hard filter, and behavior—not drift margin—selects among passing points. The
single winner then faces 48 entirely new contexts with no fallback. Screen and
confirmation content hashes are disjoint; entropy is a second hard retention
signal and varentropy is recorded diagnostically.

## 2. Does the weight contrast isolate plan supervision?

**Attack.** Independently optimized LoRA factors are non-identifiable and the
reason run's early gradients changed its whole optimization path. The endpoint
difference is not literally a separable “plan module.”

**Mitigation.** The report must call it the *learned action→reason contrast*, not
a causal plan-only vector. The endpoints nevertheless share base, data, seed,
batch order, optimizer, and action targets; plan masking is the sole recipe
difference. Interpolation answers the deployable question—whether that observed
contrast has a safe useful region—without overclaiming decomposition.

## 3. Why interpolate from action rather than scale reason from apex?

**Attack.** Scaling the full reason delta is simpler and might pass locality.

**Mitigation.** It would simultaneously erase the parent's strongest positive:
full-dose transition-balanced action learning already passed locality. Anchoring
at action preserves that signal and scales only what changed when plans were
added. Full action is an explicit control and must be beaten or meaningfully
repaired on held-out families.

## 4. Is the calibration block contaminated by prior observation?

**Attack.** Parent base/happy/action/reason calibration metrics are known, so
they cannot support a new generalization claim.

**Mitigation.** Calibration is declared selection-only and reused deliberately
to avoid spending fresh families on scale choice. No claim can be made from it.
Both four-family blocks stayed sealed in the parent and are evaluated only after
one candidate and fresh-locality confirmation are frozen.

## 5. Could failing scales leak behavior into selection?

**Attack.** Evaluating every scale behaviorally would allow post-hoc conclusions
about points that already failed the prerequisite.

**Mitigation.** The orchestrator reads the full locality receipt and evaluates
exactly its passing eligible set. The selector rejects any mismatch. The reason
endpoint is measured only as a locality curve anchor and is never behaviorally
eligible in this follow-up.

## 6. Could selection reward success while tolerating malformed tool use?

**Attack.** The parent action arm reached high success with a 19.1% invalid-turn
rate. Pure success selection could reproduce the wrong behavior.

**Mitigation.** Eligibility requires invalid actions no worse than base +0.02,
rejected-patch immediate change ≥0.60, and failed-test changed patch within two
turns ≥0.60. Invalid rate precedes transition score in tie-breaking. Transfer
again requires base-level validity and improvement in validity or immediate
recovery versus full action.

## 7. Are matched-compute and scaffold controls fair?

**Attack.** Two short samples and one deep trajectory can differ in call shape;
the explicit scaffold may receive unequal compute or a different backend.

**Mitigation.** Deep and sampling arms reserve identical maximum calls × tokens,
all use the copied pinned vLLM runner, and scaffold changes only a public
recovery instruction in the same deep loop. Backend mixing is forbidden. The
candidate must beat both sampling and scaffold by +0.03 on each transfer block.

## 8. Are the gates feasible before expensive candidate evaluation?

**Attack.** A strong control could make a delta gate mathematically impossible,
repeating the specialist-policy footgun.

**Mitigation.** Each transfer block evaluates frozen controls first, then a
machine-readable feasibility script compares every threshold with metric hard
ranges and control values. Candidate evaluation is cancelled if any bar is
unreachable. Calibration reachability is checked in the CPU smoke using frozen
parent values.

## 9. Does control-first transfer reveal tasks and invite redesign?

**Attack.** Aggregate control outcomes become visible before candidate scores.

**Mitigation.** All code, thresholds, candidate identity, and decision rules are
committed before any transfer control runs. Control outcomes may only trigger
the registered feasibility stop; they cannot change the candidate or recipe.
Transfer artifacts are generated by the orchestrator and are not inspected to
redesign within this experiment.

## 10. Does the CPU smoke consume sealed transfer seeds?

**Attack.** A convenience smoke could instantiate held-out tasks before the
registered stage.

**Mitigation.** It does not generate any calibration or transfer task. It checks
only family-set disjointness and unique split seeds, while exercising task code
with a separate smoke seed and split.

## 11. Could bfloat16 make the path non-reproducible?

**Attack.** Interpolating two saved bfloat checkpoints introduces double
rounding and may not reproduce either endpoint.

**Mitigation.** The implementation reconstructs both LoRA deltas from their
float tensors, mixes each module in float32, adds once to the common base, and
casts once to bfloat16 with TF32 disabled. Receipts preserve input hashes,
lambda, module coverage, delta norms, and output weight hash. Unit tests verify
the algebraic endpoints.

## 12. Firewall and benchmark exposure

**Attack.** A coding benchmark could silently enter training, selection, or
analysis, especially once white-box gates pass.

**Mitigation.** There is no training and the copied substrate is fresh
procedural code. The harness serializes only public state plus host-side hidden
booleans. No `benchmarks/` module or file is imported/read. Menagerie is invoked
only through its public CLI after both transfer receipts pass, with fresh paired
seeds assigned then.

## Required pre-run checks

- CPU smoke and focused unit tests pass.
- One-scale merge plus two-context vLLM/locality integration smoke succeeds.
- Parent hashes and locality block hashes match the committed config.
- `make check` passes and the design commit is rebased and pushed to `main`.
- No result-bearing scale beyond the integration checkpoint is evaluated before
  that push.
