# Decision Record: Specialize, Distill, Then Test Composition

- Date: 2026-07-11
- Status: executed; stopped at per-specialist feasibility
- Programs: agentic_breadth_installation, posttraining_and_adaptation, benchmark_generalization
- Experiments: qwen35_4b_interactive_policy_curriculum, qwen35_4b_specialist_policy_integration
- Claims: C11, C12, C21, C22, C28, C29, C44, C48, C49, C50, C52, C53

## Decision

The next expensive post-training experiment should not be vanilla privileged-hint
OPSD, a larger self-training mixture, or undifferentiated mixed-domain GRPO. It
should separate **capability production** from **capability integration**:

1. start every arm from the same merged C53 incumbent;
2. use state-aware DAgger followed by exact execution-reward RL to create four
   independently useful, same-origin specialists;
3. prove, before distillation, that each specialist is better on the student's
   own visited states and remains close enough in policy space to be a safe
   teacher;
4. train a fresh student from the common incumbent on its own rollouts, routing
   each rollout to the matching frozen specialist and minimizing MOPD's
   bias-corrected top-k reverse-KL target; and
5. require the integrated student to beat every individual specialist, joint
   RL, off-policy SFT, parameter merging, and matched-compute sampling on
   genuinely held-out compound environments.

The proposed experiment id is `qwen35_4b_specialist_policy_integration`. It
belongs primarily to `agentic_breadth_installation`, with
`posttraining_and_adaptation` and `benchmark_generalization` as secondary
programs. The closest near-duplicate is
[`qwen35_4b_interactive_policy_curriculum`](../../experiments/qwen35_4b_interactive_policy_curriculum/README.md).
That experiment creates one mixed DAgger/GRPO policy. The proposed experiment
reuses its live-state machinery but creates isolated specialists, adds an
on-policy integration stage, and makes never-trained composition—not merely a
multi-family average—the primary endpoint.

This is the strongest current bet, not a claim that the result is likely by
ordinary standards. It is the first design in this repository that gives the
fixed model a credible external capability-producing step, a distribution-
matched way to consolidate several resulting policies, and a test that can
distinguish a union of skills from their composition.

## Execution Outcome (2026-07-12)

The experiment did not reach capability production or integration. The
regenerated incumbent and compound-headroom gate passed, but the complete
paired baseline revealed that the mandatory tools core was saturated:
`ferrier = 0.9940`, so its frozen `S0 + 0.10` target was 1.0940 under a score
ceiling of 1.0. The all-four-teacher premise was therefore mathematically
unreachable before any specialist training.

The decision's mechanism argument remains open; this run is evidence about
experimental feasibility, not OPSD/MOPD efficacy. The durable amendment to
future strategy is procedural: before best-of-k or specialist production,
require independent theoretical headroom for every mandatory teacher, not
only for the downstream compound endpoint. A harder tools/provenance core and
new split require a new experiment and fresh confirmatory seeds; the current
bar is not lowered post hoc.

## What the Social Posts Actually Refer To

The quoted posts mix real methods, deliberate acronym collision, and incorrect
descriptions.

| Post | Verdict | Technical kernel |
|---|---|---|
| Replace the hinted log-probability ratio with an advantage estimator | Mostly a joke about equivalence | In policy-gradient formulations of on-policy distillation, the teacher-student log-probability difference already acts as a token-level reward or advantage. Replacing it with an outcome/group advantage moves the method back toward ordinary RL rather than producing a new algorithm. |
| Privileged information steers the student distribution during rollouts; link `2209.15189` | The description is anachronistic | [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) is a real 2022 context-distillation paper, but it generates or scores targets under extra context and fine-tunes the context-free model off-policy. It is an ancestor of the idea, not modern on-policy rollout steering. |
| “Dr. OPSD”: do RL first, then OPSD with no hint; link `2606.30406` | Joke name, substantially correct ordering | [MOPD](https://arxiv.org/abs/2606.30406) first creates same-origin domain RL specialists and then distills them into a student on the student's own rollouts. The teacher advantage comes from different trained weights, not a hidden solution in the context. |
| “OPSD + turboquant is all you need” | Shitpost | [TurboQuant](https://arxiv.org/abs/2504.19874) is an online vector/KV-cache quantization method. It could eventually change throughput or memory, but it is not a learning signal or capability mechanism and must not be mixed into a causal training comparison without separate inference-parity gates. |
| “OPSD, SDFT, SDPO, OPSD (the other one)” | Acronym joke grounded in real convergence | Several 2026 papers independently use a student's own trajectories plus a richer view of the same model to turn sparse supervision into token-level targets. They differ mainly in where the richer view comes from and whether dense distillation is the primary objective or an auxiliary to RL. |

## Literature Synthesis

Literature cutoff: 2026-07-11. These are recent preprints, generally evaluated
on public, often saturated benchmarks and often at scales unlike this
repository. Their reported scores are hypothesis generators, not evidence for
this repository's contamination-controlled mission.

### The useful common abstraction

[Generalized/on-policy knowledge distillation](https://arxiv.org/abs/2306.13649)
established the basic move: sample from the student, then have a teacher score
the student's prefixes. This fixes the exposure mismatch of training only on
teacher trajectories.

The 2026 variants differ in the source of the teacher's extra competence:

- [SDFT](https://arxiv.org/abs/2601.19897) conditions the same model on a
  demonstration and distills that view on-policy, aiming to acquire a new task
  while retaining earlier behavior.
- [OPSD](https://arxiv.org/abs/2601.18734) conditions a fixed same-model teacher
  on a reference solution, then matches its distribution along student
  rollouts. Its main result uses full-vocabulary clipped forward KL.
- [SDPO](https://arxiv.org/abs/2601.20802) conditions the current model on rich
  environment feedback or a successful sibling trajectory. The resulting
  conditional/unconditional log-probability difference becomes a dense token
  advantage.
- [SD-Zero](https://arxiv.org/abs/2604.12002) first teaches the model to revise
  an attempt given binary reward, then distills the frozen reviser's
  distribution into the generator on the generator's own attempts.
- [SDAR](https://arxiv.org/abs/2605.15155) keeps outcome RL primary and uses
  skill-conditioned OPSD only as a gated auxiliary, emphasizing positive
  teacher endorsements and attenuating negative rejections.
- [OPID](https://arxiv.org/abs/2606.26790) derives hierarchical hindsight skills
  from completed on-policy agent trajectories and combines the resulting dense
  skill advantage with the outcome advantage.
- [MOPD](https://arxiv.org/abs/2606.30406) removes privileged context entirely:
  independent specialists trained from one common checkpoint score a fresh
  student's on-policy rollouts. It directly targets multi-capability
  integration and reports more uniform inheritance than mixed RL, sequential
  RL, off-policy fine-tuning, or weight merging.

### Why vanilla privileged OPSD is not the primary bet

The positive papers establish that dense, on-policy conditional supervision can
work. The failure literature narrows when it should be trusted:

- [Why Does Self-Distillation Sometimes Degrade Reasoning?](https://arxiv.org/abs/2603.24472)
  finds that rich teacher context can suppress epistemic uncertainty, improve
  narrow in-distribution behavior, and damage out-of-distribution reasoning.
- [Rethinking OPSD for Thinking Models](https://arxiv.org/abs/2607.05184) reports
  that privileged teachers penalize reconsideration branches and reduce
  verification, backtracking, and hedging on long reasoning traces.
- [Purified OPSD](https://arxiv.org/abs/2607.02234) attributes much of the
  privileged-teacher signal to reference-specific shortcuts and explicitly
  subtracts a reference-only component.
- [Denser Is Not Better](https://arxiv.org/abs/2607.01763) finds faster
  in-distribution specialization but worse OOD behavior, stronger forgetting,
  and occasional collapse from dense self-distillation.
- [Position-Weighted OPSD](https://arxiv.org/abs/2605.21606) finds local teacher
  entropy to be a poor predictor of whether a token leads to a viable branch;
  token reliability is trajectory-structured.
- [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) shows that pure
  reverse-KL distillation can contract diversity, especially at high-entropy
  teacher states.
- [A Predictive Law for OPSD from World Feedback](https://arxiv.org/abs/2605.30070)
  finds that the *pre-training performance gap* between the ordinary student
  and feedback-conditioned self-teacher predicts the eventual gain. A dense
  signal without an initial behavioral advantage should not be trained on.

These failures match this repository unusually well. The local OPSD audits
found that a plausible hint did not supply incremental same-prefix information,
execution feedback did no better than shuffled feedback, and token-local update
objectives produced non-local logit changes. The safe inference is not “find a
more elaborate hint.” It is “first manufacture a teacher with measured
outcome-level superiority, and never let unvalidated privileged context be the
source of the supposed capability.”

### Why same-origin specialist distillation is different

MOPD separates two jobs that vanilla OPSD conflates:

- RL creates a policy that can actually obtain higher reward.
- On-policy distillation consolidates that already-demonstrated policy into a
  shared student.

The teacher and student see the same observable prompt and trajectory prefix.
Only their weights differ. There is no solution-conditioned teacher that knows
which branch wins merely because the answer was placed in its prompt. The
specialists also descend from the exact student initialization, which keeps the
initial policy divergence small; MOPD's own ablation identifies this
same-origin proximity as critical to stability.

This distinction is the main reason to take the “RL first, OPSD with no hint”
post seriously.

## Repository Evidence That Constrains the Design

The repository-wide catalog contains 209 experiments and the durable ledger
contains 53 claims. The following evidence is decisive here:

- C11 and C21: verified self-training banks what sampling can already reach but
  does not cross a new depth frontier. An explorer must supply the next rung.
- C12 and C22-C24: tool/search policies can find solutions outside the frozen
  model's sampling support; verified banking can then install them. Explorer
  and installer are distinct roles.
- C28: explicit, correct decomposition plans help; the model's own sampled
  “reasoning” about completed answers is often a rationalization. Do not use
  post-hoc self-explanations as an unverified privileged teacher.
- C29 and the constrained-DPO/pass@k-RL line: updates centered on the model's
  failures can collapse or merely trade away pass-one performance, while tuned
  sampling remains a hard baseline.
- C44: serial reasoning tokens can carry a learned procedure that a single
  forward pass cannot. Preserve thinking budget and the model's ability to
  branch, check, and backtrack.
- C45/C48: a procedure can transfer within a family while failing across
  substrate or depth. Single-family gains do not establish general capability.
- C49: runtime LoRA evaluation can silently be a no-op for this architecture.
  Every trained adapter must be explicitly merged into a loadable composite and
  pass a behavioral on/off gate.
- C52: even a token-local objective can move unrelated logits. Require the
  exact-logit locality gate before spending the full integration budget.
- C50: state/format-aware expert iteration can produce the first broad
  blackbox uplift when the gradient is placed at the emission seam.
- C53: additional variants of train-on-own-verified-outputs hit a second wall,
  and convex mixture composition exposes quick/medium tradeoffs. Its explicit
  next mechanisms are execution-reward RL, tool-found scaffolds, and targeted
  failure curricula.

The direct near-duplicates reinforce the same conclusion:

- [`qwen35_4b_opsd_pressure_locality_audit`](../../experiments/qwen35_4b_opsd_pressure_locality_audit/reports/final_report.md)
  found surface overlap but no useful incremental same-prefix hint signal.
- [`qwen35_4b_reliability_exec_opsd_audit`](../../experiments/qwen35_4b_reliability_exec_opsd_audit/reports/final_report.md)
  found no causal advantage over shuffled execution feedback.
- [`qwen35_4b_oracle_process_grpo`](../../experiments/qwen35_4b_oracle_process_grpo/reports/qwen35_4b_oracle_process_grpo_report.md)
  found the compact process-state warm start useful but the preference/RL
  increment small, making DAgger and matched SFT mandatory controls.
- [`qwen35_4b_interactive_policy_curriculum`](../../experiments/qwen35_4b_interactive_policy_curriculum/reports/preregistration.md)
  is the correct capability-production precursor: live-state DAgger, exact
  terminal reward, shuffled rewards, and unseen proxy families. It does not yet
  test whether separately optimized policies can be consolidated or composed.

## Proposed Experiment

### Primary question

Can independently execution-improved, same-origin policies be integrated into
one `Qwen/Qwen3.5-4B` policy that both retains their individual headroom and
combines their primitives on procedural compound tasks absent from every
training distribution?

### Common root and firewall

- The only model is `Qwen/Qwen3.5-4B` at the repository-pinned revision.
- Every specialist, student, control, scorer, and teacher checkpoint descends
  from the same regenerated and behaviorally verified merged C53 incumbent
  `S0`.
- Training and whitebox evaluation use only copied, self-contained procedural
  gym code under the new experiment. Nothing under `benchmarks/` is imported,
  read, or used for training.
- `patchwheel`, `spindle`, and `gatepost` are excluded from every new DAgger,
  RL, MOPD, and retention-replay row. Their historical presence in `S0` is
  disclosed; “held out” here means no incremental exposure in this experiment.
- All bulk generations for all arms use the pinned vLLM runner with identical
  decoding semantics. Transformers is used only where internals are required:
  training and frozen teacher prefills on already generated trajectories.
- Menagerie is called through its public CLI only after every whitebox gate,
  for one frozen candidate. It cannot select teachers, checkpoints, losses, or
  hyperparameters.

### Capability cores and compound substrate

The experiment should use three semantically distinct cores from the existing
firewall-clean gym and a fourth composition domain:

| Core | Specialist training families | Never-newly-trained transfer family | Required primitive |
|---|---|---|---|
| Discover/repair | `glyphgate`, `loomfix` | `patchwheel` | choose informative probes, maintain hypotheses, localize and repair a rule |
| Stateful control | `kilnrite`, `burrowmaze` | `spindle` | track latent state and execute a long legal plan under partial observation |
| Tools/provenance | `ferrier` plus `foundry_ledger` provenance replay | `gatepost` | preserve the user goal, acquire typed evidence, and chain tool outputs |
| Pairwise composition | new `cipherkiln` and `mazeferry` generators | new `patchferry` plus `tripleforge` | decide which primitive is needed now and carry its result into the next primitive |

The new generators are not narrative concatenations:

- `cipherkiln` requires active probes to infer a symbol mapping and then uses
  the inferred mapping to issue the legal state-machine sequence.
- `mazeferry` requires exploration/memory to acquire opaque typed handles and
  then uses those handles in a dependency-correct tool chain.
- `patchferry` requires diagnosing a corrupted transformation or signature
  before the repaired result can be used in a tool chain. It is fully held out.
- `tripleforge` requires discovery, stateful control, and typed tool use in one
  episode. It is fully held out and exists at pairwise-plus-one and deeper
  composition levels.

Each compound generator must have an exact programmatic oracle, disjoint
surface vocabularies across train/holdout, and automated necessity ablations:
replacing any one primitive with a random or echo policy must drive expected
success below 0.20 while the full oracle remains at or above 0.95. On a
disjoint calibration seed pool, choose structural difficulty levels at which
`S0` is below 0.60, then freeze the level distribution. Never retain, reject,
or regenerate an individual confirmatory item based on a model's output. This
prevents both one-half shortcuts and baseline-tailored item selection.

The composition specialist sees `cipherkiln` and `mazeferry` only. Neither any
teacher nor the integrated student trains on `patchferry`, `tripleforge`, the
held-out order reversals, or their surface token pools.

### Stage 1: produce four qualified specialists

Train four independent policies from `S0`: `T_discover`, `T_control`,
`T_tools`, and `T_compose`.

For each policy:

1. collect state-aware DAgger corrections on model-visited states using the
   exact programmatic expert and the existing `OBSERVE / STATE / DECIDE /
   CHECK` trace interface;
2. put loss at the reasoning/emission seam using the C50/C53 weighting and keep
   general C53 replay for retention;
3. after the DAgger gate, run grouped on-policy optimization with exact terminal
   execution reward, success-only efficiency tie-breaks, zero gradient for
   zero-variance groups, an `S0` KL anchor, and no synthetic validity reward;
4. compare against DAgger-only, compute-overmatched additional SFT, and shuffled
   group rewards; and
5. merge the adapter into a full composite and prove that vLLM outputs differ
   from `S0` on a preregistered canary while matching Transformers on frozen
   greedy behavior.

Privileged-hint OPSD is deliberately absent here. The specialist must become
better because executed actions receive better outcomes, not because a
solution-conditioned branch produces attractive token ratios.

Qualification is per core, not an average. A specialist proceeds only if, on
frozen paired seeds:

- its own-core family-macro terminal score is at least `S0 + 0.10`;
- it beats DAgger-only and additional-SFT by at least `+0.05`;
- it beats shuffled-reward training by at least `+0.03`;
- its pass@1/greedy score beats `S0`'s execution-filtered best-of-8 under the
  registered inference-token ledger;
- no non-target retention family loses more than `0.05`; and
- natural close, parse validity, entropy, and correction/backtracking marker
  rates remain inside the preregistered guard band.

If fewer than three primitive specialists and the composition specialist pass,
stop. Distillation cannot create missing teacher headroom, and averaging the
survivors would answer a different question.

### Stage 2: frozen teacher-advantage and locality audit

Before any full integration update, collect a frozen pool of `S0` trajectories
and branch from the exact student-visited prefixes. For the correct specialist,
`S0`, and a deliberately wrong-routed specialist, measure continuation outcome,
action validity, terminal reward, and per-token divergence.

The correct teacher must satisfy all of:

- paired continuation reward at least `+0.08` over `S0`, with a positive
  episode-bootstrap 95% lower bound;
- at least `+0.05` continuation reward over the wrong-routed teacher;
- among the union of the correct teacher's and `S0`'s top-four next tokens,
  force each alternative and estimate its terminal reward with four `S0`
  continuations; the top pressure quartile must beat the bottom quartile by
  `>= 0.08` and the corresponding wrong-route pressure lift by `>= 0.05`; and
- a five-update miniature MOPD run reduces correct-teacher divergence without
  exceeding C52's `0.10` per-row median non-target exact-logit drift limit or
  reducing policy entropy/correction-marker rate by more than 10%.

This is the local version of the literature's teacher-gap law. Failure stops
that teacher. It is not repaired by increasing its distillation weight.

### Stage 3: integrate in policy space

Initialize a fresh rank-matched QLoRA student from `S0`. On every step:

1. the student generates its own trajectory;
2. the prompt's training-domain metadata routes that trajectory to exactly one
   frozen, qualified specialist;
3. that specialist prefills the identical observable prompt and student prefix;
4. store the specialist's top 50 logits/probabilities; and
5. update the student with MOPD's bias-corrected top-k reverse KL and 20%
`S0`/C53 identity-retention replay.

The canonical top-k summand from MOPD equation (5) is
`p_student(v) * log(p_student(v) / p_teacher(v)) - p_student(v) + p_teacher(v)`
for each token in the teacher's top-k set. The final two terms are the
truncation-bias correction. They are not a lumped vocabulary-tail term; an
earlier “corrected tail mass” shorthand in the proposal was inaccurate. The
synthetic full-vocabulary test sets `k = |V|`, where this expression must equal
full reverse KL exactly (the linear correction cancels when summed over the
whole vocabulary).

Every rollout records the exact student checkpoint digest, is consumed by at
most one optimizer step, and is discarded; policy lag may not exceed one
update. A unit test must recover the full-vocabulary loss on a small synthetic
vocabulary before the corrected-tail implementation is allowed to train.

Routing metadata exists only in the trainer. It is not placed in the model
prompt and is not available at evaluation. The specialist receives no hidden
solution or simulator state during the prefill. This is same-observation,
different-policy distillation.

Choose the trainable adapter rank once from a memory-only smoke test and hold it
fixed across MOPD, joint RL, and off-policy SFT; rank may not be selected from
task outcomes. Give every trainable integration arm the same 20% retention
dose. Use the canonical MOPD objective as registered. Entropy-aware, purified,
position-weighted, and SDAR-like objectives are scientifically interesting but
must be separate follow-up experiments. Measure locality with batch size one
at every saved checkpoint. If entropy, exact-logit locality, or correction-
marker guards fail, stop rather than silently switching objectives.

### Arms and compute accounting

All integration arms use the same qualified teachers and frozen data splits.

| Arm | Purpose |
|---|---|
| `S0` greedy and sample-more | incumbent and mission baseline |
| MOPD | primary same-origin on-policy integration treatment |
| joint DAgger + GRPO | tests whether splitting and integration beat one mixed policy |
| off-policy specialist SFT | tests whether successful teacher trajectories alone explain the gain |
| convex/task-vector merge | cheap weight-space integration baseline |
| wrong-route MOPD | permutes teachers within initial-KL bins, preserving dense-gradient magnitude and teacher marginals while destroying semantic alignment |
| best individual specialist and oracle router | distinguishes integrated competence from specialist selection |

Report two separate ledgers:

- **Conditional integration compute:** given already trained specialists, match
  rollout tokens, teacher/student forward tokens, and optimizer target tokens
  across MOPD and off-policy controls.
- **End-to-end compute:** give joint RL the total model-forward and optimizer
  token budget consumed by specialist production plus integration. Also report
  wall time, but do not use wall time as the matching variable.

Evaluation sampling uses the same vLLM backend, prompt, parser, stop rules,
temperature, and token budgets for every arm. Seeds do not justify mixed
backends.

### Primary endpoints

#### 1. Individual capability integration

For domain `d`, define

`I_d = (score(integrated,d) - score(S0,d)) / (score(T_d,d) - score(S0,d))`.

The primary integration gate is:

- mean `I_d >= 0.75` across the four trained domains;
- minimum `I_d >= 0.50` and no negative domain;
- mean normalized integration at least `+0.10` above end-to-end matched joint
  RL and `+0.05` above off-policy SFT; and
- wrong-route MOPD at least `0.05` below correctly routed MOPD.

Report every `I_d`; do not clip values or allow one large-headroom domain to
hide a see-saw loss.

#### 2. Held-out composition

On `patchferry`, held-out order reversals, and `tripleforge`, the integrated
student must:

- improve family-macro pass@1 by at least `+0.15` over `S0`;
- exceed the best individual specialist, including `T_compose`, by at least
  `+0.10`;
- exceed joint RL and off-policy SFT by at least `+0.05`;
- beat `S0` execution-filtered best-of-8 at no greater inference-token cost;
  and
- show a positive paired-bootstrap 95% lower bound in every compound family,
  not only in their pooled average.

This is the breakthrough criterion. Recovering four specialists on four
separate prompt distributions is useful integration; beating each specialist
on a never-trained task that requires several of them is compositional
capability.

#### 3. Transfer and retention

- `patchwheel`, `spindle`, and `gatepost` receive no new training exposure.
  Their macro delta must be nonnegative, with at least two at `>= +0.05`.
- The non-target C53 gym macro, natural termination, action validity, response
  length, entropy, and explicit checking/backtracking rates must stay within
  the registered retention bands.
- Report pass@1, pass@8, unique valid coverage, and execution-filtered coverage
  so a narrowed distribution cannot masquerade as capability.

#### 4. Blackbox transfer

Only after all three whitebox gates, evaluate the one frozen MOPD candidate
through the benchmark CLI. Require at least `+0.05` absolute medium-profile
family-macro improvement over the C53 incumbent, a nonnegative quick delta,
and a positive slow-profile confirmation. No benchmark detail may be inspected
or used to revise the training design.

### Replication and analysis

- Run three independent end-to-end training seeds for the primary MOPD and
  end-to-end joint-RL arms. Cheap deterministic controls may share the frozen
  qualified specialists but must be evaluated on all paired seeds.
- Freeze all evaluation seeds and thresholds before the first update.
- Use at least 128 paired episodes per family/level cell for the confirmatory
  whitebox evaluation, or increase that number if a pre-unblinding power
  simulation cannot detect an absolute 0.08 episode-level effect at 80% power.
- Bootstrap episode ids, stratified by family and level; turns are not
  independent units.
- Plot specialist headroom, student-teacher KL, entropy, exact-logit drift,
  correction-marker rate, integration score by domain, and compound success by
  composition depth.
- Preserve negative specialists, stopped pilots, and failed controls. They
  locate whether the failure was capability production, teacher signal,
  integration, or composition.

## Interpretation Matrix

| Outcome | Conclusion |
|---|---|
| Specialists fail qualification | The current DAgger/RL explorer did not create capability beyond C53/sample-more; no OPSD variant is licensed. |
| Specialists win, but same-prefix teacher advantage or locality fails | Better endpoint policies do not provide safe dense guidance on the student's states; MOPD is unsupported on this model/interface. |
| MOPD inherits individual domains but not compounds | Policy-space integration can form a union, but composition still needs an explicitly seeded higher rung. |
| MOPD beats controls on compounds but not held-out single families or blackbox | The compound gym mechanism is substrate-local. |
| MOPD beats specialists, controls, sample-more, held-out compounds, and blackbox | Strong evidence that same-origin specialist RL plus on-policy policy integration installs broader and genuinely composable capability in the fixed 4B model. |

This matrix is why the experiment is definitive even if it fails. Every major
negative outcome points to a different missing link instead of being summarized
as “OPSD did not work.”

## Alternatives Considered

- **Vanilla reference-conditioned OPSD.** Rejected as the main run because the
  repository's hints lack incremental local signal and the recent literature
  identifies reference shortcuts, uncertainty suppression, and fork collapse.
- **SDPO/SD-Zero directly from environment feedback.** Promising when the
  feedback-conditioned branch already solves more continuations. Rejected as
  primary because the repository's execution-feedback audit did not establish
  that gap. It remains a possible specialist-training auxiliary only after a
  separate predictive-gap gate.
- **SDAR/OPID-style hindsight skills during RL.** Attractive for long-horizon
  credit assignment, but skill extraction/retrieval adds another failure source
  and post-hoc self-rationalization conflicts with C28. Programmatic state-aware
  DAgger is the cleaner first dense signal.
- **Run the existing mixed interactive curriculum unchanged.** It is still the
  right infrastructure and a useful specialist-production pilot, but one mixed
  policy cannot test the MOPD hypothesis or separate a union from composition.
- **More C53 verified-output mixing or another convex adapter blend.** Rejected
  by the second-wall evidence and retained only as a control.
- **TurboQuant in the training run.** Rejected as an orthogonal systems change.
  It may be evaluated later only through the repository's normal inference
  parity process.
- **Immediate multi-round MOPD.** Rejected for the first experiment. One round
  must first clear capability, locality, integration, and composition gates.
  A second round that seeds the next composition depth belongs in its own
  experiment.

## Consequences

- Do not spend the full GPU budget of the current mixed interactive curriculum
  until its machinery has been considered as a specialist-production component
  of this design.
- If this proposal is accepted, create a new self-contained experiment with
  `make new-experiment`, copy the required gym and training machinery, write the
  idea intake naming the interactive curriculum as the closest duplicate, and
  run a fresh adversarial design review before any GPU training.
- Treat the qualified specialist checkpoints as experimental artifacts, not
  new allowed models. All remain descendants of the one permitted model.
- Do not update the claim ledger or shared synthesis until the experiment has a
  result. This record changes the proposed next test, not durable empirical
  knowledge.

## Reversal Criteria

Reverse this priority before launch if any of the following occurs:

- the existing interactive curriculum fails to produce even one execution-
  reward specialist that beats DAgger, extra SFT, and sample-more;
- a frozen prefix audit shows that qualified specialists do not outperform
  `S0` when continuing from `S0`-visited states;
- a small mechanically verified MOPD update cannot stay under the C52 locality,
  entropy, and correction-marker guards; or
- the compound generator necessity ablations reveal a shortcut that lets a
  single primitive solve the task.

Reverse it after launch if joint RL wins the registered end-to-end comparison
or if MOPD only reshapes style/termination without increasing executed compound
success. Promote a second-round rung-seeding experiment only if MOPD clearly
integrates the individual specialists but fails specifically at held-out
composition.
