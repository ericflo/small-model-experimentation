# Preregistration: interactive policy curriculum

Frozen before any result-bearing GPU run. CPU selftests and implementation
smokes may repair plumbing, but changes to hypotheses, splits, primary metrics,
or gates require a dated amendment here before observing the affected result.

## Causal question

Does learning on current-policy state visitation plus complete-trajectory
execution reward improve a shared agentic policy beyond C53's static
completion recipe?

The treatment has two intentionally sequential components:

1. **Visited-state correction:** state-aware programmatic experts label the
   next decision at states reached by the incumbent, including recoverable
   states following malformed, refused, redundant, or merely suboptimal
   actions.
2. **Consequence optimization:** grouped trajectories from the DAgger policy
   receive exact terminal environment scores. A clipped, KL-tethered policy
   update increases complete sampled trajectories that outperform siblings
   from the same initial episode.

## Frozen environment split

Training families: `kilnrite`, `glyphgate`, `loomfix`, `ferrier`,
`burrowmaze`. These span protocol execution, active induction, repair,
dependent tools, and exploration/memory.

Unseen transfer families: `gatepost`, `patchwheel`, `spindle`. No state,
expert label, rollout reward, or replay row from these families may enter
training or curriculum selection. They may be evaluated only on frozen seeds.

Atoms from all non-heldout C53 families are retention instruments only. The
benchmark firewall is unchanged and absolute.

Exact seeds and counts live in `configs/curriculum.yaml`; code must emit a
split checksum in every receipt.

## Common process trace

Every DAgger expert target uses four short fields inside `<think>`:

```text
OBSERVE: the latest consequence relevant to the decision
STATE: compact progress, evidence, remaining budget, and unresolved goal
DECIDE: one of PROBE, TOOL, REVISE, VERIFY, or COMMIT, with a causal reason
CHECK: the expected next observation or terminal criterion
```

The answer channel contains exactly the environment's one-line action. Expert
traces are derived programmatically and capped at 120 words. The model never
receives serialized hidden state, gold reward, hidden rule identity, hidden
map edges, or the word "oracle" in its prompt. Privileged state may determine
the training label, just as a verifier determines a supervised target.

Think-token loss weight is 0.2 and action-token loss weight is 1.0, matching
the C50 emission-seam law. C53 replay rows comprise 30% of DAgger minibatch
mass, stratified across atom, recovery, oracle-trace, and episode kinds.

## Semantic uncertainty routing

Entropy is diagnostic and acquisitional, never a token-local pressure target.
For K sibling rollouts from the same episode:

- `operator_entropy`: entropy of the first semantic operator
  (PROBE/TOOL/REVISE/VERIFY/COMMIT/INVALID);
- `outcome_variance`: variance of exact terminal scores;
- `outcome_varentropy`: variance of `-log p(bucket)` over discretized outcome
  buckets;
- `confident_failure`: low operator entropy with low mean terminal score.

DAgger oversamples confident failures, where imitation supplies a missing
alternative. GRPO uses nonzero-outcome-variance groups, where consequence
credit is identifiable. High lexical entropy without semantic or outcome
variation is ignored. Thresholds are frozen in config before collection.

## DAgger data and gate

The incumbent is rolled out once greedily and once with the frozen sampling
protocol on every train seed. At each nonterminal visited state, the
state-aware expert chooses an action valid for the current mutated environment,
not merely the action index from an untouched oracle trajectory. Expert-only
demonstrations provide at most 20% of DAgger rows; the majority must be
model-visited states.

DAgger is trained as one QLoRA adapter on the regenerated merged C53 blend and
then explicitly merged into the composite checkpoint. Runtime LoRA inference
is forbidden by C49.

On frozen eval seeds, DAgger must satisfy all of:

- train-family macro terminal score `>= incumbent + 0.08`;
- unseen-family macro delta `>= 0`, with at least one unseen family `>=+0.03`;
- mean action-validity and natural-close rates no worse than `-0.03`;
- frozen atom retention macro no worse than `-0.03`;
- no family loses more than `0.10` without an offsetting, preregistered
  diagnosis.

One interface-only repair is allowed before RL if the failure is mechanical
(target truncation, parser mismatch, expert invalidity, or replay imbalance).
It must be documented before rerunning. A correctness failure is terminal for
this DAgger design.

## Execution-reward objective

For each initial episode, sample K trajectories from the current DAgger
checkpoint under one frozen vLLM protocol. Let terminal score be `s in [0,1]`.
The registered reward is:

```text
r = s * (1 - success_turn_penalty * max(0, turns - oracle_min_turns))
```

with a small penalty applying only when `s > 0`; failed trajectories receive
no synthetic validity or expert-match reward. Thus the update cannot earn
credit merely for formatting. Within each episode group, normalize rewards to
zero mean/unit standard deviation. Groups with zero reward variance receive
zero policy advantage and remain diagnostic rows.

The sequence loss covers actually sampled thinking and action tokens; injected
force-close tokens receive zero loss. Think tokens retain weight 0.2 and
answer/action tokens weight 1.0. PPO ratio clipping, gradient clipping, an
initial-policy KL penalty, and supervised replay on DAgger/C53 rows guard C29
collapse. Per-turn losses are additionally weighted by inverse trajectory
length so long-horizon episodes do not dominate solely by contributing more
training rows. A checkpoint is rejected immediately if mean sampled-token KL,
natural close, parse, or atom retention crosses its config ceiling.

## Controls

1. **Incumbent frozen:** C53 blend, same prompts/backend/budgets.
2. **Matched sampling:** incumbent K trajectories and equal tool/turn budget;
   report deployable first/greedy score and hidden oracle coverage separately.
3. **DAgger-only:** checkpoint entering RL.
4. **Compute-overmatched additional SFT:** start from DAgger and train on
   expert labels from the same newly visited states. It receives 1.5x the RL
   optimizer steps, conservatively approximating RL's extra reference forward
   while giving the static control more, not less, update opportunity.
5. **Shuffled reward:** start from DAgger and permute advantages across episode
   groups within family and level, preserving the marginal reward and length
   distributions.
6. **Oracle ceiling:** state-aware programmatic policy, clearly nondeployable.

## RL gate and iteration rule

Real-reward RL must beat DAgger and matched additional SFT by `>=+0.05` on
train-family macro terminal score, improve at least three of five train
families, be nonnegative on unseen-family macro, exceed shuffled reward by
`>=+0.03`, and pass every DAgger retention guard.

If the first RL iteration is positive but below the full gate, one further
on-policy collection/update is allowed only when outcome variance remains
nonzero and the KL/retention guards have at least 50% margin. If the first
iteration is negative versus matched SFT, no dose escalation is allowed; the
result is a mechanism failure.

## Menagerie decision

Only the first checkpoint clearing the complete RL gate is eligible. Run two
fresh paired medium events against the regenerated C53 blend, same backend and
seed within each pair. Run quick as a regression event. The strategy win is:

- mean paired medium delta `>=+0.05`, both seeds positive;
- paired quick delta `>=-0.02`;
- if those hold, paired slow confirmation delta `>0`.

Menagerie cannot select hyperparameters, curriculum families, or checkpoints.
Only aggregate/per-family score surfaces permitted by its CLI may be retained;
no benchmark item or transcript enters this experiment.

## Interpretation matrix

- DAgger wins, RL does not beat matched SFT: live-state coverage matters;
  consequence optimization is not the added mechanism.
- RL beats DAgger/SFT but not shuffled: invalid; generic update/dose explains
  the result.
- RL beats controls in-family only: interactive policy is substrate-local;
  no Menagerie run.
- RL clears proxy gate but not Menagerie: proxy-policy gain does not transfer;
  preserve as a blackbox negative.
- RL clears proxy and Menagerie gates: capability elicitation beyond C53's
  static recipe.
