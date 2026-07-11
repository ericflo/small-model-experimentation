# Adversarial design review

Self-review completed before result-bearing GPU execution. Four adversarial
lenses were used: mechanism, hidden-label/firewall, statistics, and training
safety. Verdict: **sound with required fixes**, all incorporated into the
preregistration and implementation plan below.

## Mechanism lens

### Finding: a static-successful-turn rerun would duplicate C53

If collection retained only successful episode turns, this would be another
mixture of the exact recipe C53 closed. Required fix: at least 80% of DAgger
rows must be states actually visited by the incumbent, including states after
bad or redundant actions; the expert action must be recomputed against the
mutated episode state.

### Finding: existing OraclePolicy objects are not generally state-aware

Several imported experts advance an internal action index and would repeat an
already-completed step when applied to a model-induced state. Required fix: a
new `expert_decision(episode, visible_history)` interface derives the next
action from current environment state. Expert actions are replay-tested from
random reachable prefixes before collection.

### Finding: dense validity reward would rediscover the C50 format policy

Rewarding valid syntax or expert-action match can improve the proxy without
teaching consequence-sensitive behavior. Required fix: the registered RL
reward is terminal execution score only, with a success-only efficiency
tie-break. Invalid-action and expert-match metrics are diagnostics, not reward.

### Finding: DAgger may contain the entire gain

Without a matched update, an RL improvement could be more exposure to fresh
states. Required fix: train matched additional-SFT from the same starting
checkpoint, states, optimizer-step count, and approximate supervised-token
budget; require RL to beat it.

### Finding: raw token entropy repeats the FTPO mistake

Lexical uncertainty is not equivalent to decision uncertainty and does not
make a shared-weight edit local. Required fix: route data with semantic action
entropy and terminal outcome variance only. No entropy-scaled token loss.

## Hidden-label and firewall lens

### Finding: privileged experts can leak unobservable state into prompts

The expert may use a hidden map, rule, correct program, or gold document to
choose a label. That is legitimate oracle supervision only if the serialized
model input remains deployable. Required fix: store and hash the exact visible
message transcript separately from expert metadata; lint prompts against
hidden spec keys and forbidden benchmark vocabulary. Hidden values may appear
in the target action only when that action is the supervised output, never as
an added observation.

### Finding: proxy-family holdout can be violated by C53 replay

The incumbent blend contains `gatepost` and `patchwheel` rows. Using those as
new-stage replay would make them false holdouts, even though the starting
checkpoint has historical exposure. Required fix: define holdout as **no
incremental interactive-curriculum exposure** and report that scope explicitly;
exclude those families from all new replay rows. `spindle` remains a stronger
never-trained-family holdout.

### Finding: blackbox feedback can become implicit tuning

Repeated Menagerie quick runs previously supported fast recipe search. The
current question requires a clean beyond-recipe test. Required fix: no
Menagerie call before the full proxy gate, one eligible checkpoint only, and
no post-Menagerie checkpoint selection or training.

## Statistical lens

### Finding: episode rows are correlated within seed and trajectory

Turn-level bootstrap would inflate precision. Required fix: aggregate primary
metrics per initial episode seed; paired bootstrap at the episode-group level,
stratified by family. Turns are never treated as independent evaluation units.

### Finding: group-relative reward is undefined without variation

All-fail or all-success sibling groups produce no identifiable policy
direction. Required fix: zero-variance groups get exactly zero advantage and
are reported. Curriculum selection may favor nonzero-variance groups but may
not invent process rewards.

### Finding: family macro and pooled score answer different questions

Large-horizon families contribute more turns and could dominate pooled means.
Required fix: primary proxy score is family-macro terminal score; pooled
episode mean and token-weighted training quantities are secondary.

### Finding: one lucky control seed is insufficient

Required fix: paired frozen eval seeds span every family/level cell; confidence
intervals resample episode ids. Shuffled reward is interpreted by effect size
and paired interval, not a single training-loss contrast.

## Training-safety and operations lens

### Finding: C29 collapse can occur before final evaluation

Required fix: start RL from a zero-delta LoRA on the merged DAgger checkpoint;
use ratio clipping, reference KL, supervised replay, max-gradient norm, and
intermediate closure/parse probes. Stop rather than raise dose after a
retention or KL breach.

### Finding: C49 makes runtime adapter evaluation invalid

Required fix: every result-bearing checkpoint is explicitly merged into the
full Qwen3.5 composite. vLLM receives `--model-id`/`model_override` pointing to
the merged checkpoint. Runtime LoRA is forbidden.

### Finding: injected close tokens are not policy actions

Training them with policy gradient would assign reward to a harness insertion.
Required fix: collection preserves stage-one, injected, and stage-two token
ids separately; injected ids receive zero loss. Prompt-token identity is
checked against the recorded vLLM count.

### Finding: the incumbent checkpoint is absent on this fresh host

Required fix: regenerate it from committed `sft_blend.jsonl`, record the exact
command, hashes, training receipt, and merged-checkpoint hash. All comparisons
use that regenerated artifact in the same session. Historical aggregate
numbers are context, not a substitute for the paired incumbent arm.

## Residual accepted risks

- Programmatic experts are privileged and may teach outputs not inferable from
  visible state in every single episode. Family-heldout transfer and RL's
  outcome-based comparison are the safeguards; a positive DAgger training fit
  alone is not evidence.
- One GPU limits the number of independent training seeds. The design spends
  compute on mechanism controls and paired evaluation rather than pretending
  evaluation confidence intervals include optimizer-seed uncertainty.
- The copied gym shares protocol ancestry with C53. This is intentional for a
  controlled mechanism comparison; unseen-family transfer and Menagerie are
  required before any generality claim.
