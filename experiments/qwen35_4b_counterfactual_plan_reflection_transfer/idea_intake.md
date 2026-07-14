# Idea Intake: Counterfactual Plan Reflection Transfer

## Program Fit

- Existing primary program: `posttraining_and_adaptation`.
- Cross-program relevance: `structured_execution_and_compilers` and
  `interpretability_and_diagnostics`.
- Discovery command: `make related QUERY="counterfactual reflection training
  capability reasoning procedural self reflection J-space"`.
- The material idea comes from the counterfactual reflection experiment in
  *Verbalizable Representations Form a Global Workspace in Language Models*, stripped
  of its consciousness/global-workspace interpretation.

## Prior Evidence

- Closest near-duplicate: `qwen35_4b_bank_the_thoughts`. Its correct synthetic plan
  plus code targets improved depth-3 coverage, while the model's own rationalizing
  thoughts did not. It directly supervised the deployed continuation.
- `qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay` found zero correct
  proposals after inference-time semantic plan materialization, despite healthy
  parsing, transport, exhaustive task reachability, and matched candidate pools.
- `qwen35_4b_commit_slot_semantic_power_replication` found no shared task-held-out
  J-value coordinate: overall J AUC was 0.502 and weaker than slot-margin and
  equal-width non-J controls.
- `qwen35_4b_jacobian_transport_control_replication` cleanly established the narrower
  positive mechanism: supplied target concepts in early context-local J coordinates
  can redirect a downstream consequence, with wrong-donor and random controls.
- The concurrently active
  `qwen35_4b_universal_on_policy_prefix_repair_token_match` trains corrective actual
  continuations at observed failure prefixes. It does not test reflection-only loss on
  a counterfactual branch.

## Novelty Claim

This is the first repository experiment in which the treatment arms never train the
deployed answer: only an appended reflection or auxiliary branch that names a plan
receives loss, and capability is measured on a separate unreflected action branch
sharing the same pre-action context. A separately labeled positive control does train
the action continuation and cannot support the treatment claim.

## Mechanism

Correct plan concepts needed for flexible multi-step execution may be installed in a
shared, verbalizable internal format by teaching what the model should say if
interrupted at that context. If that disposition affects the original context rather
than merely teaching a response to the reflection question, the correct-reflection
adapter should outperform an exactly matched shuffled-reflection adapter when both
are deployed on the action branch, where neither saw answer targets.

## Mechanism-Falsifying Control

The first load-bearing control is a within-family, within-optimizer-step derangement
of the reflection targets.
It keeps common contexts, reflection question, target format, number of plan steps,
training schedule, and target-token distribution matched within every optimizer step
while making the plan behaviorally wrong for each task. If correct and shuffled tie,
the branch carries no usable task-specific mechanism. The second control replaces
reflection framing with an ordinary correct auxiliary label while preserving the
target and rendered prompt-token count. A direct action-branch plan-plus-answer arm is
only a training/generalization ceiling; it cannot validate counterfactual transfer.

## Evidence Output

- Construction receipt: task IDs, compositions, behavioral signatures, prompt and
  target hashes, operation-position support, exact re-execution, answer-omission, and
  derangement checks.
- Training receipts: exact rendered tokens and loss masks, per-arm target-token and
  optimizer accounting, model revision, seeds, adapter hashes, and external paths.
- Capability result: paired family-level greedy and coverage@4/@16 on qualification
  and conditionally confirmation, with frozen and matched-sampling rows.
- Conditional successor only: a replicated behavior pass may license a separate
  experiment with disjoint J-fit, J-confirmation, and causal-confirmation evidence.

## Decision

- Run: proceed only through CPU construction, smoke publication, and adversarial
  design review.
- New program: no.
- Duplicate instead of run: no; supervision placement and evaluation branch differ
  from direct plan SFT and on-policy continuation repair.
- Model/GPU/training/Jacobian authorization: none until reviewed design and exact
  implementation are committed, pushed, and green.

Reserved seeds are construction `73301`, shuffle `73319`, schedule `73323`, retention
`73337`, training `47/53`, calibration `88027`, qualification `88031`, confirmation
`88037`, and retention evaluation `88043`. They may not be changed after an observed
model event.
