# Qwen3.5-4B Materialized Residual Sibling Search Experiment Log

## Scaffold

- 2026-07-13: Created as a new experiment scaffold under
  `structured_execution_and_compilers`.
- 2026-07-13: Read the immediate parent and closest related lines through the
  repository discovery workflow. Narrowed novelty to symmetric materialized
  sibling ranking and residual completion; interpreter materialization itself
  is prior art.
- 2026-07-13: Drafted the preregistration and launched three independent
  read-only adversarial reviews covering statistics, task construction, and
  scientific/resource confounds. No model call was made.
- 2026-07-13: All three reviews rejected the first draft. The required
  unique-first balance was impossible for `negate` and `take_k(1)`; hidden
  selection was tautological; a 512-token-per-sibling ranker was dominated by
  simply completing all siblings; suffix reachability was untested; and the
  registered qualification inference was underpowered.
- 2026-07-13: Rewrote the design around multi-label public-live siblings,
  common-panel function fingerprints, non-filtered hidden/probe outcomes,
  all-24 materialized completion as the primary explorer, cheap no-think raw
  log-probability ranking as a secondary, a public suffix ABI ceiling,
  taskwise first-over resource matching, qualification-only futility gates,
  and a 192-task paired confirmation family. Launched a second adversarial
  read before implementation. No model call was made.
- 2026-07-13: Implemented and ran the model-free construction/protocol smoke.
  It deterministically filled 264 globally disjoint exact-depth-three tasks,
  independently re-audited 34 public-live sets, verified strict protocol and
  taskwise resource machinery, and simulated 0.966 compound confirmation pass
  probability at the registered design alternative. No model was loaded or
  called and no benchmark content was read.
- 2026-07-13: The second adversarial read found and closed runner, resource,
  interface, statistical, and documentation blockers. In particular, raw
  24-way log probabilities are authenticated; mechanics covers every live
  sibling; top-four runs are independent; shuffled alignment is in the
  four-comparator confirmation family; and optional top-four failures cannot
  veto the primary all-24 decisions.
- 2026-07-13: Expanded tokenizer smoke from a mechanics subset to every frozen
  task, all 24 candidates, and every prompt family with condition-specific
  reserves and prompt/token-ID hashes. Bound the design documents,
  configuration, source, and tests into the final model-free receipt.
- 2026-07-13: Three independent reviewers accepted the construction layer after
  the final fixes. Published design/data lock still authorizes no model call;
  mechanics implementation requires a separate audit and lock.
- 2026-07-13: Implemented the complete mechanics boundary and passed two
  independent adversarial code audits. The frozen candidate reconstructs 1,984
  requests, 24 surface-control folds, and 4,032 targeted raw-logprob values;
  authenticates all 189 pinned environment distributions plus tokenizer,
  runtime, transaction, and result-chain receipts; and made zero model calls.
  Model authorization remains withheld until the prepared artifacts are pushed
  and a separate implementation lock is then published and pushed.
- 2026-07-13: Ran deterministic mechanics preparation twice under the pinned
  vLLM environment. Both passes accepted the same 1,984-request inventory and
  existing bytes; the committed preoutcome receipt records zero model loads and
  calls and no hidden, qualification, confirmation, or benchmark reads.
- 2026-07-13: Pushed the reviewed implementation at `48ef078f`, observed both
  repository-validation and research-site CI succeed, and generated a separate
  mechanics-only implementation lock binding that pushed commit and every
  critical source/prepared hash. No model was loaded or called; execution
  remains sealed until this lock is independently committed and pushed.
- 2026-07-13: Pushed the separate lock at `cd82e649` and observed both CI
  workflows pass. The exact engine then initialized, but the live preflight
  aborted before the first experimental generation request because the
  validator tried to invert vLLM's intentionally floored group-aware token
  capacity. No arm `STARTED` receipt, model output, score, or summary exists;
  ordinary internal engine profiling/warmup did occur.
- 2026-07-13: Preserved the failed preflight byte-for-byte and opened an
  append-only v2 repair. Independent incident audits required versioned active
  paths, an incident-bound lock that discloses the prior engine initialization,
  the exact 11-block Qwen hybrid-cache identity, conservative block-based arm
  fit, and validation before publishing a PASS receipt. Retry remains sealed
  pending a reviewed, committed, pushed v2 implementation and separate lock.
- 2026-07-13: Closed every v2 repair blocker. The stable candidate directly
  binds all attempt-1 evidence, uses authoritative floored capacity and exact
  11-block hybrid geometry, rejects the 703/704-block boundary, validates before
  writing PASS, isolates versioned active state, and normalizes only expected
  Git metadata while keeping the scientific runtime exact. Two independent
  final-byte adversaries returned `FREEZE`; 45 mechanics and 71 full experiment
  tests passed, and source-bound preparation reproduced twice with zero new
  model loads or calls.
- 2026-07-13: Pushed the frozen v2 repair at `fa942eef` after rebasing three
  concurrent `main` commits; both CI workflows passed. Generated the separate
  v2 lock binding 33 critical files and explicitly recording one prior engine
  initialization, zero experimental requests, and zero sampled outputs. Retry
  remains sealed until that lock is itself committed, pushed, and green in CI.

## Pending

- Commit and push the generated v2 lock, observe CI, then retry mechanics.
