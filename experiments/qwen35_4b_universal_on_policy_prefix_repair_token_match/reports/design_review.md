# Adversarial Design Review

## Scope

This review covers the model-free task, parent-deployment, rollout, failure-mining,
and prefix-masking design for
`qwen35_4b_universal_on_policy_prefix_repair_token_match`. It inspected only
experiment-owned procedural sources, inherited clean replay artifacts, public
repository contracts, and executable code. It made zero model calls and read no
benchmark source, item, transcript, private output, or score detail.

Training compute is deliberately not frozen here. Actual parent-prefix lengths and
failure-class availability do not exist until the rollout event. A second adversarial
compute review, exact-token streams, and a zero-skip tokenizer receipt are mandatory
before either training arm can be exposed.

## Question and closest duplicate

The closest near-duplicate is
`qwen35_4b_universal_state_table_compiler_token_match`. Its idealized truth tables
were executable but remained off-policy: the candidate tied replay at 16/26, lost
both execute cases, repeated both induction searches to the cap, miscounted a probe,
and propagated repair incorrectly. The present mechanism changes the supervised
state rather than the trace skin. It retains the authenticated parent's exact fresh
thinking prefix, masks that prefix from loss, and supervises only a recovery
continuation plus close and answer.

## Task truth, freshness, and shortcut audit

- Construction seed 77,113 deterministically emits 288 fresh tasks: 48 each for
  declaration-versus-operation parsing, carried state transition, bounded depth-2
  induction, probe scoring, repair propagation, and commit serialization. Source
  SHA-256 is `32589348...1172`; model-input SHA-256 is `7a643e96...a5485c`.
- All source builders compute answers procedurally. Every row carries a truth audit,
  a one-line answer, and an oracle continuation. The model-facing JSONL contains
  only task id, messages, and public routing metadata; hidden oracle and answer
  fields are absent.
- Declaration tasks contain four listed operations but no cycle-advance operation;
  their cycle line is reference data only. Commit tasks intentionally expose already
  verified scratch work and ask for immediate exact serialization. These are
  mechanism probes, not hidden-label tasks.
- Task ids, prompts, and class seeds are unique. The fresh local seed 88,009 is not
  materialized or inspected. No prior local-gate row is copied into construction.

The six classes are hypotheses about transferable recovery states. They do not prove
that the parent will fail often enough. Ten reachable failures per class are required;
an undersupplied class produces a preserved inventory and stops before training.

## Parent deployment and backend audit

The sole parent is authenticated `close_xi` (weights `16e9dc75...c179`, config
`de953bd5...7ff`) over only `Qwen/Qwen3.5-4B` revision
`851bf6e8...d0a`. The pinned vLLM stack's runtime PEFT mapping is a documented silent
no-op for this composite architecture. Passing the adapter directly would silently
collect base-model prefixes and invalidate the experiment.

The first expensive checkpoint therefore performs the repository's explicit
composite LoRA merge and verifies that every nonzero LoRA module was applied. Only a
committed merge receipt may open collection. Collection uses that full local
composite in the experiment-local vLLM runner, whose model override rejects anything
except the exact Qwen3.5-4B architecture fingerprint. Runtime LoRA and model override
are mutually exclusive.

One batched collection event runs all 288 prompts with natural thinking, greedy
decoding, one sample, seed 66,113, and a 1,024-token cap. The metadata sidecar must
record the merged path, null hub revision, no adapter, exact runner hash, input hash,
sampling contract, clean Git state, environment lock, GPU, and all token counts.

## Failure boundary and selection audit

The miner never calls a model. It compares each frozen parent output with executable
truth and records cap contact, missing answer, wrong answer, noncanonical
serialization, declaration-as-operation language, and delayed commit. It ranks only
failed rows using a frozen deterministic seed, then selects exactly ten reachable
prefixes per class.

“First failure” is defined conservatively as the first machine-observable boundary,
not an unverifiable claim about the model's first latent mistake:

- delayed commit cuts at token 33, the first token beyond the registered 32-token
  immediate-commit allowance;
- cap failures use the generation-cap boundary;
- declaration misuse uses the completed thinking boundary where the prohibited
  operation becomes auditable;
- wrong, missing, or malformed answers use the answer boundary, because correctness
  is not externally observable sooner for free-form reasoning.

This means many prefixes contain a complete wrong thought rather than a perfectly
localized semantic error. That is a real remaining limitation, but it still tests the
registered mechanism: recovery from states the parent actually visits. Calling these
latent first-error prefixes would overstate the design and is forbidden.

## Prefix-loss audit

Each selected row stores the exact generated parent token ids and their SHA-256.
The trainer concatenates chat-prompt ids, generated prefix ids, correction ids,
`</think>`, and answer ids directly. Prompt and parent-prefix weights are exactly
zero; the inherited thought/close weights apply only to the corrective continuation.
The trainer rejects empty prefixes, a missing mask flag, close/EOS tokens inside the
prefix, decode mismatches, and overlength rows. Unit tests cover exact masking,
forbidden close tokens, wrong-answer selection, correct-answer rejection, delayed
commit cutoff, declaration misuse, cap failure, and the merged-model gate.

Oracle continuations may be longer than a mathematically minimal patch. They are
permitted only after a real failed prefix and are never model inputs during
collection. A positive result belongs to the six-class prefix-repair package; it
cannot isolate a class or distinguish state placement from correction wording.

## Control, compute, and promotion audit

The planned active control remains an independent same-parent replay continuation.
Both future arms must contain 320 rows, 200 byte-identical replay rows in aligned
positions, one epoch, effective batch eight, 40 optimizer steps, seed 47, and exactly
equal encoded forward tokens with zero skips. The candidate's variable block is
planned as 60 prefix repairs plus 60 disjoint replay fillers; the control's variable
block is 120 replay rows. Those numbers do not authorize training: actual token sums
must be solved and frozen after mining.

The fresh local event remains seed 88,009 and must compare parent, replay, and the sole
candidate together in one Transformers process. Candidate admission will require the
same absolute gate as the predecessor plus strict wins over both controls overall and
over execute/induct/probe. Aggregate seed 78,139 remains sealed. A local pass can open
only one same-backend aggregate gateway and still needs strict all-family lift,
higher-tier confirmation, and a matched-compute sample-more baseline.

## Contamination and checkpoint audit

- Training construction uses only fresh procedural tasks and the inherited clean
  replay pool. No `benchmarks/` path is imported, read, or exposed by the current
  harness.
- The harness permits one stage per invocation. Every stage requires a clean
  worktree and its preceding receipt committed byte-for-byte at `HEAD`.
- Required order is design → merged parent → parent rollout → prefix inventory →
  compute freeze → control training → candidate training → local event. Each stage
  must be checked, committed, rebased, pushed to `main`, and verified in both GitHub
  workflows before the next stage.
- A failure or insufficient quota is a result and must be preserved. Seeds, class
  quotas, policy thresholds, and task rows cannot change after a model event.

## Remaining adversarial risks

1. Failure-conditioned selection emphasizes hard tails and changes the training
   distribution. That is the intervention, but the replay control must prevent
   attributing ordinary continued training to prefix repair.
2. Free-form answer-boundary localization may include irrelevant or harmful parent
   text. Masking prevents imitation loss but does not remove conditioning effects.
3. The declaration-language heuristic can misclassify a verbal discussion as an
   operation. It is supplementary; wrong/missing answer remains independently
   recorded, and the fixed heuristic cannot be tuned after observation.
4. Correct-but-delayed commit is a policy failure rather than an accuracy failure.
   Its inclusion makes the package about bounded deployed behavior as well as final
   correctness.
5. Exact forward tokens will not equalize loss-bearing tokens or gradient content.
   The later compute review must report those deltas; capability, never train loss,
   decides promotion.
6. One local seed is noisy. Strict control-relative admission can reject a useful
   method, but no post-observation threshold relaxation is allowed.

No objection makes the parent collection contaminated, structurally impossible, or
causally uninterpretable at its package-level claim. Training remains unauthorized.

**Verdict:** `PASS_PARENT_MERGE`.

This verdict authorizes only `merge-parent`, followed after its separately published
receipt by `collect-parent`, and then model-free `mine-prefixes`. It does not authorize
stream materialization, training, local evaluation, merge of trained arms, or any
benchmark access. Those require a second pushed-green compute review.

## 2026-07-14 — Post-collection operational clarification

The review's requirement that collection metadata record “clean Git state” applies to
the wrapper's preflight, before it opens experiment-owned outputs. The runner samples
Git state after that open and therefore correctly observes the new untracked log as a
dirty tree. The authenticated receipt must bind the clean preflight commit, match the
runner metadata to that commit, enumerate the permitted artifacts, and explain the
later dirty bit; it must not require the runner's post-open `git_dirty` field to be
false. This clarification repairs an impossible postcondition only. It changes no
model, input, sampling, quota, seed, promotion, or authorization decision, and it does
not broaden the `PASS_PARENT_MERGE` verdict.

## 2026-07-14 — Post-training local-protocol amendment

Paired training completed before local seed 88,009 was materialized or any local
model call ran. The repository-wide vLLM contract now supersedes this review's
prospective Transformers sentence: all three local arms will use identical pinned
vLLM runner bytes and geometry over explicit merged composites. The complete
pre-outcome task, freshness, deployment, gate, and checkpoint audit is preserved in
`local_design_review.md`. Its verdict is `PASS_CONTROL_MERGE`; it authorizes only the
separately checkpointed replay-control merge and leaves candidate merge, local
generation, and benchmark access gated in order.
