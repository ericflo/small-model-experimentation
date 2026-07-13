# Idea intake: validation-policy counterexample curriculum

## Direction

Use the predecessor's exact failure core as a residual curriculum: begin from
a near-correct failed-test state where transaction structure is already right,
then supervise only the changed patch that distinguishes malformed negative
input (`ValueError`) from ordinary missing/insufficient input (`False`).

## Closest prior work

- `qwen35_4b_transaction_invariant_recovery_curriculum` is the direct near-
  duplicate and parent. It installed copy + whole-request validation + atomic
  commit in every target first patch but did not install validation-policy
  fidelity.
- `qwen35_4b_recovery_verifier_branch_tournament` localized the shared atomic-
  reservation failure before the transaction update.
- `qwen35_4b_verifier_conditioned_recovery_bank` established that complete
  seven-transition action-only replay can preserve the local tool loop.
- `qwen35_4b_repo_search_compress_bank` and
  `qwen35_4b_interactive_policy_curriculum` show why marginal operator counts
  and broad correct traces are insufficient: failed-test revision and scarce
  verify/commit transitions must be retained explicitly.

## Non-duplication

No prior run isolates a one-line validation-policy residual inside an otherwise
correct transaction patch and contrasts that counterexample dose with matched
extra transaction/recovery dose from the same learned parent. This is not
generic transaction training, think-token steering, a policy selector, or more
sampling.

## Mechanism and falsifiers

The treatment replaces only 24 `diagnosis_to_changed_patch` rows within the
same 336-row source bank used by control. If matched extra-transaction training
equals or beats it, the counterexample semantics add no causal value. If gains
remain on six training skins but fail the three fresh representations or the
atomic sentinel, it is template learning. Locality, conditional transitions,
verification/commit, broad recovery, and sample-more are independent failure
paths.

## Contamination boundary

Repositories are fresh procedural fixtures. Hidden tests and oracle/partial
patch objects stay inside the host evaluator; the model sees only issue text,
public source, visible tests, and public tool results. Menagerie remains sealed
until every white-box gate passes and is invoked only through its public CLI;
only aggregate and per-family scores are stored.

## Expected information

- Positive: verifier-faithful semantic distinctions can be installed with a
  tiny residual update while preserving a learned agentic loop.
- Control positive too: extra update dose or predecessor replay, not the new
  counterexamples, explains improvement.
- Train-only positive: API/template memorization, not transferable policy.
- Sentinel-only positive: repair of an adaptively known failure, not breadth.
- Locality/retention failure: even one-stratum residual action SFT causes too
  much shared-weight collateral at this dose.
- White-box positive, Menagerie null: useful procedural coding transfer but no
  general capability unlock on the current instrument.

## Decision

Proceed with the isolated one-transition treatment, same-parent matched
control, content-disjoint dev/confirmation, separately gated sentinel, broad
loop retention, exact locality, and sealed paired Menagerie escalation.
