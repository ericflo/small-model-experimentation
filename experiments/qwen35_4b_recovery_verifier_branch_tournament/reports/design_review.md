# Adversarial design review

## Verdict

Proceed after the recorded corrections. The experiment now distinguishes a
publicly selectable policy portfolio from hidden-oracle union and from ordinary
same-policy sampling.

## Main objections and resolutions

1. **The 75% rule was designed after seeing both old transfer blocks.** Correct.
   Those files are checksum-frozen and labeled retrospective only. The result
   is measured twice on four new algorithm families that did not exist in any
   prior recovery block; no adaptation is allowed between dev and confirm.
2. **The selector might secretly use hidden tests.** The decision function
   accepts only two `final_visible_pass` booleans. A unit test flips hidden
   outcomes without changing the selected arm. Hidden booleans exist in the
   evaluation payload but are quarantined to scoring; firewall checks reject
   hidden code/output in artifacts.
3. **Two policies simply receive twice the compute.** The primary controls are
   two full six-call trajectories from candidate and from action-only at the
   same 12,288-token reservation. They are scored pass-if-either with hidden
   outcomes, stronger than any deployable selector. The mixed tournament must
   beat both by 3pp.
4. **A random branch choice could explain the portfolio gain.** Gate against
   the exact casewise random-choice expectation, avoiding a high-variance
   single random draw. Retain a frozen hash-random draw as a diagnostic.
5. **The candidate-default tie break may encode family knowledge.** The tie
   break was fixed because it scored 75% on both old blocks, but it never sees
   family identity. Prospective families are new, and an action-default result
   is reported as a mechanism diagnostic.
6. **Visible tests are run after the branch even when the model did not ask.**
   This is the explicit external harness intervention. The same public suite is
   available to both branches; model outputs cannot edit it. It is not counted
   as model-token compute, and same-policy controls receive the more favorable
   hidden pass-if-either score.
7. **The new families may be saturated or malformed.** CPU gates prove initial
   and partial states fail both visible and hidden tests, oracle patches pass
   both, inputs remain immutable where required, and dev/confirm manifests are
   disjoint. Control-first union feasibility cancels a block whose ceiling is
   too low relative to sample-more.
8. **Selection is plumbing rather than a curriculum.** Agreed. This experiment
   is a gated capability producer. Only a two-block positive may authorize a
   separate winner-bank experiment whose training rows are balanced at the
   required conditional transitions. Menagerie remains sealed here.
9. **Entropy/varentropy should choose the policy.** Existing evidence says
   uncertainty is useful for routing/acquisition but unsafe as token pressure,
   and same-prefix winner labels have already been unstable. The harder public
   executable signal is tested first. Entropy/varentropy may diagnose residual
   both-pass/both-fail ties after the frozen decision, never alter this rule.

## Residual limitations

- Hand-built procedural repositories are white-box transfer instruments, not
  Menagerie.
- Pass-if-either sample-more is an oracle upper bound, so the primary bar may be
  conservative.
- A positive portfolio still needs compression into one local checkpoint for
  the goal's final benchmark test.
