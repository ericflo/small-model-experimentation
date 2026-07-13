# Post-Qualification Adversarial Audit

Completed after automatic `NO_ORDER_SUPPORT_SELECTOR`. It cannot alter the
decision or authorize confirmation.

## Verdict

Accept a valid negative with a useful subthreshold mechanism clue. Ordered-minus-
shuffle probability is better than majority, but not robustly better than cheap
confidence/entropy and not specific to the exact task-matched shuffle. Keep
confirmation absent.

## 1. 38.05% is the best deployable point estimate, so call it a pass

The preregistration requires every direct comparison. Candidate minus minimum
entropy is only two net tasks (+1.77pp, lower -3.54pp); candidate minus max
confidence is three net tasks (+2.65pp, lower -2.65pp). Neither clears the 3pp
point gate or uncertainty. Best point estimate is not evidence of reliable lift.

## 2. The mismatch control cheats with labels, so ignore it

It is intentionally oracle-balanced and cannot be a deployment baseline. Its
44/113 score may benefit from label-conditioned donor selection. But removing it
still leaves three failed deployable comparisons: mean probability uncertainty,
max-confidence point/uncertainty, and minimum-entropy point/uncertainty. The
negative does not depend on treating mismatch as deployable.

## 3. Beating majority proves sample-more was beaten

No. Majority uses three generated thoughts, whereas the candidate also consumes
three full shuffle prefills. The mission comparison would charge that extra
compute against more ordered samples. Since qualification failed, that fresh
K=3-versus-K=6 experiment is not licensed.

## 4. Candidate versus first/majority replicated with positive bounds

Those two comparisons are real: +10.62pp/+8.85pp with lower +4.42pp/+2.65pp.
They show vector probability information beats hard voting. Mean probability,
confidence, and entropy are the relevant stronger baselines and stop promotion.

## 5. Tune log ratios or subtract alias means

Those transforms were named diagnostic-only and deliberately not implemented.
Choosing one after seeing raw-delta outcomes would create a new adaptive score.
It cannot rescue this experiment or access confirmation.

## 6. Drop tied tasks or analyze traces as independent units

There are 97 ties versus minimum entropy because selectors usually agree. That
is the actual operating distribution, not missing data. The task is the frozen
unit; conditioning on disagreements or resampling 339 traces would inflate the
signal.

## 7. Pool the already-collected confirmation for power

Confirmation is a separate gated stage and remains absent locally. Pooling is
forbidden, and opening it after qualification failure would erase the firewall.

## 8. Out-of-pool correct predictions prove capability creation

The candidate chose outside all three ordered argmax answers on 27 tasks and was
correct on eight. This shows weak probability aggregation can surface an answer
hard voting misses. It is still a deterministic readout of the same logits, not
a new model proposal, and it did not pass the selector gates.

## 9. Zero success on one target invalidates breadth

Candidate successes covered 10 of the 11 target aliases, above the frozen eight;
prediction choices covered 11 of 12 public aliases. `horse` had zero candidate
success and deserves reporting, but breadth was not the failed gate.

## 10. The result disproves coherent thought

It does not. The parent independently proves ordered thought changes correctness
relative to exact shuffle. This result says the raw per-alias probability
difference is not a sufficiently task-specific, cost-justified selector.

## 11. Confirmation could still pass

The stage was conditional on qualification and is now unauthorized. Its files
remain absent, `confirmation_opened=false`, and no boundary receipt exists.

## 12. Next routing

Do not tune another scalar or commit-logit selector on these rows. The durable
positive is semantic state transport; the repeated negative is value/ranking.
The next mechanism should intervene before the answer—creating systematic
counterfactual continuations or proposals—and then face a matched-compute
baseline. It needs its own idea intake, fresh data, and adversarial review.
