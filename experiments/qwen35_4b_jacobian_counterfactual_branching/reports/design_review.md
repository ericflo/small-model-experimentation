# Adversarial Design Review: Jacobian Counterfactual Branching

Verdict before implementation: **scientifically worthwhile, but continuation
work is forbidden until native-prefix controllability and exact controls pass**.

## 1. This may be another oracle donor experiment

Every possible public alias is supplied once, so no correct alias is selected or
favored. The 12 branch deltas sum to zero and the final answer averages all
branches. Oracle coverage is evaluation-only. Any branch selection by gold,
branch target score, or hindsight is prohibited.

## 2. Direct steering may survive averaging and manufacture the answer

Balanced branch targets cancel first-order mean delta, but nonlinear layers can
retain alias-dependent bias. Clean and Gram-matched non-J arms plus the full
alias distribution are mandatory. Report per-target final transition matrices;
alias collapse or a fixed identity winner invalidates the result even if
accuracy rises.

## 3. The old lens may not transfer to a native last-thought token

That is the first question, not an implementation excuse. A label-free mechanics
stage requires supplied-target selection and probability lift against non-J.
No native controllability means stop before expensive continuations.

## 4. Alpha tuning can overfit capability

Alpha is selected only on target controllability with mechanics labels sealed,
using the smallest passing member of a three-value ladder. No correctness,
oracle coverage, continuation output, or task target may enter selection.

## 5. Centered gradient directions are not donor-coordinate replacement

Correct. This is a new additive Jacobian intervention, not claimed identical to
the replicated donor clamp. Norms are anchored to that clamp, and direct
mechanics must establish what the additive edit actually does in native context.

## 6. Non-J controls can fail after bf16 even if float geometry is perfect

The independent replication found exactly this footgun. Controls are measured
after addition to each live bf16 residual. Reuse its lattice repair and require
100% norm/span validity. Float-only orthogonality is insufficient.

## 7. Matching only average norm leaves branch geometry unmatched

Non-J must preserve the complete Gram matrix, rank, per-branch norms, and zero-
sum relation by SVD/orthogonal rotation. It also uses the same layers, token,
cache, branch count, sampling seeds, and final ensemble.

## 8. Shared-prefix branching saves compute relative to sample more

That is intended, but cannot hide the baseline. Count prompt prefill, prefix
decode, cache-fork token, continuation decode, final slot, and any cache-free
audit. Compare against two pre-outcome K prefixes of a K=12 fully independent
master pool at sampled-token and total-logical-token parity; report K=12 as a
conservative overmatch. Also report an attention-shape/FLOP proxy.

## 9. Same-prefix clean is an artificially weak sample-more baseline

It is only the causal branch-structure control. Fully independent sample-more is
mandatory and primary. J must beat both, not merely recover diversity lost by
sharing the prefix.

## 10. Cached and full-prefill logits can diverge

The lineage already found cache-sensitive activation differences. Mechanics
compares cache slot logits with exact full prefill and freezes one backend path.
If max alias-logit difference exceeds 0.05, cache scoring is rejected and every
arm pays for full prefill.

## 11. Natural close creates arm-dependent token budgets

All arms use one explicit fixed-cap policy that suppresses close/EOS before
1,024, then applies the same forced commit. This is a deployed intervention and
must be named. Raw close-token rank/probability may be logged, but no arm stops
early. The parent had 100% cap contact, so this changes little absent steering.

## 12. Branch-to-seed assignment can confound aliases

Rotate the mapping by task with a frozen hash. Clean, J, and non-J use identical
row seeds. Full sample-more seeds are separate and frozen. No outcome-dependent
reruns or best seeds.

## 13. Mean probability is not necessarily the strongest selector

Candidate and every pool report mean probability, majority, max confidence, and
minimum entropy. J must beat every stronger registered deployable rule on clean,
non-J, and matched full pools—not a hand-picked weak majority.

## 14. Better oracle coverage without selection is not deployable

Coverage lift is mandatory but insufficient. Exact ensemble accuracy and paired
uncertainty must also pass. Conversely, selection gain without coverage lift
would contradict the proposal-shift mechanism and fails.

## 15. Twenty-four tasks are underpowered

Qualification is a high-effect screening gate: +10pp against every comparator
with paired lower >0. It may miss a modest useful effect, but prevents a roughly
million-token confirmation from following noise. Confirmation doubles task N
and repeats identical gates; stages never pool.

## 16. Alias breadth can hide one dead target

Report every alias row/column. Qualification requires eight predicted and six
successful target aliases; confirmation uses the same. The never-correct
`lemon` public branch remains necessary for zero-sum balance but is never counted
as target-success support.

## 17. Fresh tasks could collide with prior procedural data

Hash complete visible+hidden behavior and first operation against all direct
ancestors before model load. Reject/regenerate collisions and commit the manifest.
Never read benchmark contents.

## 18. Hooks may patch the wrong token or repeated decode steps

Patch only the single cache-fork processing of the 512th thought token, assert
query length one and exact absolute position, and count one application per
layer/branch. Subsequent 512 decode steps must see no hook mutation. Tests use
sentinel deltas and cache lengths.

## 19. Expanding KV cache may alias writable tensors across rows

Require independent batch storage or prove read-only sharing. Hash the first
511 cache tensors before/after every arm and require exact equality. Branch-only
KV rows may differ solely after the fork token.

## 20. Outcome grading can leak into resource matching

K matching consumes only recorded token counts and frozen pool order before task
labels load. Write a resource-selection receipt, hash it, then grade. Unit tests
mutate all labels and require identical K and predictions.

## 21. An intervention that only makes nonsense may appear diverse

Record finite logits, alias mass, entropy, close pressure, token repetition,
unique continuation hashes, and final alias support. Require no collapse versus
clean, and retain exact procedural answer evaluation. Diversity alone is never
a pass.

## 22. A positive is elicitation, not installation

The frozen weights are unchanged. A replicated matched-compute win would show a
test-time capability elicitation mechanism. Installing it into weights or a
controller is a later experiment with its own matched baselines.

## Mandatory pre-GPU checks

- data collision/prompt-leak audit;
- lens hash/rank/token/layer audit;
- pure zero-sum and Gram geometry tests;
- exact non-J span construction tests;
- cache expansion/fork and one-shot hook tests;
- outcome-label mutation and resource-matcher tests;
- model-free smoke with all downstream stages fail-closed; and
- outcome-blind model smoke/mechanics only after a pushed boundary.
