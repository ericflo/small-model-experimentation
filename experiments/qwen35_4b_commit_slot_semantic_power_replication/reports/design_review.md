# Adversarial Design Review: Commit-Slot Semantic Power Replication

Completed before any model call. The review starts from the possibility that the
parent's attractive 1,024 cell is noise and asks how this successor could
mistakenly promote it. CPU generation, parent aggregate analysis, power
calculation, static code review, and tests expose no new model outcome.

## Verdict

Proceed only at fixed cap 1,024 with 113 fresh task units in each seam stage and
all registered task/bootstrap/diversity/interface gates. The first draft's 64
tasks were underpowered (~59% at the parent effect); review increased both
stages to the approximate 80% requirement before model loading. A selection
miss is terminal. A pass opens exactly one equally powered untouched
confirmation, never J value directly.

## 1. The failed parent is relabeled a success

The parent passed pooled gaps but missed mixed tasks.

**Hardening:** preserve terminal `COMMIT_SLOT_SEAM_FAIL`. This is a new
qualification plus confirmation; no parent holdout is opened.

## 2. The mixed-task gate is quietly relaxed

Changing six to five would manufacture a pass.

**Hardening:** do not change the parent. The successor scales the floor to
28/113, below the parent's observed proportion but requiring far more units.

## 3. Winner's-cap bias is ignored

Cap 1,024 was the best of three parent cells.

**Hardening:** acknowledge selection and test only 1,024 on fully fresh tasks.
No claim treats the parent point estimate as unbiased.

## 4. A larger cap chases the trend

Moving to 2,048 could turn budget search into endless tuning.

**Hardening:** cap list is exactly `[1024]`; smoke aborts otherwise.

## 5. Decoder calibration chases alias bias

Post-hoc residual decoders were tempting.

**Hardening:** preserve the original constrained argmax. No bias subtraction,
temperature, alpha, prompt, or alias remapping exists.

## 6. “Powered” is branding rather than arithmetic

The first draft used 64 tasks without calculating power.

**Hardening:** parent task SD and effect imply 113 for one-sided alpha .05,
80% normal-approximation power. Runtime hash-verifies the receipt.

## 7. Normal power is optimistic for a discrete clustered outcome

Only 16 parent task units estimate variance.

**Hardening:** call it planning only. The actual gate is a nonparametric
10,000-resample task bootstrap; replication can fail despite planned power.

## 8. Winner's-curse effect inflates the power calculation

The observed +8.33pp may overestimate truth.

**Hardening:** state power is conditional on the parent effect. Absolute +5pp
and bootstrap gates remain; underestimation of N makes failure more likely, not
a license to relax.

## 9. No-thought is also treated as statistically powered

Its parent variance was larger and N=113 may not give 80% power.

**Hardening:** power claim is only real-minus-shuffled. No-thought remains a
preregistered +3pp point gate and receives a bootstrap diagnostic, not a powered
claim.

## 10. Three traces are counted as 339 independent tasks

Within-task paths share prompt and label.

**Hardening:** primary lower bound resamples 113 tasks; metrics macro-average
task differences. Trace count is descriptive.

## 11. A pooled point gain passes despite broad uncertainty

The parent failure had exactly this shape.

**Hardening:** require one-sided 95% task-bootstrap lower bound above zero in
each seam stage.

## 12. Qualification and confirmation are pooled to rescue one another

Combining 226 tasks could hide a negative split.

**Hardening:** identical gates must pass independently. Pooled analysis is
diagnostic only after both decisions.

## 13. Confirmation thresholds are softened

The parent design used smaller confirmation gaps.

**Hardening:** this replication uses identical +3/+5pp, bootstrap, diversity,
headroom, and interface thresholds in both stages.

## 14. Another seed is tried after a miss

Large cost can create pressure to rerun.

**Hardening:** one stable seed block per stage. Any retry is a new experiment;
raw failure remains.

## 15. Pooled accuracy is driven by one easy alias

Parent `tiger` rows were saturated.

**Hardening:** balance 11 target operations and require correct successes across
at least eight gold aliases.

## 16. The model chooses only one or two aliases

A prior-biased classifier can score above chance on balanced data.

**Hardening:** require at least eight distinct real chosen aliases.

## 17. Alias support is counted per trace rather than semantic class

Many successes for one alias could inflate support.

**Hardening:** support is the set cardinality of distinct alias strings, not row
count.

## 18. Alias lexical frequency changes across experiments

New words would create a new interface.

**Hardening:** reuse the exact 12 aliases, mapping, tokenizer IDs, and prompt.

## 19. Multi-token aliases make the mask incomparable

Tokenization differences can dominate choice.

**Hardening:** model smoke requires 12 distinct leading-space single tokens.

## 20. The mask guarantees apparent format success

Every finite row emits an alias after masking.

**Hardening:** never gate parse. Gate semantic accuracy and require unmasked
top-is-alias >=75% and total alias mass >=50%.

## 21. The mask chooses among negligible alternatives

Even a correct forced choice may be unnatural.

**Hardening:** derive constrained and full-vocabulary metrics from the same
forward; both interface thresholds are load-bearing.

## 22. Fixed `First:` syntax is called a reasoning gain

The slot itself repairs answer mode.

**Hardening:** every real comparison uses the identical suffix. Scope the slot
as a deployment interface, never an internal intervention.

## 23. No-thought still performs transformer computation

The final slot can solve from the prompt.

**Hardening:** call it zero-generated-thought, not zero-compute. Shuffled thought
is the primary coherent-content control.

## 24. Shuffled thought changes length or answer mentions

That would make it an easy straw control.

**Hardening:** exact token-multiset/length equality, source/shuffle hashes, and
moved-position rate are runtime assertions.

## 25. Shuffled thought is out of distribution

It tests order versus bag, not every notion of coherent reasoning.

**Hardening:** make only the ordered-over-shuffled inference. No necessity claim
or free-form generalization follows.

## 26. Correct alias verbalization is copied into the slot

That is answer transport, not certainty.

**Hardening:** preserve mention diagnostics and the same mention tokens in the
shuffle. Mention slices cannot rescue gates.

## 27. Natural-close rows get longer duplicate prefixes

This can inflate evidence.

**Hardening:** replay only thought before natural close and label it. One trace
remains one path.

## 28. EOS-before-cap is force-replayed

That creates an unregistered state.

**Hardening:** malformed rows are incorrect/nonfinite in denominators and never
enter slot/free-form replay.

## 29. Exact-depth tasks are secretly depth one

Then long thinking is mischaracterized.

**Hardening:** exhaustively reject every visible depth-one fit.

## 30. First operation is not identifiable

Several pipelines may imply different answers.

**Hardening:** enumerate all concrete depth-two matches and require one shared
first-operation type.

## 31. `negate` algebraic reordering returns

The parent already found this target invalid.

**Hardening:** exclude it from first-operation target support.

## 32. Hidden examples enter the prompt

This would leak evaluation information.

**Hardening:** render visible examples only; hidden rows serve construction and
fingerprint audits.

## 33. Benchmark content contaminates later capability work

Reading benchmark suites cannot be undone.

**Hardening:** self-contained procedural data only; never read/import
`benchmarks/`.

## 34. Fresh rows overlap one of five parents

Repeated tasks could reproduce the hint mechanically.

**Hardening:** 322 unique fingerprints and zero overlap with all five direct
parents before design freeze.

## 35. Seed blocks overlap across stages

Shared traces would invalidate confirmation.

**Hardening:** distinct selection/confirmation trace, free-form, and shuffle
base seeds plus disjoint future-stage seeds.

## 36. Parent aggregate data leaks new outcomes

Power planning uses the parent result.

**Hardening:** parent is terminal and public in-repo evidence. The receipt hashes
only its aggregate diagnostic; all successor tasks/seeds are fresh.

## 37. Model smoke reveals correctness

Even one example can influence design.

**Hardening:** smoke stores no chosen alias, task correctness, trace text, or
comparison. It runs only after immutable design anchoring.

## 38. Batch/backend differences manufacture a gap

Qwen3.5 is batch-sensitive and HF/vLLM seeds differ.

**Hardening:** unpadded Transformers batch one, bf16 SDPA, one revision, one
runner for every arm.

## 39. Cache behavior differs across real and shuffled slots

Cached history could make the comparison asymmetric.

**Hardening:** cache samples native tokens only. Every slot arm recomputes its
complete exact sequence cache-free.

## 40. Trace generation cache silently fails

Full-prefix forwards at every token alter cost/numerics.

**Hardening:** require one initial prefill then only length-one forwards on every
native and free-form row.

## 41. Real and shuffled logits come from separate decoding rules

Different temperature or sampling would confound order.

**Hardening:** both are deterministic argmax from the same slot method and alias
IDs; no seed affects the readout.

## 42. Close-only free-form rescues a failed slot

Its sampled decoding is not matched to constrained argmax.

**Hardening:** free-form remains diagnostic and is absent from `seam_gate`.

## 43. Partial long-run results are summarized

Two-hour execution invites early peeking.

**Hardening:** exact 339/339/339/339/113 cardinality plus cache/finite/multiset
checks precede raw scientific files and summary. Progress prints counts only.

## 44. A crash is followed by selective manual completion

That could change row membership.

**Hardening:** no manual row injection. A rerun must reproduce all stable IDs or
become a separately documented experiment.

## 45. The configuration changes after preregistration

Machine gates can drift while prose looks frozen.

**Hardening:** design boundary verifies ancestor commit plus README,
preregistration, design-review, and semantic-config hashes.

## 46. Data files change after CPU smoke

Task swapping could tune the result.

**Hardening:** runtime verifies manifest and every split hash before model load.

## 47. The power receipt changes with the desired conclusion

N could be retrofitted after outcomes.

**Hardening:** hash parent analysis, N, planned power, and receipt into the
pre-model design; runtime refuses mismatch.

## 48. The bootstrap seed is chosen for a positive lower bound

Monte Carlo quantiles vary slightly.

**Hardening:** stable stage-specific bootstrap seeds and 10,000 resamples are
frozen. No seed sweep or analytic substitution.

## 49. A one-sided interval is described as two-sided certainty

Directional testing can sound stronger than it is.

**Hardening:** every artifact says one-sided 95% lower bound; report two-sided
diagnostics separately if computed.

## 50. The 20% floor is near chance

Chance is 8.33%, but absolute success can still be weak.

**Hardening:** 20% is paired with +5pp shuffled, +3pp no-thought, positive task
lower bound, 28 mixed tasks, and eight-alias support.

## 51. The 70% ceiling unnecessarily kills a strong interface

A saturated seam could itself be useful.

**Hardening:** headroom is required specifically for later value causality. A
ceiling result is preserved as constrained elicitation and branched separately;
it cannot open this value ladder.

## 52. Diversity gates are impossible at observed accuracy

Too few successes could span eight aliases.

**Hardening:** CPU gate receipt verifies integer compatibility: at least 68
successes are required by the accuracy floor across 339 rows, ample for eight
aliases and 28 mixed tasks.

## 53. Alias balancing is only global, not per split

One stage could omit hard operations.

**Hardening:** each 113-task seam split has 10 or 11 rows per eligible target;
CPU manifest records counts.

## 54. J value is fit immediately after qualification

That would skip replication.

**Hardening:** value stages remain fatal-unavailable; only an untouched seam
pass can license later implementation.

## 55. Reserved value tasks leak into seam work

Their existence might tempt exploratory use.

**Hardening:** seam runner reads only its named split. Future splits prove
freshness and remain unopened.

## 56. A seam pass is called J-space certainty

No activations are measured in the scientific seam.

**Hardening:** call it behavioral qualification only. The lens is hash-checked
but unused.

## 57. Constrained choice is called installed capability

Gold labels evaluate a closed vocabulary.

**Hardening:** require a later label-free controller and matched-compute held-out
gain before capability language.

## 58. More test-time compute is ignored

Real/shuffled prefixes cost 1,024 generated tokens plus prefills.

**Hardening:** report generated/prefill/answer tokens, forwards, runtime, and
peak memory. This stage has no capability endpoint.

## 59. Secondary metrics rescue the primary

Probability, margin, free-form parse, or a subgroup may look positive.

**Hardening:** only the conjunction in `seam_gate` determines labels. Preserve
all other metrics without rescue.

## 60. Claim pressure or concurrent work causes premature numbering

This is a high-interest line on shared `main`.

**Hardening:** no claim ID during re-grade; synchronize, check, commit, and push
experiment-local artifacts before shared terminal updates.

## Required assertions before model loading

1. 322 unique exact-depth tasks; zero overlap with five parents;
2. balanced target counts and no benchmark content;
3. exact model/lens/token/backend contracts;
4. passing integer gate and 113-task power receipts;
5. immutable prose/config/data/parent-analysis hashes;
6. outcome-blind smoke schema; and
7. later stages fatal-unavailable.

## Required assertions before untouched confirmation

1. complete 339-trace qualification and all 1,130 slot/control rows;
2. every qualification gate independently passed;
3. all five raw-row hashes plus summary hash frozen;
4. no cap/decoder/threshold/seed change; and
5. confirmation remains 113 fresh task-bootstrap units.
