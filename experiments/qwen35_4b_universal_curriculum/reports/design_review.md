# Adversarial Design Review

## Verdict

Revise before GPU work. The original scaffold is a useful doctrine prototype, but its
v1 corpus and runner are not admissible evidence. The revised pilot may run only after
the blocking items below have automated gates.

## Blocking Findings

1. **Contradictory induction supervision.** In 16/600 v1 induction rows, the trace uses
   the first probe-consistent second operation returned by `find_single`, while the final
   answer is computed with the hidden sampled operation; those operations disagree on
   the query. A correct reasoning target and answer therefore conflict.
2. **Nominal depth is not behavioral depth.** At least 33/600 sampled two-operation
   rules are indistinguishable from one primitive on a strong deterministic witness set.
   This repeats the C12/C13 nominal-depth footgun.
3. **The documented smoke command is dead.** README and artifact manifest pass
   `--n-induct/--n-execute/--n-select`, but the generator accepts only `--mix`.
4. **The advertised run never started.** There is no run directory or log, and the
   `universal1` adapter directory is empty. README's “in flight” status is false.
5. **The shell chain is fail-open.** It omits `set -e`, can continue after training
   failure when a stale adapter exists, and authenticates success through a log being
   written by the same process.
6. **The search path violates the current benchmark firewall.** It invokes the suite
   through the older `bench.py`, which captures raw child streams and reads result files.
   Current policy requires the trusted aggregate gateway, private temporary raw output,
   and merged-checkpoint same-backend evaluation.
7. **Adaptive benchmark overfitting is possible.** Repeatedly optimizing on public
   per-family deltas can fit ten aggregate cells even without item access. Freeze the
   initial factorial, use fresh seeds once, and reserve independent quick and medium
   confirmation. Any post-result curriculum change belongs in a successor experiment.
8. **“All families improved” is noise-sensitive.** Quick cells are discrete and some
   family deltas have high event variance. A single event may screen candidates, but the
   claim requires independent seeds and paired uncertainty; zero is not a strict boost.

## Required Repairs and Gates

- Generate every value from a recorded executable specification. Validate every row,
  reproduce the checked-in corpus byte-for-byte, and fail on duplicates or malformed
  answer seams.
- For induction, require query-identifiability across all probe-consistent candidate
  compositions, a real dead-end candidate, and a witness that no primitive implements
  the sampled composition.
- Record actual tokenizer lengths and require zero skipped targets at the frozen maximum
  length. Report forward-token exposure and unique-row/task support per arm.
- Include broad replay as a retention control. Compare designed-only, replay-only, and
  combined arms before attributing any movement to a universal feature.
- Use Qwen/Qwen3.5-4B at the pinned revision only. Base and candidate must use merged
  composite checkpoints and the same `qwen_vllm` gateway protocol.
- Run a real train smoke and non-benchmark held-out generator smoke before consuming a
  benchmark seed. Preserve failed controls and adapters through the artifact manifest.
- Initial promotion gate: positive aggregate and no negative family delta on a fresh
  quick event. Confirmation gate: positive mean aggregate and strictly positive mean
  delta for all ten public families across independent quick events, followed by the
  same condition on medium. A durable claim additionally needs paired uncertainty and a
  matched-compute sampling comparison.

## Mechanism-Falsifying Read

If replay-only matches the combined arm, the new synthetic lessons added no transferable
feature. If designed-only learns its local substrate but hurts benchmark families, the
result is format-local installation or capture, not universality. If the combination
beats replay only on the axes mirrored by its lesson types, call it axis transfer, not a
universal feature. Only broad, replicated held-out movement licenses the stronger term.

## Scope Decision

The initial clean factorial remains inside this not-yet-result-bearing experiment.
After its first aggregate result, any adaptive curriculum, new substrate, replication,
or confirmation is created as a separate experiment directory, with fresh intake and
design review, so result-bearing history is never rewritten.
