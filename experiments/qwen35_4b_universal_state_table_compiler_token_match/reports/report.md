# Natural-Language State-Table Universal Curriculum Report

## Summary

The truth-audited state-table curriculum failed its fresh local gate. The candidate
tied exact-token replay at 16/26 correct, lost to its 19/26 parent, and was worse than
both controls on the targeted execute/induct/probe subtotal. No arm was promoted and
the aggregate benchmark remained sealed.

## Research Program Fit

The experiment belongs to `agentic_breadth_installation` and must beat same-parent
exact-token replay before it can contribute broad evidence.

## Method

The 80-row intervention contains 20 each of variable-depth natural-language execution
tables, independent three-hypothesis/five-probe score tables, first-error repair, and
concise commit from a verified table. Both candidate and replay start independently
from authenticated `close_xi`. Each arm has 320 rows, 286,814 exact forward tokens,
zero skips, 40 updates, and 200 position-aligned identical replay rows.

## Results

Construction source SHA-256 is `a7b453af...e88bb`; replay/candidate streams are
`2727e29a...a2b5` / `8e1b8fdc...1355`; token and design receipts are
`163e40a6...f0b8` / `0bac3340...ef837`. All truth audits recompute, all absolute gates
are reachable, and 48 tests plus the frozen smoke pass. The replay control completed
40/40 updates over 320 rows with zero skips, final loss 0.4226, and 294.1 wrapper wall
seconds. Its 169,903,320-byte external adapter has weights/config hashes
`83a741e4...409a` / `13838f2e...843`; the preserved receipt/log hashes are
`b05dc72e...e99a` / `5f4d1fe3...60ba`. The candidate independently restarted from
the same authenticated parent and completed 40/40 updates over 320 rows with zero
skips, final loss 1.059, and 290.9 wrapper wall seconds. Its 169,903,320-byte adapter
weights/config hashes are `36e54804...5d0f` / `7101cc87...4b34`; receipt/log hashes
are `6aab42b3...2be2` / `26907944...c059`.

At fresh seed 88,008, parent/replay/candidate scored 19/16/16 correct, 23/21/22
parsed, 3/5/5 cap contacts, and 438.1/508.1/522.5 mean generated tokens. The target
subtotal over execute, induct, and probe was 4/6, 2/6, and 1/6. Candidate execute was
0/2, induction 0/2, and probe 1/2. It failed the absolute accuracy, parse, cap,
execute, and induction checks; every strict win over parent and replay was false.
Promotion is empty. Local/promotion receipt hashes are `027c0f63...f2869` /
`429770fd...70f5`; no merge or benchmark ran and seed 78,138 remains sealed.

## Controls

The required active control is an independently trained same-parent replay
continuation matched on exact forward tokens, optimizer steps, and 200 aligned replay
positions. Candidate minus control target-token deltas are +1,196 prompt, -1,955
thought, 0 close, and +759 answer. The failed predecessor adapter is not a parent or
control.

## Oracle Versus Deployable Evidence

Generator execution and table recomputation are construction oracles. Model outputs
must be scored without supplying hidden state. Benchmark data stays behind the
aggregate-only firewall and is unavailable unless the sole candidate passes every
frozen local check. It did not, so no benchmark source, item, transcript, result
detail, or aggregate event was accessed.

## Interpretation

The package hypothesis is rejected at this dose and interface. Paired forensics show
two narrow improvements—one trace and one optimization flip versus both controls—but
five parent wins regressed, including both execute cases and one probe. One state
answer was semantically exact but failed serialization only; one execute thought held
the exact target but never committed before cap. Those seam failures coexist with
semantic failures: a cycle declaration became a spurious operation, both induction
cases repeated to cap, and a probe counted two distinct outputs as three. Idealized
truth-audited traces therefore did not align training with the model's deployed
failure prefixes. See `analysis/local_failure_forensics.md`.

## Next Experiments

Preserve and publish this negative. A new experiment may test fresh on-policy
failure-prefix correction with executable oracle continuations, explicit bounded
commit targets, and exact answer serialization. It must use new seeds, exclude this
held-out local event from training, retain a same-parent exact-token replay control,
and pass the unchanged local gate before any aggregate event.

## Artifact Manifest

The parent identity, data/receipt hashes, reserved seeds, and one-stage checkpoint
order are recorded in `artifact_manifest.yaml` and `preregistration.md`.
