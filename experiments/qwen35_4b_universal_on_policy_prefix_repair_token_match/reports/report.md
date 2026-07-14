# On-Policy Failure-Prefix Universal Curriculum Report

## Summary

Model-free design, explicit parent deployment, authenticated rollout collection,
failure-only mining, and the second exact-compute review are complete. Every fixed
class quota passed, and two frozen 320-row streams match at exactly 304,313 forward
tokens with zero skips. Both independently initialized arms have now trained and
authenticated, and the fresh same-backend local gate is frozen. No capability result
exists yet.

## Research Program Fit

The experiment belongs to `agentic_breadth_installation`. It changes the intervention
state from idealized truth traces to on-policy deployment prefixes while retaining
the universal line's exact-token replay and strict promotion contract.

## Method

Construction seed 77,113 produces 288 truth-audited tasks balanced across six failure
classes. The authenticated `close_xi` adapter is explicitly merged because runtime
vLLM LoRA is a verified silent no-op. One greedy natural-thinking vLLM event collected
288 parent outputs at seed 66,113 and cap 1,024. The frozen model-free miner selected
ten reachable failures per class and masks every generated parent-prefix token from
loss. Deterministic stream construction combines 200 position-aligned shared replay
rows with either 60 repairs plus 60 replay fillers or 120 disjoint replay-control
rows. The actual training encoder and pinned tokenizer measure every final row.

## Results

CPU feasibility, deterministic generation, and adversarial design review passed.
Source/model-input hashes are `32589348...1172` / `7a643e96...a5485c`; design receipt
hash is `98c6a168...5638`. The parent composite merge then applied 128/128 nonzero
LoRA modules. Its single weight shard is `4933f2dd...eb373` and its external merge
receipt is `1fbc84b3...5557`. The frozen parent event then completed all 288/288
rollouts with 170,252 sampled tokens at 849.923 tokens/s. Rollout/metadata/log hashes
are `8010632f...3b17f` / `9fe81276...664` / `ed0d4fc4...26b7` and the authenticated
receipt is `c6b98b79...74fa`. The initial postvalidator rejected only its own
post-open dirty-tree condition; explicit recovery reran no generation and bound the
completed event to commit `21e1eb59`. At that rollout checkpoint, no failure grading
or downstream event had run.

The separately published rollout then opened model-free grading. Of 288 rows, 230
failed at least one registered condition and 58 passed; all 230 failures had a clean
reachable prefix. Available failures by bounded-induction/commit/declaration/probe/
repair/state class were 46/48/35/24/36/41, clearing every quota of ten. The 60-row
repair source is `30141538...d84b8`; full inventory is `7230af52...dfe7`.
Selected prefixes contain 47,123 masked tokens, with min/mean/max
33/785.383/1,024. Forty-two cut at the generation-cap boundary, ten at the first
token beyond the commit budget, and eight at the answer boundary.

The final control/candidate hashes are `541805df...be6` / `9a43f3be...03f1`, and
the exact token receipt is `eb08026f...e0cfc`. Both arms contain 320 rows, 304,313
forward tokens, zero skips, 200 aligned common rows, and 40 updates. The longest row
is 2,991 of 4,096 tokens. Forward compute is equal, but target composition is not:
candidate minus control is +33,421 masked-context, −33,949 think-target, zero
close-target, +528 answer-target, and −33,421 total target tokens. Candidate/control
nonzero-weight tokens are 111,983/145,404; absolute loss masses are
25,049.4/31,311.2. The second review records this ambiguity and authorizes only the
control. No model load, adapter training, capability measurement, or benchmark event
ran during the freeze.

After compute-freeze commit `a8529c04` passed both workflows, the replay control
trained for one epoch and exactly 40 updates. It encoded 320/320 rows with zero skips,
finished at loss 0.4588, and took 272.8 trainer seconds. Log/receipt hashes are
`a49076ec...3501` / `f78f2069...d6de`; adapter config/weights hashes are
`0dfd9bda...120f` / `bb59d3bd...5154d`. The 169,903,320-byte adapter contains 256
finite, nonzero tensors and 42,467,328 elements. This authenticates the control
artifact but is not capability evidence.

After that control checkpoint passed both workflows as commit `b690a4b3`, the
prefix-repair candidate independently restarted from the same parent. It likewise
encoded 320/320 rows with zero skips and completed exactly 40 updates over one epoch.
Final loss was 1.288; trainer/wrapper times were 282.4/298.2 seconds. Candidate
log/receipt hashes are `e895c546...ca0` / `846d8107...7098`; adapter config/weights
hashes are `91b7db57...37de` / `85811191...0f14`. Its 169,903,320-byte adapter also
contains 256 finite, nonzero tensors and 42,467,328 elements. This completes the
paired operational training stage, not a capability comparison.

After paired training, local seed 88,009 froze 26 truth-audited tasks, two per
registered universal skill. Source/model-facing/receipt hashes are
`9682744e...acdee` / `ff407551...ce988` / `3982d5b8...6e85a`. Model input contains
only ids, messages, and public metadata. It has zero canonical-message overlap with
658 training/collection messages and 234 regenerated messages at prior reserved
local seeds. No model was called.

Before any local outcome, the active repository inference contract required a
symmetric amendment from the prospective Transformers process to the pinned vLLM
runner. Parent, replay, and candidate will all deploy as explicit composites with
identical natural-thinking, greedy, seed, token cap, batch geometry, and runner bytes.
The frozen absolute and strict control-relative promotion rules are unchanged. Local
review verdict `PASS_CONTROL_MERGE` authorizes only the separately checkpointed
replay-control merge.

After local-design commit `6dc0e677` passed both workflows, that replay-control merge
applied 128/128 nonzero LoRA modules. Tracked receipt/log hashes are
`bc78f332...d550` / `7ab404b8...8995`; external merge receipt and 9,078,620,536-byte
weight shard hash to `aa763255...45a3` / `7ab4c419...6e2e`. The saved composite
passes the exact Qwen3.5 architecture and frozen local engine-request gate. This is
an authenticated deployment artifact, not capability evidence.

## Controls

Baseline is authenticated `close_xi`. The mechanism-falsifying control is an
independent same-parent replay continuation matched on exact encoded forward tokens,
optimizer steps, seed, and aligned shared replay. It must train and publish first.
Both arms have now trained independently from the parent, and candidate preflight
authenticated the committed control receipt, log, and external adapter before model
load. The local deployment order is published parent composite, replay-control merge,
candidate merge, then one three-arm vLLM local stage; the replay-control merge is now
complete, and every remaining transition keeps its own published receipt.

## Oracle Versus Deployable Evidence

Executable truth is permitted only to construct tasks, grade parent failures, and
build corrections or the fresh local gate. Hidden oracle fields are excluded from
both rollout and local model input; commit tasks deliberately expose verified work as
their public task substrate. Local seed 88,009 was materialized only after training,
and its messages are disjoint from training and prior reserved local seeds.
`benchmarks/` remains read-forbidden and the aggregate gateway stays sealed.

## Interpretation

The parent supplies enough failures in every registered class, and exact forward
compute is matched. The fresh local gate and same-backend deployment path are now
also frozen. That closes substrate and runnability risks, not the mechanism claim.
The selected set is dominated by long capped prefixes, and the candidate has fewer
supervised tokens and lower loss mass than replay. A candidate win would show targeted
repair beats additional replay under equal forward compute, but would not separate
prefix-state conditioning from target-composition effects. “First failure” still
means the first machine-observable boundary rather than an unobservable latent error.

## Next Experiments

Publish and CI-verify the replay-control composite. Then merge and publish the
candidate. Only after both merge receipts are committed may the three-arm local event
run. Aggregate access remains conditional on the strict local gate.

## Artifact Manifest

Parent identity, frozen task hashes, replay hashes, the staged external merged
checkpoint, parent-rollout hashes, stream hashes, and both trained adapters are
recorded in `artifact_manifest.yaml`; local task/input/protocol hashes are tracked and
the prospective merged-arm paths are registered. No capability result exists.
