# On-Policy Failure-Prefix Universal Curriculum Report

## Summary

Model-free design, explicit parent deployment, authenticated rollout collection,
failure-only mining, and the second exact-compute review are complete. Every fixed
class quota passed, and two frozen 320-row streams match at exactly 304,313 forward
tokens with zero skips. Replay-control training is authorized; no training or
capability result exists yet.

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

## Controls

Baseline is authenticated `close_xi`. The mechanism-falsifying control is an
independent same-parent replay continuation matched on exact encoded forward tokens,
optimizer steps, seed, and aligned shared replay. It must train and publish first.
Candidate training is fail-closed on that committed receipt; both arms independently
restart from the parent.

## Oracle Versus Deployable Evidence

Executable truth is permitted only to construct tasks, grade parent failures, and
build corrections. Hidden oracle fields are excluded from the rollout input; commit
tasks deliberately expose verified work as their public task substrate. Prior local
events remain held out; `benchmarks/` remains read-forbidden and the aggregate gateway
stays sealed.

## Interpretation

The parent supplies enough failures in every registered class, and exact forward
compute is matched. That closes substrate and runnability risks, not the mechanism
claim. The selected set is dominated by long capped prefixes, and the candidate has
fewer supervised tokens and lower loss mass than replay. A candidate win would show
targeted repair beats additional replay under equal forward compute, but would not
separate prefix-state conditioning from target-composition effects. “First failure”
still means the first machine-observable boundary rather than an unobservable latent
error.

## Next Experiments

Publish and CI-verify the exact-compute freeze. Train only the replay control, publish
its durable receipt, and wait for both workflows before training the sole candidate.
Local capability design and execution remain separate later checkpoints.

## Artifact Manifest

Parent identity, frozen task hashes, replay hashes, the staged external merged
checkpoint, parent-rollout hashes, stream hashes, and prospective adapter paths are
recorded in `artifact_manifest.yaml`; no capability result exists.
