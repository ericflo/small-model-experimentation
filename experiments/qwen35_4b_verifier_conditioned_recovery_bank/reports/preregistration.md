# Preregistration — verifier-conditioned recovery banking

Frozen on 2026-07-12 before any result-bearing harvest, model training, or evaluation. Host-oracle CPU checks and a two-step GPU integration smoke were run only to validate plumbing and are excluded from evidence.

## Primary question

Does balancing supervision at public verifier-conditioned state→action transitions install transferable recovery behavior in a looping Qwen3.5-4B coding agent, beyond matched happy-path training, an explicit runtime recovery scaffold, and matched-compute sampling, without erasing ordinary solve/verify/commit behavior or broadly perturbing unrelated logits?

## Fixed lineage and firewall

- Only `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` may be loaded.
- Every arm warm-starts from the immutable C54 `apex_replay` merged checkpoint, itself a derived checkpoint of that exact model. The same checkpoint is the frozen incumbent and harvest policy.
- Bulk generation uses this experiment's pinned vLLM runner. Compared arms never mix inference backends.
- All repositories are newly procedural. Benchmark source, items, transcripts, and family modules remain unread; no benchmark content enters training.
- The model sees issue text, public source, tool receipts, and visible-test output. Hidden test code is host-only. It is used only to admit executable final repairs and to score final workspaces; serialized private state is limited to booleans in host receipts.

## Fresh substrate and splits

Every fixture contains two independent defects. The initial workspace and a registered host-only partial repair must both fail visible and hidden tests; the oracle must pass both. The partial repair creates a real visible `FAIL` state without exposing a solution.

- Harvest: six train families × 12 tasks, seed 84610; four sampled trajectories/task; eight turns; 512 think + 256 answer tokens/call.
- Calibration/arm selection: the six train families × 5 new tasks, seed 84700.
- Transfer dev: four never-trained algorithm families × 10 tasks, seed 84800.
- Transfer confirmation: the same family names but 10 new tasks/family, seed 84900.
- Locality: 48 frozen unrelated contexts, seed 85000.

Task IDs and manifests must be disjoint across harvest, calibration, transfer dev, and confirmation.

## Harvest and executable banking

For each covered harvest task, select the shortest successful model trajectory by replay-minimized patch count, sampled tokens, then turns. Collapse the source-changing edits to one full-file patch and admit it only when a fresh visible+hidden replay passes.

Construct seven rows per covered task:

1. `start→inspect`
2. `inspect→patch`
3. `rejected_patch→changed_patch`
4. `failed_test→diagnose`
5. `diagnosis→changed_patch`
6. `patch_ok→verify`
7. `passed_test→commit`

The failure prefixes are deterministic and public: an exact-anchor patch rejection, or application of the registered partial repair followed by a visible failure. Recovery targets are derived from the selected model repair, not the host oracle. A bank is invalid unless every initial/partial/final state has the registered executable truth values, all transitions are present, no row exceeds 4096 tokens, every admitted final patch replays, and coverage is ≥35% overall, ≥15% per family, and ≥24 tasks.

Set each operator's total weighted action mass equal to the whole bank's raw action-token mass. Within an operator, divide that mass equally among its transition strata. This balances conditional transitions, not merely row counts. The three arms have the same task and transition counts and equal operator/transition action mass:

- `happy_action`: the same seven transition labels and operator counts, but all contexts are ordinary successful-path states; plan loss is zero.
- `recovery_action`: verifier-conditioned recovery contexts; plan loss is zero.
- `recovery_reason`: byte-identical to `recovery_action`; plan supervision adds exactly 5% of action loss mass at every transition stratum.

Happy-path and recovery patch bytes are not forced identical where doing so would make an edit non-executable in its context; exact weighted action mass is the registered match. The action/reason contrast is byte-identical.

## Training

Each arm receives a fresh r32/alpha64/dropout0.05 QLoRA over all seven projection modules from the same warm start. Use LR 5e-5 cosine, batch 4, accumulation 7, seed 42, max length 4096, and eight complete bank epochs. If the covered task count is not divisible by four, duplicate only complete seven-transition task blocks. Every optimizer accumulation window contains one microbatch from each transition stratum; padding and exposure are recorded. The action-token denominator is identical across arms, so the reason arm differs only by its registered plan numerator.

This is one frozen dose. No transfer or Menagerie outcome may retune it.

## Calibration selection

Evaluate frozen base, happy, action, and reason arms on both controlled recovery scenarios in the trained-family calibration block. Select action or reason lexicographically by (overall recovery success, worst-scenario success, mean of rejected immediate change / failed immediate diagnose-or-revise / failed changed-patch-within-two); exact ties prefer action-only recovery.

Stop unless the selected arm is no worse than base by more than 0.02 and either beats happy recovery success by ≥0.03 or beats its transition composite by ≥0.08. This block selects one arm; transfer remains untouched.

## Evaluation and matched controls

Controlled recovery begins after either a rejected patch or a partial patch plus visible test failure. The deep policy receives one greedy six-turn trajectory. Matched sampling receives two independently sampled three-turn trajectories; both reserve six calls and 4,608 sampled tokens/case. Normal retention uses one greedy eight-turn trajectory. The explicit scaffold uses the frozen base and appends a one-line changed-patch/diagnose-revise rule after the injected failure.

On both transfer blocks, the selected candidate must satisfy every gate:

- recovery success ≥ base +0.05, happy +0.03, base matched sampling +0.03, and base scaffold +0.03;
- paired-bootstrap 95% lower bound ≥0 against base deep and base matched sampling;
- rejected-patch immediate changed-edit rate ≥0.60 and ≥base +0.05;
- failed-test changed-edit-within-two rate ≥0.60 and ≥base +0.05;
- candidate-base success is nonnegative in at least three of four families, with no family below -0.10;
- normal-start success delta ≥-0.03;
- normal verification among successes ≥0.70 and no more than 0.05 below base;
- normal commit among verified cases ≥0.65 and no more than 0.05 below base;
- invalid-action rate no more than base +0.02;
- median centered non-target logit drift on the frozen unrelated contexts ≤0.15.

Exact equality passes. Transfer dev is evaluated once; if it passes, the same frozen candidate and all controls repeat on confirmation.

## Entropy and varentropy

Teacher-forced next-token entropy, surprisal varentropy, target-token log probability, and target rank at the plan and action seams are exploratory diagnostics. They may describe where recovery states are uncertain and how training moves them, but they cannot select examples, change loss weights, choose an arm, or rescue a failed gate in this experiment.

## Menagerie license

Menagerie remains sealed unless calibration, locality, transfer dev, and transfer confirmation all pass. Only then assign two fresh registry-checked paired seeds and run aggregate-only `quick` and `medium` tiers for candidate and frozen base on the same backend/decode. A positive requires ≥0.02 on at least one tier, no tier regression worse than 0.03, and the direction repeated on the second seed. A prerequisite failure consumes no benchmark seed.

## Interpretation boundaries

- Recovery ≈ happy means marginal action training, not verifier conditioning, caused the change.
- Recovery ≤ scaffold means the behavior is promptable process control, not a justified weight intervention.
- Recovery ≤ matched sampling means the method is not a compute-efficient capability gain.
- Better transitions without final hidden-test gains are process changes, not capability elicitation.
- Train-family gains without both transfer blocks are protocol memorization.
- Excess locality drift invalidates a capability headline even if recovery rises.
