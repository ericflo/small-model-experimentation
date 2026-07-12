# Payload-capable recovery agent harness report

## Summary

**Status: `TRANSFER_CONFIRM_FAIL`; Menagerie sealed.** The payload-capable
interface validated the candidate's recovery mechanism and passed development
transfer, but the independent confirmation block tied rather than beat the
action-only incumbent.

## Research Program Fit

The predecessor located a high-success, local checkpoint but its 256-token
answer slot truncated tool payloads. This follow-up resolves whether that is the
remaining deployment bottleneck before spending the untouched transfer and
Menagerie instruments.

## Method

Candidate, base, happy, action, scaffold, and sample-more arms all use 512
thinking + 512 answer tokens/call on the same vLLM backend. A fresh locality
block precedes behavior. Controls and mathematical feasibility precede the
candidate on calibration and each transfer block.

## Results

Fresh locality passed: centered non-target logit drift was 0.114 against the
0.15 ceiling, mean entropy changed −0.0059 nats, and mean varentropy changed
−0.0105. Calibration then passed all gates at 60/60 candidate recovery versus
58/60 action-only, with no invalid actions and one answer-cap hit.

On `transfer_dev`, candidate recovery was 57/80 (71.25%) versus base 47/80,
happy 49/80, action 53/80, matched sample-more 40/80, and scaffold 53/80.
Paired bootstrap intervals versus base and sample-more were respectively
[+3.75,+22.5]pp and [+8.75,+33.75]pp. Candidate and base both solved 10/40
normal tasks, with perfect verify/commit conditional on success. Every gate
passed.

On independent `transfer_confirm`, candidate scored 55/80 (68.75%) versus base
48/80, happy 45/80, action 55/80, matched sample-more 37/80, and scaffold
51/80. The candidate improved every family versus base, retained normal success
at 10/40, reached 100% valid two-turn recovery for both controlled states, and
cut answer-cap hits from base's 29.0% to 7.9% of turns. It failed only the
registered action-only contrast: 0.0pp versus a required +3pp. The stop label is
therefore `TRANSFER_CONFIRM_FAIL`.

## Controls

All frozen controls were rerun before the candidate on each block. Equal-reserved
sample-more used two three-call trajectories versus one six-call deep loop;
the explicit scaffold used the base checkpoint and identical deep budget. The
candidate beat both on both transfer blocks, but did not beat action-only on
confirmation.

## Oracle Versus Deployable Evidence

Procedural hidden tests are host-side oracles only. Familiar-family calibration
cannot establish breadth. Menagerie stays sealed until two family-held-out
blocks pass. An exploratory oracle union of candidate and action-only reaches
63/80 (78.75%) on both dev and confirm, versus 57/80 and 55/80 for the
candidate alone; it is not deployable evidence until a public verifier selects
the branch without hidden outcomes.

## Interpretation

The answer cap and immediate-only proxy were real harness bugs, not evidence
against recovery learning. Repairing them reveals a local, transferable policy
that strongly beats base and sample-more. The remaining failure is policy
complementarity: reason mixing helps some algorithmic repairs and hurts others.
The paired disagreement pattern replicates in union size across both blocks,
which argues for verifier-guided branching rather than another scalar weight
dose.

## Next Experiments

A new result-bearing experiment should fork action-only and λ=.18 from the same
public recovery state, execute bounded branches, select using only visible
verifier/rejection signals, and compare against equal-compute independent
sampling. Only if that capability producer transfers should its winning traces
be compressed into a conditionally balanced curriculum.

## Artifact Manifest

See `artifact_manifest.yaml`.
