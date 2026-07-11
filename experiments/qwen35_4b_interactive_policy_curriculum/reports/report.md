# Interactive policy curriculum: oracle DAgger to execution-reward RL Report

## Current Verdict

`PREREGISTERED — RESULT-BEARING RUN NOT YET STARTED.` The experiment is not a
positive or negative result at this point. Its implementation and adversarial
review are complete enough for the frozen CPU smoke; GPU stages remain gated
in the order recorded below.

## Research Program Fit

This is the first beyond-C53 mechanism test in `agentic_breadth_installation`.
It reuses the firewall-clean gym for a controlled comparison but changes both
the state distribution and objective: current-policy visited states replace
successful static turns, then complete-trajectory execution reward replaces
completion imitation.

## Registered Method

Regenerate the C53 blend, collect live DAgger corrections on five interactive
families, train and proxy-gate that warm start, collect grouped on-policy
episodes, and compare guarded sequence-GRPO with compute-overmatched additional
SFT and shuffled rewards. Three incremental families remain unseen. Menagerie
is opened only for a checkpoint clearing the complete whitebox gate.

Exact hypotheses, metrics, thresholds, reward definitions, and stop rules are
frozen in [preregistration.md](preregistration.md). The required pre-run review
and its fixes are in [design_review.md](design_review.md).

## Smoke Evidence

- State-aware experts solve every supported family at every L1–L6 level.
- They recover from a malformed visited state where the environment's score is
  recoverable; spindle's intentionally irreversible first-try penalty is
  handled explicitly.
- The injected `</think>` close sequence receives zero policy-loss weight.
- Shuffled control construction preserves the marginal advantage vectors.
- All fourteen copied gym family selftests pass.
- The copied vLLM wrapper's thirteen non-GPU unit tests pass.

Smoke proves plumbing and oracle validity only. It is not evidence for the
capability hypothesis.

## Results

No result-bearing metric has been observed. Each reached stage will be added
without deleting failed gates or stopped downstream stages.

## Controls And Evidence Boundary

The registered controls are incumbent frozen sampling, DAgger-only,
compute-overmatched new-state SFT, shuffled rewards, and a programmatic expert
ceiling. Expert labels and proxy rewards are privileged training/evaluation
signals; deployment inputs remain visible transcripts only. The benchmark
firewall remains closed.

## Artifact Manifest

Large adapters and merged checkpoints will remain in the gitignored path
declared by `artifact_manifest.yaml`; small trajectory summaries, paired gate
receipts, and aggregate-only benchmark records remain in the experiment.
