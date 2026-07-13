# Qwen3.5-4B Balanced-Core Answer-Potential SFT Report

## Summary

Experiment in progress. The balanced raw pool is complete and the failed candidate scoring instrument has
been replaced prospectively by its single-context reference. A later selector-balance repair is explicitly a
post-score, pre-official-selection deviation and is machine-sealed before selection; no trained
capability result has been observed.

## Plain-Language Question

If we sample many complete ways the model thinks through a problem, can the correct answer's likelihood tell
us which reasoning to teach back—or is simply choosing the shortest complete reasoning better?

## Method

Finish a checksum-preserved 360-task, three-family N=64 bank; compare answer-potential, joint-potential,
random, successful, shortest, and task-shuffled full-thought SFT; evaluate every arm on fresh core, harder,
and family-held tasks before any optional expansion.

## Results

Operational results: 360/360 tasks, 23,040 traces, 108,759,239 sampled thought tokens, 22,681 natural closes,
four loops, and no top-ups. The task-diverse joint HF/vLLM gate failed at 0.692447 > 0.15 before any bulk
score. The frozen threshold was preserved; vLLM likelihood scoring was retired in favor of the
single-context Transformers reference. Exact scoring completed for all 22,681 eligible traces in 17,296
seconds, followed by one R1 answer rollout per trace in 10,915 seconds. A read-only whole-bank validation
passes exact scope, artifact, source-link, trace-join, and eligibility-set checks. The resulting retrospective
seal binds the original and final index hashes, exact operation contracts, frozen code/data, and deviation
disclosure. Capability results remain pending; official selection has not run.

Exact reference scoring subsequently completed for all 360 tasks and 22,681 eligible traces. Applying the
original helper in memory exposed an unintended 116-task filter; because those scores were already observed,
the balance fallback is an exploratory post-score deviation. Partial R1 labels were subsequently inspected
for cost planning before commit but did not determine the fallback; no official SFT row, adapter, or held-out
outcome informed it.

## Controls

Pending SFT matrix.

## Oracle Versus Deployable Evidence

Reference answers curate training traces only. The primary result will be autonomous natural-thinking exact
accuracy; answer likelihood is not itself a deployment metric.

## Interpretation

Pending.

## Next Experiments

None licensed before the Stage-A verdict.

## Artifact Manifest

See `artifact_manifest.yaml`.
