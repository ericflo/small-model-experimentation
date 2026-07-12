# Qwen3.5-4B Balanced-Core Answer-Potential SFT Report

## Summary

Experiment in progress. The balanced raw pool is complete and the failed candidate scoring instrument has
been replaced prospectively by its single-context reference; no trained capability result has been observed.

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
single-context Transformers reference. Capability results remain pending.

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
