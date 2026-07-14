# Qwen3.5-4B Balanced-Core Answer-Potential SFT Report

## Summary

Experiment finished at its selection-only stop. The balanced raw pool is complete and the failed candidate
scoring instrument was replaced prospectively by its single-context reference. A later selector-balance
repair is explicitly a post-score, pre-official-selection deviation and is machine-sealed before selection;
no SFT ran and no trained capability result was observed.

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
disclosure. Official selection has now run; capability results remain pending because SFT has not.

Exact reference scoring subsequently completed for all 360 tasks and 22,681 eligible traces. Applying the
original helper in memory exposed an unintended 116-task filter; because those scores were already observed,
the balance fallback is an exploratory post-score deviation. Partial R1 labels were subsequently inspected
for cost planning before commit but did not determine the fallback; no official SFT row, adapter, or held-out
outcome informed it.

Official selection contains 720 rows per arm and is byte-deterministic on rerun. Five arms cover all 360
tasks; the success-RFT control has 97 unique successful source traces from 58 tasks and repeats each source
seven or eight times to reach matched optimizer exposure. Selected potential thoughts are not length-capped
at 512: answer/joint maxima are 14,240/14,325 tokens. The frozen two-epoch matrix is 34,446,994 forward
tokens, with an estimated 9.6--18.1 GPU-hour training envelope before merge/evaluation. No adapter, merge,
deployment probe, or evaluation artifact exists.

## Controls

Six exact selected datasets are banked. Random-natural, shortest-natural, success-RFT, and task-shuffled
potential remain the controls; their training has not started.

## Oracle Versus Deployable Evidence

Reference answers curate training traces only. The primary result will be autonomous natural-thinking exact
accuracy; answer likelihood is not itself a deployment metric.

## Interpretation

Selection alone is not a capability result. The full frozen matrix is too slow for the current budget, and
the success control's narrow task support must be considered when choosing a smaller prospective fork. This
experiment is closed; any such fork must receive a new experiment directory and prospective boundary.

## Next Experiments

A lower-cost prospective fork may be created after an explicit compute/design choice. The frozen full-matrix
claim is unchanged; no subset result may be relabeled as its confirmatory verdict.

## Artifact Manifest

See `artifact_manifest.yaml`.
