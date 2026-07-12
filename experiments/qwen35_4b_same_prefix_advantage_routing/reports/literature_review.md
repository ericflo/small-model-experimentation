# Literature Review: Why Outcome Advantage Must Precede Dense Distillation

Primary sources rechecked on 2026-07-12. This note motivates the frozen design;
it does not license changing a gate after output.

## Capability integration

[MOPD](https://arxiv.org/abs/2606.30406) supplies the central positive result:
same-origin domain-RL teachers score exact student rollouts and provide dense
reverse-KL targets, outperforming mixed RL, cascade RL, off-policy fine-tuning,
and parameter merging at much larger scale. The paper's top-k equation adds
`-p_student + p_teacher`, restoring a stationary point at the teacher despite
truncation. It also provides the sharpest warning: swapping in a stronger but
distributionally different teacher increased initial per-token KL about fivefold
and caused entropy contraction or catastrophic top-k collapse. This experiment
therefore keeps same-origin teachers, corrects top-k exactly, and measures
initial distribution/locality rather than trusting endpoint score.

MOPD routes by known prompt domain. The repository's preceding experiment
showed that C54's coarse quick/deep labels are not reliable local teacher labels
on the procedural proxy. The present experiment changes only that missing
premise: continuation value at the exact student state determines whether a
teacher exists there.

## On-policy distillation is conditional, not automatically safe

[Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016) finds that
teacher/student thinking-pattern compatibility and genuinely new teacher
capability jointly govern success. Effective runs progressively align shared
high-probability tokens; failing runs have stagnant overlap and persistent
entropy mismatch even when the teacher has a higher benchmark score. That
supports initializing from the source-policy soup, logging top-k overlap, and
requiring a verified continuation gap rather than assuming score rank transfers
to student states.

[Denser Is Not Better](https://arxiv.org/abs/2607.01763) reports that dense
self-distillation can accelerate specialization yet increase parameter/response
drift, formatting feedback loops, forgetting, and collapse. On-policy history
alone is not a stability proof. Exact-logit locality, anchor pressure, round
loss/entropy guards, held-out families, and matched sparse/off-policy controls
are consequently mandatory here.

[Rethinking On-Policy Self-Distillation for Thinking Models](https://arxiv.org/abs/2607.05184)
shows why privileged hints are especially dangerous at high-entropy forks:
an informed teacher can penalize verification, hedging, and self-correction
that are rational under the student's information state. This experiment uses
no hint, solution, hidden state, or extra context. Every policy continues the
identical student-visible state; only external verifier outcomes choose whether
the dense target is trusted.

## Advantage and sample routing

[Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
shows that environment feedback can become a dense logit-level signal by
re-scoring the student's own rollout with a feedback-conditioned self-teacher.
It motivates teacher scoring on student trajectories, but its privileged
feedback route is not used here because the two frozen same-origin policies
already provide candidate teachers.

[Sample-Routed Policy Optimization](https://arxiv.org/abs/2604.02288) traces
late SDPO collapse to ambiguous dense updates on already-correct samples and
degrading teacher reliability. It routes correct samples to reward-aligned RL
and failed samples with teacher information to dense correction, with entropy
used to suppress unreliable tokens. The present route adopts the conservative
kernel: acquire residual failed states, require a teacher to beat the current
student on independent verified continuations, and abstain otherwise. It does
not copy the paper's GRPO branch; the frozen-soup dense anchor is the matched
retention mechanism and fixed-deep/coarse/off-policy arms test attribution.

[Self-Supervised On-Policy Distillation](https://arxiv.org/abs/2605.17497) and
[Pass-Rate Weighted Self-Distillation](https://arxiv.org/abs/2605.27765) likewise
narrow dense supervision to failure-conditioned or difficulty-weighted states.
They reinforce the principle that a teacher logit discrepancy is not itself a
capability advantage.

## Older context-distillation and the social-post misread

[Learning by Distilling Context](https://arxiv.org/abs/2209.15189) is genuine
context distillation: a student internalizes behavior available with
instructions, demonstrations, or scratchpads. It is not a new 2026 on-policy
rollout-steering algorithm. The social post attaching it to “privileged
information steering student rollouts” mixes a real ancestor with a different
mechanism.

The useful kernel behind “replace the hinted logprob ratio with an advantage
estimator” is therefore not a new theorem or acronym. It is a falsifiable
design principle: before applying dense pressure at a student state, measure
whether following that teacher from the same information state improves
verified return. This experiment isolates that principle with an independent
audit split.

## Repository evidence corrected before lock

During the pre-output implementation window, C54 completed a larger-n audit of
the source pair. Pooling nine apex medium events revised the earlier n=3
`+0.345` estimate to `+0.321 ± 0.017 SE`; the fixed model's medium ceiling
straddles the old `+0.32` target. Its directly evaluated quick/deep tier router
scores quick `+0.338` and medium `+0.321`. This strengthens the router baseline
but weakens any claim that the deep endpoint has a decisive aggregate margin.
The present state-level gate is unchanged in principle: it assumes neither
teacher is locally superior, and it now freezes at least eight medium events if
the procedural mechanism earns benchmark access.

## Explicit deviations and scope

- MOPD initializes from the common pre-specialization checkpoint. This run
  starts from the strongest 40/60 soup because the scientific target is an
  improvement over the current best single checkpoint, not recovery from a
  weaker root.
- MOPD uses prompt-domain routing. This run uses a training-only outcome-value
  router because coarse labels failed locally.
- The route selection uses empirical continuation return only as a binary
  teacher/abstain decision; noisy per-state advantage magnitude does not scale
  token loss.
- Teachers stay frozen. A later teacher-retraining round is out of scope unless
  this first integration conclusively passes.
