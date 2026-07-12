# Literature Review: What Is Real in the OPSD Acronym Cloud?

Reviewed against primary papers on 2026-07-12. This note is interpretive
context, not an amendment to the frozen experiment.

## The real mechanism families

1. **Context distillation predates OPSD.** [Learning by Distilling
   Context](https://arxiv.org/abs/2209.15189) (Snell, Klein, Zhong, 2022)
   trains a model to internalize behavior available under instructions,
   scratchpads, or examples. Its core evidence is real, but it is not the
   modern student-rollout OPSD algorithm. Treating it as “privileged context
   steering the student distribution during rollouts” conflates context
   distillation with on-policy distribution matching.
2. **OPSD is privileged same-model distillation.** [Self-Distilled
   Reasoner](https://arxiv.org/abs/2601.18734) conditions one copy of the model
   on verified reasoning information and minimizes token-level divergence from
   that teacher on trajectories sampled by the unprivileged student.
3. **SDFT makes demonstrations on-policy.** [Self-Distillation Enables
   Continual Learning](https://arxiv.org/abs/2601.19897) uses a
   demonstration-conditioned self-teacher so the learning targets live on the
   student's state distribution rather than an expert's static trajectories.
4. **SDPO turns feedback into dense credit.** [Reinforcement Learning via
   Self-Distillation](https://arxiv.org/abs/2601.20802) conditions the current
   policy on environment feedback (or a successful peer attempt), stops the
   teacher gradient, and distills its next-token distribution into the
   feedback-free policy.
5. **MOPD is the actual capability-integration paper.** [Multi-Teacher
   On-Policy Distillation](https://arxiv.org/abs/2606.30406) first produces
   independently RL-specialized, same-origin teachers and then routes each
   student rollout to its domain teacher. It reports stronger integration than
   mixed RL, cascade RL, off-policy fine-tuning, and parameter merging. Its
   top-k loss includes the correction term that makes the truncated objective
   stationary at the teacher distribution.
6. **TurboQuant is real but orthogonal.** [TurboQuant: Online Vector
   Quantization with Near-optimal Distortion
   Rate](https://arxiv.org/abs/2504.19874) is an inference/storage compression
   method for high-dimensional vectors, including KV-cache applications. It
   neither supplies a teacher signal nor installs a capability.

## The failure literature matters just as much

- [Rethinking On-Policy Self-Distillation for Thinking
  Models](https://arxiv.org/abs/2607.05184) finds that privileged context can
  hurt long-thinking policies, especially at high-entropy forks: the informed
  teacher can penalize verification, hedging, and self-correction tokens that
  are reasonable under the student's information state.
- [Denser Is Not Better](https://arxiv.org/abs/2607.01763) separates the source
  of data from the update objective. On-policy histories do not automatically
  make dense distillation conservative; teacher noise, formatting artifacts,
  parameter drift, response drift, and collapse can all be amplified.
- [Sample-Routed Policy Optimization](https://arxiv.org/abs/2604.02288)
  attributes late SDPO collapse to ambiguous updates on already-correct
  samples and degrading teacher reliability, then routes correct samples to
  reward optimization and failed samples to entropy-weighted distillation.
- [Self-Supervised On-Policy
  Distillation](https://arxiv.org/abs/2605.17497) narrows supervision to a
  successful peer completion and a failed on-policy prefix rather than
  treating every token equally.
- [Pass-Rate Weighted Self-Distillation](https://arxiv.org/abs/2605.27765)
  restores a difficulty curriculum by weighting the dense loss with online
  success-rate information.

These results do not refute MOPD. They say dense supervision is credible only
when the teacher is aligned with the student's information state and policy
distribution, and when drift is measured rather than assumed away.

## Reading the social posts

- “Remove the hinted logprob ratio and replace it with an advantage
  estimator” is a plausible debugging slogan assembled from real collapse
  findings, but not a generally established OPSD theorem.
- The post attached to `2209.15189` mislabels real context-distillation work as
  a new on-policy rollout method.
- “Dr. OPSD” is a joke expansion, but `2606.30406` is a real MOPD paper and the
  substantive summary—specialized RL first, multi-teacher distillation without
  a privileged hint afterward—is accurate.
- “OPSD + turboquant” splices a post-training algorithm and an orthogonal
  compression method into an “all you need” joke; both names are real, but the
  conjunction is not an evidenced capability recipe.
- The acronym pile-up is comedic, while OPSD, SDFT, SDPO, SSOPD, and MOPD are
  distinct real proposals with overlapping ingredients.

## Why this experiment takes the MOPD branch

The repository already supplies the fact MOPD normally assumes but OPSD must
manufacture with privileged context: two independently trained, same-origin
policies with measured complementary behavior. Therefore the cleanest test is
to use each policy directly as a frozen teacher and never expose a solution or
hint.

The implementation deliberately adds safeguards demanded by the failure
literature:

- exact current-student rollouts and same-prefix frozen-teacher scoring;
- corrected teacher-top-50 reverse KL with full-softmax probabilities;
- same-origin teachers only, with qualification before distillation;
- a five-update entropy and non-target-logit locality pilot;
- a frozen per-round loss ceiling and non-finite-gradient stop;
- 25% quick-policy retention pressure and explicit anchor/transfer checks;
- wrong-route, off-policy, parameter-merge, and compute-overmatched SFT
  controls;
- three training seeds and a matched best-of-8 terminal hurdle.

Two deviations from the paper are explicit. The paper initializes its student
from the shared pre-specialization checkpoint; this experiment initializes
from the quick policy so the already-installed short-regime capability is the
retention baseline. Also, this experiment's four rounds refresh rollouts while
keeping the two teachers frozen; they are not the paper's later
teacher-retraining iteration. The causal controls determine whether those
choices help or merely preserve one side of the original frontier.
