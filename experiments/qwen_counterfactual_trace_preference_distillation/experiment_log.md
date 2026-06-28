# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: counterfactual trace preference distillation.
- Core idea: build candidate groups from the frozen-Qwen compiler, execute all
  candidate programs, label each candidate by a strict quality order, train a
  preference selector on groups where a better candidate exists than the base
  decode, then use the learned selector as a no-answer repair teacher.
- Main comparison plan:
  - seed supervised compiler;
  - answer-verified repair distillation;
  - learned preference-selected distillation;
  - best-quality distillation control;
  - full-supervised ceiling.
- Large artifacts will be stored in
  `large_artifacts/qwen_counterfactual_trace_preference_distillation/checkpoints/`.

## Iteration Notes

- Implemented the standalone VM core and experiment script.
- Smoke run `smoke_counterfactual_pref` completed end to end, including frozen
  Qwen feature extraction, seed training, candidate group construction,
  preference training, answer-verified distillation, preference-selected
  distillation, best-quality control, full-supervised control, metrics, and
  checkpoint writing.
- Smoke diagnostic: the tiny candidate surface had no true counterfactual
  repair groups because answer-correct candidates only appeared when the base
  decode was already answer-correct. The next step is a larger pilot with a
  stronger seed/search surface so the preference objective has real repairable
  failures.
- Pilot `pilot_pref_s96_c256_m192` produced real counterfactual repair groups
  (`17.2%` of unlabeled prompts had a better candidate than the base), but the
  learned selector was weak (`7.8%` best validation selection vs `31.2%`
  oracle). This exposed a missing channel: the selector could execute a
  candidate but had only an indirect representation of the prompt-implied
  answer.
- Patched the preference model with no-answer candidate features: normalized
  program prior, prompt answer-head logprob of the candidate's VM final value,
  VM validity, and normalized VM final value.
- Smoke run `smoke_counterfactual_pref_v2` verified the patched feature path.
- Pilot `pilot_pref_s96_c256_features` validated the feature bridge. The
  selector reached `15.6%` validation selection against a `31.2%` oracle, and
  on held-out splits selected correct candidates at `20.3%` fresh-paired and
  `21.9%` hard-composition from a seed compiler with `0%` direct accuracy on
  those two splits. Preference-selected distillation improved deployable direct
  accuracy to `9.4%` fresh-paired and `17.2%` hard-composition. This is still
  far from the oracle, but it is the first useful no-answer selector signal in
  this experiment, so the main run will scale this version.
- Main run `main_counterfactual_trace_preference_s192_c1024` completed with
  192 seed examples, 1024 candidate prompts, 1024 full-supervised examples, and
  128 examples per evaluation split.
- Main candidate surface: 246,784 sampled candidates, 64.2% valid candidate
  rate, 7.6% answer-correct candidate rate, 45.5% prompt-level oracle found
  rate, and 31.7% true counterfactual repair groups.
- Main selector result: the learned no-answer preference selector reached 14.8%
  validation selection accuracy against a 41.4% oracle, but did not generalize
  robustly across held-out splits. Fresh-paired selector accuracy was 11.7%
  against a 47.7% oracle.
- Main distillation result: preference-selected distillation slightly improved
  direct accuracy over answer-verified distillation on fresh-paired (15.6% vs
  14.8%) and hard-composition (14.8% vs 10.2%), but its selected targets were
  noisy (12.2% answer-correct), and its search accuracy was worse than
  answer-verified distillation on the key fresh-paired split (46.1% vs 53.9%).
- Full-supervised ceiling remained high: 91.4% fresh-paired direct, 97.7%
  fresh-paired search, and 68.0% hard-composition direct.

## Final Artifacts

- Markdown report:
  `reports/qwen_counterfactual_trace_preference_distillation_report.md`.
- HTML report:
  `reports/qwen_counterfactual_trace_preference_distillation_report.html`.
- Analysis summary: `analysis/summary.md`.
- Figures: `analysis/figures/`.
- Main run files: `runs/main_counterfactual_trace_preference_s192_c1024/`.
- Checkpoints:
  `large_artifacts/qwen_counterfactual_trace_preference_distillation/checkpoints/`.
