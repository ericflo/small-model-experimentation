# Evidence

## Seed Experiments

- [qwen35_4b_constrained_coverage_dpo](../../experiments/qwen35_4b_constrained_coverage_dpo/reports/final_report.md)
- [qwen35_4b_offline_hard_negative_coverage_dpo](../../experiments/qwen35_4b_offline_hard_negative_coverage_dpo/reports/final_report.md)
- [qwen35_4b_live_tool_dagger](../../experiments/qwen35_4b_live_tool_dagger/reports/report.md)
- [qwen35_4b_oracle_process_grpo](../../experiments/qwen35_4b_oracle_process_grpo/reports/qwen35_4b_oracle_process_grpo_report.md)

## Key Result

- [qwen35_4b_meta_induction (general arm)](../../experiments/qwen35_4b_meta_induction/reports/report_general.md) (claim C45): GENERAL induction-via-reasoning IS installable -- a general hypothesize-and-verify CoT trained on families {a=1,3,9} transfers to held-out a=7 (0.905, as high as in-family). The fixed 4B can be taught general induction, but only as a serial reasoning procedure (C44: forward-pass 0.01). Teach the general strategy across diverse cases + deploy with chain-of-thought.

- [qwen35_4b_meta_induction (reasoning arm)](../../experiments/qwen35_4b_meta_induction/reports/report_reasoning.md) (claim C44): the forward-pass induction wall is a SERIAL-COMPUTE limit, not a knowledge limit -- reasoning-SFT induces held-out shifts perfectly via generation (1.00) but at chance in one forward pass (0.01); the CoT is 100% load-bearing. Give the model serial tokens and induction works; it cannot be compressed into a forward pass.

- [qwen35_4b_meta_induction](../../experiments/qwen35_4b_meta_induction/reports/report.md) (claim C43): can SFT install the induction skill? PARTIALLY -- shift induction 0.087 (chance) -> 0.40 (data-limited) but plateaus below the execute ceiling (0.72), is shift-specific (OOF affine 0.30), and answer-only SFT catastrophically forgets execution (0.72 -> 0.09). The wall is neither a hard bound nor cleanly liftable; trained to induce, the model learns a specific procedure, not the general skill.

- [qwen35_4b_learn_from_failures](../../experiments/qwen35_4b_learn_from_failures/reports/report.md) (claim C29): preference training on the model's OWN failures does NOT close the coverage->deployable gap -- DPO collapses generation (greedy@1 & coverage crash; pre-DPO 2AFC=0.81 verifier but preference-optimizing it destroys the model). The gap closes with MORE SFT: SFT_2x triples greedy@1 (0.037->0.113). Extends prior MBPP DPO work to the controlled depth-3 substrate. Limits: DPO not heavily constrained, single seed.

- [qwen35_4b_bank_the_thoughts](../../experiments/qwen35_4b_bank_the_thoughts/reports/report.md) (claim C28): banking correct decomposition PLANS beats banking ANSWERS on deployable depth-3 -- three fresh QLoRA on matched data (A=prompt->code, T=prompt->plan->code, T_corrupt=mismatched plan); T coverage@16 0.325 vs A 0.200; content-causal (T_corrupt collapses to 0.113, below A); test-time channel (T no-think 0.013). Resolves C26/C27 (thinking helps once the reasoning is banked). Limits: synthetic plans (Phase 2 = model's own thoughts), step-1-think eval incomplete, single seed.

- [qwen35_4b_decompose_compose_frontier](../../experiments/qwen35_4b_decompose_compose_frontier/reports/report.md)
  (claim C12): banking search+interpreter-harvested solutions (QLoRA-SFT, no teacher) that the model could NOT
  monolithically sample EXTENDS its frontier into the weights — monolithic held-out pass@5 0.125→0.237
  (+0.112, ~2.6 SE), depth-3 pass@5 4×. This breaks M4's coverage bound: self-training on data from OUTSIDE
  the sampling support (harvested by tool-augmented search) pulls it into the distribution. Frontier
  extension without a teacher.
- [qwen35_4b_neurosymbolic_repl_substrate](../../experiments/qwen35_4b_neurosymbolic_repl_substrate/reports/report.md)
  (claim C11): the corpus's **first self-training WIN**. QLoRA-SFT on the 4B's OWN 189 execution-verified
  solutions (no teacher) improved held-out fresh single-shot on a contamination-free substrate: think-greedy@1
  0.224→0.319 (+0.095, ~2.2 SE, N=210), pass@5 up (NO diversity collapse), two seeds. This **works where the
  corpus's MBPP self-improvement regressed** (`qwen35_4b_verifier_guided_self_improvement`) — implicating
  contamination/substrate, not the method. (Test-time execution-feedback self-correction, by contrast, did
  NOT beat matched-compute sampling — same experiment, M2.)

- [qwen35_4b_answer_potential_trace_sft](../../experiments/qwen35_4b_answer_potential_trace_sft/reports/report.md)
  (claim C51): a dense reference-answer score did not earn the right to become an SFT curriculum. It
  carried modest within-task signal and passed shuffled/foreign/format controls, but missed its AUROC and
  practical top-choice uplift gates. Almost every thought was cap-bound and only 13.2% of forced-close
  answers parsed. The preregistered guard stopped before N=128, selection, or training, so this is a scorer
  negative—not an SFT negative.

- [qwen35_4b_think_ftpo_round2](../../experiments/qwen35_4b_think_ftpo_round2/reports/report.md)
  (claim C52): selecting only low-entropy, non-degenerate-varentropy confident wrong turns did not rescue
  single-token preference training. Positive-only chosen-token uplift was safer than conventional demotion
  and true labels separated from shuffled labels on the parent gym (+6.25pp) and fresh repository agent
  (+13.89pp, CI touching zero), but every LoRA arm failed the exact-logit locality ceiling. Held-out coding
  remained below base (39/72 vs 43/72), so the registered result is a low-dose capability null, not a new
  adaptation recipe.

- [qwen35_4b_specialist_policy_integration](../../experiments/qwen35_4b_specialist_policy_integration/reports/report.md):
  the first same-origin specialist/MOPD test stopped before best-of-8 or any
  specialist update. Its full paired baseline put the only tools family at
  0.994, making the mandatory `S0 + 0.10` target 1.094 under a hard score cap
  of 1.0. This is a posttraining-design feasibility negative, not evidence for
  or against MOPD; every mandatory arm now needs a ceiling/headroom check before
  production.

- [qwen35_4b_pareto_policy_integration](../../experiments/qwen35_4b_pareto_policy_integration/reports/report.md):
  the corrected successor removed the fixed effect-size floor and completed
  two paired qualification blocks. The assumed quick teacher was negative in
  both (`-0.00693`, `-0.03789`; pooled `-0.02241`), while the deep teacher had
  a credible `+0.04563` capability advantage but failed six retention cells.
  Every protocol check passed. The stop occurred before teacher audit or MOPD,
  establishing a teacher-transport prerequisite rather than a distillation
  negative.

## Current Read

Adaptation is useful only when the target behavior is well specified and controls expose whether training
changed the intended mechanism — AND on a substrate where a gain is even measurable. C11 shows honest
self-training (own verified solutions, no teacher) banks capability into single-shot on a CONTAMINATION-FREE
substrate, reversing the corpus's earlier "self-training loses to sample-more" reads that were likely
confounded by contaminated/saturated benchmarks. Priority: scale the self-training loop (expert iteration),
test cross-substrate transfer, and re-run the failed MBPP self-improvement with contamination controls.

C51 sharpens the curation prerequisite: before comparing posttraining arms, prove that the proposed dense
label selects deployably better traces at useful effect size. A teacher-forced answer state after an injected
close is not automatically a valid SFT target source, even when corruption controls say the score notices
relevant content.

C52 adds a separate intervention prerequisite: a label can contain real
directional information while its shared-parameter update is too non-local to
transfer. Confident-outlier geometry and entropy/varentropy routing do not
replace an exact-logit locality gate. Positive-only pressure is preferable to
pairwise demotion, but do not scale it until the update clears ≤0.10 median
non-target drift on independent contexts.

The specialist stop adds a still earlier prerequisite: prove that each
mandatory arm can mathematically clear its frozen gain rule. Aggregate endpoint
headroom can coexist with a saturated arm, so average calibration cannot license
a multi-arm adaptation run by itself.

The corrected successor adds the next prerequisite: even a source policy that
wins an external aggregate tier may not be locally better on the student's
clean rollout distribution. Distillation needs replicated same-prefix teacher
advantage, not a checkpoint label. The most informative continuation is a
fresh outcome-routed pilot that estimates both teachers' verified continuation
values before any update; reusing coarse quick/deep routing would repeat the
measured mismatch.
