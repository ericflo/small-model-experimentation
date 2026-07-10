# Adversarial Design Review

Verdict before GPU-scale work: **sound with mandatory fixes integrated into the frozen design**.

## 1. Near-Duplicate And Mechanism Risk

The obvious objection is C28: the repository already banked successful self-generated thoughts and
found them inert. The new independent variable must therefore be the trace *selector*, not merely
larger N, shorter traces, another substrate, or extra training. Resolution: `success_rft` uses the
same thought pool and one sampled answer; all matched arms preserve the canonical answer and SFT
recipe. The confirmatory contrast is potential versus binary RFT and length-matched random traces.

## 2. Tautological Scorer Risk

Canonical-answer likelihood trivially measures canonical-answer likelihood; this does not prove that
the trace helps open-ended generation or SFT. Resolution: G0 uses eight fresh sampled continuations
per trace, procedural grading, new seeds, within-task metrics, and top-one selection. Only those
independent outcomes can license training.

## 3. Pooled Difficulty Leakage

C47 showed pooled AUROC can be entirely task difficulty. Resolution: all gates are within-task/task-
macro. Pooled scores are diagnostic only. The empty-thought subtraction aids interpretability but
does not replace within-task evaluation.

## 4. Answer Leakage And Lucky Guessing

A trace can obtain excellent score merely by writing the correct answer, especially in small answer
domains. Resolution: measure the earliest exact answer mention and require positive pre-mention gain;
use procedural decoy margins; report answer-domain strata; exclude ambiguous equivalence sets; and
retain C50's small-domain agreement guard in descriptive forensics. A gain appearing only after the
answer mention is classified as answer-copying, not reasoning progress.

## 5. Length And Formatting Confounds

Shorter thoughts may score better because they cause less context dilution, while fixed `ANSWER:`
syntax can dominate the first token. Resolution: exclude fixed boundary tokens, store first-content
and full-answer scores, compare length as a ranker, use same-task length-matched random traces, test a
harmless formatting variant, and apply brevity only *after* a frozen quality threshold.

## 6. Extreme-Value / Reward-Hacking Risk At N=128

Hard argmax over a large pool can find rare pathological traces. Resolution: trace prior-likelihood
floor, duplicate/loop removal, top-quality set rather than naked global argmax, deterministic
diversity selection, nested N curves, fresh-rollout validation, and comparison of selected-trace
quality as N grows. Do not increase N after inspecting treatment evaluation.

## 7. Diversity Metric Risk

Lexical distance can reward incoherent noise or numeric substitutions. Resolution: normalize
identifiers/numbers, quality-gate first, use farthest-first only inside the quality set, select the
shortest near-best medoid, and cap normalized templates globally. No external embedding model is
permitted.

## 8. Pivot Search Compute And Greedy-Path Risk

Branching from a promising prefix gets extra information and repeatedly pays long prefill; matching
only sampled suffix tokens would favor it. Greedy potential climbing can also lock onto a spurious
prefix. Resolution: match total forward tokens including repeated prefill, retain independent-128 as
the comparator, branch only at natural boundaries before measured drops, preserve task/family
coverage, and gate the branch arm on independent fresh rollouts before SFT.

## 9. SFT Token/Task/Channel Confounds

Trace arms differ in length and task coverage; answer-only and thinking-trained models can differ in
chat channel. Resolution: channel-matched empty-thought floor, same-task length-matched random control,
hard-trim matched matrix, report applied versus matched-task intersection, identical canonical
answers, steps and QLoRA recipe, plus the full think/no-think evaluation grid. Exact target-token and
forward-token counts are reported rather than asserting perfect token matching.

## 10. Near-Self-Distillation And Emission-Seam Failure

C50 found full-weight training on verbatim own chains installed nothing; the useful change came from
answer-seam weighting and recovery states. Resolution: prompt 0 / thought 0.2 / close-answer 1.0,
canonical terse answers, recovery thought as context only, broad families, and trace-adoption plus
parse-conditional reporting. A parse-only gain is labeled interface repair, not reasoning.

## 11. Evaluation Power And Selective Replication

Five arms x two training seeds is expensive, while one seed cannot establish a robust win. Resolution:
frozen seed-42 screen for all arms, then seed-43 replication only under an explicit >=0.03/no-parse-
loss rule. The positive threshold is >=0.05 with task-paired CI lower bound >0 on 400 IID items;
held-family/hard outcomes are untouched secondary tests.

## 12. Matched-Compute Sample-More

An adapter delta alone does not meet the repository mission. Resolution: preserve a base candidate
stream and compare cumulative actual forward tokens, including prompt/prefill, generation, scoring,
and branch recomputation where relevant. Report deployable selectors and the oracle ceiling
separately; training cost is an amortization axis rather than silently ignored.

## 13. Backend And Adapter Hazard

Generation/scoring through different inference backends would invalidate candidate and compute
comparisons, and C49 shows runtime vLLM LoRA is a silent no-op. Resolution: vLLM for every scientific
generation/scoring arm; a tiny HF probability parity check is infrastructure-only; trained adapters
are merged into composite checkpoints and must pass a real on/off behavioral gate.

## 14. Firewall And Held-Family Contamination

Copying a prior gym is permitted, but importing held-family modules in a generic registry during
training could accidentally expose their code or metadata to curation logic. Resolution: the
training-stage registry contains only the ten train families; held-family loaders live behind an
evaluation-only module and are never imported in calibrate/harvest/select/train. No `benchmarks/`
path may appear in experiment Python.

## Mandatory Pre-Run Checks

- Commit README, preregistration, review, intake, config, and artifact plan before GPU scale.
- CPU selftest every copied generator and verify split/digest disjointness.
- Unit-test answer-span boundaries, observed-token logprob extraction, equivalence rendering,
  pre-answer mention, bootstrap, selector determinism, derangement, and compute accounting.
- GPU smoke thought-only stop, answer scoring, R=2 continuation grading, tiny SFT, merged-checkpoint
  on/off behavior, and end-to-end analysis before G0.
- The orchestrator must hard-stop full harvest without both the design commit receipt and G0 pass.
