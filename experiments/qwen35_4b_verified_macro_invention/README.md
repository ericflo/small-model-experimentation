# Qwen3.5-4B Verified Macro Invention

**Status:** finished

smoke or full evaluation was run; the verified-macro hypothesis remains unresolved.

**2026-07-10 follow-up:** the separate
[`qwen35_4b_verified_macro_long_context_rerun`](../qwen35_4b_verified_macro_long_context_rerun/)
later cleared a broader, disjoint plan-given interface gate on 16/16 records under adequately
budgeted vLLM inference. This directory's historical outputs and reported numbers remain unchanged;
the additive forward link does not rewrite its registered low-budget stop. That stop should not be
read as a durable model-level shortcut-interface failure.

## Research program

- Primary: `operator_and_skill_inventories`
- Secondary: `structured_execution_and_compilers`, `benchmark_generalization`
- Intake: [idea_intake.md](idea_intake.md)
- Preregistration: [reports/preregistration.md](reports/preregistration.md)
- Fresh-smoke interface amendment: [reports/preregistration_amendment_1.md](reports/preregistration_amendment_1.md)
- No-think transcription amendment: [reports/preregistration_amendment_2.md](reports/preregistration_amendment_2.md)
- Design review: [reports/design_review.md](reports/design_review.md)
- Preserved smoke-v1 failure: [reports/smoke_v1_failure.md](reports/smoke_v1_failure.md)
- Final interface failure: [reports/interface_v3_failure.md](reports/interface_v3_failure.md)

## Question

Can a verified abstraction library built only from prior solved programs turn fresh,
behaviorally deep programs into shallower decisions and improve visible-only selected accuracy
beyond matched-compute sampling over the original primitives?

## Hypothesis

The fixed-vocabulary assumption is part of the composition bottleneck. Recurring depth-2/3
motifs can be packaged as exact callable macros, reducing the surface decision depth of unseen
behaviorally true-depth-5 combinations. If that representation is useful, frequent train-only macros should
beat both base-primitive sampling and matched random composites; correct treatment-only
solutions should actually call macros; and the advantage should concentrate on a preregistered
motif-reuse split.

## Why this is not another operator-bank experiment

The closest experiments grow or shortlist fixed, human-authored atomic inventories. Here the
inventory itself is learned from prior programs. The main scientific comparison separates:

- deterministic tool-mined abstractions;
- Qwen-proposed, locally verified abstractions;
- highlighted-but-not-callable subsequences;
- random composite entries matched on count, length, and train support;
- generator-known motifs as a clearly labeled ceiling.

The experiment therefore asks both whether abstraction helps the system and whether Qwen adds
anything to a deterministic miner.

## Setup

- **Only model:** `Qwen/Qwen3.5-4B`, repository-pinned revision.
- **Inference:** experiment-local [src/vllm_runner.py](src/vllm_runner.py) under
  `.venv-vllm`; no Transformers inference and no mixed backend.
- **Substrate:** contamination-free procedural list transformations with exact execution.
- **Construction corpus:** primitive-rendered programs from a frozen latent-motif grammar.
- **Full evaluation:** 80 motif-reuse and 40 primitive-multiset-matched no-reuse tasks at
  behaviorally verified depth 5, plus a
  disjoint smoke set.
- **Examples:** eight visible, eight hidden-grade, and eight unlabeled probe inputs per task.
- **Sampling:** budgeted thinking, K=12 per macro arm; base K=24 supplies the matched-token
  sample-more curve.
- **Primary metric:** visible-only selected hidden-all accuracy, pooled on the reuse split.
- **Oracle-only metric:** whether any sampled candidate passes all hidden cases.
- **Hidden boundary:** macro construction sees train programs only; prompts see visible I/O;
  hidden outputs enter only the committed analyzer.

The latent-motif generator is necessary rather than cosmetic: a uniform primitive generator has
no real recurring abstraction to discover. The no-reuse control preserves each paired task's
primitive multiset while permuting away the three evaluation-recurrent motifs; train-only decoy
motifs may remain, making this a conservative rather than artificially macro-hostile control.

## Gates

1. CPU split/min-depth/leakage checks and an oracle compression check.
2. Adversarial design review saved before model generation.
3. Separate vLLM smoke: parse rate at least 0.50 and nonzero macro use in the designed-ceiling
   arm, with no tuning on full evaluation outputs. Smoke v1 failed the interface gate and is
   preserved. The task-independent plan-given attempts 2 and 3 then failed before any fresh smoke
   prompt. Amendment 2's stop rule is now final.
4. Full result only after the smoke gate passes. It never did, so full generation is forbidden.

The macro mechanism clears only under the decision rule in the preregistration: at least +0.10
selected-accuracy lift over base with a positive paired interval, survival at matched token cost,
macro use carrying treatment-only successes, and a materially smaller effect on no-reuse tasks.

## Run

All commands use the uv-managed vLLM environment.

CPU preparation and tests:

```bash
.venv-vllm/bin/python -m unittest discover -s experiments/qwen35_4b_verified_macro_invention/tests -v
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_invention/scripts/run.py --prepare
```

Historical reproduction of the stopped interface path (expected to terminate at the failed gate;
earlier attempts are preserved under versioned archive paths):

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_invention/scripts/run.py --smoke
```

Full (registered for provenance but never run and now forbidden by the stop rule):

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_invention/scripts/run.py --full
```

Analyze an existing run without loading the model:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_invention/scripts/analyze.py --run full
```

## Results

Smoke v1 failed the interface gate. The matched base/designed pool parsed at 0.5972 overall
(base 0.6111; designed 0.5694), but answer truncation was 0.40046, the designed arm produced zero
valid macro-using candidates, and its oracle coverage was 0 versus base 0.0833. All 1,440 solver
samples force-closed; 607 answer stages truncated. The sole base oracle solve was a no-reuse task.

The macro-proposal whole-answer parser also accepted 0/16 samples. A post-failure, exploratory
line-local audit found 18 behaviorally unique train-supported candidates, showing parser loss but
not establishing a usable Qwen library. Neither diagnostic is evidence against or for verified
macros. Full generation has not run, so the research hypothesis remains unresolved.

Amendment 1 freezes a fresh-seed v2 smoke at the already planned full thinking budget, with a
shared surface-first procedure, a strict solver parser, a train-only plan-given interface probe,
and matched K=12 base/designed arms. Full generation remains blocked until that gate passes.

The first execution of that task-independent probe, interface attempt 2, failed before any fresh
induction prompt was shown. Of 4 records and 16 samples, records `00` and `02` succeeded, 4 samples
strictly parsed and all 4 used macros, but answer truncation was 12/16 = 0.75. Amendment 2 therefore
retries only the identical plan-transcription gate with vLLM `thinking: off`, n=4, and the unchanged
128-token answer cap and 3/4-plus-truncation gate. If it passes, the still-unseen induction smoke
remains think@768 exactly as frozen in amendment 1.

Interface attempt 3 removed the mechanical failures but still failed exact alias fidelity. All
16/16 samples strictly parsed, all 16 used macros, and none truncated, yet only record `00`
succeeded; record coverage was 1/4 against the registered 3/4 gate. Qualitatively, most inspected
errors called the intended leading alias and then hallucinated extra aliases for primitive suffix
operations that those aliases did not match. The committed raw-row audit makes the pattern exact: all 13
failed samples used multiple aliases and expanded past depth five; 10/13 included the correct
designated alias and 3/13 omitted it.

The stop rule fired. No fresh induction-smoke or full prompt was ever generated. Consequently this
experiment provides a strong interface lesson—syntax and macro invocation can be perfect while
literal expansion fidelity fails—but no evidence for or against the macro-invention hypothesis.
There is no claim-ledger update. Further work must start as a new material follow-up, not amendment
4 in this directory.

## Artifacts

- `data/`: frozen corpus, task splits, libraries, prompts, manifests, and the smoke-v1 snapshot.
- `runs/`: raw vLLM generations and exact runtime/token metadata, including versioned failed-v1
  proposal/solver outputs and failed interface attempts 2 and 3.
- `analysis/`: derived task-level tables and machine-readable verdicts, including smoke v1 and
  both task-independent interface gates.
- `archive/`: exact source snapshots needed to explain the historical v1, interface-v2, and
  interface-v3 outputs.
- `reports/`: preregistration, amendments, design review, failure/final reports, and manifest.
