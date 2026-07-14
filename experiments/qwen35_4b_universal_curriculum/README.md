# Qwen3.5-4B: Installing Universal Features via Designed Synthetic Curricula

**Status:** finished

Outcome: sequential arm failed aggregate transfer; from-base replay union failed its
frozen local gate before benchmark.

This experiment turns the doctrine in
[`docs/installing_universal_features.md`](../../docs/installing_universal_features.md)
into a contamination-controlled search. It asks whether hand-designed, executable
synthetic lessons can add procedures that transfer beyond their surfaces while retaining
the strongest existing broad emission-policy install.

## Research program

- Primary program: `agentic_breadth_installation`.
- Supporting programs: `benchmark_generalization` and
  `structured_execution_and_compilers`.
- Prior anchors: C11 (self-harvest is coverage-bounded), C49 (runtime LoRA is a
  silent no-op for this composite), C53 (the broad emission install is a strong but
  saturating control), C56 (executable exploration transfers while answer narration
  does not), and C59 (reasoning content—not nominal token count—crosses serial walls).
- Intake and closest duplicates: [`idea_intake.md`](idea_intake.md).

## Frozen pilot question

Does a truth-audited, surface-varied curriculum of generic search, execution,
verification, repair, uncertainty, state, and routing procedures add broad held-out
transfer beyond the existing C53 `blend` install?

The pilot succeeds only if candidate-minus-base aggregate is positive and none of the
ten public family deltas is negative on a fresh quick event. That is a screening result,
not a universal-feature claim. Confirmation requires independent quick seeds, strictly
positive mean deltas for every family, medium-tier transfer, paired uncertainty, and a
matched-compute sampling baseline.

## Design

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Generator: [`scripts/gen_curriculum.py`](scripts/gen_curriculum.py) emits 13 lesson
  types—induction, execution, selection, tracing, verification, counting, repair,
  optimization, abstention, state carry, ordering, probe choice, and routing—over six
  randomized abstract surfaces.
- Truth controls: every row comes from an executable specification; induction rows
  require query-identifiability, a real dead end, and a deterministic witness that the
  claimed two-step rule does not collapse to one primitive.
- Frozen doses: the full v2 corpus has 2,300 rows. The fast search tier is exactly 800
  rows and one epoch. Exact tokenizer receipts require zero boundary merges, truncations,
  or skipped targets.
- Arms: pinned base, frozen C53 `blend`, designed-only, designed-plus-replay, and the
  first sequential arm `blend_then_designed_fast`. The initial event compares base,
  `blend`, and the sequential arm.
- Firewall: benchmark access is exclusively through
  [`scripts/run_benchmark_aggregate.py`](../../scripts/run_benchmark_aggregate.py).
  Each promoted arm is explicitly merged and evaluated with the same `qwen_vllm`
  backend, tier, and seed. Raw suite output never enters this experiment.

The adversarial review and frozen interpretation rules are in
[`reports/design_review.md`](reports/design_review.md) and
[`reports/preregistration.md`](reports/preregistration.md). Hardware caps are frozen in
[`reports/protocol_amendment_001.md`](reports/protocol_amendment_001.md); the quantitative
from-base replay-union local gate is frozen in
[`reports/protocol_amendment_002.md`](reports/protocol_amendment_002.md).

## Reproduction

Generate and test the frozen corpus:

```bash
.venv/bin/python experiments/qwen35_4b_universal_curriculum/scripts/gen_curriculum.py
.venv/bin/python -m unittest \
  experiments.qwen35_4b_universal_curriculum.tests.test_curriculum -v
.venv/bin/python experiments/qwen35_4b_universal_curriculum/scripts/validate_curriculum.py \
  --data experiments/qwen35_4b_universal_curriculum/data/sft_universal_fast.jsonl \
  --mix induct=80,execute=60,select=50,trace=60,verify=60,count=30,repair=90,optimize=70,abstain=70,state=80,order=50,probe=50,route=50 \
  --max-length 2048 \
  --receipt experiments/qwen35_4b_universal_curriculum/data/sft_universal_fast.receipt.json
```

The first training command is preserved verbatim in
[`runs/training/blend_then_designed_fast.json`](runs/training/blend_then_designed_fast.json).
The fail-closed entry point is [`scripts/train_trial.py`](scripts/train_trial.py); it
authenticates corpus hashes, exact token exposure, model lineage, trainer row counts,
and the final adapter.

Fresh synthetic screening uses [`scripts/eval_curriculum.py`](scripts/eval_curriculum.py).
Promoted adapters use [`scripts/merge_trial.py`](scripts/merge_trial.py), and paired
aggregate events use [`scripts/run_benchmark.py`](scripts/run_benchmark.py). Both refuse
stale or unauthenticated artifacts.

## Results so far

### Corpus and training gates

- The original v1 corpus was rejected before training: 16/600 induction traces
  contradicted their answers and at least 33/600 nominal two-step rules collapsed to a
  primitive. The advertised historical run had never started.
- The deterministic v2 full corpus contains 2,300 valid rows; the fast corpus contains
  800. All six generator tests pass across process hash seeds.
- `blend_then_designed_fast` consumed 800/800 rows with zero skips at max length 2,048.
  It ran for 346.7 seconds, ending at train loss 0.3495. Adapter weights SHA-256:
  `72af43458777245f7236f0df6edf89013eeade67140a9a16c4aee279f95e6e77`.
- `designed_plus_replay_fast_b1` consumed all 3,040 frozen designed-plus-replay
  rows from base with zero skips at max length 4,096. The batch-2 launch hit a
  first-step WSL/CUDA residency failure; the exact effective-batch-8 recovery used
  batch 1 / accumulation 8, ran 380 steps in 2,622.6 seconds, and ended at finite
  train loss 1.366. Adapter weights SHA-256:
  `e551c1d291fca993f94bdd03c1cbaeef43b1b7b7bd4f4f16d277cfa12bc6412b`.

### Fresh synthetic screen

On 26 unseen generator tasks at seed 88001 and a 1,024-token cap:

| arm | exact accuracy | parse rate | cap contacts | mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| frozen `blend` | 0.500 | 0.615 | 10/26 | 763.7 |
| `blend_then_designed_fast` | 0.692 | 0.962 | 1/26 | 231.1 |

This passes installability/emission screening but is not transfer evidence. Induction
and repair remain imperfect locally, and the sequential arm over-abstained on two route
items; those negatives are retained in the local result.

The from-base replay union was screened prospectively on fresh seed 88002:

| arm | exact accuracy | parse rate | cap contacts | mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| `designed_plus_replay_fast_b1` | 0.692 | 0.846 | 4/26 | 394.4 |

It met the 0.65 accuracy bar and recovered routing to 2/2, but failed the frozen parse
rate (required at least 0.90) and cap-contact (required at most 2/26) gates. Induction
was 0/2 and execution 0/2. Benchmark seed 78132 was therefore never consumed.

### Aggregate transfer

Native quick@8,192 was rejected before any score: the permitted estimator marks it over
the 60-second hardware gate on this RTX 4090. Two raw-suppressed failures and one
interrupted diagnostic are retained. Protocol amendment 001 freezes quick@1,024 and
medium@2,048, the highest power-of-two caps that pass their tier estimates.

Quick@1,024 seed 78131 completed through the paired `qwen_vllm` gateway:

| arm | aggregate | delta vs. base | positive / nonnegative families | minimum family delta |
| --- | ---: | ---: | ---: | ---: |
| base | 0.1667 | — | — | — |
| frozen `blend` | 0.4458 | +0.2791 | 8 / 9 | -0.1250 |
| `blend_then_designed_fast` | 0.3073 | +0.1406 | 6 / 7 | -0.1250 |

The sequential candidate fails the pilot gate. It moves real axes—chronicle and
siftstack are each +0.75 versus base, and aggregate is +0.1406—but rites, stockade, and
warren are negative. Relative to `blend`, it loses 0.1385 aggregate and sharply regresses
lockpick, mirage, rites, stockade, and toolsmith. This is specialization plus catastrophic
displacement, not a universal feature. The full authenticated event is
[`runs/benchmark/quick_tb1024_seed78131_pilot1_tb1024/summary.json`](runs/benchmark/quick_tb1024_seed78131_pilot1_tb1024/summary.json).

## Artifacts

- Checked in: deterministic corpora, tokenizer receipts, source, tests, intake,
  preregistration, design review, training/local receipts, and aggregate-only events.
- External: adapters and 9 GB composite checkpoints under
  `large_artifacts/qwen35_4b_universal_curriculum/`.
- Manifest: [`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).

Negative controls and failed infrastructure attempts are never overwritten. Any
post-result adaptive curriculum or confirmation moves to a successor experiment. The
parent factorial is complete; the next registered mechanism is
`qwen35_4b_universal_replay_anchor`.
