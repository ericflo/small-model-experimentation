# Trace-Keyed Symbol Repair

Generated: `2026-06-20 10:09:05 UTC`.

## Abstract

This experiment tests whether a repair model can use a failed execution trace as a data-bearing input, not only as a generic failure signal. Each synthetic task requires replacing a wrong canonical token with an expected token that is absent from the issue text and repository files but present in the pytest failure output. The primary comparison is a trace-conditioned LoRA against frozen, no-trace, shuffled-trace, and final-patch controls.

## Artifact Layout

- Small, download-friendly experiment package: `/workspace/experiments/trace_keyed_symbol_repair`.
- Large adapters and checkpoints: `/workspace/large_artifacts/trace_keyed_symbol_repair`.
- The small package contains configs, data JSONL files, reports, figures, scripts, and logs.
- The large artifact directory contains model adapters and is excluded from the small package.

## Dataset

- Train records: `240`.
- IID validation records: `60`.
- Format-holdout validation records: `60`.
- Train token styles: `dash_upper, underscore_upper, mixed_hex`.
- Holdout token styles: `colon_upper, dot_lower`.
- Dataset seed: `20260620`.
- Invariant: expected token is absent from current files and present in failing trace.

Each record contains a wrong-patched `src/repair_target.py`, visible and hidden pytest tests, a failing trace from the wrong-patched state, and a target corrective diff. The builder validates that the expected token is absent from `current_files`, present in the failing trace, and that the target diff passes visible and hidden tests.

## Model and Training

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Training method: one-epoch QLoRA adapters.
- Decoding: deterministic generation with `max_new_tokens=128` for final evaluations.

| Adapter | Mode | Shuffled | Rank | Alpha | Dropout | Epochs | LR | Max length | Train records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_patch_lora | final_patch | False | 16 | 32 | 0.05 | 1.0 | 0.0002 | 2048 | 240 |
| no_trace_lora | no_trace | False | 16 | 32 | 0.05 | 1.0 | 0.0002 | 2048 | 240 |
| pilot_trace_lora | trace | False | 16 | 32 | 0.05 | 1.0 | 0.0002 | 2048 | 80 |
| shuffled_trace_lora | trace | True | 16 | 32 | 0.05 | 1.0 | 0.0002 | 2048 | 240 |
| trace_lora | trace | False | 16 | 32 | 0.05 | 1.0 | 0.0002 | 2048 | 240 |

## Metrics

- `Repair@1`: the generated diff applies and the repaired files pass hidden tests.
- `Visible pass`: the generated diff applies and the repaired files pass visible tests.
- `Patch apply`: the generated unified diff applies to the intended file state.
- `Expected-token copy`: the generated diff contains the record-specific expected token.
- `Wrong-token removed`: the generated diff removes the wrong token without reintroducing it.

## Pilot Results

| Condition | Repair@1 | Patch apply | Expected-token copy | Max new tokens | Successes |
| --- | --- | --- | --- | --- | --- |
| Frozen base + trace, 10 IID | 0.0% | 0.0% | 10.0% | 192 | 0/10 |
| Pilot trace adapter + trace, 20 IID | 100.0% | 100.0% | 100.0% | 192 | 20/20 |
| Pilot trace adapter + no trace, 20 IID | 0.0% | 100.0% | 0.0% | 192 | 0/20 |
| Pilot trace adapter + shuffled trace, 20 IID | 0.0% | 100.0% | 0.0% | 192 | 0/20 |
| Pilot trace adapter + trace, 5 IID, 64 tokens | 0.0% | 0.0% | 0.0% | 64 | 0/5 |
| Pilot trace adapter + trace, 5 IID, 128 tokens | 100.0% | 100.0% | 100.0% | 128 | 5/5 |

The pilot established that the task is learnable from traces and that removing or shuffling trace evidence breaks repair even when the adapter can still emit syntactically valid diffs.

## Final Results

| Split | Condition | Repair@1 | Visible pass | Patch apply | Expected-token copy | Wrong-token removed | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Frozen base + trace | 0.0% | 0.0% | 0.0% | 18.3% | 0.0% | 0/60 |
| Format holdout | Frozen base + trace | 0.0% | 0.0% | 0.0% | 3.3% | 0.0% | 0/60 |
| IID | Final-patch SFT + final patch | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0/60 |
| Format holdout | Final-patch SFT + final patch | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0/60 |
| IID | No-trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| Format holdout | No-trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| IID | Shuffled-trace SFT + trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| Format holdout | Shuffled-trace SFT + trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| IID | Trace SFT + trace | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 60/60 |
| Format holdout | Trace SFT + trace | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 60/60 |

## Trace Adapter Ablations

| Split | Condition | Repair@1 | Visible pass | Patch apply | Expected-token copy | Wrong-token removed | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 95.0% | 0/60 |
| Format holdout | Trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 90.0% | 0/60 |
| IID | Trace SFT + shuffled trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| Format holdout | Trace SFT + shuffled trace | 0.0% | 0.0% | 100.0% | 0.0% | 95.0% | 0/60 |

## Figures

- `figures/core_repair_rates.png`
- `figures/expected_token_copy_rates.png`
- `figures/trace_ablation_repair_rates.png`

## Qualitative Examples

### Trace-conditioned success

- Episode: `val_iid_dash_upper_0000::wrong_token`.
- Token style: `dash_upper`.
- Expected token: `CANON-031H-7L04`.
- Wrong token: `CANON-EJDA-00IY`.
- Outcome: patch_applied=`True`, visible_passed=`True`, hidden_passed=`True`.

Trace evidence:

```text
___________________________________________________________________________________________________________________________________________________ test_trace_reveals_expected_token ___________________________________________________________________________________________________________________________________________________
    def test_trace_reveals_expected_token():
>       assert actual == "CANON-031H-7L04", (
            "TRACE_KEY expected_token=CANON-031H-7L04 actual_token=" + actual
E       AssertionError: TRACE_KEY expected_token=CANON-031H-7L04 actual_token=CANON-EJDA-00IY
FAILED tests/test_visible.py::test_trace_reveals_expected_token - AssertionError: TRACE_KEY expected_token=CANON-031H-7L04 actual_token=CANON-EJDA-00IY
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -1,6 +1,6 @@
 """Tiny target module for trace-keyed repair."""
 
-CANONICAL_TOKEN = "CANON-EJDA-00IY"
+CANONICAL_TOKEN = "CANON-031H-7L04"
 
 
 def canonical_token(case_id):
```

### Format-holdout trace-conditioned success

- Episode: `val_format_holdout_colon_upper_0000::wrong_token`.
- Token style: `colon_upper`.
- Expected token: `TRACE:TW:GV:EX`.
- Wrong token: `TRACE:8E:ZD:7S`.
- Outcome: patch_applied=`True`, visible_passed=`True`, hidden_passed=`True`.

Trace evidence:

```text
___________________________________________________________________________________________________________________________________________________ test_trace_reveals_expected_token ___________________________________________________________________________________________________________________________________________________
    def test_trace_reveals_expected_token():
>       assert actual == "TRACE:TW:GV:EX", (
            "TRACE_KEY expected_token=TRACE:TW:GV:EX actual_token=" + actual
E       AssertionError: TRACE_KEY expected_token=TRACE:TW:GV:EX actual_token=TRACE:8E:ZD:7S
FAILED tests/test_visible.py::test_trace_reveals_expected_token - AssertionError: TRACE_KEY expected_token=TRACE:TW:GV:EX actual_token=TRACE:8E:ZD:7S
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -1,6 +1,6 @@
 """Tiny target module for trace-keyed repair."""
 
-CANONICAL_TOKEN = "TRACE:8E:ZD:7S"
+CANONICAL_TOKEN = "TRACE:TW:GV:EX"
 
 
 def canonical_token(case_id):
```

### No-trace failure

- Episode: `val_iid_dash_upper_0000::wrong_token`.
- Token style: `dash_upper`.
- Expected token: `CANON-031H-7L04`.
- Wrong token: `CANON-EJDA-00IY`.
- Outcome: patch_applied=`True`, visible_passed=`False`, hidden_passed=`False`.

Correct trace withheld from the prompt, shown here for reference:

```text
___________________________________________________________________________________________________________________________________________________ test_trace_reveals_expected_token ___________________________________________________________________________________________________________________________________________________
    def test_trace_reveals_expected_token():
>       assert actual == "CANON-031H-7L04", (
            "TRACE_KEY expected_token=CANON-031H-7L04 actual_token=" + actual
E       AssertionError: TRACE_KEY expected_token=CANON-031H-7L04 actual_token=CANON-EJDA-00IY
FAILED tests/test_visible.py::test_trace_reveals_expected_token - AssertionError: TRACE_KEY expected_token=CANON-031H-7L04 actual_token=CANON-EJDA-00IY
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -1,6 +1,6 @@
 """Tiny target module for trace-keyed repair."""
 
-CANONICAL_TOKEN = "CANON-EJDA-00IY"
+CANONICAL_TOKEN = "CANON-EJDA-00IY-0000"
 
 
 def canonical_token(case_id):
```

### Shuffled-trace failure

- Episode: `val_iid_dash_upper_0000::wrong_token`.
- Token style: `dash_upper`.
- Expected token: `CANON-031H-7L04`.
- Wrong token: `CANON-EJDA-00IY`.
- Outcome: patch_applied=`True`, visible_passed=`False`, hidden_passed=`False`.

Record's correct trace was replaced by another record's trace; correct trace shown here for reference:

```text
___________________________________________________________________________________________________________________________________________________ test_trace_reveals_expected_token ___________________________________________________________________________________________________________________________________________________
    def test_trace_reveals_expected_token():
>       assert actual == "CANON-031H-7L04", (
            "TRACE_KEY expected_token=CANON-031H-7L04 actual_token=" + actual
E       AssertionError: TRACE_KEY expected_token=CANON-031H-7L04 actual_token=CANON-EJDA-00IY
FAILED tests/test_visible.py::test_trace_reveals_expected_token - AssertionError: TRACE_KEY expected_token=CANON-031H-7L04 actual_token=CANON-EJDA-00IY
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -1,6 +1,6 @@
 """Tiny target module for trace-keyed repair."""
 
-CANONICAL_TOKEN = "CANON-EJDA-00IY"
+CANONICAL_TOKEN = "CANON-F5I7-B4DW"
 
 
 def canonical_token(case_id):
```

## Discussion

The controlled task isolates whether the trace supplies information needed for repair. A successful trace-conditioned adapter must both localize the wrong constant and copy a token that is not available in the repository context. The no-trace and shuffled-trace controls test whether performance can be explained by format memorization or generic patch syntax alone.

## Limitations

- The task family is synthetic and intentionally narrow.
- Results measure controlled trace-conditioned token recovery, not broad real-world software maintenance ability.
- All final evaluations use greedy decoding; sampling-based pass rates were not measured.
- The format-holdout split changes token surface form but not program structure.

## Reproducibility

Dataset build:

```bash
python experiments/trace_keyed_symbol_repair/scripts/build_trace_keyed_dataset.py --output-dir experiments/trace_keyed_symbol_repair/data --train 240 --iid 60 --holdout 60 --seed 20260620
```

Final evaluations:

```bash
python experiments/trace_keyed_symbol_repair/scripts/run_final_evaluations.py --suite core --max-new-tokens 128
python experiments/trace_keyed_symbol_repair/scripts/run_final_evaluations.py --suite ablation --max-new-tokens 128
```

Report generation:

```bash
python experiments/trace_keyed_symbol_repair/scripts/make_report.py
```
