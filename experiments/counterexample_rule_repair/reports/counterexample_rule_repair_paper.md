# Counterexample Rule Repair

Generated: `2026-06-20 20:16:10 UTC`.

## Abstract

This experiment tests whether a code-repair model can use failed-test counterexamples as evidence for a compact behavioral rule. Each task presents a wrong-patched implementation and visible pytest failures that print concrete input, expected output, and actual output triples. The desired rule is not stated in the issue text. A correct patch must infer the rule from visible counterexamples and pass hidden tests on unseen inputs.

## Artifact Layout

- Small, download-friendly experiment package: `/workspace/experiments/counterexample_rule_repair`.
- Large adapters and checkpoints: `/workspace/large_artifacts/counterexample_rule_repair`.
- The small package contains configs, data JSONL files, reports, figures, scripts, and logs.
- The large artifact directory contains LoRA adapters and is excluded from the small package.

## Dataset

- Train records: `240`.
- IID validation records: `45`.
- Format-holdout validation records: `45`.
- Rule-family-holdout validation records: `45`.
- Train families: `affine_int, slug_affix, threshold_label`.
- Withheld rule families: `parity_offset_holdout`.
- Dataset seed: `20260620`.
- Invariants: wrong-patched implementation fails visible counterexamples; target corrective diff applies to the wrong-patched implementation; target implementation passes visible and hidden tests; hidden test inputs do not overlap visible trace inputs; visible expected outputs appear in the failed execution trace.

Each record contains `src/repair_target.py`, visible tests, hidden tests, the failed trace from the wrong-patched implementation, and the target corrective diff. The visible trace emits `COUNTEREXAMPLE input=... expected=... actual=...` lines. Hidden cases are disjoint from visible inputs, so copying only the visible cases is insufficient.

## Model and Training

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Training method: QLoRA adapters.
- Final training recipe: 3 epochs, rank 32, alpha 64, dropout 0.05, learning rate 1.5e-4, max length 3072.
- Decoding: deterministic generation with `max_new_tokens=256` for final evaluations.

| Adapter | Mode | Shuffled | Rank | Alpha | Dropout | Epochs | LR | Max length | Train records | Eval records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_patch_lora | final_patch | False | 32 | 64 | 0.05 | 3.0 | 0.00015 | 3072 | 240 | 45 |
| no_trace_lora | no_trace | False | 32 | 64 | 0.05 | 3.0 | 0.00015 | 3072 | 240 | 45 |
| pilot_trace_lora | trace | False | 16 | 32 | 0.05 | 1.0 | 0.0002 | 3072 | 90 | 45 |
| shuffled_trace_lora | trace | True | 32 | 64 | 0.05 | 3.0 | 0.00015 | 3072 | 240 | 45 |
| trace_lora | trace | False | 32 | 64 | 0.05 | 3.0 | 0.00015 | 3072 | 240 | 45 |

## Conditions

- `Frozen base + trace`: base model with the wrong-patched file and failed trace, no fine-tuning.
- `Trace SFT + trace`: adapter trained and evaluated with failed counterexample traces.
- `No-trace SFT + no trace`: adapter trained and evaluated without failed trace text.
- `Shuffled-trace SFT + trace`: adapter trained on mismatched trace evidence, evaluated with the real trace.
- `Final-patch SFT + final patch`: adapter trained to reproduce final diffs from the original buggy state rather than repair from the wrong-patched state.
- `Trace SFT + no trace` and `Trace SFT + shuffled trace`: input ablations for the trace adapter.

## Metrics

- `Repair@1`: the generated diff applies and repaired files pass both visible and hidden tests.
- `Visible pass`: repaired files pass the visible counterexample tests.
- `Hidden pass`: repaired files pass hidden tests on unseen inputs.
- `Patch apply`: the generated unified diff applies to the intended file state.
- `Marker match`: the diff contains all target rule markers recorded by the dataset builder.
- `Input literal`: the diff contains at least one visible input literal, a diagnostic for hardcoding visible cases.

## Iteration Log Summary

| Condition | Repair@1 | Visible pass | Hidden pass | Patch apply | Successes |
| --- | --- | --- | --- | --- | --- |
| Frozen base + trace, 6 IID | 0.0% | 0.0% | missing | 16.7% | 0/6 |
| Pilot trace adapter + trace, 20 IID | 45.0% | 45.0% | 65.0% | 100.0% | 9/20 |
| Pilot trace adapter + no trace, 20 IID | 0.0% | 0.0% | 0.0% | 100.0% | 0/20 |
| Pilot trace adapter + shuffled trace, 20 IID | 0.0% | 0.0% | 0.0% | 100.0% | 0/20 |
| Full trace adapter + trace, 20 IID check | 85.0% | 85.0% | 85.0% | 100.0% | 17/20 |

The initial frozen pilot showed near-zero patch application and no successful repairs. A small trace adapter made the task learnable, while no-trace and shuffled-trace prompts remained at zero repair. The full trace adapter then improved the 20-record IID check enough to justify training the full control adapters.

## Final Results

| Split | Condition | Repair@1 | Visible pass | Hidden pass | Patch apply | Marker match | Input literal | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Frozen base + trace | 0.0% | 0.0% | 0.0% | 4.4% | 2.2% | 48.9% | 0/45 |
| IID | Final-patch SFT + final patch | 0.0% | 0.0% | 0.0% | 100.0% | 17.8% | 64.4% | 0/45 |
| IID | No-trace SFT + no trace | 8.9% | 8.9% | 8.9% | 100.0% | 22.2% | 53.3% | 4/45 |
| IID | Shuffled-trace SFT + trace | 4.4% | 4.4% | 4.4% | 100.0% | 17.8% | 53.3% | 2/45 |
| IID | Trace SFT + trace | 91.1% | 91.1% | 91.1% | 100.0% | 95.6% | 55.6% | 41/45 |
| Format holdout | Frozen base + trace | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 48.9% | 0/45 |
| Format holdout | Final-patch SFT + final patch | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 42.2% | 0/45 |
| Format holdout | No-trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 62.2% | 0/45 |
| Format holdout | Shuffled-trace SFT + trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 60.0% | 0/45 |
| Format holdout | Trace SFT + trace | 53.3% | 53.3% | 53.3% | 100.0% | 68.9% | 66.7% | 24/45 |
| Rule-family holdout | Frozen base + trace | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0/45 |
| Rule-family holdout | Final-patch SFT + final patch | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/45 |
| Rule-family holdout | No-trace SFT + no trace | 2.2% | 2.2% | 2.2% | 100.0% | 2.2% | 100.0% | 1/45 |
| Rule-family holdout | Shuffled-trace SFT + trace | 0.0% | 0.0% | 0.0% | 100.0% | 2.2% | 100.0% | 0/45 |
| Rule-family holdout | Trace SFT + trace | 0.0% | 0.0% | 0.0% | 100.0% | 6.7% | 100.0% | 0/45 |

## Trace Adapter Ablations

| Split | Condition | Repair@1 | Visible pass | Hidden pass | Patch apply | Marker match | Input literal | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 11.1% | 53.3% | 0/45 |
| IID | Trace SFT + shuffled trace | 0.0% | 0.0% | 0.0% | 100.0% | 8.9% | 53.3% | 0/45 |
| Format holdout | Trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 57.8% | 0/45 |
| Format holdout | Trace SFT + shuffled trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 55.6% | 0/45 |
| Rule-family holdout | Trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/45 |
| Rule-family holdout | Trace SFT + shuffled trace | 0.0% | 0.0% | 0.0% | 100.0% | 2.2% | 100.0% | 0/45 |

## Trace Adapter Family Breakdown

| Condition | Family | Repair@1 | Visible pass | Hidden pass | Successes |
| --- | --- | --- | --- | --- | --- |
| Trace SFT + trace / IID | affine_int | 73.3% | 73.3% | 73.3% | 11/15 |
| Trace SFT + trace / IID | slug_affix | 100.0% | 100.0% | 100.0% | 15/15 |
| Trace SFT + trace / IID | threshold_label | 100.0% | 100.0% | 100.0% | 15/15 |
| Trace SFT + trace / Format holdout | affine_int | 0.0% | 0.0% | 0.0% | 0/15 |
| Trace SFT + trace / Format holdout | slug_affix | 60.0% | 60.0% | 60.0% | 9/15 |
| Trace SFT + trace / Format holdout | threshold_label | 100.0% | 100.0% | 100.0% | 15/15 |
| Trace SFT + trace / Rule-family holdout | parity_offset_holdout | 0.0% | 0.0% | 0.0% | 0/45 |

## Figures

- `figures/core_repair_rates.png`
- `figures/trace_ablation_repair_rates.png`
- `figures/visible_pass_rates.png`

## Qualitative Examples

### Trace-conditioned IID success

- Episode: `val_iid_threshold_0002::counterexample_rule`.
- Family: `threshold_label`.
- Outcome: patch_applied=`True`, visible_passed=`True`, hidden_passed=`True`.

Visible counterexamples:

```text
f"COUNTEREXAMPLE input={value!r} expected={expected!r} actual={actual!r}"
E       AssertionError: COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=8 expected='LOW_DYNF' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=9 expected='HIGH_C68M' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=11 expected='HIGH_C68M' actual='BADH_BTX1'
E       assert not ["COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'", "COUNTEREXAMPLE input=8 expected='LOW_DYNF' actual='...LE input=9 expected='HIGH_C68M' actual='BADL_1KL8'", "COUNTEREXAMPLE input=11 expected='HIGH_C68M' actual='BADH_BTX1'"]
FAILED tests/test_visible.py::test_visible_counterexamples - AssertionError: COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -2,6 +2,6 @@
 
 
 def apply_rule(value):
-    if value < 11:
-        return "BADL_1KL8"
-    return "BADH_BTX1"
+    if value < 9:
+        return "LOW_DYNF"
+    return "HIGH_C68M"
```

### Trace-conditioned format-holdout success

- Episode: `val_format_holdout_slug_0003::counterexample_rule`.
- Family: `slug_affix`.
- Outcome: patch_applied=`True`, visible_passed=`True`, hidden_passed=`True`.

Visible counterexamples:

```text
f"COUNTEREXAMPLE input={value!r} expected={expected!r} actual={actual!r}"
E       AssertionError: COUNTEREXAMPLE input='  Alpha Beta  ' expected='pre:ckmalpha.beta:doneje' actual='WP_8SSCalpha/beta_WSW7JH'
E         COUNTEREXAMPLE input='MIXED_case Word' expected='pre:ckmmixed.case.word:doneje' actual='WP_8SSCmixed/case/word_WSW7JH'
E         COUNTEREXAMPLE input='two   spaces' expected='pre:ckmtwo.spaces:doneje' actual='WP_8SSCtwo/spaces_WSW7JH'
E       assert not ["COUNTEREXAMPLE input='  Alpha Beta  ' expected='pre:ckmalpha.beta:doneje' actual='WP_8SSCalpha/beta_WSW7JH'", "COUNT..._WSW7JH'", "COUNTEREXAMPLE input='two   spaces' expected='pre:ckmtwo.spaces:doneje' actual='WP_8SSCtwo/spaces_WSW7JH'"]
FAILED tests/test_visible.py::test_visible_counterexamples - AssertionError: COUNTEREXAMPLE input='  Alpha Beta  ' expected='pre:ckmalpha.beta:doneje' actual='WP_8SSCalpha/beta_WSW7JH'
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -4,5 +4,5 @@
 def apply_rule(value):
     text = str(value).strip().lower().replace("_", " ")
     pieces = [piece for piece in text.split() if piece]
-    body = "/".join(pieces)
-    return "WP_8SSC" + body + "_WSW7JH"
+    body = ".".join(pieces)
+    return "pre:ckm" + body + ":doneje"
```

### Withheld-rule-family trace-conditioned example

- Episode: `val_rule_holdout_parity_0027::counterexample_rule`.
- Family: `parity_offset_holdout`.
- Outcome: patch_applied=`True`, visible_passed=`False`, hidden_passed=`False`.

Visible counterexamples:

```text
f"COUNTEREXAMPLE input={value!r} expected={expected!r} actual={actual!r}"
E       AssertionError: COUNTEREXAMPLE input=0 expected=4 actual=2
E         COUNTEREXAMPLE input=1 expected=12 actual=9
E         COUNTEREXAMPLE input=4 expected=8 actual=6
E         COUNTEREXAMPLE input=7 expected=18 actual=15
E       assert not ['COUNTEREXAMPLE input=0 expected=4 actual=2', 'COUNTEREXAMPLE input=1 expected=12 actual=9', 'COUNTEREXAMPLE input=4 expected=8 actual=6', 'COUNTEREXAMPLE input=7 expected=18 actual=15']
FAILED tests/test_visible.py::test_visible_counterexamples - AssertionError: COUNTEREXAMPLE input=0 expected=4 actual=2
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -2,6 +2,6 @@
 
 
 def apply_rule(value):
-    if value % 2 == 0:
-        return value + 2
-    return value + 8
+    if value % 2 == 0:
+        return value + 4
+    return value + 6
```

### Trace adapter with trace removed

- Episode: `val_iid_threshold_0002::counterexample_rule`.
- Family: `threshold_label`.
- Outcome: patch_applied=`True`, visible_passed=`False`, hidden_passed=`False`.

Visible counterexamples:

```text
f"COUNTEREXAMPLE input={value!r} expected={expected!r} actual={actual!r}"
E       AssertionError: COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=8 expected='LOW_DYNF' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=9 expected='HIGH_C68M' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=11 expected='HIGH_C68M' actual='BADH_BTX1'
E       assert not ["COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'", "COUNTEREXAMPLE input=8 expected='LOW_DYNF' actual='...LE input=9 expected='HIGH_C68M' actual='BADL_1KL8'", "COUNTEREXAMPLE input=11 expected='HIGH_C68M' actual='BADH_BTX1'"]
FAILED tests/test_visible.py::test_visible_counterexamples - AssertionError: COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -2,6 +2,6 @@
 
 
 def apply_rule(value):
-    if value < 11:
-        return "BADL_1KL8"
-    return "BADH_BTX1"
+    if value < 7:
+        return "BADL_1KL8"
+    return "BADH_BTX1"
```

### Trace adapter with shuffled trace

- Episode: `val_iid_threshold_0002::counterexample_rule`.
- Family: `threshold_label`.
- Outcome: patch_applied=`True`, visible_passed=`False`, hidden_passed=`False`.

Visible counterexamples:

```text
f"COUNTEREXAMPLE input={value!r} expected={expected!r} actual={actual!r}"
E       AssertionError: COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=8 expected='LOW_DYNF' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=9 expected='HIGH_C68M' actual='BADL_1KL8'
E         COUNTEREXAMPLE input=11 expected='HIGH_C68M' actual='BADH_BTX1'
E       assert not ["COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'", "COUNTEREXAMPLE input=8 expected='LOW_DYNF' actual='...LE input=9 expected='HIGH_C68M' actual='BADL_1KL8'", "COUNTEREXAMPLE input=11 expected='HIGH_C68M' actual='BADH_BTX1'"]
FAILED tests/test_visible.py::test_visible_counterexamples - AssertionError: COUNTEREXAMPLE input=7 expected='LOW_DYNF' actual='BADL_1KL8'
```

Generated diff:

```diff
--- a/src/repair_target.py
+++ b/src/repair_target.py
@@ -2,6 +2,6 @@
 
 
 def apply_rule(value):
-    if value < 11:
-        return "BADL_1KL8"
-    return "BADH_BTX1"
+    if value == '  Alpha Beta  ':
+        return "P_512Talpha+beta_SX05E"
+    return "P_512Ttwo+spaces_SX05E"
```

## Discussion

The core contrast is whether the model can transform failed counterexample traces into a general rule rather than merely producing syntactically plausible diffs. The visible and hidden pass split is important: a patch can sometimes satisfy hidden cases while violating visible counterexamples, so the primary metric requires both. The rule-family holdout is a harder extrapolation test because the parity-offset structure is absent from training.

## Limitations

- The tasks are synthetic and intentionally focused on one-file rule repair.
- The experiment measures greedy single-sample repair, not sampling-based pass rates.
- Hidden tests are generated from known templates, so they are controlled probes rather than open-ended software behavior.
- The withheld rule family tests structural transfer to one unseen family only.

## Reproducibility

Dataset build:

```bash
python experiments/counterexample_rule_repair/scripts/build_counterexample_dataset.py --output-dir experiments/counterexample_rule_repair/data --train-per-family 80 --iid-per-family 15 --format-per-family 15 --rule-holdout 45 --seed 20260620
```

Final evaluations:

```bash
python experiments/counterexample_rule_repair/scripts/run_final_evaluations.py --suite all --max-new-tokens 256
```

Report generation:

```bash
python experiments/counterexample_rule_repair/scripts/make_report.py
```
