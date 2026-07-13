# Qwen Register Trace Refiner

**Status:** finished

This experiment tests whether a learned verifier can improve a Qwen3-4B register-program compiler by searching local edits around the compiled latent program and selecting a better execution trace.

The experiment is standalone: it packages the fixed input compiler and the trained refiner under `large_artifacts/qwen_register_trace_refiner/`, while the experiment directory contains only source, logs, metrics, figures, and write-ups.

## Layout

- `src/qwen_register_trace_refiner_experiment.py` - builds candidate repairs, trains the verifier, evaluates base/learned/guarded/oracle selection.
- `src/qwen_register_trace_refiner_core.py` - local register compiler, data generator, and modular runtime utilities.
- `src/analyze_qwen_register_trace_refiner.py` - regenerates CSV summaries and figures.
- `runs/` - smoke, pilot, and main run metrics.
- `analysis/` - aggregate CSVs, summary, and figures.
- `reports/` - standalone experiment log and paper-style report.
- `checkpoint_manifest.csv` - exact large artifacts used by the main run.

## Large Artifacts

Download or preserve these separately from the experiment folder:

- `large_artifacts/qwen_register_trace_refiner/checkpoints/input_register_compiler`
- `large_artifacts/qwen_register_trace_refiner/checkpoints/main_register_trace_refiner_s512/register_trace_refiner.pt`

The input compiler directory is about 84 MB. The trained refiner checkpoint is about 2.5 MB.

## Main Result

Fresh length-24 modular programs, top-3/two-edit local repair search:

| split | base | learned/guarded | oracle |
|---|---:|---:|---:|
| standard | 23.4% | 26.6% | 37.1% |
| paraphrase | 4.7% | 4.7% | 7.0% |
| paired | 12.3% | 12.9% | 18.6% |

The learned refiner recovers a small part of the available oracle gap. The larger result is diagnostic: the correct program is often not present in the local repair set, especially for paraphrases, and selecting repaired candidates robustly remains difficult.

## Reproduce

Smoke:

```bash
PYTHONPATH=experiments/qwen_register_trace_refiner/src \
python experiments/qwen_register_trace_refiner/src/qwen_register_trace_refiner_experiment.py \
  --run_name smoke_register_trace_refiner_guarded \
  --train_examples 8 --val_examples 4 --eval_examples 4 --eval_pairs 4 \
  --verifier_epochs 1 --qwen_batch_size 2 --repair_topk 2 --repair_max_edits 1 \
  --trace_d_model 64 --trace_layers 1 --trace_heads 4
```

Main:

```bash
PYTHONPATH=experiments/qwen_register_trace_refiner/src \
python experiments/qwen_register_trace_refiner/src/qwen_register_trace_refiner_experiment.py \
  --run_name main_register_trace_refiner_s512 \
  --train_examples 512 --val_examples 128 --eval_examples 256 --eval_pairs 256 \
  --verifier_epochs 18 --qwen_batch_size 8 \
  --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 24 \
  --trace_d_model 128 --trace_layers 3 --trace_heads 4 --trace_ff_mult 4
```

Analysis:

```bash
python experiments/qwen_register_trace_refiner/src/analyze_qwen_register_trace_refiner.py
```

