# Qwen Register Trace Refiner Experiment Log

## Objective

Train a learned repair verifier for a Qwen3-4B register-program compiler. The compiler emits a fixed register program for modular arithmetic. The refiner enumerates local program edits, executes each candidate with a deterministic modular runtime, and learns to select a better candidate without access to the target answer or target trace at evaluation time.

## Artifact Discipline

- Experiment code, metrics, reports, and figures are in `experiments/qwen_register_trace_refiner/`.
- Large model artifacts are in `large_artifacts/qwen_register_trace_refiner/`.
- `checkpoint_manifest.csv` records the input compiler checkpoint and trained main refiner checkpoint.
- Analysis includes CSV summaries and three figures:
  - `analysis/figures/executor_accuracy_by_split.png`
  - `analysis/figures/oracle_gap_recovered.png`
  - `analysis/figures/candidate_set_profile.png`

## Iteration Notes

### 1. Scaffold

Created a standalone experiment directory with local source files:

- `qwen_register_trace_refiner_core.py`
- `qwen_register_trace_refiner_experiment.py`
- `analyze_qwen_register_trace_refiner.py`

Copied the fixed input compiler into:

`large_artifacts/qwen_register_trace_refiner/checkpoints/input_register_compiler`

### 2. Smoke Run

Command used a tiny dataset, top-2 one-edit search, and one verifier epoch.

Outcome: the register interface, candidate construction, verifier training, metrics writing, and figure generation all worked.

### 3. Pilot Run

Run: `pilot_register_trace_refiner_s128`

Settings:

- 128 train examples
- 64 validation examples
- 64 fresh standard examples
- 64 fresh paraphrase examples
- 64 paired evaluation pairs
- top-3/two-edit repair search
- 6 verifier epochs

Finding: the candidate set had oracle headroom, but the learned verifier kept choosing the base candidate. Fresh standard was 20.3% base and 40.6% oracle, but learned stayed at 20.3%.

### 4. Oversampled Pilot

Run: `pilot_register_trace_refiner_oversample_s128`

Change: oversampled repairable training groups where the base candidate was wrong but a local edit was exact.

Finding: validation improved from 15.6% base to 18.8% learned, but fresh standard dropped from 20.3% to 17.2%. This showed the verifier could learn repair choices but needed a base-preserving selection rule.

### 5. Guarded Selection

Added validation-tuned guarded selection:

Keep the learned candidate only when its score beats the base candidate by a tuned margin; otherwise keep the base program.

Smoke run `smoke_register_trace_refiner_guarded` verified the guarded path and paired metrics.

### 6. Main Run

Run: `main_register_trace_refiner_s512`

Settings:

- 512 train examples
- 128 validation examples
- 256 fresh standard examples
- 256 fresh paraphrase examples
- 256 paired evaluation pairs
- top-3/two-edit repair search, 1,299 candidates per example
- 18 verifier epochs
- repairable-group oversampling = 10
- guard threshold selected on validation = 0.25

Runtime: 1087.1 seconds on NVIDIA RTX 6000 Ada Generation.

## Main Metrics

| split | base | learned | guarded | oracle |
|---|---:|---:|---:|---:|
| train_len24 | 14.8% | 16.6% | 16.6% | 21.9% |
| val_len24 | 15.6% | 17.2% | 17.2% | 20.3% |
| fresh_standard_len24 | 23.4% | 26.6% | 26.6% | 37.1% |
| fresh_paraphrase_len24 | 4.7% | 4.7% | 4.7% | 7.0% |
| fresh_paired_len24 | 12.3% | 12.9% | 12.9% | 18.6% |

## Interpretation

The refiner produced a real but small gain. Standard fresh L24 improved by 3.2 points and recovered 22.9% of the oracle gap. Paired L24 improved by 0.6 points and recovered 9.4% of the oracle gap. Paraphrase L24 did not improve.

The candidate set is the main limiter for paraphrase robustness: oracle availability is only 7.0% on paraphrase and 18.6% on paired evaluation. Selection is still a limiter on standard prompts: oracle reaches 37.1%, while learned/guarded reaches 26.6%.

