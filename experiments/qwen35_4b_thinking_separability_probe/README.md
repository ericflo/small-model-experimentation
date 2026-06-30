# Qwen3.5-4B Thinking Separability Probe

## Research Program

- Program: `test_time_reasoning_budget`
- Program question: is the native-thinking gain genuine reasoning, or compute/scaffold? And can
  the model's internal state predict its own correctness (attacking C2 from inside the model)?
- Prior anchors: [`qwen35_4b_thinking_budget_scaling`](../qwen35_4b_thinking_budget_scaling/reports/report.md)
  (the shuffle control showed much of the gain is compute/scaffold, and said it could not isolate the
  coherent-reasoning contribution); [`qwen35_4b_thinking_budget_controller`](../qwen35_4b_thinking_budget_controller/reports/report.md)
  (bounded by visible-test false-passes — C2); the corpus's synthetic hidden-state probes
  (`qwen_readable_candidate_verifier`, `qwen_candidate_conditioned_trace_verifier`,
  `qwen_prefix_state_process_verifier`).

## Question

Does native thinking make the model's own correctness more **linearly decodable** from its
activations? Concretely: train a linear probe on the answer-token hidden state of Qwen3.5-4B's
**own** generated MBPP solution (labels = execution pass) and compare its per-layer AUC across
`no_think`, real `think`, and `shuffled-think` at matched budgets. "Does it know it's right
better after reasoning?"

## Hypothesis

The shuffle puzzle left two possibilities. If real thinking raises probe separability **beyond
shuffled-think** (which preserves token-count/scaffold but destroys coherent order), that is the
genuine-reasoning-content signal the scaling report could not isolate — reasoning reorganizes the
model's internal state toward a more decodable correctness representation. If real ≈ shuffled,
then thinking's effect on internal "knowing" is also mostly compute/scaffold, reinforcing C9.

## Setup

- Model: Qwen3.5-4B (frozen, bf16, sdpa, fast path enabled). 32 layers → 33 hidden states.
- Dataset: MBPP sanitized `test`, first 100 tasks; k=8 sampled solutions per task per condition.
- Conditions: `no_think`, `think_512`, `shuffle_512`, `think_1024`, `shuffle_1024` (s1-style budget
  forcing on `</think>`; shuffle = permute the model's own thinking tokens before the answer).
- Signal: the **answer-token** (last-token) hidden state of the model's own generated sequence
  (prompt + thinking + answer), extracted per layer with a clean **right-padded** forward pass
  (keeps the linear-attention recurrence uncorrupted).
- Probe: per-layer logistic regression (standardized), **GroupKFold by task** (no task-identity
  leakage), out-of-fold AUC for predicting full-test pass; bootstrap CI by resampling tasks.
- Controls: shuffled-label probe (must be ~0.5); think vs shuffle at matched budget; per-layer sweep.
- Deployable / oracle boundary: probe trained on hidden-test outcomes is a non-deployable
  **diagnostic** of decodability. The deployable angle is the **false-pass test**: among candidates
  that pass the *visible* test (what the controller commits), can the probe rank the true full-test
  passes above the C2 false-passes?

## Run

Smoke (4 tasks, 2 conds): `../../.venv/bin/python scripts/run.py --smoke`

Full:

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python scripts/run.py --tasks 100 --k 8      # generate + extract activations
../../.venv/bin/python scripts/verify_probe.py                 # torch-free execution labels
../../.venv/bin/python analysis/probe.py                       # per-layer probes + figure
```

## Results

Full results in [reports/report.md](reports/report.md); figure `analysis/auc_vs_layer.png`.
Best-layer probe AUC (predict full-test pass from the answer-token activation; n=100, 800/cond):

| condition | probe AUC | shuffled-label | visible-passer AUC |
| --- | ---: | ---: | ---: |
| no_think | 0.642 | 0.514 | 0.518 |
| think_512 | 0.708 | 0.482 | 0.684 |
| shuffle_512 | 0.733 | 0.511 | 0.626 |
| think_1024 | 0.720 | 0.456 | 0.598 |
| shuffle_1024 | 0.755 | 0.520 | 0.669 |

1. **Correctness is moderately decodable** from one answer-token activation (AUC 0.64–0.76 vs
   shuffled-label ~0.50) — "models know more than they say," novel on self-generated real code.
2. **Thinking raises decodability** across essentially every layer (no_think ~0.5–0.64 → thinking
   ~0.67–0.75): "knows it's right better after thinking."
3. **But not via reasoning — hypothesis falsified.** *Shuffled* thinking matches/exceeds *real*
   thinking at both budgets and across all layers. So the active ingredient is compute/scaffold/
   token-presence, not coherent content — **converging with the behavioral C9 shuffle finding at
   the representational level.**
4. Deployable spinoff: among visible-passers (C2 false-pass regime), the probe has weak signal under
   thinking (~0.60–0.68) but ~chance under no-think.

## Interpretation

The cleanest evidence yet that this 4B's thinking benefit is largely **not coherent reasoning** —
now shown at two independent levels (behavioral accuracy C9 + internal representation). Thinking
makes the model more internally aware of its own correctness, but scrambled thinking does so equally.
A weak verifier-free signal (probe vs C2 false-passes) appears only under thinking. Caveats: moderate
AUCs, n=100 single seed, real-vs-shuffle within overlapping CIs (robust claim is the across-depth
ordering), last-token probe only.

## Knowledgebase Update

- Program evidence updated: yes (`research_programs/test_time_reasoning_budget/evidence.md`).
- Claim ledger updated: C9 extended (reasoning-vs-compute confirmed at the representational level).

## Artifacts

- `src/probe_lib.py` (generation + activation extraction), `src/tasks.py` (MBPP + sandbox verifier).
- `scripts/run.py` (extract), `scripts/verify_probe.py` (labels), `analysis/probe.py` (probes + figure).
- `data/records.jsonl`, `data/labels.jsonl`, `data/tasks.json` (small, in-repo).
- Activations (~0.7 GB) in `large_artifacts/qwen35_4b_thinking_separability_probe/` (external, gitignored;
  regenerable by `scripts/run.py`) — see `reports/artifact_manifest.yaml`.
