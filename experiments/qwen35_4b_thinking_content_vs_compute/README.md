# Qwen3.5-4B Thinking Content vs Compute

## Research Program

- Program: `test_time_reasoning_budget`
- Program question: of the native-thinking gain, how much is coherent reasoning vs token-presence/
  relevance vs pure compute + scaffold?
- Prior anchors: `qwen35_4b_thinking_budget_scaling` (shuffle control: coherent order reproduced part
  of the gain) and `qwen35_4b_thinking_separability_probe` (shuffled ≈ real in separability). Both kept
  the same thinking tokens; neither removed relevance — this experiment does.

## Question

The shuffle control destroys coherent *order* but keeps the *same thinking-token multiset* (relevant
variable names, values, operations stay in context). This experiment adds the **foreign-task-thinking**
control — splice a *different* task's thinking into this task — which removes **relevance/token-presence**
while keeping count + scaffold + compute. Where does foreign land on the ladder?

```
no_think  →  foreign  →  shuffle  →  real
            +compute    +relevance   +coherent
            +scaffold   (presence)   order
```

- foreign ≈ real ≈ shuffle → thinking is **pure compute + format**; even irrelevant thinking helps as much.
- foreign ≈ no_think (≪ shuffle) → **relevance/token-presence** is the active ingredient (just not order).

## Hypothesis

Given C9 (order doesn't matter), the open question is whether the *relevant tokens* matter. If foreign
collapses to (or below) no_think while shuffle stays high, the thinking benefit is "the relevant tokens
being in context," not reasoning and not pure compute.

## Setup

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8, thinking budget 512.
- **Real thinking generated once** (capturing its thinking tokens); shuffle (permute those tokens) and
  foreign (a cyclically-shifted *other* task's thinking tokens, same sample slot) reuse the tokens and
  **regenerate only the answer** from the modified prefix — so all conditions share the same
  thinking-token multiset and matched thinking length (compute).
- Measures: behavioral full-test pass (k=8) AND per-layer answer-token separability probe (as in the
  separability experiment: right-padded activations, GroupKFold-by-task logistic, bootstrap CI,
  shuffled-label control).

## Run

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python scripts/run.py --tasks 100 --k 8 --budget 512
../../.venv/bin/python scripts/verify.py        # execution labels
../../.venv/bin/python analysis/probe.py        # per-layer separability
../../.venv/bin/python analysis/decompose.py     # behavioral + separability ladder
```

## Results

Full results in [reports/report.md](reports/report.md). Behavioral ladder (full-pass, n=100):

| no_think | foreign | shuffle | real |
| ---: | ---: | ---: | ---: |
| 0.764 | **0.043** | 0.739 | **0.859** |

- **Foreign thinking collapses accuracy to 4%** — the model *follows* the foreign reasoning to the
  wrong problem (verified: a string task + a matrix-sort thought emits `sort_matrix`). So the model
  uses thinking as **content**, not a content-free compute/format crutch.
- **Shuffle ≈ no_think** (0.739 vs 0.764): scrambled relevant thinking ≈ no thinking (sampled full-pass).
- **Real beats shuffle by +12pp**: at the efficient 512 budget the gain **is coherent reasoning**.
- Separability is noisy here (no_think 0.682, shuffle 0.636, real 0.676 — overlapping CIs; foreign
  0.994 is an imbalance artifact), so only the behavioral ladder is robust.

## Interpretation

This **corrects** the earlier "much of the gain is compute/scaffold, not reasoning" (a greedy-metric
artifact that held mainly at high budgets / in the noisy decodability slice). At the efficient budget,
the behavioral gain is genuine coherent reasoning over relevant content. Remaining piece: a
filler/pause-token arm to isolate pure compute (foreign adds *misleading* content, not contentless compute).

## Artifacts

- `src/ladder_lib.py` (gen_real captures thinking tokens; gen_answer regenerates answers; activations),
  `src/tasks.py`. `scripts/run.py`, `scripts/verify.py`. `analysis/probe.py`, `analysis/decompose.py`.
- `data/records.jsonl`, `data/labels.jsonl`, `data/tasks.json` (small, in-repo).
- Activations in `large_artifacts/qwen35_4b_thinking_content_vs_compute/` (external, gitignored, regenerable).
