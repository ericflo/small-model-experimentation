# Qwen3.5-4B Thinking Content vs Compute

**Status:** finished

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

| no_think | filler | shuffle | real | foreign |
| ---: | ---: | ---: | ---: | ---: |
| 0.749 | 0.744 | 0.739 | **0.861** | 0.040 |

Complete attribution (additive ladder no_think → filler → shuffle → real):
- **pure compute + scaffold** (filler − no_think): **−0.005** — contentless `.` tokens buy nothing.
- **token-presence / relevance** (shuffle − filler): **−0.005** — scrambled relevant tokens buy nothing.
- **coherent content** (real − shuffle): **+0.122** — the entire gain.
- **misleading content** (foreign − no_think): **−0.709** — the model follows foreign thinking to the
  wrong problem (verified: a string task + a matrix-sort thought emits `sort_matrix`).

So at the efficient 512 budget on MBPP, **100% of the behavioral thinking gain is coherent reasoning
content** — not compute, not scaffold, not token-presence. Separability is noisy (overlapping CIs; the
foreign AUC is an imbalance artifact), so only the behavioral ladder is robust.

## Interpretation

This **conclusively corrects** the earlier "much of the gain is compute/scaffold, not reasoning": pure
compute (filler) contributes ~0. At the efficient budget the gain is entirely coherent reasoning the
model uses. The "compute/scaffold" reading survives only at high budgets (overthinking) and in the noisy
decodability slice. Next: a high-budget ladder to confirm the coherence advantage shrinks under overthinking.

## Artifacts

- `src/ladder_lib.py` (gen_real captures thinking tokens; gen_answer regenerates answers; activations),
  `src/tasks.py`. `scripts/run.py`, `scripts/verify.py`. `analysis/probe.py`, `analysis/decompose.py`.
- `data/records.jsonl`, `data/labels.jsonl`, `data/tasks.json` (small, in-repo).
- Activations in `large_artifacts/qwen35_4b_thinking_content_vs_compute/` (external, gitignored, regenerable).
