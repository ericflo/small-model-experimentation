# Qwen3.5-4B Thinking-Budget Scaling

## Research Program

- Program: `test_time_reasoning_budget`
- Program question: Is the native thinking-token budget a controllable test-time-compute
  axis that raises deployable accuracy, or only the oracle ceiling / cost — on a model the
  corpus always ran in no-think mode?
- Prior anchors: `qwen_python_shaped_silent_executor` (the only corpus run that enabled
  thinking; fixed 768-token CoT, never swept), `qwen35_4b_adaptive_evidence_budget_policy`
  and `qwen35_4b_humaneval_adaptive_budget` (STOP/MORE controllers over *evidence*, not thinking),
  `qwen35_4b_real_sample_verify_commit` / `qwen35_4b_retrieval_adapt_verify_scale` (C2:
  coverage ≫ deployable selection on code).

## Question

When Qwen3.5-4B is allowed its native thinking mode, how does MBPP accuracy scale with the
thinking-token budget, and does extra thinking raise the **oracle ceiling** (pass@k), the
**deployable line** (greedy / visible-test-selected pass@1), or **only cost**? Does the
*content* of thinking matter, or only the `<think>` scaffold plus extra compute?

## Hypothesis

From the corpus's central confirmed bottleneck (C2), the prior is that extra thinking will
raise the oracle ceiling (more diverse correct candidates → higher pass@k) more than the
deployable line (a single greedy or visible-selected answer). If so, the thinking budget is a
new instance of coverage-without-selection, and the payoff shifts to a thinking-budget
controller and thinking-as-verifier (program backlog). A clean alternative is that thinking
lifts greedy pass@1 directly (genuine deployable reasoning gain).

## Setup

- Model: **Qwen/Qwen3.5-4B** (the repo standard), bf16, `AutoModelForCausalLM`, `attn_implementation=sdpa`.
- Dataset/task source: MBPP **sanitized**, `test` split (held out), first 100 tasks.
- Protocol: prompt = NL description + one example assert (signature anchor); candidates
  verified by executing the full `test_list` in a sandboxed subprocess. Visible-test selector
  uses only the first assert (deployable), then verifies the chosen candidate on the full set.
- Budgets (s1-style budget forcing on `</think>`=248069): `no_think` (enable_thinking=False),
  `think_{256,512,1024,2048}`, `think_unbudgeted`. Greedy (deployable pass@1) + k=8 sampled
  candidates (oracle ceiling). Thinking decode temp 0.6/top_p0.95/top_k20; no-think 0.7/0.8/20.
- Baseline: `no_think` (the corpus's universal setting).
- Controls (planned follow-up run, `--controls`): **shuffled-thinking** and **truncated-thinking**
  (isolate thinking *content* from the scaffold + compute); matched-total-token comparison.
- Primary metric (deployable): greedy pass@1 and visible-test selector@1 vs budget.
- Oracle-only metrics: pass@k (k=8) — labeled non-deployable (uses hidden test outcomes to pick).
- Hidden-label boundary: pass@k and any "oracle" selection use hidden test results and are
  reported separately from deployable (greedy / visible-test-only) numbers.

## Run

Smoke (proves the path; ~5 min):

```bash
.venv/bin/python scripts/run.py --smoke
```

Full (background; ~hours on one RTX 4090):

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run.py --tasks 100 --k 8 \
  --budgets no_think,256,512,1024,2048,unbudgeted --out runs/main
# then
.venv/bin/python analysis/analyze.py --tag main
```

Generation (GPU) and verification (a separate torch-free process, so candidate-sandbox forks
never inherit a CUDA context) are split inside `run.py`.

## Results

Full results, controls, and limitations in [reports/report.md](reports/report.md). Headline
(n=100 MBPP, k=8), deployable greedy pass@1 by thinking budget:

| no_think | think_256 | think_512 | think_1024 | think_2048 | unbudgeted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.760 | 0.870 | 0.870 | **0.910** | 0.860 | 0.840 |

- **Native thinking is a deployable win the corpus disabled:** greedy +15pp (0.76→0.91), moving
  the deployable line *more* than the oracle ceiling (pass@8 +5pp, 0.91→0.96) and *closing* the
  selection gap — opposite of the C2-based prior. Paired: 17 fail→pass vs 2 pass→fail (McNemar p≈0.001).
- **Not monotonic:** broad optimum ~512–1024 then decline; `unbudgeted` (0.84) < a cap (0.91).
- **Scales with difficulty:** +22pp on middling tasks (n=40); +33pp on never-solved (n=9, suggestive).
- **Shuffled-thinking control:** scrambling the model's own thinking reproduces much of the gain
  (so a large share is scaffold + compute + token-presence); evidence that coherent reasoning
  *order* adds more is weak/budget-dependent. Numbers independently recomputed from raw data + audited.

## Interpretation

For a 4B on basic code, native thinking buys real deployable capability the corpus forfeited by
construction, and it reframes C2 (here the deployable line moves, not just coverage). But the
shuffle control shows much of the "thinking" benefit is compute/scaffold rather than coherent
reasoning. The practical knob is a thinking *budget* with a real overthinking cost — motivating a
learned budget controller and a stronger content control (program backlog).

## Knowledgebase Update

- Program evidence updated: yes (`research_programs/test_time_reasoning_budget/evidence.md`).
- Program backlog updated: yes (controller, distillation, silent-executor budget sweep, stronger content control, thinking-as-verifier).
- Claim ledger updated: yes (C9, see `knowledge/claims/`).

## Artifacts

- `src/` runtime (model + budget forcing), tasks (MBPP + sandbox verifier), metrics.
- `scripts/` run (sweep), verify_runs (torch-free verification), launch_main, status.
- `configs/` default sweep config.
- `runs/` per-run `summary.json`, `generations.jsonl`, `verified.jsonl` (raw small artifacts).
- `analysis/` `analyze.py`, generated tables and figures.
- `reports/` final report + `artifact_manifest.yaml`.
- Model weights are external (HF cache), not committed; see `reports/artifact_manifest.yaml`.
