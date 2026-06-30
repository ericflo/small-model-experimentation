# Qwen3.5-4B Thinking-Budget Scaling — Report

## Summary

The imported 155-experiment corpus runs Qwen3.5-4B exclusively in **no-think** mode
(`enable_thinking=False` ×48, `True` ×0 besides one fixed-768-token CoT foil) and treats
"budget" as evidence/probe/sample budget, never the model's **native reasoning-token budget**.
We turn thinking on and sweep the thinking budget on MBPP, decomposing the **deployable line**
(greedy / visible-test-selected pass@1) from the **oracle ceiling** (pass@8). Three findings:
(1) native thinking is a large **deployable** win the corpus forfeited — greedy pass@1
**0.76 → 0.91 (+15pp)**, with the deployable gain *exceeding* the oracle-ceiling gain and
*closing* the selection gap (opposite of our C2-based prior); (2) **more thinking is not
monotonically better** — accuracy rises to a broad optimum around 512–1024 tokens and then
*declines*, so the naive `unbudgeted` default (greedy 0.84) is worse than a capped budget
(0.91), a rise-then-fall shape that holds across both greedy and sampled pass@1; (3) a
**shuffled-thinking control** reproduces much of the gain — scrambling the model's own thinking
tokens still recovers most of the lift — so a large share of the benefit is "scaffold + extra
compute + relevant-token-presence," and the evidence that *coherent reasoning order* adds beyond
that is weak and budget-dependent.

## Research Program Fit

First experiment of `test_time_reasoning_budget`. It establishes the reasoning-token budget as
a real, controllable test-time-compute axis (a verified corpus-wide blind spot) and motivates
the program's controller / distillation / thinking-as-verifier lines. It also pressure-tests the
corpus's "silent latent compute beats CoT" framing (`qwen_python_shaped_silent_executor`) and its
central selection bottleneck (C2).

## Method

- **Model:** Qwen/Qwen3.5-4B (the repo standard; `Qwen3_5ForCausalLM`, bf16, sdpa), frozen.
- **Benchmark:** MBPP *sanitized*, `test` split (held out), **first 100 tasks**. Each task =
  NL spec → one Python function, checked against its 3–5 `assert`s (median 3).
- **Protocol:** prompt = NL spec + one example assert (signature anchor); a candidate passes iff
  it executes and satisfies the **full** `test_list` in a sandboxed subprocess (fork, CPU/AS
  rlimits, 10s timeout + one retry to remove load-induced jitter).
- **Independent variable — thinking budget (s1-style budget forcing on `</think>`=248069):**
  `no_think`, `think_{256,512,1024,2048}` (cap thinking at B tokens; if `</think>` is not emitted,
  inject it and regenerate the answer), `think_unbudgeted` (cap 4096, no forcing).
- **Decoding:** Qwen presets — thinking temp 0.6/top_p 0.95/top_k 20; no-think 0.7/0.8/20.
  Greedy (deployable pass@1) + k=8 sampled candidates (oracle pass@8).
- **Controls:** **shuffled-thinking** at 512 and 2048 — permute the model's own thinking tokens
  before forcing the answer (same token count, same multiset, same scaffold; coherent *order*
  destroyed). Isolates reasoning *content* from compute + scaffold + token-presence.

## Results

### Main sweep (n=100, k=8)

| budget | think tok | greedy@1 | sampled pass@1 | visible-sel@1 | pass@8 (oracle) | forced-close |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_think | 0 | 0.760 | 0.776 | 0.840 | 0.910 | 0.00 |
| think_256 | 246 | 0.870 | 0.833 | 0.880 | 0.930 | 0.85 |
| think_512 | 408 | 0.870 | 0.858 | 0.890 | 0.940 | 0.42 |
| **think_1024** | 530 | **0.910** | 0.879 | 0.910 | 0.960 | 0.13 |
| think_2048 | 629 | 0.860 | 0.864 | 0.920 | 0.960 | 0.07 |
| think_unbudgeted | 572 | 0.840 | 0.864 | 0.920 | 0.950 | 0.00 |

Figures: `analysis/main_scaling_curve.png`, `analysis/main_accuracy_vs_compute.png`,
`analysis/main_passk_vs_k.png`.

### Finding 1 — thinking is a deployable win, not just a ceiling win

Greedy deployable pass@1 rises **0.76 → 0.91 (+15pp)**; visible-selector +8pp. The oracle ceiling
rises only **+5pp** (0.91 → 0.96 — little headroom). So thinking **closes** the oracle−deployable
gap (0.07 → 0.03–0.05), the opposite of our prior (from C2 we expected thinking to inflate
coverage but not deployment). Robust on a *paired* basis: at think_1024 vs no_think, 17 tasks flip
fail→pass and only 2 pass→fail (McNemar on 17/2 discordant pairs → p≈0.001).

### Finding 2 — more thinking is not monotonically better (a broad optimum, then decline)

Deployable greedy rises to a broad optimum around 512–1024 tokens (think_1024 0.910 is the
single best cell) and then *declines* at think_2048 (0.860) and think_unbudgeted (0.840).
Sampled pass@1 traces the same shape (0.776 → 0.833 → 0.858 → **0.879** → 0.864 → 0.864, peaking
at 1024). Forced-close fraction (0.85 → 0.42 → 0.13 → 0.07) shows the model "wants" ~500–1000
thinking tokens; forcing below costs accuracy, but letting it run free (`unbudgeted`) is **worse**
than a cap. The oracle ceiling plateaus at ~0.96. Caveat: gaps *among* the thinking budgets are
2–7pp at n=100 single-seed, so we do **not** pin the optimum to exactly 1024 — but the
rise-then-fall *shape* is corroborated across two metrics (greedy and pass@1), and the actionable
lesson is robust: **cap thinking; do not deploy `unbudgeted`.**

### Finding 3 — thinking's gain scales with task difficulty

Greedy, sliced by no-think oracle pass@8, comparing no_think to a **fixed** think_1024 budget
(using a fixed budget rather than a per-bucket argmax avoids selection bias; think_1024 happens
to be the best or tied cell in every bucket):

| no_think difficulty | n | no_think greedy | think_1024 greedy | Δ |
| --- | ---: | ---: | ---: | ---: |
| already easy (8/8) | 51 | 0.941 | 1.000 | +0.06 |
| middling (1–7/8) | 40 | 0.700 | 0.925 | +0.23 |
| never solved (0/8) | 9 | 0.000 | 0.333 | +0.33 |

The benefit grows with difficulty. The robust part is the **middling** bucket (n=40): +22pp. The
**never-solved** bucket is only **9 tasks** (3 of 9 become greedy-solvable), so treat +0.33 as
suggestive — but it does indicate thinking helps the model solve tasks it otherwise fails on
every sample, i.e. some genuine capability, not only re-selection.

## Controls

Shuffled-thinking (content vs compute), greedy@1:

| budget | no_think | shuffled thinking | real thinking |
| --- | ---: | ---: | ---: |
| 512 | 0.760 | 0.800 | 0.870 |
| 2048 | 0.760 | 0.860 | 0.860 |

The striking result is how **much of the gain scrambled thinking reproduces**. At 512, shuffling
the model's own thinking tokens still recovers 0.76 → 0.80 of the 0.76 → 0.87 greedy lift (~⅓ of
the greedy gain from scaffold + compute + relevant-token-presence alone); at 2048, shuffled
*equals* real (0.86 = 0.86) and even slightly exceeds it on sampled pass@1 (0.866 vs 0.864). So a
large share of the "thinking" benefit for this small model on this task is **not** coherent
reasoning — it is the extra forward compute, the `<think>` scaffold, and having relevant tokens in
context. The evidence that coherent reasoning *order* adds beyond that is **weak and
budget-dependent**: a ~7pp greedy edge at 512 (within single-seed noise; and only ~1pp on the more
robust sampled pass@1) that disappears by 2048. This is the report's most provocative and most
caveated result; a stronger control (substitute a *different* task's thinking, destroying
token-presence too) is needed to settle how much is genuine reasoning.

## Oracle Versus Deployable Evidence

Deployable (visible-only): greedy@1 and visible-selector@1 — these carry the headline. Oracle
(non-deployable, uses hidden test outcomes): pass@8 — reported only as a ceiling. Unusually for
this corpus, the *deployable* metric moved more than the oracle ceiling, and the gap *narrowed*
with thinking. No hidden-label leakage into deployable numbers.

## Interpretation

For a 4B on basic code, native thinking buys real **deployable** capability the corpus discarded
by construction. It also reframes C2 (coverage ≫ deployable selection): for the thinking axis on
MBPP, C2 does *not* hold — thinking moves the deployable line and closes the gap. The practical
knob is a budget with a clear optimum (~1024) and a real overthinking cost, directly motivating a
learned thinking-budget controller.

### Limitations

- One benchmark (code), one difficulty band ("basic" Python), n=100, one model, single seed/cell.
  Differences ≲7pp are within unpaired noise; the large effects and paired flip counts are robust,
  the exact optimum (1024) is suggestive.
- The shuffle control destroys token *order* but not *presence* (same multiset incl. any partial
  answer fragments), so it **understates** the content contribution; substituting a different
  task's thinking is the stronger control.
- Throughput: Qwen3.5-4B's linear-attention ran in the slow torch fallback (fast path needs
  `causal-conv1d`, buildable with the on-box CUDA 13.2 toolkit; deferred). Cost was generation
  time, not GPU memory.

## Next Experiments

- Thinking-budget sweep on `qwen_python_shaped_silent_executor` (its CoT collapsed to 0% at len-24
  at a fixed 768 budget — does ≥1024 thinking rescue it?).
- Learned STOP/MORE controller over the thinking budget vs the fixed ~1024 optimum.
- Stronger content control: substitute a different task's thinking (destroy token-presence too).
- Harder substrates (full MBPP, LiveCodeBench, math): does the optimum move, does C2 reappear?
- Thinking-as-verifier vs thinking-as-generator under matched token budget.

## Artifact Manifest

See `artifact_manifest.yaml`. Model weights are external (HF cache, ~9.3 GB); small run artifacts
(`runs/*/summary.json`, `generations.jsonl`, `verified.jsonl`) are kept in-repo.

## Reproducibility

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run.py --tasks 100 --k 8 \
  --budgets no_think,256,512,1024,2048,unbudgeted --out runs/main
.venv/bin/python scripts/run.py --tasks 100 --k 8 --only-controls --out runs/controls
.venv/bin/python analysis/analyze.py --tag main
.venv/bin/python analysis/analyze.py --tag controls
.venv/bin/python analysis/deeper_analysis.py --tag main
```

Env: torch 2.12.1+cu130, transformers 5.12.1 (native `qwen3_5`), single RTX 4090 (24 GB), WSL.

## Refinement (added after the foreign-thinking control)

This report's "much of the gain is scaffold + compute, not coherent reasoning" (Finding 3 /
shuffle control) was later found **overstated**. A foreign-task-thinking ladder
([`qwen35_4b_thinking_content_vs_compute`](../../qwen35_4b_thinking_content_vs_compute/reports/report.md))
showed the model *uses thinking as content* (splicing a different task's thinking collapses accuracy
to ~4% — it solves the wrong problem), that scrambled thinking ≈ no-think on **sampled** full-pass
(the "shuffle recovers ~⅓" here was a **greedy-metric** artifact), and that coherent thinking adds
+12pp at the 512 budget. So at the efficient budget the behavioral gain **is** coherent reasoning;
the compute/scaffold reading holds mainly at high budgets (the 2048 shuffle ≈ real overthinking
result above) and at the representational level. See claim C9 for the corrected statement.
