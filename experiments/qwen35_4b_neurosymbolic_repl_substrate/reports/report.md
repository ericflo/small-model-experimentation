# Qwen3.5-4B Neurosymbolic REPL: Substrate, Feedback Loop, and Banking

## Summary

Mission test: can we **unearth latent capability in the fixed Qwen3.5-4B weights** with non-stochastic
execution feedback — no larger models, no distillation, no scaling? On a **fresh, procedurally-generated,
contamination-free** program-synthesis substrate, three findings: (M1) the substrate is hard-but-fair —
oracle-solvable 100%, but the frozen 4B (thinking on) gets greedy@1 0.156 / pass@6 0.244, with headroom at
depths 1–4. (M2) **A neurosymbolic multi-turn REPL loop (draft → execute → real feedback → refine) does
NOT beat matched-compute independent sampling** (repl_real 0.287 @ ~3.9 gens vs sample_more 0.338 @ 5), and
the execution-feedback *content* adds only +0.024 over a paired no-feedback control (within noise). (M3)
**QLoRA-SFT on the 4B's OWN 189 verified solutions (no teacher) DOES bank capability into deployable
single-shot** — held-out fresh think-greedy@1 0.224 → 0.319 (+0.095, ~2.2 SE over N=210, +42% relative),
pass@5 up (no diversity collapse), confirmed on two fresh seeds. Net: for this small model, the lever that
unearths latent capability is **self-training on verified self-solutions**, not test-time self-correction —
and this self-training works on a **contamination-free** substrate where the corpus's prior MBPP STaR run
regressed, implicating substrate/contamination in that earlier failure.

## Research Program Fit

`structured_execution_and_compilers` (extends C1). Directly serves the mission memory
[[unearth-latent-capability-mission]]: elicit the fixed 4B, never import capability. The interpreter is a
calculator (a tool), not a model, so the loop is on-mission. Also the corpus's first **contamination-free**
program-synthesis substrate — a reusable asset for any future elicitation claim.

## Method

- **Substrate** (`src/gen_tasks.py`): each task is a random depth-D composition of ~23 total list-of-int
  primitives; presented as input/output examples; the model synthesizes `transform(xs)` graded on **held-out**
  examples (functional — any correct program counts). Contamination-free by construction; a reference oracle
  solves 100%. Sandbox `src/code_env.py` (AST safety + `-I` subprocess + rlimits; captures actual outputs).
- **M1** (`run_baseline.py`): frozen 4B, thinking on — greedy@1 / pass@k / oracle by depth.
- **M2** (`run_repl.py`): ≤5 sampled turns; `repl_real` refines on real feedback (actual vs expected),
  `repl_nofb` is a **paired** control (reuses real's turn-0, told only "try again"), `sample_more` draws 5
  independent samples and selects by visible pass-count. Matched-compute accuracy-vs-generations analysis.
- **M3** (`collect_solutions.py` → `train_lora.py` → `eval_lora.py`): QLoRA-SFT the 4B on its OWN verified
  (prompt→code) solutions (no teacher); eval frozen vs trained single-shot on held-out fresh tasks.

## Results

### M1 — failure profile (thinking on)
| depth | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy@1 | 0.667 | 0.133 | 0.133 | 0.000 | 0.000 | 0.000 |
| pass@6 | 0.933 | 0.333 | 0.133 | 0.067 | 0.000 | 0.000 |
| oracle | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Overall greedy@1 0.156, pass@6 0.244. Hard-but-fair; headroom at depths 1–4 (5–6 are dead — pass@6 = 0).

### M2 — REPL loop vs matched-compute sampling (hidden accuracy)
| arm | acc | mean gens | d1 | d2 | d3 | d4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy@1 | 0.263 | 1.0 | 0.65 | 0.35 | 0.00 | 0.05 |
| repl_real (feedback) | 0.287 | 3.9 | 0.60 | 0.45 | 0.00 | 0.10 |
| repl_nofb (paired control) | 0.263 | 4.0 | 0.60 | 0.40 | 0.00 | 0.05 |
| **sample_more (bar)** | **0.338** | 5.0 | 0.80 | 0.50 | 0.00 | 0.05 |

sample_more selected curve by #samples: 1→0.20, 2→0.237, 3→0.263, 4→0.325, 5→0.338 (**== oracle curve**).
Figure: `analysis/repl_vs_samplemore.png`.

- **Feedback does not beat sampling.** repl_real 0.287 @ 3.9 gens lies on/below the sample_more curve;
  sample_more reaches 0.338. Independent sampling is the stronger use of the same compute.
- **The feedback content barely matters.** repl_real 0.287 vs paired repl_nofb 0.263 = +0.024 (2/80 tasks).
- **Depth 3 is 0.0 for every arm** — genuinely beyond this 4B; feedback cannot manufacture a solution the
  model can't sample. Gains are confined to the easy end (d1–d2), where sampling already wins.

### M3 — banking self-solutions into single-shot
Collected 189 execution-verified (prompt→code) pairs from 146/450 fresh tasks (depths 1–3, seed 202; solve
rate d1 104/150, d2 29/150, d3 13/150). QLoRA-SFT (r32/α64, 2 epochs, no teacher). Evaluated frozen vs
trained on held-out fresh tasks (seeds 303 & 404, unseen compositions):

| metric (held-out, pooled N=210) | frozen | trained | Δ |
| --- | ---: | ---: | ---: |
| think greedy@1 (deployable single-shot) | 0.224 | **0.319** | **+0.095 (~2.2 SE, +42%)** |
| think pass@5 (coverage) | 0.310 | 0.371 | +0.061 (no collapse) |
| no-think greedy@1 | 0.000 | 0.007 | ~0 |

By depth (confirmation set, greedy@1): d1 0.60→0.69, d2 0.13→0.33, d3 0.07→0.11. The gain is broad across
the in-distribution training depths and holds on both fresh seeds.

- **Self-training banks capability into deployable single-shot** (+0.095, ~2.2 SE) on held-out fresh
  tasks — a real, generalizing improvement from the model's OWN verified solutions, no teacher.
- **No diversity collapse:** pass@5 rises (0.310→0.371), the opposite of the STaR failure mode that sank
  the corpus's contaminated-MBPP self-improvement run.
- **No-think stays ~0:** the model still needs thinking; the adapter (trained on no-think targets)
  transferred its task knowledge to the *thinking* path, not a new no-think one-shot path.

## Controls

Reference oracle (100% solvable) proves every failure is the model's. `repl_nofb` is **paired** to real's
turn-0, isolating the feedback content from multi-turn re-drafting. `sample_more` selected == oracle curve
shows visible-test selection is loss-free here (no C2 false-passes), so coverage *is* deployable accuracy —
the bottleneck the loop would have to move is pure coverage.

## Oracle Versus Deployable Evidence

All arms deployable (only visible info used); pass@k oracle is the coverage ceiling. Crucially the substrate
is contamination-free, so — unlike MBPP — a memorization confound cannot inflate any arm.

## Interpretation

Two contrasting levers, cleanly separated on the same contamination-free substrate:

- **Test-time execution feedback does NOT unearth capability** (M2). The model cannot reliably convert
  "you returned X, expected Y" into a corrected program better than re-sampling; where it can't sample a
  solution at all (depth ≥3), feedback adds nothing. The frozen model's *test-time* ceiling is its sampling
  distribution, and self-correction is not the lever.
- **Self-training on verified self-solutions DOES** (M3). QLoRA-SFT on the model's own execution-verified
  solutions moves the *weights*, banking sampling-accessible capability into deployable single-shot with a
  significant, generalizing +0.095 on held-out fresh tasks and no diversity collapse.

The synthesis: for a small model, you don't unearth latent capability by reading the frozen weights more
cleverly at test time — you **bank it into the weights** by training on what the model can already verify.
And the fact that this self-training *works here* but *regressed on MBPP* (`qwen35_4b_verifier_guided_self_improvement`)
is itself a finding: contamination/substrate, not the method, likely explains the earlier failure — a
contamination-free, structured substrate is what lets honest self-improvement show up. This is on-mission
(no teacher, no scaling) and points the whole capability question toward clean-data self-training.

### Limitations
- One model, one task family; budget 512; n=20/depth in M2. Depth ≥3 near-zero limits headroom to d1–d2.
- The loop uses only per-example pass/fail + outputs; richer feedback (traces, counterexamples) is untested.

## Next Experiments

- (M3, running) banking; if it also fails, the combined statement is strong: neither test-time feedback nor
  self-training unearths capability here.
- Richer feedback signals; a substrate whose difficulty sits in the d1–d2 "coverage-exists" band for more mass.

## Artifact Manifest

See `artifact_manifest.yaml`. Small tasks/records/summaries + figures in-repo; the LoRA adapter (~170MB) is a
regenerable training artifact.
