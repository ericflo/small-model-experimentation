# Qwen3.5-4B Neurosymbolic REPL — Experiment Log

## Design

Mission: unearth latent capability in the FIXED Qwen3.5-4B weights via non-stochastic execution feedback
(an interpreter is a calculator, not a bigger model). Fresh procedurally-generated, contamination-free
substrate (compositional list transforms, PBE, graded by held-out execution; difficulty = depth).

- M1 (`run_baseline.py`): failure profile. Substrate solvable (oracle 100%) but hard; coverage->deployment
  headroom at depths 1-4 (depths 5-6 dead: pass@6=0).
- M2 (`run_repl.py`): REPL loop (draft -> execute visible -> real feedback -> refine, sampled turns) vs
  matched-compute sample-more + visible-select, with a paired no-feedback control. Matched-compute
  accuracy-vs-generations analysis. Central question: does execution feedback beat independent sampling.
- M3 (`collect_solutions.py` + `train_lora.py` + `eval_lora.py`): bank the 4B's OWN verified solutions via
  QLoRA-SFT (no teacher), test single-shot generalization on held-out FRESH tasks + depth extrapolation.

## Env / decisions

- Installed **peft 0.19.1** into the .venv (was missing; needed for M3 LoRA). bitsandbytes 0.49.2 already present.
- Trainer uses `AutoModelForCausalLM` (NOT the corpus's `AutoModelForImageTextToText`) to MATCH the eval
  runtime (`gen_lib` loads CausalLM), so the LoRA adapter's module paths are compatible frozen<->trained.
- M2 uses budget 512 (short synthesis tasks; ~halves thinking cost vs 1024) and SAMPLED REPL turns (greedy
  per-turn got stuck in the smoke -- 0/5 fixed; sampling gives exploration and a fair match to sample-more).
- REPL no-feedback control is PAIRED (reuses repl_real's turn-0 draft), isolating the feedback *content*.

## M1 results (see runs/baseline.json)

Depths 1-6, 15/depth, thinking on. greedy@1 / pass@6 / oracle by depth: d1 .667/.933/1.0, d2 .133/.333/1.0,
d3 .133/.133/1.0, d4 .0/.067/1.0, d5 .0/.0/1.0, d6 .0/.0/1.0. Overall greedy .156, pass@6 .244
(coverage->deployment gap +.089). Substrate is hard-but-fair with real headroom at depths 1-4.

## M2 results (see runs/repl_summary.json, analysis/repl_vs_samplemore.png)

80 tasks (depths 1-4, 20/depth), turns 5, budget 512, seed 101. Hidden accuracy:
greedy@1 0.263 (1 gen), repl_real 0.287 (3.9 gens), repl_nofb 0.263 (4.0 gens), sample_more 0.338 (5 gens).
sample_more selected curve: n1 .20, n2 .237, n3 .263, n4 .325, n5 .338 (== oracle curve: visible-select is
perfect here -- 6 visible I/O examples is a near-perfect correctness signal, no C2 false-passes).

**NEGATIVE for the loop, cleanly controlled:** (1) execution-grounded self-correction does NOT beat
matched-compute independent sampling -- repl_real 0.287 @ 3.9 gens sits on/below the sample_more curve;
sample_more reaches 0.338. (2) The execution-feedback CONTENT adds only +0.024 over the paired no-feedback
control (repl_real 0.287 vs repl_nofb 0.263), within noise. (3) By depth: gains are at d1-d2 (d2: greedy .35
-> repl_real .45 -> sample_more .50); d3 is 0.0 for ALL arms (unsolvable by this 4B, feedback or not); d4
~.05-.10. So for this 4B on a clean substrate, the capability ceiling is its own sampling distribution and
execution feedback does not push past it -- replicating the corpus's "sample-more is hard to beat" theme on
contamination-free data.

## M3 results (see runs/eval_frozen*.json vs runs/eval_trained*.json)

Collected 189 verified (prompt->code) pairs from 146/450 fresh tasks (depths 1-3, seed 202; d1 104/150,
d2 29/150, d3 13/150). QLoRA-SFT r32/a64, 2 epochs, no teacher (train_loss 0.113). Held-out fresh eval
(seed 303 n=75 depths 1-5, seed 404 n=135 depths 1-3):

POOLED N=210 -- think_greedy@1: frozen 0.224 -> trained 0.319 (+0.095, ~2.2 SE, +42% rel). think_pass@5:
0.310 -> 0.371 (+0.061, no collapse). no-think greedy ~0 both (model still needs thinking; adapter trained
on no-think targets transferred to the thinking path). By depth (confirm greedy): d1 .60->.69, d2 .13->.33,
d3 .07->.11 -- broad, both seeds.

**POSITIVE, confirmed:** self-training on own verified solutions banks capability into deployable single-shot
on held-out fresh tasks, no diversity collapse. Contrast with M2 (test-time feedback negative) and with the
corpus's MBPP verifier_guided_self_improvement (regressed) -- works on contamination-free data. Note: adapter
saved to runs/lora_adapter (~170MB), gitignored + removed before commit (regenerable via scripts/m3_chain.sh).
