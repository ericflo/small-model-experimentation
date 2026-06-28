# Latent Fast-Weight Qwen Experiment Log

## Objective

Test whether a frozen Qwen 4B-class model augmented with an invisible recurrent fast-weight hyperadapter learns useful internal computation, operationalized as higher multiple-choice accuracy when given more latent recurrent steps `K`.

## Primary Hypotheses

1. If the adapter learns real internal computation, validation accuracy should improve with recurrent budget: `K=0 < K=1 < K=2 < K=4`, with possible plateau by `K=8`.
2. If fast-weight memory contributes to computation rather than acting as noise, disabling memory should reduce hard-set accuracy or weaken K-scaling.
3. If activation-programmed low-rank transforms contribute distinct programmable operators, disabling the dynamic low-rank bank should reduce accuracy or weaken K-scaling.
4. If the result is a narrow training artifact, improvements will disappear on harder held-out examples with longer operation chains.

## Initial Environment Findings

- Machine: NVIDIA RTX 6000 Ada Generation, about 48 GB VRAM.
- Python: 3.12.3.
- PyTorch: 2.8.0+cu128, CUDA available, bf16 supported.
- Installed missing dependencies: `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece`, `safetensors`, `einops`, `matplotlib`, `pandas`, `seaborn`, `scipy`, `scikit-learn`, `markdown`, `reportlab`.
- Hugging Face API check showed `Qwen/Qwen3.5-4B` and fallback `Qwen/Qwen3-4B` are public and ungated.
- `Qwen/Qwen3.5-4B` loaded successfully with `AutoModelForMultimodalLM` in 4-bit; detected `model.language_model.layers`, 32 layers, hidden size 2560, hook index 28 for `--hook_layer -4`.

## Script Changes Made

- Added unhooked frozen-model multiple-choice baseline evaluation.
- Added step-0 untrained-adapter evaluation.
- Added run metadata with software versions, GPU details, and script SHA-256.
- Added ablations: `--disable_fast_memory` and `--disable_dynamic_lowrank`.
- Added optional hook-mode workspace auxiliary value loss: `--aux_value_loss`.
- Verified the modified hook path with a tiny random Llama smoke test.

## Planned Experimental Sequence

1. Short Qwen pilot: estimate step/eval wall-clock cost and catch runtime failures.
2. Main recurrent adapter run with full fast-weight memory and dynamic low-rank bank.
3. Static/no-recurrence control using `train_k=0`.
4. Fast-memory ablation.
5. Dynamic-low-rank ablation.
6. Analyze K-scaling on normal and harder held-out tasks.
7. Generate figures and write a paper-style report.

## Control Runs

### K=0-Only Training Control

Run: `../runs/control_qwen35_hook_traink0_seed7`

Purpose: train the same prompt-conditioned hook/readout path with no recurrent steps. At evaluation, `K>0` invokes an untrained recurrent core. Any K gains here would weaken the interpretation that recurrent training caused the gains in the full run.

Step 100 eval:

- Validation: `K=0 25.0%`, `K=1 25.0%`, `K=2 24.0%`, `K=4 22.0%`, `K=8 22.0%`
- Hard: `K=0 20.0%`, `K=1 20.0%`, `K=2 18.0%`, `K=4 20.0%`, `K=8 23.0%`

Interpretation: unlike the full run's step-100 validation result, untrained recurrent steps do not create a small-K validation improvement. This mildly supports the recurrent-training interpretation.

Step 200 eval:

- Validation: `K=0 19.0%`, `K=1 19.0%`, `K=2 17.0%`, `K=4 16.0%`, `K=8 21.0%`
- Hard: `K=0 26.0%`, `K=1 25.0%`, `K=2 28.0%`, `K=4 30.0%`, `K=8 28.0%`

Interpretation: the untrained recurrent core can still create small apparent K bumps on 100-example evals, especially on hard. This weakens any interpretation based only on isolated checkpoint/K improvements.

Step 300 final eval:

- Validation: `K=0 22.0%`, `K=1 22.0%`, `K=2 20.0%`, `K=4 17.0%`, `K=8 20.0%`
- Hard: `K=0 19.0%`, `K=1 17.0%`, `K=2 19.0%`, `K=4 16.0%`, `K=8 20.0%`

Interpretation: final K=0-only control is mostly flat or negative with K. The control does not show the same repeated small-K validation gains as the full run, but it did show enough noise at step 200 that small 100-example K bumps cannot be treated as conclusive.

### Auxiliary Value-Loss Run

Run: `../runs/main_qwen35_hook_aux02_seed7`

Purpose: strengthen supervision by adding `0.2 * CE(value mod 97)` from the final workspace, while keeping final evaluation as answer-letter log-likelihood. This tests Claude's critique that letter-only supervision is too low-bandwidth for learning arithmetic-like latent computation.

Step 100 eval:

- Validation: `K=0 18.0%`, `K=1 20.0%`, `K=2 23.0%`, `K=4 19.0%`, `K=8 20.0%`
- Hard: `K=0 22.0%`, `K=1 22.0%`, `K=2 21.0%`, `K=4 18.0%`, `K=8 22.0%`

Auxiliary value loss remains close to random (`log(97) ~= 4.57`) through this checkpoint. Interpretation: the auxiliary head has not yet learned arithmetic, and the eval pattern remains validation-only and unstable.

Step 200 eval:

- Validation: `K=0 20.0%`, `K=1 20.0%`, `K=2 18.0%`, `K=4 17.0%`, `K=8 19.0%`
- Hard: `K=0 28.0%`, `K=1 26.0%`, `K=2 25.0%`, `K=4 26.0%`, `K=8 26.0%`

Auxiliary value loss remains near random. Interpretation: the auxiliary objective is not producing useful recurrent computation under this budget; recurrence is now negative.

Step 300 final eval:

- Validation: `K=0 17.0%`, `K=1 18.0%`, `K=2 18.0%`, `K=4 20.0%`, `K=8 19.0%`
- Hard: `K=0 15.0%`, `K=1 18.0%`, `K=2 20.0%`, `K=4 18.0%`, `K=8 17.0%`

Auxiliary value loss still did not move convincingly below random. Interpretation: the final checkpoint has a small positive K effect, but the auxiliary mechanism failed at its intended numeric-prediction target, so this is not strong evidence for learned arithmetic computation.

Large retest of final auxiliary checkpoint:

Run: `../runs/eval_aux_final_n250`

- 250 validation and 250 hard examples.
- Validation: `K=0 17.6%`, `K=2 15.2%`, `K=4 16.8%`, `K=8 15.6%`
- Hard: `K=0 21.2%`, `K=2 22.8%`, `K=4 20.4%`, `K=8 22.8%`

Interpretation: the validation K effect reversed under larger-sample retesting. The hard-set bump is only +1.6 percentage points and not compelling at this sample size. This reinforces the conclusion that the current implementation does not demonstrate robust latent recurrent K-scaling.

## Pilot Results

- Tiny random Llama hook smoke passed after setting `--hook_layer -1` for the 2-layer toy model.
- Qwen3.5 load-only check passed with 4-bit quantization and hook index 28.
- Qwen3.5 batch-1 pilot passed.
- Qwen3.5 batch-4 pilot passed with `candidate_batch_size=20`; no OOM. Three optimizer steps plus two small eval passes took about 32 seconds including reload overhead. This supports running the main sweep with `batch_size=4`, `grad_accum=1`, `eval_batch_size=4`, and `candidate_batch_size=20`.

## Main Full-Adapter Run: Early Evidence

Run: `../runs/main_qwen35_hook_full_seed7`

- Initial frozen-model baseline over 100 validation and 100 hard examples:
  - Validation: 18.0%
  - Hard length-generalization set: 21.0%
- Initial untrained-adapter eval:
  - Validation: `K=0 19.0%`, `K=1 19.0%`, `K=2 18.0%`, `K=4 18.0%`, `K=8 19.0%`
  - Hard: `K=0 28.0%`, `K=1 26.0%`, `K=2 26.0%`, `K=4 29.0%`, `K=8 29.0%`

Interpretation: the default task is already in the right headroom regime; frozen Qwen3.5 is near 5-way chance. Difficulty tuning remains useful as a robustness check, but not as a prerequisite for observing a signal.

Step 50 eval:

- Validation: `K=0 24.0%`, `K=1 26.0%`, `K=2 27.0%`, `K=4 27.0%`, `K=8 25.0%`
- Hard: `K=0 26.0%`, `K=1 28.0%`, `K=2 28.0%`, `K=4 27.0%`, `K=8 27.0%`

Interpretation: early result shows a small positive K effect around `K=2..4`, but the effect is weak enough that it requires later checkpoints, controls, and binomial uncertainty intervals before making any claim.

Step 100 eval:

- Validation: `K=0 21.0%`, `K=1 23.0%`, `K=2 23.0%`, `K=4 22.0%`, `K=8 22.0%`
- Hard: `K=0 25.0%`, `K=1 24.0%`, `K=2 23.0%`, `K=4 24.0%`, `K=8 23.0%`

Interpretation: the validation K bump persists weakly, but the hard-set length-generalization result is negative. This argues against a strong current-form recurrent-computation claim.

Step 150 eval:

- Validation: `K=0 22.0%`, `K=1 25.0%`, `K=2 28.0%`, `K=4 25.0%`, `K=8 24.0%`
- Hard: `K=0 29.0%`, `K=1 32.0%`, `K=2 31.0%`, `K=4 27.0%`, `K=8 28.0%`

Interpretation: the strongest positive effect so far appears at small recurrent budgets, especially `K=2` on validation and `K=1..2` on hard. This is evidence for a compute-budget interaction, but not for monotonic K-scaling.

Step 200 eval:

- Validation: `K=0 23.0%`, `K=1 27.0%`, `K=2 26.0%`, `K=4 25.0%`, `K=8 27.0%`
- Hard: `K=0 22.0%`, `K=1 27.0%`, `K=2 24.0%`, `K=4 26.0%`, `K=8 26.0%`

Interpretation: another positive `K>0` result, strongest at `K=1`. The effect is inconsistent across K and checkpoints, but recurrent steps have not been purely decorative.

Step 250 eval:

- Validation: `K=0 18.0%`, `K=1 24.0%`, `K=2 21.0%`, `K=4 21.0%`, `K=8 24.0%`
- Hard: `K=0 24.0%`, `K=1 23.0%`, `K=2 23.0%`, `K=4 21.0%`, `K=8 24.0%`

Interpretation: validation retains a positive recurrent-step effect, but hard-set length generalization does not. This weakens the serial-computation interpretation.

Step 300 final eval:

- Validation: `K=0 26.0%`, `K=1 22.0%`, `K=2 21.0%`, `K=4 22.0%`, `K=8 24.0%`
- Hard: `K=0 15.0%`, `K=1 15.0%`, `K=2 15.0%`, `K=4 14.0%`, `K=8 15.0%`

Interpretation: the final checkpoint does not support K-scaling. Since intermediate checkpoints did show small positive effects, the current evidence points to unstable or undertrained recurrent refinement, not a conclusive latent-computation win.

Preserved checkpoints:

- `../../../large_artifacts/qwen_fastweight_hook/checkpoints/main_qwen35_hook_full_seed7/latent_adapter_step200.pt`
- `../../../large_artifacts/qwen_fastweight_hook/checkpoints/main_qwen35_hook_full_seed7/latent_adapter_step250.pt`
- `../../../large_artifacts/qwen_fastweight_hook/checkpoints/main_qwen35_hook_full_seed7/latent_adapter.pt` (step 300)

Large retest of step 200 checkpoint:

Run: `../runs/eval_main_step200_n250`

- 250 validation and 250 hard examples.
- Validation: `K=0 20.8%`, `K=1 19.2%`, `K=2 18.4%`, `K=4 19.2%`, `K=8 20.0%`
- Hard: `K=0 26.0%`, `K=1 24.0%`, `K=2 23.6%`, `K=4 23.6%`, `K=8 24.4%`

Interpretation: the most favorable recurrent checkpoint did not survive larger-sample retesting. `K=0` is best on both splits. This is strong evidence against a robust K-scaling effect in the current implementation.
