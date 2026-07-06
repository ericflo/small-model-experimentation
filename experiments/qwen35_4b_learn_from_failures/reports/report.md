# Learn from your own failures: DPO collapses the model; the coverage→deployable gap closes with MORE SFT, not preference training

## Summary

Across the whole arc, SFT-on-positives (banking) raises depth-3 coverage@16 (~0.3–0.5) but not deployable
greedy@1 (stuck 0.05–0.19). Does **preference/contrastive training on the model's OWN (correct, incorrect)
samples** — learning from its failures — raise greedy@1? Design hardened by an adversarial review (which noted
this **extends prior MBPP DPO work** here — `constrained_coverage_dpo`, `offline_hard_negative_coverage_dpo` —
which found constrained DPO preserved pass@1 but didn't beat sample-more on coverage; and demanded the compute
control + the load-bearing shuffled control + early-stopping).

## Result (no-think depth-3, frozen held-out, n=80)
| arm | greedy@1 | cov@16 |
|---|---|---|
| base | 0.000 | 0.000 |
| SFT (banking, 3 epochs) | 0.037 | 0.113 |
| **SFT_2x (6 epochs, compute control)** | **0.113** | **0.212** |
| DPO learn-from-failures (0.25 ep) | 0.050 | 0.113 |
| DPO (0.5 ep) | 0.000 | 0.075 |
| DPO (3 ep) | 0.013 | 0.013 |
| DPO-shuffled (loss-shape control) | 0.037 | 0.062 |

Harvested 174 (chosen=verified-correct, rejected=verified-wrong) same-task pairs from banked_1280's own no-think
samples; chosen/rejected identical median length (151 chars — no length heuristic).

- **The model is a strong latent verifier of its own samples: pre-DPO 2AFC = 0.810** (the SFT model assigns
  higher logp to its correct sample than its wrong one 81% of the time; matches C13's ~0.73).
- **But preference-optimizing that discrimination COLLAPSES generation.** DPO greedy@1 bumps to 0.050 at 0.25
  epochs (within noise of SFT's 0.037; coverage flat = within-support), then craters: 0.000 by 0.5 ep, 0.013
  by 3 ep — both greedy@1 AND coverage@16 crash. Classic DPO over-optimization (the margin logp_c−logp_r blew
  to ~61 unchecked). It never beats SFT_2x at any point.
- **The effective lever is just MORE SFT:** SFT_2x (6 vs 3 epochs) triples greedy@1 (0.037 → 0.113) and doubles
  coverage (0.113 → 0.212). The "gap" was partly UNDERTRAINING.
- **The shuffled control (0.037) confirms it's not pure loss-shape** — and real DPO (0.013) is even *worse*, so
  the same-task correct-vs-wrong signal made the collapse worse, not better.

## Implication

You cannot close the coverage→deployable gap by teaching the model to PREFER its correct over its wrong samples
(DPO) — that destroys its generation. Just train longer on the correct samples (SFT). The model's strong latent
sample-discrimination (2AFC 0.81) is a **"read-only" verifier ability that does not transfer to a "write"
(generation) improvement via preference training.** This extends the prior MBPP DPO finding (DPO didn't beat
sample-more) to the controlled depth-3 substrate, adding: DPO is *fragile/collapses*, and the deployable gap is
best closed by more SFT-on-positives.

## Honest limits

My DPO recipe (cached-ref-SFT DPO, β=0.05, NLL anchor 0.05, LR 2e-5, no dev-early-stop) was not heavily
constrained; the prior MBPP run used ~10 steps + heavier anchoring and avoided collapse. So a more carefully
constrained DPO might not collapse — but across the tested range (0.25–3 epochs) DPO never beat SFT_2x, and
collapsed by 0.5 epochs, so the negative is fairly robust for "does preference-on-failures beat more-SFT."
Single seed, n=80.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Adapters (~180MB each) moved out of repo.
