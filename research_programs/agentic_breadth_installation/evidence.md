# Evidence

## Seed Experiments

- Experiment: `qwen35_4b_gauntlet_breadth_round1` — gym built (12 families,
  10 trained + 2 held out), two fast training rounds run, first
  menagerie-arbitrated install in the corpus.

## Confirmed Claims

- Claim: C49 (Confirmed) — vLLM 0.24 runtime LoRA is a silent no-op for
  Qwen3.5-4B PEFT adapters; on-vs-off behavioral gate required; deploy via
  merged composite checkpoints or the HF backend.
- Claim: C50 (Promising) — breadth-first expert iteration with
  emission-seam-weighted loss installs substrate-general agentic competence:
  menagerie quick +0.223/+0.294 on two fresh paired seeds (HF backend,
  deterministic), gym +0.518 including never-trained held-out families.

## Negative Findings

- Finding: full-weight SFT on the model's own naturally-closed verified
  chains (round 1) installs nothing measurable — near-self-distillation; the
  deployment-critical post-force-close state must be in-distribution and the
  gradient concentrated on the answer/action emission seam.
- Finding: stallwright (bounded optimization) is unharvestable at round 1 —
  the base model never concludes its optimization deliberation even at a
  4096-token think budget; the axis moved only by transfer (+0.395 gym) and
  its menagerie analogue (stockade) did not move.

- Claim: C53 (Promising) — the second wall: nine paired events across six
  escalation levers land the treated model at 0.375-0.447 aggregate; the
  emission-policy install is a one-time step and no self-verified-output
  recipe variant moves the band; a single rich family installs nearly the
  full effect (breadth is an axis-aligned increment, not the mechanism).

## Current Read

Breadth + strict verifiers + emission-seam supervision is the first recipe in
the corpus to move the blackbox instrument, and the locality laws
(C43/C45/C48) do not extend to it. The binding deployed constraint at
current difficulty is the truncation cascade (consume budget → force-close →
verbose restart → no parseable answer); repairing commit-from-partial-
reasoning transfers across substrates the model never trained on. Trust only
paired same-backend menagerie comparisons; never a vLLM adapter arm (C49).
Next: the residual is a capability core — scaffold-distillation of
tool-found solutions, on-policy RL at residual failures, and
failure-forensics curricula are the queued beyond-recipe mechanisms
(see C53 next tests). Same-recipe scaling is closed.
