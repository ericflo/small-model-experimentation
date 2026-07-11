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

## qwen35_4b_think_ftpo_round1 (2026-07-11, C52 — Negative)

The first different-mechanism recipe after C50's re-saturation: single-position
preference training (FTPO) on outcome-conditioned think-block pivot points
(prefix-tree divergence of n=16 verifier-scored rollouts). Preregistered
mechanism gate FAILED (−0.039/−0.076 vs a +0.05 bar on held-out band tasks);
the shuffled-label control degraded identically, so the harm is the training
regime, not the steering signal. Guards localize the channel: no C29-style
collapse (the two-tier logit tether works), no-think channel clean — the
damage is think-flow convergence (natural close halves at think@2048). Read:
FTPO's safety/efficacy requires the rejected token to be a CONFIDENT OUTLIER
(loop initiators, lexical attractors); near-parity pivot tokens violate the
precondition and the ε-margin objective's collateral dominates. Menagerie was
correctly never exposed (mechanism-gate rule). Census bonus: repetition loops
are ~0.1% at deployed budgets — the loop-FTPO variant belongs to 16k+ only.
That result queued confident-wrong-turn filtering (failing branch's token also
locally dominant); round 2 below has now tested it.

## qwen35_4b_think_ftpo_round2 (2026-07-11, C52 — Low-dose null)

The registered confident-wrong-turn rescue also failed, but isolated the next
bottleneck. A frozen entropy/varentropy selector retained 155 failed-argmax
pivots. Conventional demotion, bounded positive-only uplift, and shuffled
uplift all failed exact-logit locality (mean per-row median non-target drift
0.229/0.145/0.120 logits vs a 0.10 ceiling). Pull-up was materially safer than
push-down and true labels beat shuffled labels on the gym (+6.25pp) and fresh
repository agent (+13.89pp, paired-bootstrap CI touching zero), so the steering
directions contain some signal. They did not elicit breadth: repository hidden-
test pass was base 43/72, uplift 39/72, demote 34/72, shuffled 29/72; fresh
whitebox uplift was +0.26pp at think@1024 and −3.06pp at think@2048. Every
coarse C49/collapse/no-think/gym guard passed, but P1/P2/P3 did not, so
menagerie remained sealed (zero seeds consumed).

Read: confident-outlier geometry is necessary but not sufficient; the active
constraint is context locality of the parameter update. Entropy/varentropy can
route and diagnose pivots, but higher varentropy was not safer (lowest-V
quartile had the cleanest uplift drift). Do not scale this LoRA recipe. Require
a lower-dose or context-gated mechanism to clear P1 before another harvest or
agentic transfer run.
