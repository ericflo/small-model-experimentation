# Think-block FTPO round 1: outcome-conditioned pivot steering as an agentic install recipe

## Research Program

- Program: `agentic_breadth_installation` (primary; menagerie-arbitrated install recipes),
  cross-cutting `posttraining_and_adaptation` (preference-objective mechanics; C29 prior)
  and `test_time_reasoning_budget` (whose loop-control mandate this experiment's census
  phase informs).
- Program question: does self-training on diverse verifier-gated agentic substrates
  install capability that transfers to the blackbox menagerie instrument? Round 3 of the
  SFT recipe re-saturated (C50); the open frontier is whether a **different-mechanism
  recipe** moves the instrument where same-recipe iteration cannot.
- Prior anchors: C50 (breadth SFT installs once, then re-saturates; binding constraint =
  the truncation cascade — non-repetitive verbose non-convergence), C9 (thinking content
  is load-bearing), C29 (sequence-level DPO on own preference pairs COLLAPSED generation —
  the strongest in-corpus prior against preference training; FTPO's single-position,
  tether-constrained design is the published escape and this experiment adjudicates it),
  C44/C45 (serial-compute law), C48 (trained think-channel adapters can interfere outside
  their substrate — drives the format-transfer slice).

## Question

Can **outcome-conditioned single-token preference training** on the think channel install
deployed agentic capability from the model's own generations? Concretely: sample n=8
verifier-scored think trajectories per task, find the prefix-tree divergence nodes where
sibling branches have a large verified success-rate gap, and FTPO-train exactly those
positions — rejected = the failing branch's next token, chosen = the succeeding branches'
next tokens. This distills Monte-Carlo process supervision into surgical token
preferences: steering thinking toward empirically fruitful continuations, with no gold
reasoning, no teacher, no answer-seam supervision.

**Census finding already banked (zero GPU):** the v1 design's loop-repair premise is
false at deployed budgets — the published fingerprint detector flags **0.08%** of greedy
base gym atoms (1/1200, think@1024) and **0.00%** of episode turns (0/786). Loops
dominate only at 16k+ (81/144 at think@32,768). Loop-FTPO is therefore requeued as a
long-context follow-up; round 1's trained recipe is pivot steering.

## Hypothesis

Think-block trajectories contain a small number of **decision points** where the
next-token choice measurably changes the probability of eventual verified success. The
alternatives at those points sit inside the model's own sampled behavior, so
single-position preference training can move the policy at exactly those points without
disturbing anything else — capability *elicitation by trajectory steering*. The
falsifiable chain (P0–P4) with all gates and constants is frozen in
[`reports/preregistration.md`](reports/preregistration.md) (v2); the adversarial review
that reshaped v1 → v2 is in [`reports/design_review.md`](reports/design_review.md).

Headline predictions: **P0** ≥30% of n=8 sampling groups on learnable-band tasks yield an
eligible divergence node; **P1** pivot lifts greedy success on held-out band tasks by
≥ +0.05 absolute with the shuffled control showing < half that; **P4** paired menagerie
deltas clear a null-calibrated three-seed quick gate (+ conditional medium confirmation).

## Setup

- Model: `Qwen/Qwen3.5-4B` @ pinned revision, thinking mode, two-stage budget protocol,
  vLLM 0.24 for every generation arm (same-backend rule).
- Arms: `base`, `pivot` (primary), `pivot-shuffled` (within-prompt outcome labels
  permuted before mining — the required shuffled-label control), plus a labeled
  NON-DEPLOYABLE base n=8 coverage reference (C2/C5 discipline).
- Elicitation (harvest P): 10 TRAINED gym families only (brinework/spindle preserved as
  held-out controls) ≈60% + list-transform code tasks ≈40%; learnable band calibrated in
  smoke (base greedy success ∈ (0.1, 0.9) per cell); temperature 0.6/top-p 0.95/top-k 20,
  n=8, think@1024 (quick-tier deployment budget), adaptive dose in 800-prompt slices to a
  projected ≥1,200-row pool (5h cap). Closed seed ranges; raw token IDs archived.
- Mining: prefix tree per prompt; nodes at depth ≥16 with ≥2 rollouts per sibling branch
  and success-rate gap ≥ 0.5; ≤2 nodes/prompt; rejected/chosen from observed branch
  tokens only; contexts ≤ 6,144 tokens. Rejected-token flattening 0.3, chosen flattening
  0.5, train ≤ 70% of pool.
- Trainer: standalone FTPO (torch+peft, repo `.venv`): bf16 LoRA r=256 α=128 on
  q/k/v/o/gate/up/down_proj (attention exists on 8/32 layers of this hybrid; MLP on all —
  documented coverage; no lm_head), hinged softplus margin ε=2.0 + two-tier logit-space
  MSE tether (0.4 / 0.05, dead zone 0.5), reference = adapter-disabled weights,
  final-position-only logits (248,320-token vocab — full-sequence logits are forbidden),
  RIGHT-padded batches with last-real-index gather + preregistered padding-equivalence
  gate, gradient checkpointing, lr 1.5e-5, 1 epoch, early stop chosen_win ≥ 0.4.
- Deployment: merged composite checkpoints, **per-arm C49 on-vs-off behavioral gate**.
- Evals: whitebox think-economy + P1 band-task success + format-shifted slice + full
  termination triple (N=500/arm/budget at think@{1024,2048}); gym-internal (all 12
  families, fresh seed); collapse guard (120 code tasks, greedy + pass@8, C29 watch);
  no-think guard (120 atoms); menagerie via the null-calibrated conditional rule in the
  preregistration, fresh seeds union-checked incl. 31337.
- Hidden-label boundary: verifier outcomes label whole rollouts (success/failure) for
  mining only; no gold content enters any prompt or target; menagerie via run.py CLI +
  aggregate scores only.

## Run

Smoke (CPU selftests + config-freeze check, then tiny GPU path):

```bash
python3 scripts/run.py --smoke                              # selftests + config==prereg
../../.venv-vllm/bin/python scripts/band_calibrate.py       # learnable-band table (~20 min)
../../.venv-vllm/bin/python scripts/harvest.py --smoke      # 80 prompts x n=8
```

Full pipeline (single-tenant GPU, staged):

```bash
../../.venv-vllm/bin/python scripts/harvest.py               # adaptive slices + census
python3 scripts/build_rows.py --arm pivot                    # CPU mining + regularize
python3 scripts/build_rows.py --arm shuffled
../../.venv/bin/python scripts/train_ftpo.py --arm pivot \
    --out ../../large_artifacts/qwen35_4b_think_ftpo_round1/adapters/pivot
../../.venv/bin/python scripts/train_ftpo.py --arm shuffled \
    --out ../../large_artifacts/qwen35_4b_think_ftpo_round1/adapters/shuffled
../../.venv/bin/python scripts/merge_ftpo.py --adapter <adapters/ARM> --out <merged/ARM>
../../.venv-vllm/bin/python scripts/eval_whitebox.py --arms base pivot shuffled
../../.venv-vllm/bin/python scripts/eval_gym.py --arms base pivot shuffled
python3 scripts/bench.py --tier quick --arms base base      # null calibration first
python3 scripts/bench.py --tier quick --arms base pivot     # x3 fresh seeds
```

## Results

Preregistered verdict: **training-recipe FAILURE (P1 mechanism fail) — no capability
read, no menagerie spend.** Deployable evidence (greedy, held-out seeds, merged bare
checkpoints, same backend/geometry per comparison):

- **Census:** loops are absent at deployed budgets (0.08% of greedy atoms, 0/786
  episode turns) — the v1 loop-repair premise is dead below 16k and was descoped
  pre-training. Pivot census PASSED: 25.6% of n=16 groups yield an eligible divergence
  node, 49.5% outcome-mixed; 879-row pool → 615 training rows from 2,800 prompts.
- **P1 FAIL:** pivot arm's greedy success on held-out band tasks −0.039 (think@1024)
  and −0.076 (think@2048) vs base — the bar was +0.05. Natural-close rate halves at
  2048 (0.167 → 0.076); answer-limit contacts +6pp.
- **The shuffled-label control degrades nearly identically** (−0.022/−0.055; identical
  natural-close 0.076): the damage is generic to the training regime, not the
  outcome-conditioned signal.
- **Guards:** C29 collapse guard CLEAN (greedy +14% rel, pass@8 −7.7% rel on the code
  substrate — the tether prevents sequence-level collapse); no-think guard CLEAN
  (0.367 → 0.408); gym-internal guard FAIL for pivot (0.517 → 0.484 vs shuffled 0.514).
- Oracle reference (NON-DEPLOYABLE): base best-of-8 coverage 0.69 vs greedy 0.50 on the
  same tasks — the headroom the signal did not convert.
- Full tables: `analysis/headline.md`, `reports/report.md`.

## Interpretation

**The attractor precondition:** published FTPO successes reject confident outlier
tokens (loop initiators, lexical attractors — argmax-dominant); demoting an extreme is
a small local edit. Pivot rows reject near-parity tokens (initial chosen_win ≈ 0.43),
so the ε=2.0 margin objective must manufacture separations at ~600 scattered
early-think positions — producing think-flow collateral (delayed closure, more budget
exhaustion) that permuted labels reproduce exactly. More likely now: single-position
preference training is safe/effective only against confident distributional outliers
(C29 hardened one level down). Less likely: naive re-dosing flips the sign. Still
unknown: whether filtering to "confident wrong turns" (failing branch's token also
locally dominant) restores both the precondition and the benefit — round 2's sharpest
lever — and what the 879-row UNDERDOSED caveat hides.

## Knowledgebase Update

- Program evidence updated: `research_programs/agentic_breadth_installation/evidence.md`
- Program backlog updated: `research_programs/agentic_breadth_installation/backlog.md`
  (+ `test_time_reasoning_budget` backlog: long-context loop-FTPO follow-up)
- Claim ledger updated: C52 (Negative — single-position preference training requires
  the rejected-token attractor precondition; think-flow collateral otherwise)

## Artifacts

- `src/` — pivot miner, loop detector (census), gym + harness + runner (copied,
  self-contained), code-task generator + sandbox
- `scripts/` — staged pipeline (run --smoke, band_calibrate, harvest, build_rows,
  train_ftpo, merge_ftpo, eval_whitebox, eval_gym, bench)
- `configs/default.yaml` — every preregistered constant (asserted by --smoke)
- `runs/` — census, band table, yields, eval tables, menagerie event log (aggregates
  only)
- `reports/` — preregistration.md (v2), design_review.md, report.md,
  artifact_manifest.yaml (adapters + merged checkpoints external under
  `large_artifacts/`)
