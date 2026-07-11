# Think-block FTPO round 1 — outcome-conditioned pivot steering: report

## Summary

**Verdict (preregistered decision rule): training-recipe FAILURE for the pivot arm —
mechanism prediction P1 failed, so no capability read and no blackbox spend.** The round
nevertheless resolves three questions cleanly and localizes exactly why the recipe
fails, banking the strongest available guidance for any future preference-training
round. All constants, gates, and the decision rule were frozen in
[`preregistration.md`](preregistration.md) (v2 + amendments 1–4, every amendment
pre-training and evidence-cited); the adversarial design review that shaped v2 is
[`design_review.md`](design_review.md).

## Research Program Fit

`agentic_breadth_installation` sought a different-mechanism install recipe after C50's
SFT recipe re-saturated. This round tested single-position preference training (FTPO)
driven by outcome-conditioned pivot mining — and rules out its naive form, while the
census phase also closes the loop-repair variant at deployed budgets for
`test_time_reasoning_budget`. For `posttraining_and_adaptation`, the result hardens C29
one level down (see Interpretation).

## Method

Sample n=16 verifier-scored think trajectories per learnable-band task (base greedy
success ∈ (0.1, 0.9); 14/33 calibrated cells across 8 trained gym families + code
depth-2), build a prefix tree per prompt over exact think-token IDs, and at divergence
nodes where ≥2 sibling branches each carry ≥2 rollouts with a verified success-rate gap
≥ 0.5, emit an FTPO row: context = prompt + shared think prefix; rejected = the failing
branch's next token; chosen = the succeeding branches' next tokens. Regularize
(rejected flattening 0.3, chosen flattening 0.5, train ≤70% of pool), train the
published FTPO objective (softplus margin ε=2.0, two-tier logit MSE tether
λ=0.4/0.05 τ=0.5, reference = adapter-disabled weights) as bf16 LoRA r=256/α=128 on the
7 projection modules, 1 epoch lr 1.5e-5, merge into the composite checkpoint, gate
on-vs-off (C49). Control: identical pipeline with per-prompt outcome labels permuted
(seed 3407), row count matched (615). All generation arms share one backend and engine
geometry; whitebox evals N=500/arm/budget on held-out seeds.

## Results

**Census (deployable, zero/low GPU):**

- Repetition loops are absent at deployed budgets: 1/1200 greedy base gym atoms at
  think@1024 (0.08%), 0/786 episode turns, 0.2–0.4% on this round's own evals
  (`runs/census_existing.json`). The corpus's loop pathology (81/144 at think@32,768)
  does not exist two orders of magnitude below its regime — the v1 loop-repair arm was
  descoped on this evidence.
- P0 PASS: 2,800 prompts (7 slices, 7.7h) → 25.6% of groups yield an eligible node
  (gate 15%), 49.5% outcome-mixed (gate 30%), pool 879 → 615 training rows (gate 600).
  Median pivot depth 22 think-tokens; mean gap 0.59; 95% of rows have one chosen token.

**Headline (greedy, held-out band tasks, N=500/arm/budget; `analysis/headline.md`):**

| arm | budget | success | Δ vs base | natural-close | answer-limit |
|---|---|---:|---:|---:|---:|
| base | think@1024 | 0.498 | — | 0.014 | 0.443 |
| pivot | think@1024 | 0.459 | **−0.039** | 0.002 | 0.476 |
| shuffled | think@1024 | 0.476 | −0.022 | 0.002 | 0.480 |
| base | think@2048 | 0.541 | — | 0.167 | 0.420 |
| pivot | think@2048 | 0.465 | **−0.076** | 0.076 | 0.490 |
| shuffled | think@2048 | 0.486 | −0.055 | 0.076 | 0.484 |

- **P1 mechanism: FAIL** (bar +0.05 absolute; measured −0.039/−0.076).
- **P3 / gym guard: FAIL for pivot** — 12-family aggregate 0.517 (base) vs 0.484
  (pivot) vs 0.514 (shuffled) at think@1024 (pivot−shuffled within noise at n≈168/arm;
  pivot−base crosses the −0.02 guard).
- **P4 menagerie: NOT RUN** — the preregistered rule cancels blackbox spend on
  mechanism failure (no benchmark seeds consumed).
- Trainer facts: 39 optimizer steps/arm (13 min); batch-of-1 forwards (the
  padding-equivalence gate failed at 0.30–0.44 logits on this hybrid architecture even
  with right padding); chosen_win ≈0.43 → noisy 0.39–0.60; both C49 gates PASS.

## Controls

- **Shuffled-label control degrades nearly identically** (natural-close falls to the
  same 0.076 at 2048 in both trained arms): the damage is generic to the training
  regime, not the outcome-conditioned signal; the signal-specific residual
  (pivot − shuffled ≈ −1.7pp/−2.0pp) is itself non-positive.
- **C29 collapse guard: CLEAN** (code substrate greedy 0.058→0.067, pass@8 0.108→0.100,
  inside ±10%): the two-tier tether prevents sequence-level collapse; the damage
  channel is think-flow (delayed closure, +6pp answer-limit contacts), not
  distributional destruction. Loop rate unchanged (0.002→0.002).
- **No-think guard: CLEAN** (0.367→0.408): think-channel-specific effect, no
  answer-channel interference.

## Oracle Versus Deployable Evidence

Deployable: every headline number (greedy, single-shot, deployed budgets, merged bare
checkpoints, same backend/geometry per comparison). Oracle-only, labeled
NON-DEPLOYABLE: base best-of-8 coverage at harvest sampling on the same held-out band
tasks = 0.69 (`runs/whitebox_base_coverage.json`) — the sampling headroom the steering
signal was supposed to convert into greedy performance, and did not. Hidden-label
boundary held throughout: verifier outcomes labeled whole rollouts during mining only;
no gold content entered any prompt or target.

## Interpretation

**The attractor precondition.** The published FTPO successes (loop initiators,
22.9%→1% on this exact model; lexical over-use, 83–92% suppression) share a structural
feature this arm lacks: the rejected token is a **confident outlier** — the argmax of
its context by a wide margin — so demoting it is a small, local edit. Our pivot rows
reject tokens sitting **near parity** with their siblings (both sampled at T=0.6 from
the same context; initial chosen_win ≈ 0.43 ≈ chance). For such rows the ε=2.0 margin
objective must *manufacture* a 2-logit separation at ~600 scattered early-think
positions; the measured consequence is think-flow disturbance indistinguishable between
true and permuted labels — pure collateral from the edit, with n=16 Monte-Carlo labels
too weak to show a steering benefit on top at this dose. More likely now: single-token
preference training is safe and effective only against confident distributional
outliers; C29's lesson (preference-on-own-samples damages) extends to the
single-position regime whenever that precondition is violated. Less likely: that any
simple re-dose of this exact recipe flips the sign (the shuffled control tracks the
damage, so dose scales harm and signal together at best). Still unknown: whether
"confident wrong turns" (pivot nodes where the failing token is also locally dominant)
restore the precondition and the benefit — that filter is the sharpest round-2 lever —
and whether the UNDERDOSED caveat (879-row pool ≪ published 15–20k) hides a weak
positive signal beneath the collateral.

## Next Experiments

1. **Confident-wrong-turn filter (round 2, highest information):** keep the pipeline,
   add a logit-readout filter selecting only pivot nodes where the rejected token is
   the locally dominant continuation (restores the attractor precondition); expect far
   fewer rows — dose gate accordingly.
2. ε ablation (0.25–0.5): demote toward parity rather than manufacture margins.
3. n=32 / gap=1.0-only labels: cleaner Monte-Carlo signal at matched GPU.
4. **Long-context loop-FTPO** (separate experiment, `test_time_reasoning_budget`):
   the loop pathology lives at 16k+; the loop-control mandate and the ≤6k training
   context cap make this its own design problem.
5. Menagerie remains unexposed for whichever variant first passes its mechanism gate.

## Artifact Manifest

`artifact_manifest.yaml` is current: adapters + merged checkpoints external under
`large_artifacts/qwen35_4b_think_ftpo_round1/`; harvest shards omitted (regenerable;
mined `data/rows_*.jsonl.gz` committed as the training-input anchor); menagerie log
empty this round.
