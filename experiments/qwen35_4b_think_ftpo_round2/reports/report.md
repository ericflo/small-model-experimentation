# Entropy-routed think-pivot optimization round 2 — final report

## Verdict

`LOW_DOSE_NULL` (preregistered label). Entropy-routed, outcome-conditioned
single-token updates did **not** elicit a general capability gain from
Qwen3.5-4B. The primary positive-only `uplift` arm was less destructive than
conventional `demote` and separated from shuffled labels on two local
substrates, but it did not beat the frozen model on the held-out repository
agent: 39/72 versus 43/72 (`−5.56pp`, paired-bootstrap 95% CI
`[−19.44,+8.33]`). It also missed both fresh whitebox gates.

The experiment resolves the round-1 follow-up: selecting confident wrong
turns is not sufficient to make weight-space FTPO local. Pulling the successful
token up changes the damage profile and preserves some outcome-label signal,
but the shared LoRA update still moves non-target logits too far. No menagerie
event was run; zero blackbox benchmark seeds were consumed.

Here `LOW_DOSE_NULL` does not mean “nothing moved.” It is the frozen analyzer's
label for missing the capability gates when the real-label arm separates from
the shuffled arm enough that the stricter `GENERIC_TRAINING_HARM` rule does not
apply. The absolute held-out result remains negative.

## Design

The same 155 frozen-base-qualified real pivot rows fed two arms:

- `demote`: the published pairwise FTPO objective with a two-logit margin;
- `uplift`: a bounded positive-only objective targeting a +0.5 chosen-token
  logit gain while treating the failed token as non-target.

An otherwise identical `uplift_shuffled` arm used outcome-permuted parent
labels. Rows qualified at the actual harvest temperature, T=0.6, only when
the failed token was base argmax, led the successful sibling by at least 0.5
logits, had P(failed)≥0.5, entropy≤1.5 nats, varentropy≥0.1 nats², and retained
a plausible successful alternative. Full thresholds, gates, and outcome
labels were frozen in [preregistration.md](preregistration.md).

The capability north star was a fresh, procedural six-family repository-
repair suite. An eight-turn tool loop could inspect files, search, run visible
tests, apply exact patches, and submit; final workspaces were scored by hidden
tests. The matched-compute sample-more baseline unioned two independent
four-turn base trajectories under the same eight-call / 6,144-reserved-token
ceiling.

## Gate results

| Gate | Result | Evidence |
| --- | --- | --- |
| P0 geometry | PASS | 155 matched rows; minimum 128 |
| P1 targeted mechanism | FAIL, all arms | hit-rate bars passed, but mean per-row median absolute non-target drift was 0.229/0.145/0.120 logits for demote/uplift/shuffled; ceiling 0.10 |
| P2 fresh whitebox | FAIL | uplift `+0.26pp` at think@1024 and `−3.06pp` at think@2048 vs base; required ≥+3pp plus control separation and termination guards |
| P3 repository agent | FAIL | uplift 39/72 vs base 43/72; required ≥+8pp and a sample-more win |
| P4 broad guards | PASS | C49, gym floor, collapse, and no-think guards all passed |
| P5 menagerie | NOT ELIGIBLE | P1, P2, and P3 failed |

## Geometry and training mechanism

The selector premise was real but sparse. Of 615 real and 661 shuffled parent
rows, 155 (25.2%) and 166 (25.1%) qualified; seeded matching retained 155 per
arm. Full-pool medians for the real rows were P(failed)=0.397, failed-minus-
best-successful gap=0.0 logits, entropy=0.692 nats, and varentropy=0.441 nats².

All arms hit the training safety stop early: demote after 8/20 optimizer steps,
uplift and shuffled after 5/20. The exact-logit audit shows why downstream
evaluation remained unsafe despite intended-target movement:

| Arm | Objective hit | Chosen gain | Failed-token drift | Pair-gap shift | Median non-target drift | P95 non-target drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| demote | 40.0% | +0.382 | −1.043 | +1.425 | 0.229 | 0.547 |
| uplift | 75.5% | +0.781 | +0.502 | +0.280 | 0.145 | 0.363 |
| uplift_shuffled | 76.1% | +0.762 | +0.550 | +0.212 | 0.120 | 0.304 |

Positive-only uplift cut median collateral by 36.6% relative to demotion and
avoided manufacturing a large chosen-over-failed margin. It did not isolate
the edit: even the tethered failed token rose +0.50 logits on average. The
shuffled arm's similar movement identifies a generic shared-parameter update,
not outcome-conditioned steering, as the dominant P1 failure.

### Entropy and varentropy routing

Entropy and varentropy were useful instruments for locating and auditing
forks, but they were not monotone measures of editability. For the real uplift
arm, entropy quartiles Q1→Q4 had objective-hit rates
`79.5%, 89.5%, 76.9%, 56.4%` and non-target drifts
`0.163, 0.134, 0.137, 0.147`. Varentropy quartiles Q1→Q4 had hit rates
`89.7%, 71.1%, 84.6%, 56.4%` and drifts
`0.122, 0.148, 0.176, 0.136`.

The cleanest stratum was **lowest** varentropy, not highest. The preregistered
minimum-varentropy filter removed deterministic grooves, but increasing
varentropy beyond that did not create safer or more fruitful weight edits.
This is a routing result, not permission to post-hoc train on Q1: any such
filter needs an independent experiment and still must clear P1.

## Fresh whitebox outcomes

The nominal N=400 allocation produced 392 paired tasks because integer
per-cell allocation was used (a 2% protocol deviation; no task was selected or
removed by outcome).

| Budget | Base | Demote Δ | Uplift Δ | Shuffled Δ | Uplift−shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| think@1024 | 53.57% | −2.81pp `[-6.63,+0.77]` | +0.26pp `[-3.57,+4.08]` | −2.30pp `[-6.38,+1.79]` | +2.55pp `[-1.53,+6.89]` |
| think@2048 | 58.16% | −3.32pp `[-7.40,+0.77]` | −3.06pp `[-7.40,+1.28]` | −1.02pp `[-5.87,+3.83]` | −2.04pp `[-6.63,+2.81]` |

Uplift did not transfer consistently across budgets. At think@2048 it raised
natural closure from 15.05% to 19.13%, but answer-limit contacts also rose from
37.24% to 40.56%, violating the ≤2pp termination guard. Demotion raised
answer-limit contacts by 3.32pp/2.81pp at 1024/2048. Loops remained rare in
every arm (0–0.77%), reinforcing that repetition was not the deployed-budget
bottleneck.

## Repository-agent north star

| Arm | Hidden-test pass | Δ vs deep base (95% CI) | Submit | Invalid actions / turn | Mean sampled tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| base, eight-turn deep | 43/72 (59.72%) | — | 45.83% | 9.98% | 2,510 |
| demote | 34/72 (47.22%) | −12.50pp `[−23.61,−1.39]` | 38.89% | 13.84% | 2,597 |
| uplift | 39/72 (54.17%) | −5.56pp `[−19.44,+8.33]` | 38.89% | 12.27% | 2,643 |
| uplift_shuffled | 29/72 (40.28%) | −19.44pp `[−33.33,−5.56]` | 26.39% | 14.26% | 2,816 |
| base sample-more, 2×4 turns | 22/72 (30.56%) | — | 1.39% | 4.86% | 2,341 |

Uplift exceeded shuffled by +13.89pp, with its paired-bootstrap interval
touching zero `[0.00,+27.78]`. This is compatible with useful information in
the true outcome directions, as is the gym separation below, but it is not a
general capability gain: uplift remained 5.56pp below the stronger serial
base agent. It beat matched-compute branching by +23.61pp
`[+11.11,+36.11]`, but the preregistration required beating both sample-more
and deep base. Here, preserving eight serial feedback turns was much stronger
than splitting them into two short trajectories.

## Broad guards and controls

Gym aggregate success was base 48.55%, demote 49.29%, uplift 53.27%, and
shuffled 47.02%. Thus true-label uplift exceeded shuffled by 6.25pp on the
parent-style gym, while held-out-family uplift was 50.0% versus base 53.57%.
The direction signal is substrate-local, not held-out breadth.

All merged checkpoints passed the mandatory C49 on-vs-off behavioral gate
(0/8 outputs identical to base). Collapse guards passed: base greedy/pass@8
were 4.17%/9.17%, versus demote 5.00%/12.50%, uplift 5.83%/10.83%, and
shuffled 6.67%/11.67%. No-think success was base 36.67%, demote 39.17%,
uplift 37.50%, and shuffled 36.67%. This is not C29-style collapse; the damage
is specific enough to evade coarse guards yet broad enough to erase transfer.

## Interpretation

Round 2 falsifies the simple rescue proposed after round 1. A confident failed
argmax plus low entropy and nonzero varentropy restores the published
*geometry*, but not the required *parameter locality*. Conventional demotion
is decisively harmful on the coding agent. Positive-only pressure is the
better direction—it retains a real-label advantage over shuffled training and
reduces collateral—but shared LoRA weights still change neighboring logits
and agent behavior more than the sparse label signal can repay at 155 rows.

The next experiment should therefore not merely harvest more rows or increase
varentropy. First earn locality with a smaller (+0.25) uplift or a genuinely
context-gated last-layer/activation intervention, using P1 as a hard preflight.
Only a mechanism below 0.10 median non-target drift should receive a fresh,
larger outcome harvest and another agentic transfer test. Long-context loop
FTPO remains a separate 16k+ question; this experiment provides no reason to
apply it at deployed budgets.

## Reproducibility and artifacts

The result is regenerated by `scripts/run.py --full`; machine-readable gates
are in [`analysis/summary.json`](../analysis/summary.json). Training rows and
all small run receipts are committed. Adapters and merged checkpoints are
external, checksummed in [artifact_manifest.yaml](artifact_manifest.yaml).
The full run used only `Qwen/Qwen3.5-4B`, kept HF exact-logit work separate
from vLLM generation, and evaluated every vLLM arm as a merged checkpoint.
