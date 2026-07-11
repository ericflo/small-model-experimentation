# Entropy-routed think-pivot optimization round 2

This completed follow-up tested whether entropy-routed, outcome-labeled
thought pivots make Qwen3.5-4B better at held-out reasoning and multi-turn
coding, comparing bounded successful-token uplift against conventional FTPO
demotion and a shuffled-outcome control. The preregistered verdict is
`LOW_DOSE_NULL`: true labels retained a local signal, but no trained arm
elicited general capability.

## Research Program

- Primary: `agentic_breadth_installation`; cross-cutting
  `posttraining_and_adaptation` and `test_time_reasoning_budget`.
- Program question: after breadth-SFT's one-time C50 gain and round 1 FTPO's
  C52 failure, can a different, localized update mechanism move general
  agentic capability rather than merely suppress surface pathologies?
- Prior anchors: C52 (near-parity FTPO harms think flow), C50 (signal placement
  can transfer broadly), C29 (preference training is fragile), C41/C42
  (uncertainty localizes useful choices), and the positive-pressure locality
  audit (incremental signal must exist at the exact fork).

## Question

Round 1 pushed successful and failed sibling tokens two logits apart even when
the base regarded them as peers; its shuffled labels caused the same delayed
closure and degradation. This round asks two separable questions:

1. Does conventional FTPO work when restricted to true confident wrong turns?
2. At those same turns, is it safer or more useful to **pull up** the empirically
   fruitful continuation by a bounded amount while pinning the failed token?

Entropy and varentropy are measured at the actual harvest temperature. They do
not supply correctness labels; they distinguish a focused attractor from broad
noise and a deterministic groove from a spiky distribution with a plausible
alternative tail.

## Hypothesis

A failed argmax token with P≥0.5, a ≥0.5-logit lead, low entropy, and nonzero
varentropy has the outlier geometry missing from round 1. Pairwise demotion may
therefore become safe. The primary hypothesis is stronger: a +0.5-logit,
positive-only lift of its successful sibling will preserve think flow better
than manufacturing a two-logit chosen-over-rejected margin, and the gain will
not appear under shuffled outcomes.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision.
- Training source: round 1's exact committed, already-regularized real and
  shuffled FTPO rows. No new labels, teacher, benchmark content, or gold
  reasoning enters training.
- Geometry: frozen-base exact logits, one context per forward (the hybrid
  architecture fails padded equivalence); entropy/varentropy at T=0.6.
- Arms: base, `demote`, `uplift` (primary), `uplift_shuffled`.
- Trainer: LoRA r256/α128, two epochs maximum, final-position logits only,
  two-tier raw-logit tether. `demote` is published FTPO; `uplift` targets a
  +0.5 chosen-logit gain and treats rejected as non-target.
- Fresh evaluation: parent whitebox/gym substrates on new seeds plus a new
  held-out six-family repository-repair agent.
- Coding agent: iterative tree/read/search/test/exact-patch/submit tools over
  materialized Python repos; final hidden-test grading after eight turns.
- Matched-compute baseline: two independent four-turn base trajectories, the
  same maximum eight model calls and 6,144 sampled tokens per task. Explicit
  submit rate is separate from final-workspace correctness.
- Blackbox: menagerie only through `run.py` CLI and aggregate scores, and only
  after a preregistered whitebox gate.
- Full predictions and outcome labels: [preregistration](reports/preregistration.md).

## Run

CPU smoke:

```bash
python3 scripts/run.py --smoke
```

GPU smoke (two exact-logit rows + one task per repository family):

```bash
python3 scripts/run.py --gpu-smoke
```

Full staged pipeline through analysis:

```bash
python3 scripts/run.py --full \
  --artifact-root ../../large_artifacts/qwen35_4b_think_ftpo_round2
```

If a future exact reproduction sets `analysis/summary.json` to
`menagerie_eligible=true`, run the two frozen
paired quick events and regenerate analysis:

```bash
python3 scripts/bench.py --tier quick --seed 62011 --arms base uplift \
  --merged uplift=../../large_artifacts/qwen35_4b_think_ftpo_round2/merged/uplift
python3 scripts/bench.py --tier quick --seed 62012 --arms base uplift \
  --merged uplift=../../large_artifacts/qwen35_4b_think_ftpo_round2/merged/uplift
python3 scripts/analyze.py
```

## Result

P0 and every broad P4 guard passed, but P1 locality, P2 fresh whitebox, and P3
agentic coding all failed. The selector retained 155 confident-wrong-turn rows.
Positive-only uplift moved its target on 75.5% of them and reduced median
non-target drift from demotion's 0.229 logits to 0.145, but missed the 0.10
locality ceiling.

On 72 fresh repository repairs, deep base passed 43, `uplift` 39, `demote` 34,
and shuffled uplift 29. Uplift's +13.89pp separation from shuffled labels
(paired-bootstrap 95% CI `[0.00,+27.78]`) and +6.25pp gym separation show that
the outcome directions were not empty; neither effect overcame the generic
shared-weight update. Whitebox uplift was +0.26pp at think@1024 and −3.06pp at
think@2048 versus base. Menagerie was ineligible and zero benchmark seeds were
consumed. See the [final report](reports/report.md).

Entropy/varentropy were diagnostically useful but not monotone steering
coordinates: the lowest-varentropy uplift quartile had the cleanest update
(0.122 median non-target drift), while the third quartile was worst (0.176).
The next lever is parameter locality—a lower-dose or genuinely context-gated
intervention that clears P1—not simply more rows or higher-varentropy pivots.

## Interpretation boundary

The parent pool can supply only a few hundred qualified rows. A flat outcome
is therefore a low-dose null, not proof that thought steering cannot work.
Control-equivalent harm, termination damage, or a positive signal with clean
guards are informative at this dose and decide whether a larger harvest is
worth the GPU time.

## Artifacts

- `data/rows_{pivot,shuffled}.jsonl.gz`: exact parent inputs; selected rows are
  generated after P0.
- `src/repo_{tasks,agent}.py`: procedural repositories and iterative harness.
- `scripts/score_rows.py`: dominance/entropy/varentropy census and matched set.
- `scripts/train_sparse.py`, `audit_logits.py`: objectives and locality audit.
- `scripts/eval_whitebox.py`, `eval_gym.py`, `eval_repo_agent.py`: fresh gates.
- `analysis/`: regenerated paired-bootstrap verdict after the full run.
- External adapters and merged checkpoints are declared in
  `reports/artifact_manifest.yaml`; model weights never enter git.
