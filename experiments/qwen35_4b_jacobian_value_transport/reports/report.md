# Averaged J coordinates can overwrite a concept report, but do not transport through a new mapping

## Verdict

**Terminal frozen label: `NO_J_WRITING` (G0 failed).** The label hides a useful
split: the targeted J direction was strongly writable at one late layer, but it
did not behave like a reusable intermediate that downstream computation could
consume.

On the untouched 24-item confirmation half, a layer-24 source-to-target J swap
changed direct concept reports from 0/24 target answers at baseline to 18/24
(75%). A logit-lens swap reached 5/24 (20.8%); random reached 0/24. Yet the same
J swap changed a prompt-local consequence of the concept on **0/24**, at every
layer. Earlier layers were inert, so the required adjacent-layer criterion also
failed. Per preregistration, prefix-value mapping and task-level causal patches
were not run.

This is a negative for the paper's transferable-workspace premise on this fixed
4B under an averaged single-token lens and pairwise coordinate swap. It is not a
negative for context-local Jacobians, true set-to-target clamping, or
counterfactual reflection training.

## Design and data

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: Transformers 5.13.0, torch 2.11.0, bf16 SDPA, with the Qwen hybrid
  fast path available.
- Lens corpus: 64 fresh procedural prompts, disjoint from all control and task
  splits.
- Dictionary: 24 concepts verified as single leading-space tokens.
- Source layers: 8, 12, 16, 20, 24; target block output: 31.
- Estimator: equal weighting over every valid causal source/target-position pair,
  rather than summed-future-target weighting per source.
- Positive controls: 48 fresh prompt-local items, split deterministically into
  24 selection and 24 confirmation items.
- Clean tasks: direct report of a selected concept and a separate arbitrary
  mapping from the concept to a digit consequence.

The selection half chose alpha 4 and the nominal layer pair [20, 24]. The
confirmation half remained untouched until that choice was frozen. All 5,088
per-item condition rows are in `runs/positive_control_rows.jsonl`.

## G0 confirmation results

Target-answer rate at the selected alpha:

| layer | direct J | direct random | direct logit lens | mapped consequence J |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0% | 0.0% | 0.0% | 0.0% |
| 12 | 0.0% | 0.0% | 0.0% | 0.0% |
| 16 | 0.0% | 0.0% | 0.0% | 0.0% |
| 20 | 4.2% | 0.0% | 0.0% | 0.0% |
| 24 | **75.0%** | 0.0% | 20.8% | **0.0%** |

Baseline source accuracy and parse rate were 100% in both tasks. Layer-24 J
writing preserved a 100% direct parse rate while reducing source answers to 25%
and increasing target answers to 75%. A wrong-concept J swap also reduced source
answers to 25% without increasing the registered target, consistent with writing
the wrong concept rather than causing generic corruption.

The consequential task did not merely miss the argmax threshold. At layer 24 on
the selection half, mean target-minus-source digit margin stayed flat:

| alpha | direct-report margin | consequence margin |
| ---: | ---: | ---: |
| 0.5 | -8.82 | -8.57 |
| 1.0 | -6.98 | -8.57 |
| 2.0 | -3.43 | -8.56 |
| 4.0 | **+3.22** | **-8.57** |

The intervention monotonically crossed the direct report boundary while having
essentially zero effect on the consequence. That is the central result.

## Frozen gate audit

- Clean accuracy >= 0.70: **pass** (1.00 direct and consequence).
- Direct target shift >= +0.20: **pass at layer 24 only** (+0.75).
- Consequence target shift >= +0.15: **fail at every layer** (+0.00).
- J minus random >= +0.10: direct layer 24 passes; consequence fails.
- Two adjacent tested layers pass: **fail**.
- Parse-rate drop <= 0.10: **pass**.

Overall G0: **fail**. G1 and G2 were ineligible by the frozen decision rule.

## What the intervention appears to be

The layer profile is late and motor-like. Nothing meaningful happens through
layer 16; a weak direct effect appears at layer 20; layer 24 strongly controls
the imminent concept token. If this were a broadly readable intermediate
workspace coordinate, changing the selected concept should also have changed
the random mapping consequence. It did not.

The result therefore favors “late token-aligned output control” over “reusable
causal reasoning variable” for this averaged lens on Qwen3.5-4B. It also explains
why a readable coordinate need not unlock capability: making a word easier to
say is different from making downstream circuitry recompute with its meaning.

## Honest protocol notes

1. A tokenizer preflight initially stopped before any result artifact because
   space-plus-digit was two tokens. The corrected, pre-result contract puts the
   space in the fixed `Value: ` prefix and scores one bare digit token.
2. The random coordinate write used normalized random dictionary pairs but did
   not exactly match the realized per-example J delta norm. At direct layer 24,
   mean J delta norm was 13.64 versus 2.64 random. This weakens the strength of
   the direct J-versus-random specificity claim. It cannot rescue G0: the much
   larger J perturbation still had zero consequence effect, and consequence
   layer-24 norms were closer (4.96 J versus 4.04 random).
3. Applying a swap independently at both layers [20, 24] can swap a coordinate
   toward the target and then swap it back. Its low band result is not treated
   as a test of the paper's set-to-target clamping operation and does not enter
   the individual-layer frozen gate.
4. The positive control is next-token consequential inference, not multi-step
   native thinking. G0 was deliberately required before the more expensive
   thought-prefix stages.

## Decision and next experiment

Do not run prefix-value or task-patching stages with this lens. Preserve the
negative and start a new result-bearing experiment that changes the mechanism:

- use a context-local Jacobian from the selected concept to the actual
  consequence margin;
- set source coordinates low and target coordinates high across an earlier
  layer band instead of repeatedly swapping them;
- apply the clamp across the positions where the concept is represented;
- construct orthogonal controls with the exact realized J delta norm per item;
- retain the direct-versus-consequence firewall before entering native thinking.

Only if that consequence gate passes should the program return to think-prefix
value and exact task success.

## Compute and artifacts

- Targeted fit: 29.3 seconds, 12.3 GB peak allocated, 617,627-byte lens artifact.
- G0 sweep and confirmation: 40.5 seconds after model load.
- No training, adapter, benchmark seed, G1 continuation, or G2 task-patching
  compute was consumed.
- See `reports/artifact_manifest.yaml` for the conditional full-J artifacts that
  were correctly not generated after G0 failed.
