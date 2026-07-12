# Preregistration: Jacobian Value Transport

Frozen before any result-bearing GPU call. CPU unit tests, tokenizer inspection,
environment validation, and two-item plumbing smokes may precede the scientific
stages, but their outputs cannot tune the gates below.

## Claim ladder

This experiment separates four claims that are often conflated:

1. **Readable:** a target concept ranks in a J-lens readout.
2. **Writable:** replacing its coordinate changes a direct verbal report.
3. **Transported:** the same replacement changes a downstream consequence and
   local Jacobian gain predicts rollout value beyond coordinate strength.
4. **Capability-causal:** patching during native thinking changes exact verified
   task success specifically against the full control set.

Only claim 4 authorizes a non-oracle capability follow-up. None of these oracle
stages is itself deployable.

## Data and contamination boundary

All data are generated locally from fresh procedural grammars copied into this
experiment. The anchor family composes list transformations; string and three-
register families are held-family checks. Lens-fit, calibration, IID, held-family,
and hard-depth seeds are disjoint. No file below `benchmarks/` is read, imported,
or used for training or task generation.

The positive control uses fresh prompt-local mappings among single-token concepts.
One arm asks for the selected concept directly; a second maps that concept to a
random prompt-local consequence and tests whether the intervention changes the
consequence. The second arm prevents a direct-output-only effect from satisfying G0.

## Jacobian estimators

The generic layer transport is

`J_l = E[d h_target,t' / d h_l,t]`,

with equal weighting over declared valid source/target pairs. The implementation
must store the weighting convention in every lens receipt. Targeted token directions
are vector-Jacobian products of final-layer unembedding covectors. A full matrix is
fitted only after the targeted positive-control plumbing works.

Source block outputs are indexed `[8, 12, 16, 20, 24]`; the target is block output
31. A direction is always paired with its layer. Multi-token concepts are excluded
from the confirmatory positive control rather than averaged post hoc.

Coordinate swaps normalize the two dictionary vectors, read their least-squares
coordinates with a pseudoinverse, exchange source and target coordinates, and write
the signed delta back. Norm-matched random and wrong-token swaps use the same write
operator and coefficient. Logit-lens controls substitute the source-layer
unembedding direction. ActAdd uses a calibration-only high-minus-low mean direction.

## G0: causal positive control

Forty-eight calibration items are fixed before model evaluation. The clean model
must answer at least 70% correctly. A J swap must:

- raise the target concept report rate by at least 0.20;
- raise the target downstream-consequence rate by at least 0.15;
- exceed the norm-matched random control by at least 0.10;
- satisfy the effect at two adjacent tested layers; and
- reduce parse rate by no more than 0.10.

All conditions and layer coefficients are reported. The confirmatory coefficient is
selected on half the calibration items and applied unchanged to the other half.
Failure stops the value and causal-patch stages. Exploratory diagnostics may be
reported but cannot retrospectively pass G0.

## G1: think-prefix value and transport

On 48 disjoint depth-2 anchor tasks, sample four natural native-thinking traces per
task. Choose checkpoints at the frozen fractions of the generated think span, with
at least 16 thought tokens. From each prefix sample four disjoint-seed continuations
without forcing a close. Exact hidden execution defines

`V(prefix) = fraction of continuations that solve the task`.

Whole-trace labels are never assigned to every token. Confirmatory analysis is
within task and position-detrended. The primary transport score is the alignment of
the correct-operation J direction with the context-local gradient of the eventual
correct-answer margin; coordinate activity alone, answer entropy, prefix length,
and shuffled outcome labels are comparators.

G1 passes only with at least 24 mixed-value tasks and 100 scored prefixes, task-macro
transport AUROC at least 0.65, transport AUROC at least 0.03 above coordinate-only
activity, and shuffled-label AUROC within 0.05 of chance. If the natural-close or
parse rate makes these sample requirements impossible, the result is a seam failure,
not permission to inject `</think>` or lower the gate.

## G2: causal thought-coordinate patch

For common-task high-value and low-value branches, patch the low-value branch only
inside its natural think span and continue with full-prefix recomputation and
`use_cache=False`. The confirmatory layer band and coefficient are frozen from G0.

Conditions are:

- frozen baseline;
- correct J-coordinate swap;
- wrong-operation J swap;
- outcome-shuffled J swap;
- norm-matched random write;
- logit-lens swap;
- C20-style mean-difference ActAdd;
- raw high-minus-low donor patch;
- sparse J component of that donor delta; and
- its norm-matched non-J remainder.

All arms use the same model revision, HF backend, prompt, prefix, maximum generated
tokens, and decoding seeds. Exact hidden-task success and parse/termination are
primary; direct operation naming is diagnostic.

G2 passes only if the J arm improves exact success by at least +0.10 with paired
bootstrap lower bound above zero, exceeds random and wrong controls by at least
+0.08, exceeds the non-J remainder by at least +0.05, and loses no more than 0.05
parse rate. A direct-report shift without exact success is `WRITABLE_NOT_CAPABLE`.

## Decision labels

- `NO_J_WRITING`: G0 fails.
- `WRITABLE_NOT_VALUED`: G0 passes and G1 fails.
- `VALUED_NOT_CAUSAL`: G1 passes and G2 fails.
- `ORACLE_CAUSAL_TRANSPORT`: G2 passes; create a separate non-oracle experiment.

No claim number is reserved in advance. Shared claim/index edits occur only after a
fresh pull/rebase of `origin/main` following the terminal result.
