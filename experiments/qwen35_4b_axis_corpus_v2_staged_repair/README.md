# Axis Corpus V2 with Staged Repair

Rebuild the axis curriculum's two dead blocks from quantified failure forensics of the line's own 1,296 graded completions: staged repair lessons whose think targets DEMONSTRATE the bounded search (localize-only, repair-given-step, and trace-audit formats on the best-measured formalism, with rejected op-TYPE candidates and explicit checkpoint rules), hygiene with the co-located-injection share raised, explore unchanged — trained on the line's best local artifact against an exact-exposure replay control, judged under the corrected detectability bar with a prospectively documented answer-normalization fix, medium pilot behind it.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: three fresh-instrument replications established which lessons install (hygiene, explore, termination) and which do not (v1 trace-repair: an asserted search the model had to improvise; v1 protocol: redundant dose); the forensics document quantifies every failure class.

## Question

Does demonstrating the bounded search — instead of asserting its result — install the repair skill that three events show the model improvising and failing, at matched exposure against replay, without disturbing the installed blocks?

## Hypothesis

The v1 model faithfully ran the taught prefix and diverged exactly at the hand-wave. Staged formats separate the cheap sub-skill (the correct instruction content, already present in 16 wrong-step failures) from the binding one (localization); the demonstrated rejections supply the op-TYPE and write-target candidate classes the truth requires; the checkpoint rules kill the two measured heuristic errors; bounded thinks kill the 25% enumeration spirals.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: the `axis_on_replay` composite (tree `77e4858f...4ea`, weights `7ebcad39...0e4`); warm start from its adapter (`87cdebde...801`).
- Corpus (construction seed 77,118): `u_bugfind` 30 (localize-only, `STEP <k>`; early bugs oversampled), `u_bugmend` 25 (corrected instruction given the step), `u_retrace` 25 (first wrong transition + correct final state), `u_explore` 40 (unchanged), `u_hygiene` 40 (co-located injections ≥ 40% of injected rows). All executable-truth audited; repair uniqueness enforced by exhaustive grammar enumeration.
- Arms: `replay_repeat3` (control) and `axis_v2` (candidate); 1,280-core + 240-block exact three-axis MILP (slot seed 55,120); training seed 54; 190 updates; zero skips.
- Gate: fresh seed 88,017; 50-task axis holdout (10 per v2 kind) + 104-task retention screen; the corrected detectability bar (undetectable kinds excluded and reported; ⌈2/3 × detectable⌉ wins required; fail-closed if none detectable); retention bands unchanged; ANSWER NORMALIZATION applied identically to every arm (whitespace collapsed; spaces adjacent to '>' and ';' removed), documented in the receipt with the measured 21-row rationale.
- Conditional pilot: sealed seed 78,147, MEDIUM tier, think budget 1,024; candidate aggregate strictly above base, replay control, and parent; every-family-versus-base recorded as the goal gate.
- Kill rule (frozen): if neither `u_bugfind` nor `u_bugmend` registers a holdout win over both controls, the trace-repair axis closes for this model at this dose; no v3 of the same mechanism.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_axis_corpus_v2_staged_repair/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_axis_corpus_v2_staged_repair/scripts/run.py --stage train-control
.venv/bin/python -B experiments/qwen35_4b_axis_corpus_v2_staged_repair/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_axis_corpus_v2_staged_repair/scripts/run.py --stage merge-arms
.venv/bin/python -B experiments/qwen35_4b_axis_corpus_v2_staged_repair/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_axis_corpus_v2_staged_repair/scripts/run.py --stage benchmark
```

## Results

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending.
- Program backlog updated: this trial claims the queued v2 slot.
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis_v2.jsonl`, `data/corpus_manifest.json`: frozen v2 corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88017.jsonl`, `data/local_input_seed88017.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
