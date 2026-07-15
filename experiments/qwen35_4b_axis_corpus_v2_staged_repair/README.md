# Axis Corpus V2 with Staged Repair

Rebuild the axis curriculum's two dead blocks from quantified failure forensics of the line's own 1,296 graded completions: staged repair lessons whose think targets DEMONSTRATE the bounded search (localize-only, repair-given-step, and trace-audit formats on the best-measured formalism, with rejected op-TYPE candidates and explicit checkpoint rules), hygiene with the co-located-injection share raised, explore unchanged — trained on the line's best local artifact against an exact-exposure replay control, judged under the corrected detectability bar with a prospectively documented answer-normalization fix, medium pilot behind it.

**Status:** finished · 2026-07-15 · not promoted; the frozen kill rule FIRED (neither staged-repair lesson won) — the trace-repair axis is closed for this model at this dose; third-dose interference measured on the adapter lineage; seed 78,147 permanently sealed

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

Both arms trained cleanly (control 0.3323, candidate 0.4647 train loss; 0 skips) and merged. The frozen 154-task gate event at seed 88,017 (normalized grading, four of five kinds detectable, required wins 3): axis holdout of 50 — candidate 19, parent 19, replay_repeat3 25. Per-kind candidate/parent/replay: bugfind 3/0/3 (tie), bugmend 3/4/2 (loss), retrace 1/2/5 (loss), explore 5/7/9 (loss), hygiene 7/6/6 (the only win). Retention: candidate 66/98/4 versus parent 71/98/3 and replay 69/95/8. Failed checks: the breadth bar AND both axis-total strict comparisons. KILL RULE: `u_bugfind_win: false`, `u_bugmend_win: false`. No promotion; seed 78,147 permanently sealed; the medium pilot never ran.

## Interpretation

Two laws land, both preregistered readings. (1) The trace-repair kill rule fires cleanly: demonstrating the bounded search did not install what asserting it failed to install — two content designs, four measurement events, zero robust repair installs; the axis is closed for this model at this dose, and any future attack requires a genuinely different mechanism argument. (2) Third-dose interference: the candidate — the third consecutive designed dose continued in place on one rank-32 adapter lineage — tied its parent on the axis total, lost its previously-installed explore edge, and dropped retention by five, while the third replay round WON the whole axis holdout (25/50, explore 9/10, retrace 5/10). Content stacking on this adapter has hit diminishing-to-negative returns exactly where replay refreshing keeps gaining; the lineage looks saturated as a vehicle for further designed doses.

## Terminal Disposition

No later event is authorized here. Seed 78,147 is spent-by-sealing. The published composites and raw gate outputs are preserved. The trace-repair axis may not be reopened at this dose/mechanism family; any future designed dose should start from a fresh adapter on a clean parent rather than a fourth continuation of this lineage.

## Knowledgebase Update

- Program evidence updated: kill-rule closure and third-dose interference recorded.
- Program backlog updated: the axis-dose line closes; successor directions carry calibration notes.
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis_v2.jsonl`, `data/corpus_manifest.json`: frozen v2 corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88017.jsonl`, `data/local_input_seed88017.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
