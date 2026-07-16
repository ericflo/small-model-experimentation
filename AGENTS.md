# Agent Operating Guide

This repository is a compounding research system. The current experiments are examples and seed evidence; your job is to extend the frontier while preserving what has already been learned.

This file plus `docs/` and `knowledge/` are the complete operating context: everything an agent needs to work here must live in the repository itself, not in any agent's private memory or notes. If you learn something operational the hard way (a footgun, a recovery recipe, a convention), codify it here — preferably as an automated check, otherwise in the matching doc.

## First Pass

1. Read [research_programs/README.md](research_programs/README.md).
2. Read [knowledge/research_program_index.md](knowledge/research_program_index.md).
3. Read [knowledge/program_scorecards.md](knowledge/program_scorecards.md).
4. Read [knowledge/claims/index.md](knowledge/claims/index.md).
5. Read [knowledge/synthesis.md](knowledge/synthesis.md).
6. Read [docs/model_playbook.md](docs/model_playbook.md) — how to elicit and evaluate Qwen3.5-4B correctly (distilled from the claim ledger).
7. Read [docs/compute_environment.md](docs/compute_environment.md) — how to run on the current box, including GPU failure recovery.
8. Read [docs/vllm_inference.md](docs/vllm_inference.md) before bulk generation — the pinned high-throughput runner, LoRA path, parity gates, and backend-mixing prohibition.
9. Use `make related QUERY="<rough idea>"`, [knowledge/experiment_catalog.md](knowledge/experiment_catalog.md), and [knowledge/tag_index.md](knowledge/tag_index.md) to find close prior work.
10. Before adding work, read [docs/discovery_workflow.md](docs/discovery_workflow.md), [docs/idea_intake_protocol.md](docs/idea_intake_protocol.md), [docs/experiment_lifecycle.md](docs/experiment_lifecycle.md), [docs/research_program_lifecycle.md](docs/research_program_lifecycle.md), [docs/artifact_policy.md](docs/artifact_policy.md), and [docs/knowledgebase_protocol.md](docs/knowledgebase_protocol.md).
11. Use [docs/quality_gates.md](docs/quality_gates.md) to understand what `make check` enforces and how to fix its common failures.

## Non-Negotiables

- **One model, always `Qwen/Qwen3.5-4B` — absolute.** Never load, run, or even suggest any other model, for ANY purpose: not as a distillation teacher, trace generator, judge, or capability source, and never an older Qwen (3-4B, 2.5) for tooling convenience. Distillation and scaling are off-mission because they push the capability problem up the scaling stack. The only acceptable switch is a strictly newer, better Qwen ~4B if one is released. See the constraint section in [README.md](README.md).
- **"Sample more" is the baseline to beat, not the answer.** A method counts only if it beats matched-compute sampling on held-out, contamination-controlled tasks.
- **Use the vLLM template for bulk generation unless the measurement requires Transformers internals.** Keep every arm and matched-compute baseline on the same inference backend; equal seeds do not make HF and vLLM samples comparable. Preserve the exact backend and runner metadata with results.
- **Contamination invalidates self-training claims.** Training-on-own-solutions results must use contamination-free (procedural/fresh) substrates — self-training that gains on clean data has regressed on contaminated MBPP (C11). Saturated public benchmarks are acceptable for *measurement* studies (calibration, confidence), not for capability-gain claims.
- **Never train on benchmarks/ content.** benchmarks/ holds held-out measurement instruments (see benchmarks/README.md). Experiments may RUN a suite via its run.py and record scores — nothing else: never import family modules, never read family sources, generated items, transcripts, or result details, and never put suite items/transcripts in training data. Directory and family names are public metadata; contents are read-forbidden because reading contaminates agent context and can leak into later experiments. Leaked benchmark content cannot be un-leaked.
- **In ambiguity, follow the repo's evident convention** — not whatever is easiest to set up. If the corpus overwhelmingly does X, do X and solve the tooling friction.
- Treat the imported tracks as prototypes, not as the repo boundary.
- **Every experiment is STANDALONE — fully reproducible from its own
  directory (owner directive, 2026-07-15).** An experiment must never
  require walking another experiment's directory, receipts, or
  `large_artifacts/` tree to be reproduced. If a cell merges adapters or
  continues training from any prior checkpoint, it carries the complete
  reproduction package itself: a clearly delineated ORDERED list of SFT
  datasets copied into its own `data/` (e.g. `data/lineage/stage01_*.jsonl`),
  the exact per-stage recipe (base model revision, adapter rank/alpha,
  hyperparameters, fixed seeds, merge order), and a rebuild script such
  that the same datasets in the same order with the same seeds reproduce
  the same checkpoints. Cross-experiment tree/weights SHAs may appear as
  *verification aids* (asserting the rebuilt artifact matches what was
  measured) but never as the reproduction path or a load-bearing
  dependency. Evaluation-only cells that measure a composite carry that
  composite's full lineage package too. Scope boundary: shared measurement
  instruments (`benchmarks/` suites, the trusted aggregate gateway) are
  repo-level infrastructure referenced in place — the standalone rule
  covers the model-reproduction path (datasets, training recipes, seeds,
  merges, and their scripts), not the instruments.
- Keep experiments self-contained under `experiments/<id>/`.
- Follow-up benchmarks, replications, and design variants get their own experiment
  directory. Copy the prior harness or artifacts you need into the new experiment
  and modify there; do not extend an existing result-bearing experiment with a new
  substrate or follow-up result.
- Attach every new experiment to a research program, or create a new program.
- Use an idea intake note for material new directions and name the closest near-duplicate before running.
- Run an adversarial design review before the expensive run and save it as `reports/design_review.md` (reviews have caught peeking bugs, redundant framings, and missing controls before they burned GPU-days).
- Preserve negative results and failed controls; they are part of the map.
- Keep `reports/artifact_manifest.yaml` or a historical large-artifact manifest current when outputs are external or omitted.
- Update program evidence and shared synthesis when a result changes strategy.
- Run `make check` before committing, and check `gh run list` after pushing — local checks can pass while CI diverges (see [docs/quality_gates.md](docs/quality_gates.md)).
- Treat `in-progress` as a live execution state, never a backlog label. In the same commit that records a terminal result, deliberate stop, deferral, or handoff to a successor, set the experiment README to `**Status:** finished` and state the disposition honestly. Before every push, review the roster printed by `make active-experiments`; follow-up work belongs in a new experiment rather than keeping the concluded parent open.
- Capture and validate a clean Git preflight **before** an in-repo runner wrapper opens logs or outputs. Runner metadata sampled afterward will correctly report a dirty tree caused by those new artifacts; bind the original commit and allowed artifact paths instead of asserting post-open `git_dirty=false`.

## When Starting A New Experiment

Pick the program first. If no program fits, create one with:

```bash
make new-program PROGRAM=<program_id> TITLE="<Title>" FOCUS="<one-sentence focus>"
```

Then create the experiment with:

```bash
make new-experiment EXPERIMENT=<experiment_id> PROGRAM=<program_id> TITLE="<Title>"
```

Fill in the README, make the smallest runnable smoke path real, and only then run expensive work.

## When Starting A New Program

Create `research_programs/<program-id>/` with a charter, backlog, and evidence ledger. Add it to `research_programs/registry.yaml`; `make new-program` does the mechanical pieces. The charter must explain why the program is not just a variant of an existing line.

## When Editing Imported Work

Prefer additive notes in `README.md`, `experiment_log.md`, `analysis/`, or `reports/`. Do not rewrite historical outputs unless you are regenerating them from code and can explain the change.
