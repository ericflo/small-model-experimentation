# Qwen3.5-4B Partial-Structure Recognition-Guided Search Experiment Log

## Scaffold

Created as a new experiment scaffold after the user selected the top-ranked forest-review proposal.

## Pre-run audit

- `make related` routed the line to Structured Execution and found C25/C35/C47/C48 plus the older prefix
  verifier experiments.
- Re-graded the load-bearing C25 and C35 overclaims before relying on them: C25 base versus random was 1/80
  versus 2/80 (no established directional difference), and C35 did not run depth 5 or control banked-model
  dose across depths.
- Three read-only adversarial audits independently required the same two-stage gate: exact CPU oracle utility,
  then actionable within-parent recognition calibration.
- The audit caught a depth-validation footgun before data creation: inherited `min_depth_leq` silently stops at
  a 60,000-state cap and the inherited depth-5 generator only excludes solutions through depth 3. This
  experiment requires an exact exhaustion receipt through depth 4.
- Frozen pre-registration uses a type-only prefix, exact alternative-completion labels, vLLM-only model arms,
  thinking budget 256, beam width 4, depth-4 calibration, and disjoint depth-5 primary tasks.

## Pre-primary hardening amendment

Before any depth-5 model search, an independent compute audit found that the original 320-sample direct arm
matched only sampled decode tokens. It did not match the recognition controller's much larger repeated
prefill bill, and 320 samples were provably insufficient to cover the projected total-token cap. The frozen
pool was therefore raised to 512 and a second, stronger total-logical-model-token prefix was added from the
same generated pool. A G4 result must beat both direct arms; pool exhaustion is an invalid comparison. This
is a conservative amendment made without observing any primary-search model output.

Further read-only audits made the search path fail-closed before launch:

- removed the scientific-gate override;
- tied gate receipts to their exact source files and cache receipts to config/data/code/output hashes;
- added the missing budget-truncated brute control and a depth-pooled surface fallback;
- added pool-exhaustion, task alignment, direct-basis, and total-token parity checks;
- removed semantic-oracle reads from non-oracle frontier construction;
- required both sampled-token and total-logical-token direct controls for any G4 verdict.

The audit also caught that the initial oracle gate consumed primary hidden grades. A separate 12-task
exact-depth-5 development partition (seed 9001) was created, checked for behavioral collisions against both
other partitions, and made the only depth-5 gate basis. The corrected gate loads no primary artifact.

## Executed gates and runs

Commands were run from the repository root unless noted.

```bash
python3 experiments/qwen35_4b_partial_structure_search/scripts/build_data.py --workers 8
python3 experiments/qwen35_4b_partial_structure_search/scripts/data_audit.py --workers 8
python3 experiments/qwen35_4b_partial_structure_search/scripts/oracle_gate.py
python3 experiments/qwen35_4b_partial_structure_search/scripts/build_calibration.py
.venv-vllm/bin/python experiments/qwen35_4b_partial_structure_search/scripts/run_calibration.py
python3 experiments/qwen35_4b_partial_structure_search/scripts/compact_calibration_outputs.py
.venv-vllm/bin/python experiments/qwen35_4b_partial_structure_search/scripts/run_calibration.py --upgrade-receipt
python3 experiments/qwen35_4b_partial_structure_search/scripts/analyze_calibration.py
python3 experiments/qwen35_4b_partial_structure_search/scripts/full_brute.py --workers 8
```

- Data integrity: 120/120 uncapped minimum-depth receipts independently recomputed; 120 task/oracle pairs;
  zero collisions on the frozen 64-input common behavior bank; hidden excluded from semantic labels.
- Corrected oracle-development gate: PASS, 12/12 path coverage and selected hidden success, 262,144x completed-
  leaf compression. Primary artifacts loaded: false.
- Full visible-only brute reference: 60/60 pool hidden coverage, 56/60 selected hidden success, 111.85 seconds
  parallel wall on eight CPU workers.
- Model calibration: 7,200 children / 450 sibling groups. vLLM wall 2,563.24 seconds. Thinking used
  11,880,928 logical model tokens and forced-close rate 1.0.
- Recognition verdict: `G1_unreadable_partial_state`. Thinking macro within-task AUROC 0.506 [0.470, 0.543];
  strongest control no-think 0.556; paired difference -0.049 [-0.090, -0.010]. Thinking recall@4 0.251;
  no-think 0.303; paired difference -0.052 [-0.110, +0.007]. All six gate checks failed.
- Canary: on the first eight tasks, original-prompt thinking AUROC 0.450 and task-shuffled 0.476; original minus
  shuffled -0.025 [-0.146, +0.090].

The two-task smoke had suggested thinking AUROC 0.662 but already failed its recall point threshold (lift
0.071 < 0.10). The full result reversed the AUROC signal, confirming smoke is infrastructure-only.

## Stop decision

The full depth-5 model-guided search was not authorized and was not run. No `--recompute`, gate override, or
post-hoc alternative judge was used. Banking was not scaffolded. A separate end-to-end *smoke* search was run
only to verify the reusable stopped-branch harness; its outputs are explicitly non-scientific. Every arm and
both direct compute matches completed with sealed receipts and no pool-exhaustion violation; thinking,
score-shuffle, next-op, and both direct arms all selected correctly on 0/2 smoke tasks.

## Artifact handling

The full raw model traces totaled 177,176,351 bytes and included rendered prompts, token IDs, thoughts, and
targeted-logprob diagnostics. They were moved to
`large_artifacts/qwen35_4b_partial_structure_search/calibration/`. Compact analysis-complete rows were written
back under `runs/`, and the schema-v2 receipt was sealed against their ordered IDs and checksums. Raw and
compact checksums are in `analysis/calibration_compaction.json` and the artifact manifest.

## Protocol deviations retained in the record

- The implemented calibration parents are max/min-completion live parents plus random dead parents. The design
  review's broader likelihood-frontier, uniform-frontier, hard-negative, and one-edit mixture was not built.
- Exact-depth construction uses all frozen examples to reject shallower-equivalent targets; semantic prefix
  labels use only visible plus label-probe examples.
- The initial primary-consuming oracle gate was superseded by the dedicated development gate after calibration
  infrastructure was underway but before any primary model search.
