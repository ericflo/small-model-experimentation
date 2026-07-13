# Qwen3.5-4B: Externalize the Latent Readout (probe-to-prompt)

**Status:** finished

## Research Program
- Program: `interpretability_and_diagnostics` / `structured_execution_and_compilers`
- Question: C19 showed the first-op is linearly DECODABLE from the base residual but C20 showed it is NOT steerable. Can we EXTERNALIZE the readout instead -- decode it with the probe and inject it as a PROMPT hint (shift the proposal, the lever C17 allows)?

## Setup
- Refit C19's linear first-op probe (standardize+PCA128+L2-logistic) on 1500 training tasks (replicates C19).
- On FRESH fsig-disjoint eval tasks (n=100/depth 2,3), decode the first-op from the base model's own activation; generate under 6 arms: no-hint, neutral(placebo), oracle-type, oracle-full(+param), probe(decoded), wrong(random). Metrics greedy@1 + coverage@6, no-think.

## Run
`python scripts/fit_probe.py --n-per-depth 500` then `python scripts/run_hints.py --n-per-depth 100 --depths 2 3` then `python scripts/analyze.py`.

## Results
Externalization ELICITS deployable depth-2 where steering (C20) failed -- oracle_full lifts depth-2 greedy@1 6x (0.03->0.19). BUT the deployable bottleneck is the PARAMETER, not the op-TYPE C19 decodes (oracle_type lifts coverage only); the type-only probe nets to ~zero. Graded by depth (fades at depth-3, thread). Controls clean (neutral~=no-hint, wrong hurts, layer-0 at chance). See `reports/report.md`, `analysis/probe_to_prompt.png`.

## Knowledgebase Update
- Claim ledger: C30

## Artifacts
- `scripts/fit_probe.py` (fits + saves C19 probe, mid-layer + layer-0), `scripts/run_hints.py` (6-arm hint experiment + leak control), `scripts/analyze.py`, `scripts/capture.py` (from C19)
- `data/train_fsigs.json`, `runs/hint_results.json`, `runs/verdict.json`, `analysis/probe_to_prompt.png`, `reports/{report,design_review}.md`
- Probe pickle + activations moved out of repo.
