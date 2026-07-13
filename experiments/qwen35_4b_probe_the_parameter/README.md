# Qwen3.5-4B: Is the Parameter Latent? (probe the full first op)

**Status:** finished

## Research Program
- Program: `interpretability_and_diagnostics` / `structured_execution_and_compilers`
- Question (C30 follow-up): C30 found the deployable bottleneck is the concrete first op's PARAMETER, not the op-TYPE C19 decodes. Is the parameter model-LATENT (elicitable) or just surface-readable off the I/O?

## Setup
- Fit 16-way op-TYPE + 32-way CONCRETE-op probes on residual activations (600/depth training).
- REAL surface control (the layer-0 probe is degenerate under RoPE): an external classifier on raw I/O features (lengths, sums, min/max, elementwise diffs) with NO 4B.
- Decodability on a large fsig-disjoint eval (activation-only); deployability arms (n=130/depth) split by param vs non-param first ops: no-hint, oracle_type, oracle_full, probe_full, surface_full, wrong_param.

## Run
`python scripts/fit_probe_full.py --n-per-depth 600` then `decode_eval.py` (decodability + surface baseline) then `run_hints_full.py` (deployability) then `analyze.py`.

## Results
The op-TYPE is MODEL-LATENT (probe 0.41 > surface 0.27) but the PARAMETER is SURFACE-READABLE (probe 0.49 vs surface 0.53). Deployability: the param is the bottleneck (oracle_full 0.095 >> oracle_type 0.007) but the cheap surface pipeline (0.027) delivers more than the model probe (0.014). See `reports/report.md`, `analysis/probe_the_parameter.png`.

## Interpretation
Sharp localization: the forward pass COMPUTES the op-type (latent, elicitable) but only READS the parameter off surface I/O -- no privileged model knowledge to elicit. The training-free latent-elicitation ceiling is the op-type.

## Knowledgebase Update
- Claim ledger: C31

## Artifacts
- `scripts/fit_probe_full.py`, `scripts/decode_eval.py` (external-I/O surface baseline), `scripts/run_hints_full.py`, `scripts/analyze.py`, `scripts/capture.py`
- `data/{train_fsigs,concrete_vocab}.json`, `runs/{decode_results,full_results,verdict}.json`, `analysis/probe_the_parameter.png`, `reports/{report,design_review}.md`
- Probe pickle + activations moved out of repo.
