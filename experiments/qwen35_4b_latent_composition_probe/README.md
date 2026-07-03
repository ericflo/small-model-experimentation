# Qwen3.5-4B Latent Composition Probe: is the wall representational or expressive?

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: when the model fails to identify a composition, is the answer linearly PRESENT in its
  activations (latent, an expression failure) or ABSENT (a representation/information gap)?
- Prior anchors: C13/C16 (the wall is generation), C17 (coverage not selection), C18 (banking installs it).

## Question

The C13–C18 arc measured the generation wall only behaviorally. Look INSIDE: can a linear probe on
residual-stream activations decode the composition the model cannot generate?

## Hypothesis

Pre-registered (`reports/prereg.md`): depth-1 probe ≥ 0.80 (methodology); depth-3 probe ≥ 3× chance and ≥
behavioral naming + 0.15 (LATENT) or ≈ chance (ABSENT); monotonic decay with depth.

## Setup

- Model: Qwen3.5-4B (only permitted model). Inference only (forward passes + linear probes).
- Tasks: fresh verified-depth, collapse-rejected `list`, depths 1/2/3, 500 each, disjoint.
- Capture: last identification-prompt-token residual stream at every layer (`gen_lib.activations`,
  `[1500, 33, 2560]`). Probe: standardize → PCA(128) → L2 logistic, stratified 70/30 held-out; decode first-op.
- Baselines: chance, shuffled-label (overfit floor), layer-0 (surface), behavioral first-op naming +
  identification pass@1 (150 tasks/depth).

## Run

Smoke: `python scripts/capture.py --smoke && python scripts/probe.py`
Full: `python scripts/capture.py --n-per-depth 500 --n-behavioral 150 --depths 1 2 3 && python scripts/probe.py && python scripts/analyze.py`

## Results

**The wall's nature changes with depth.** Depth 1: first-op probe **0.99** vs model naming 0.44 / generation
0.68 → representation ≫ expression (latent). Depth 2: probe 0.42 vs behavior ~0.13. Depth 3 (the wall): probe
0.27 but shuffled floor 0.14 → real signal ~0.13 ≈ behavior. So EXPRESSION failure when shallow, REPRESENTATION
failure when deep. See `reports/report.md`, `analysis/latent_probe.png`, `runs/probe_results.json`.

## Interpretation

Activation steering has headroom at depth 1–2 (info present, unexpressed) but nothing to steer toward at the
deep wall (info not computed). Explains why banking (C18) was necessary — it installs the representation the
base lacks. Only proposal-installation, not readout, crosses the deep wall. Verdict: GRADIENT/crossover.

## Knowledgebase Update

- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C19)
- Claim ledger updated: C19 added

## Artifacts

- `scripts/capture.py` (activations + behavioral), `scripts/probe.py` (linear probes), `scripts/analyze.py`
- `data/{present.npy, labels.json, tasks.jsonl}`; `data/acts.npy` (241MB) moved out of repo (regenerable)
- `runs/probe_results.json`, `analysis/latent_probe.png`, `reports/prereg.md`, `reports/report.md`
