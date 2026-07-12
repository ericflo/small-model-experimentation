# Qwen3.5-4B Jacobian Value Transport Report

## Status

Pre-run scientific state. No scientific result has been observed. The immutable
design boundary, tokenizer/layout audit, targeted vector-Jacobian product, and a
cache-free coordinate-write smoke all pass. The smoke result is explicitly
non-scientific.

## Research Program Fit

This is a causal diagnostic with an explicit decision: determine whether a
Jacobian-aligned coordinate intervention can cross the representation/expression gap
left open by C20. It does not claim that readable coordinates are capabilities.

## Method

The experiment fits token-targeted and, conditionally, full averaged Jacobians at
selected residual layers; validates coordinate writing on direct and consequential
positive controls; estimates natural think-prefix value from exact-verifier sibling
continuations; and finally compares causal coordinate patches with the complete
frozen control set.

## Results

The real-model smoke used the pinned Qwen3.5-4B revision on an RTX 6000 Ada. Four
single-token concepts produced finite nonzero pullback directions at layers 8, 16,
and 24. Peak allocated memory was 9.25 GiB. A cat-to-dog coordinate swap at layer 16
produced a nonzero mean residual delta norm of 0.4045 without cache use. These checks
validate plumbing only and cannot satisfy G0.

The preregistered 64-prompt targeted fit also completed: 24 single-token
concepts at five source layers produced 120 finite, nonzero directions using
equal weighting over valid causal source/target pairs. It took 29.3 seconds and
12.3 GB peak allocated memory. This establishes the fitted intervention artifact;
it is not an outcome result.

## Controls

Frozen baseline, logit lens, ActAdd, raw donor, sparse J component, non-J remainder,
wrong donor, shuffled outcome, and norm-matched random conditions are preregistered.

## Oracle Versus Deployable Evidence

Correct-operation directions, reference margins, and high-value donors are oracle
diagnostics. A successful oracle causal gate would start a separate non-oracle
experiment; it would not itself meet the repository's deployable-capability bar.

## Interpretation

Pending the terminal gate. No inference should be made from design artifacts alone.

## Next Experiments

Conditional only: a specific G2 pass opens a non-oracle activation controller and
verifier-grounded counterfactual-reflection experiment. Any failure selects the next
mechanistic branch documented in the terminal report.

## Artifact Manifest

See `artifact_manifest.yaml`; large Jacobians and activation caches are external and
must be checksummed before a result is reported.
