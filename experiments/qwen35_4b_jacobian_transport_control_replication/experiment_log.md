# Qwen3.5-4B Jacobian Transport Control Replication Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — intake and adversarial review

- Named the invalid 48/48 context-local clamp result as the direct parent.
- Froze its exact lens, band 4–8, alpha one, prompt grammar, and model revision.
- Registered fresh calibration/confirmation mappings and two independent random
  controls per item.
- Added post-bf16 gates for realized norm (1e-5 relative) and J-span projection
  fraction (0.01), plus wrong-donor same-span specificity.
- Completed the adversarial review before implementation or any model call.

## 2026-07-12 — immutable design boundary

- Pushed design commit `27b9da2a0973dbddbdfd2b6f7acddbfc7f4f736f`
  before model inference.
- Recorded exact frozen README/preregistration hashes and the byte-identical
  parent lens hash.

## 2026-07-12 — quantization-aware control implementation

- Copied the parent's cache-free batch-one Qwen patching and coordinate code,
  then added a numeric-only post-bf16 control optimizer.
- For 32 fixed random candidates per layer, the hook alternates realized-span
  removal/renormalization with 64-step scale search and chooses the first
  candidate meeting both frozen constraints. Candidate selection cannot access
  logits or labels.
- Implemented a model smoke and the 480-layer numeric calibration gate. The
  calibration writer rejects outcome-like fields and discards every forward's
  logits before serialization.
- CPU suite passes 24 tests plus 24 subtests; no model call has occurred in this
  replication.
