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
