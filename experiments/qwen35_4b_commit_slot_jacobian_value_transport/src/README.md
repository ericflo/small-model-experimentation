# Source

- `task_data.py`: copied self-contained exact-depth generator with new seeds.
- `model_ops.py`: cached native traces, close-only free-form control, and exact
  full-prefill constrained plus unmasked alias-slot logits.
- `scripts/run.py`: stage/hash gates, no-thought and exact-token-multiset shuffle
  controls, seam selection/confirmation, and fail-closed later stages.

Value/patch/control code is gated on slot replication; unavailable stages fail.
