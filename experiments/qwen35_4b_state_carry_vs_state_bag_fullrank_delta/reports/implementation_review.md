# Implementation Review

**Review state: CPU-complete, GPU-unverified.**

- The direct delta bank owns one FP32 full-shape linear per discovered Qwen
  target and keeps target references non-owning. Hooks are explicitly scoped.
- Live construction rejects any target count/shape drift and zero-init mismatch.
- K=1 uses raw first-R state and the direct frozen coda; delta call counters and
  direct-model parity guard both Carry and Bag.
- Checkpoints separate `delta_state.pt` from `loop_state.pt`, bind both hashes to
  source/config/model/runtime identity, and verify the live target manifest.
- Pilot checkpoints persist the exact G0 licensing receipt lineage; full
  checkpoints persist both G0 and G1 lineage. Loading and analysis require the
  exact phase-appropriate lineage shape and pass statuses.
- G0 includes actual backwards for both arms, the scheduled first Adam step,
  exact per-delta FP32 Adam moment audits, reserved-memory headroom, post-step
  K=1/K=12, and observable two-payload/logit restoration.
- The prior counterfactual `b_to_a` tuple arity and duplicate checkpoint-raise
  regressions have static tests.
- Full data preparation is content-bound to frozen parent rows; model-bearing
  stages recompute current row receipts and the full frozen parity metadata.
  Reduced smoke data is explicitly nonconfirmatory and cannot enter such a
  stage.
- CPU unit, compilation, analysis, generator, and static-contract tests pass.

Not yet verified: real target discovery against the pinned model, realized GPU
memory/headroom, Adam allocation on the target environment, or any scientific
metric. Those are deliberately left to G0 and later authorized stages.
