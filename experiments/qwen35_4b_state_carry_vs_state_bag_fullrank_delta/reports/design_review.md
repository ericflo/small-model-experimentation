# Adversarial Design Review

Completed before data generation or any model-bearing call.

## Verdict

**Proceed through the gated run.** The successor cleanly tests the
registered LoRA-capacity alternative only if the parent trigger, exact task
parity, target geometry, K=1 path, actual optimizer allocation, and checkpoint
round trip all pass. The largest unresolved risk is physical feasibility on the
47.5 GiB device; G0 is designed to answer that with an actual step rather than a
paper estimate.

## Failure attacks and hardening

1. **The follow-up is post-hoc architecture search.** It is not: the parent
   preregistration mandated this exact successor after a valid failure to form
   deep state. G0 reads the content-addressed, identity-bound parent summary and requires the correct parent,
   pilot phase, complete checks, reachable gate, and
   `joint_state_sufficient=false`.

2. **“Full rank” is secretly high-rank LoRA.** It is not. Each of the 62 target
   linears owns one direct `[out,in]` FP32 parameter tensor. Construction checks
   892,272,640 parameters against live Qwen module shapes. No PEFT import, rank,
   factorization, merge, or adapter checkpoint is executable.

3. **Deltas alter ordinary inference.** Hooks are context-disabled for P, the
   first R call, and C. Carry and Bag K=1 are counted independently and must each
   make zero active delta calls. Direct base logits must match to `1e-5` before
   and after the optimizer step.

4. **Zero initialization hides a path mismatch.** G0 compares enabled-zero and
   explicitly suspended K=4 paths bit-exactly, proves nonzero gradients in both
   Carry and Bag, then uses the final Bag backward for one real finite Adam update.

5. **Carry gets more parameters or calls.** Both modes use the same wrapper and
   parameter receipt. Active calls must be 186 at K=4 and 682 at K=12 for each
   mode. Only the carried-state edge differs.

6. **The task merely looks the same as the parent.** Seeds alone are not the
   proof. The config freezes canonical decompressed row hashes for every parent
   split. Preparation hashes IDs, content, and order; if parent artifacts exist,
   it also compares their canonical rows. Training refuses data without the
   pass receipt.

7. **Lazy or partial Adam allocation makes the memory estimate fictitious.** G0
   executes a schedule-correct AdamW step and requires every delta parameter to
   own finite, shape-matched FP32 `exp_avg` and `exp_avg_sq` tensors. It records
   current/peak allocated and reserved GiB plus headroom.
   Static estimates are 3.324 GiB parameters + 3.324 gradients + 6.648 moments
   = 13.296 GiB before base weights, activations, and temporaries.

8. **A tiny input understates the peak.** G0 includes K=4 backward for both arms
   and finite Carry/Bag K=12 forwards using the registered worst-case prose/depth
   geometry. It still cannot guarantee every later allocator transient; reserved
   headroom is reported diagnostically and is not a post-hoc pass threshold.

9. **Checkpoint code saves tensors but cannot restore behavior.** G0 saves delta
   and loop-state payloads, hashes them, destroys both live parameter families,
   reloads them, and requires bit-exact recurrent answer logits.

10. **Dropout changes more than capacity.** Dropout 0.05 and scale 2.0 are held
    to the parent's LoRA branch semantics. They apply only to the added delta
    branch. Evaluation disables dropout.

11. **Learning rate 2e-4 is too aggressive for full matrices.** It is intentionally
    held fixed to isolate parameterization; changing it would confound the
    capacity question. Nonfinite loss/update is a valid optimization failure of
    this registered recipe, not a license for in-directory tuning.

12. **The extra dense matmul breaks compute matching.** Carry and Bag execute the
    identical added matmuls at every extra call, so the primary counterfactual is
    matched. This successor is not compute-matched to its LoRA parent and does
    not claim otherwise; the parent comparison is about capacity diagnosis.

13. **A partial pilot is mistaken for a scientific miss or promoted into a full
    claim.** Seed 7401 only gates the transition to seeds 7411–7413. Incomplete
    diagnostics are `PILOT_INCOMPLETE`; complete failures outside joint-state
    formation are `PILOT_PROMOTION_BLOCKED`. All original complete-cell,
    state-sufficiency, unseen-K, holdout, edge-cut, and swap requirements remain
    fail-closed.

14. **A second negative leaves the user's LoRA concern hanging.** This successor
    is the direct answer only when the pilot is complete, the answer gate is
    reachable, and joint-state sufficiency specifically fails. That
    `PILOT_STATE_FORMATION_MISS` means the observed failure is not explained by
    LoRA rank under the held-fixed design. It closes that capacity branch; it
    does not warrant unregistered tuning.

15. **G4 absence is quietly treated as success.** The analyzer always reports
    sample-more and deployment unavailable. The strongest possible verdict is
    `FULLRANK_CAUSAL_DEPTH_POSITIVE`, never a deployment breakthrough.

16. **Benchmark leakage contaminates the substrate.** Source and static tests
    prohibit benchmark imports; generation is procedural. Parent parity compares
    artifacts only and imports no parent implementation.

## Required stop conditions

Stop before training on wrong/missing parent trigger, data-parity mismatch,
target-count mismatch, any K=1 delta call, K=1 logit mismatch, base gradient,
missing gradient group, nonfinite update/logit, insufficient Adam state, G0 OOM,
or failed behavioral checkpoint round trip. None is a scientific
negative about serial representations.
