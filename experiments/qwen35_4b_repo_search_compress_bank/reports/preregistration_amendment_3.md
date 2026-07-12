# Preregistration implementation amendment 3 — exact chunked loss

Frozen after the memory-safe 2 × 8 fallback completed three unsaved preflight steps and before any adapter/checkpoint or trained evaluation existed.

Microbatch 2 × accumulation 8 avoids the allocation failure, but measured ~19–23 seconds per optimizer step because it doubles forward/backward microbatches. Inspection pinned the batch-4 OOM to `cross_entropy`, not the model forward: dense loss upcast every sequence position across the 248,320-token vocabulary and requested a single 9.54 GiB allocation.

The final implementation restores the originally registered batch 4 × accumulation 4 geometry and replaces only dense loss materialization with exact 128-position chunks. Each chunk's softmax is gradient-checkpointed, so its FP32 temporary is discarded and recomputed during backward instead of retaining all positions at once. A CPU regression test proves scalar loss and every logits gradient match ordinary dense weighted cross-entropy to 1e-6, including negative and zero weights.

This preserves the original example order, effective batch, optimizer steps, padding, target weights, learning rate, and objective exactly. Expandable CUDA segments remain enabled. Neither failed/preflight attempt saved a model or produced an evaluation.

Before the result-bearing rerun, a dedicated two-step canary used the eight longest encoded apex rows (maximum 3,193 tokens) to exercise the formerly failing allocation path at batch four. It passed both steps, peaked at 48,375,846,912 CUDA bytes, and saved only a smoke-tagged adapter that is forbidden from evaluation. The receipt is [loss_stress_receipt.json](loss_stress_receipt.json).
