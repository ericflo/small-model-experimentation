# Preregistration implementation amendment 2 — memory-feasible microbatching

Frozen after an apex replay training preflight failed and before any adapter/checkpoint was saved or any trained evaluation ran.

The registered batch 4 × gradient accumulation 4 geometry reached optimizer step 52, then a 3,193-token microbatch required a 9.54 GiB logits allocation with only 9.09 GiB free on the 48 GB GPU. The process stopped with CUDA OOM. Its output directory contains only the deterministic pre-training encoding receipt; no adapter, checkpoint, evaluation, or benchmark result exists.

All arms now use microbatch 2 × gradient accumulation 8. This preserves:

- effective batch size 16;
- exactly 584 optimizer steps and 9,344 examples processed;
- exactly three deterministic apex padding duplicates, hence exactly two apex control epochs;
- identical ordering algorithm, seed, learning rate, model, rank, target weights, and repository/C54 data;
- source-homogeneous microbatches and complete four-operator repository task blocks across paired microbatches.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is also set before importing Torch to reduce fragmentation. This is a compute-equivalent feasibility correction caused solely by an observed memory allocation failure, not an outcome-driven recipe change.
