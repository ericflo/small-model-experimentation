# Preregistration amendment 12: capacity-fit long-context scheduling

Date: 2026-07-10. Frozen while the max-seqs-64 K=4 base probe at think@49,152 was still
inside vLLM, before its receipt existed and before any termination result, token IDs, decoded
output, parser result, or score was available.

## Pre-result observation

The engine-start audit reported 995,328 KV-cache tokens. The installed vLLM 0.24 scheduler admits
up to `max_num_seqs`, but when a running request cannot allocate another cache block it frees that
request's blocks, resets its computed-token count to zero, and requeues it. Prefix caching is off,
so this is full-prefix recomputation rather than a harmless scheduling pause.

The active engine used `max_num_seqs=64`. That value was calibrated for think@16,384, not the two
longest rungs. At think@49,152 the largest complete solver sequence is 50,726 tokens, so at most 19
such sequences fit (`19 * 50,726 = 963,794`; 20 require 1,014,520). At think@61,440 the registered
global context maximum is 65,432 tokens, so at most 15 fit (`15 * 65,432 = 981,480`; 16 require
1,046,912). The active K=4 probe exposes 48 sequences and a complete K=12 arm exposes 144. The
max-seqs-64 protocol therefore permits predictable cache overcommit and recomputation at both long
rungs.

This is a throughput/protocol defect discovered from engine capacity, registered prompt lengths,
installed scheduler source, and prior wall-time/token accounting. It does not depend on the active
probe's generated tokens or quality.

## Frozen repair

1. A receipt watcher may stop the active process group only after the think@49,152 probe has
   atomically committed its last-written receipt. It may not interrupt or inspect the in-flight
   generation.
2. Preserve that max-seqs-64 probe verbatim as a non-scored, non-selectable
   `scheduler_overcommitted` diagnostic. It cannot choose a budget, authorize a matrix, be promoted,
   be pooled, or be decoded/scored.
3. Do not run think@61,440 or a complete matrix with max-seqs 64.
4. Continue in the independent follow-up
   `qwen35_4b_verified_macro_capacity_fit_rerun`, with a fresh external artifact root and protocol
   binding. Repository policy requires the concurrency design variant to have its own experiment.
5. Freeze the capacity-fit map to `max_num_seqs=19` at think@49,152 and `max_num_seqs=15` at
   think@61,440. Keep max batched tokens 32,768, synchronous scheduling, prefix caching off, the
   model/revision, prompts, tasks, libraries, seeds, sampling, answer allowance, K values,
   termination classifier, and decision thresholds unchanged.
6. Start the follow-up with a fresh K=4 base-only probe at think@49,152. A probe is never promoted
   or scored. At the first adequate probe, generate a fresh complete K=12 base/designed matrix at
   the same rung and the same mapped concurrency for both arms. Only that complete, adequate matrix
   may reach semantic analysis.
7. The max-seqs-64 16k and 32k arms remain valid rejected-rung evidence under their original
   protocol, but no row crosses into the capacity-fit experiment.

## Interpretation boundary

Concurrency changes dynamic batch composition and can change sampled trajectories on this Ada GPU,
even with fixed per-request seeds. Capacity-fit outputs are therefore independent protocol rows,
not continuations or replacements for max-seqs-64 rows. This amendment repairs wasted recomputation;
it does not alter the verified-macro hypothesis or create evidence for or against macro utility.

If the tuned 61k probe is inadequate, the follow-up must branch on the failing termination gate.
More context is justified only for unresolved reasoning-boundary contacts; excess exact loops call
for a separate symmetric loop-control experiment, and excess answer-limit contacts call for a
separate answer-envelope experiment.
