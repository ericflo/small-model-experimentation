# Adversarial design review

**Initial verdict:** `BLOCK`  
**Review date:** 2026-07-13  
**Model/GPU access:** none  
**Benchmark access:** none

An independent adversarial audit reviewed the fresh-replication draft and its
copied construction code before any construction artifact or model request was
authorized. The initial design did not pass.

## Blocking findings

1. `task_data.py` still emitted the parent's unversioned task IDs. All 264 task
   IDs therefore overlapped, and copied request-domain logic would have
   reproduced 631 parent request IDs, including seven IDs in the terminal
   52-row invocation.
2. The draft required zero parent behavior-function, concrete-triple, and
   two-operation-suffix reuse. This is infeasible under the frozen finite DSL:
   only 49 four-live function groups exist, and parent-function rejection left
   five usable groups for a 44-task quota. A model-free attempted construction
   failed closed with `exact task pool exhausted before quotas: {'quad': 39}`.
3. A seed-only fresh construction is substantively new at the task-instance
   level but necessarily reuses parts of the finite grammar. The audit measured
   56 shared behavior functions after re-evaluating parent triples on the fresh
   panel, 41 shared concrete triples, and 181 shared suffixes. It also measured
   zero exact public task payloads and zero terminal prompt texts before task-ID
   namespacing.
4. The copied fresh-panel capacity test froze parent values. The actual fresh
   panel has 354 shallow functions, 3,526 exact-depth-three groups, and group
   histogram `{1: 1553, 2: 1601, 3: 323, 4: 49}`.
5. The successor bound only two incident hashes in its administrative scaffold
   and lacked a full fail-closed lineage verifier. The construction,
   preoutcome, and implementation lock must authenticate the complete parent
   incident chain.

## Required response

- Make the declared namespace operational in task IDs and every later request
  seed key/record ID.
- Replace the impossible parent function/triple/suffix rejection with measured,
  descriptive reuse. Preserve within-successor function/triple/suffix
  disjointness and retain rejection of compatible unrelated prior art.
- Gate zero parent overlap in identity-free public instances, model-facing
  mechanics prompts, task IDs, and later actual request IDs/seed keys/derived
  seeds.
- Authenticate an exact 13-file parent lineage allowlist, reject symlinks,
  absent files, schema drift, inventory drift, malformed JSONL, and hash drift.
- Freeze the actual fresh-panel capacity and add mutation/fail-closed tests.
- Re-review the corrected implementation before setting the design boundary to
  passed or running construction.

## Resolution state

Corrections are in progress. This document does not authorize construction or
model use. A follow-up independent verdict must be appended after the tests and
freshness receipt implementation are complete.
