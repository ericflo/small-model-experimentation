# Is the compositional wall STRUCTURE or VALUES? It is STRUCTURE — the model can't propose the op-sequence; once it's known, values are free

## Motivation
C31 showed the model computes the op-TYPE (structure) but reads the PARAMETER off surface I/O (values). That
raised a sharp question about the WALL itself, which the entire arc (C13–C31) never tested: when the model fails
depth-3, does it fail on the STRUCTURE (wrong op-type sequence) or the VALUES (right skeleton, wrong constants)?
If it's a value-binding failure, eliciting the skeleton + cheap value-search would crack depth-3.

## Design pivot (after adversarial review + smoke)
The natural design — ask the model to output the op-sequence — **failed**: op-seq generation solves **0.00 even
at depth-1** (the model cannot emit DSL op-sequences; a format handicap the repo had hit before). The review also
flagged the headline "skeletonfill ≫ direct ⇒ values" as a **false dichotomy** (a strong value-search fills many
wrong skeletons). So the design pivoted to answer the question cleanly on **min-depth-verified** true-depth tasks:
1. **model native Python** greedy@1 + cov@k (the baseline; op-seq format not required).
2. **model STRUCTURE-coverage** (format-immune): run each model program, check if its *behavior* matches the true
   op-type skeleton with **any** params (right structure, maybe wrong values).
3. **oracle-skeletonfill**: true op-type skeleton + enumerate params + execute-filter-on-visible + check hidden
   (ceiling: if you *know* the structure, does value-search finish?).
4. **random-skeletonfill @ R**: R random op-type skeletons, each param-filled (value-fungibility control).

## Result (min-depth-verified, n=120/depth)
| depth | mono greedy@1 | mono cov@8 | model STRUCTURE-cov@8 | value tax | oracle-skelfill | random R8/R50/R200 |
|---|---|---|---|---|---|---|
| 2 | 0.033 | 0.092 | 0.108 | +0.017 | 1.000 | 0.033 / 0.225 / 0.600 |
| 3 | 0.008 | 0.017 | **0.017** | **+0.000** | **1.000** | 0.000 / 0.042 / **0.108** |

**The depth-3 wall is STRUCTURE, decisively:**
- **No value tax.** The model's STRUCTURE-coverage (right op-type sequence, any param) **equals** its concrete
  coverage (depth-3: 0.017 = 0.017; depth-2: +0.017). Its failures are **wrong-skeleton**, not
  right-skeleton-wrong-param. There is no hidden pool of "right structure, wrong values" solutions to unlock.
- **Values are trivial given structure.** oracle-skeletonfill = **1.000** — knowing the op-type sequence, cheap
  value-search *always* finishes (consistent with C31: the param is surface-readable).
- **The DSL is not value-fungible.** Random structure barely works (R200 = 0.108 at depth-3): it's not that "any
  skeleton fills," so structure genuinely matters.

## Implication
The compositional wall is a **STRUCTURE-PROPOSAL** problem: the model can't propose which operations in which
order at depth-3; once the structure is known, values are free (oracle 1.0; C31 surface-readable). This unifies
the arc — C19 (depth-3 first-op is a representational "thread"), C25 (no step-1 lookahead), C31 (param
surface-readable) all point to the same thing: **the model reads/computes VALUES easily but cannot propose deep
STRUCTURE.** It is exactly why tool-enumerated *structure* seeds (C22) and banking (which installs structure) were
necessary, and why value-side interventions (C31 param-hint, DPO on values) don't move the wall. The deployable
recipe is structure-search (tool/enumeration) + cheap value-fill — a tool-augmented search (C12/C17 family), not a
forward-pass gain.

## Honest scope
- "Structure" = op-type sequence on the list DSL; "values" = the arity-1 parameters. Generalization to other
  substrates/parameter-types is untested.
- op-seq generation = 0.00 is a separate FORMAT failure (documented), not evidence about structural knowledge —
  the structure signal uses the model's native Python behavior.
- Depth-2 has a tiny value tax (+0.017, within noise at n=120) — even at depth-2 the wall is mostly structure.

## Artifact Manifest
See `reports/artifact_manifest.yaml`.
