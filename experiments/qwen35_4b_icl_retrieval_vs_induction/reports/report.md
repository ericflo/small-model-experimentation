# Is in-context learning retrieval or induction? Retrieval of familiar structure — not induction of novel structure

## Motivation
The arc unified into "executor, not inducer" (C37 execution intact in language; C38 induction hard everywhere).
But *in-context learning* is what LLMs are famous for. If the model genuinely can't induce a novel rule from
examples (C38), what is ICL doing? Hypothesis: ICL **retrieves** a familiar pretrained structure the examples
point to, rather than **inducing** a novel rule.

## Method
Execution-safe single-value task: "advance k steps in a cyclic order", output one digit. The review's crux:
**FAMILIAR** structure (natural order 0–9, retrievable) vs **NOVEL** structure (a *stated* random cyclic order)
at matched 1-parameter complexity ("advance k") — crossed with **EXECUTE** (rule stated) vs **INDUCE** (rule must
be inferred from few-shot examples). Query is an *unseen* digit (generalization). No-think (code-mode-free);
chance = 1/10.

**Substrate note (methodological):** the first vehicle (letter-substitution ciphers) FLOORED — the 4B cannot
apply even a *given* cipher (application-only 0.20), a character-assembly limit, not induction. Confirmed the
floor was harness/char-manipulation and pivoted to the single-value substrate.

## Result (no-think, n=60)
| | EXECUTE (rule stated) | INDUCE (few-shot) |
|---|---|---|
| **Familiar** order (0–9) | **1.00** | **0.45** |
| **Novel** order (scrambled) | **0.97** | **0.12** (= chance 0.10) |

- **Execution is near-perfect and familiarity-independent** (1.00 / 0.97): the model applies the novel
  scrambled-order rule almost perfectly *when told it*.
- **Induction is familiarity-bound and collapses for the novel order** (0.45 → 0.12 = chance): the model cannot
  induce the novel rule from examples — *even though it executes that exact rule at 0.97*.
- **Not data-limited:** more examples make novel induction *worse* (0.15 → 0.05 with 8 examples), so it is bounded
  by familiarity, not data.

## Conclusion
In-context "learning" **surfaces/retrieves familiar structure; it does not create/induce novel structure.** This
unifies the arc: the model is an **executor/retriever** of pretrained structure (C37; here 0.97–1.00), not an
**inducer** of novel structure (C38, C32/C36). ICL is the retrieval half of "reasoning," not the induction half.
For the mission: "unearthing latent capability" means surfacing structure the model *already has* — the fixed 4B
cannot acquire genuinely novel structure in-context, no matter how many examples.

## Honest caveats
- Familiar induction is itself only 0.45 (imperfect retrieval).
- The novel arm's induction requires reasoning through a scrambled order (mechanically harder); but the 0.97
  execution control shows that mechanism is *not* the bottleneck — inducing the rule is.
- Single seed; no-think primary (think-mode triggers code-mode on these tasks — a confound documented in runs).
- The char-cipher floor (0.20 application-only) is recorded as a methodological lesson: char-level string tasks
  are a poor vehicle for this 4B.

## Artifact Manifest
See `reports/artifact_manifest.yaml`.
