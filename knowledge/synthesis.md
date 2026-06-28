# Cross-Track Synthesis

This synthesis is deliberately evidence-linked. It should be updated whenever a new result changes the recommended research direction.

## Executive Takeaways

1. `Confirmed`: executable or structured intermediates are the strongest repeated path to lift. They work best when the experiment can use visible execution, typed slots, latent registers, or candidate programs as evidence instead of asking the small model to directly emit the final answer.
2. `Confirmed`: candidate generation is often easier than deployable selection. Several experiments find additional hidden-correct candidates, but public evidence or weak rerankers fail to choose safely.
3. `Confirmed`: controls are not optional. Shuffled labels, random retrieval, corrupted memories, and probe ablations repeatedly prevent overclaiming.
4. `Promising`: external memory, retrieval, operator banks, and active evidence can add coverage, but the useful version is evidence-conditioned and verifier-aware rather than plain retrieval into a prompt.
5. `Open`: the next frontier is not one new trick. It is a repeatable loop: generate diverse candidates, gather stronger deployable evidence, train or search with that evidence, and feed failures back into the library.

## Evidence

### Structured Execution And Latent Programs

The structural latent compiler line shows that small models can benefit from explicit executable structure. In [qwen_structural_latent_compiler_expansion](../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md), a Qwen-attached compiler expanded to 24 slots and reached 82.8% on the standard length-24 split, 100.0% on the paraphrase length-24 split, and 93.8% on paired length-24. This supports continuing typed-slot, latent-register, and state-supervised compiler work.

The Foofah fallback experiments point in the same direction from a different substrate. [qwen35_4b_foofah_selective_program_fallback](../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md) improved strict held-out table transformation accuracy from 55.2% direct JSON to 62.4% by committing a visible-passing executable program, with no direct-correct losses in that candidate pool.

The operator inventory pilot makes the search-side result even clearer. [qwen35_4b_operator_inventory_search_pilot](../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md) found 100.0% target coverage with the full operator inventory and reached 100.0% held-out selection with a small active-query budget. The lesson is that growing and disambiguating the operator bank can be more valuable than asking the model to infer everything from type signatures.

### Selection Is The Bottleneck After Coverage

[qwen35_4b_retrieval_adapt_verify_scale](../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md) is the cleanest example. Semantic retrieval plus adaptation recovered 8 of 24 residual direct-sampling misses and could raise all-task coverage from 70.0% to 80.0% under an oracle selector. But the deployable selectors still committed many hidden-wrong visible-pass candidates. The result is positive for external algorithmic memory and negative for weak public-test selection.

This should guide future work away from "retrieve and hope" and toward stronger evidence: generated counterexamples, independent implementation consensus, public-test augmentation, process verifiers, and calibrated abstention.

### More Context Helps Modestly Unless The Selection Signal Improves

[qwen_active_example_acquisition](../experiments/qwen_active_example_acquisition/reports/qwen_active_example_acquisition_report.md) improved full-task exact from 66.7% to 70.0% with one actively selected example, while the oracle among tested acquisitions reached 73.3%. That is useful but not transformative; it says additional examples can help, but the acquisition policy must find genuinely informative evidence.

[qwen_verified_skill_memory_rag](../experiments/qwen_verified_skill_memory_rag/reports/qwen_verified_skill_memory_rag_report.md) is a valuable negative result. Top-1 same-family retrieval was often non-random, but the skill-card RAG method scored 47.5% full-task exact versus 50.0% for direct row inference. This argues against assuming that analogous verified examples automatically improve a frozen model.

### Negative Controls Are A Research Asset

Several experiments are valuable because they disproved tempting shortcuts:

- Foofah probe agreement did not improve fallback selection; simple visible execution was stronger in the tested pool.
- Skill-memory RAG did not beat direct row inference under top-1 retrieval.
- Retrieval adaptation found hidden-correct candidates, but public visible-pass evidence was not enough to choose safely.
- Active acquisition had a real but small lift, with random/order/diversity controls showing the limits.

Future reports should keep these failed mechanisms visible. They define the boundary between plausible and already-tested ideas.

## Working Model

The strongest shared model of the corpus is:

1. Small models can often generate or score useful partial structure.
2. External structure can expose more candidates than direct answer generation.
3. The hard part is choosing under deployable evidence, not hidden oracle evidence.
4. The most reliable progress comes from making the evidence channel stronger, then training or searching against that channel.

## Implications

- Prefer experiments that convert hidden oracle wins into deployable evidence.
- Treat visible-pass as useful but insufficient until false-pass rates are measured.
- Add random, shuffled, corrupted, or frozen controls before interpreting a gain.
- Record oracle ceilings separately from deployable selectors.
- Build reusable evaluation surfaces only after several self-contained experiments converge on the same need.

