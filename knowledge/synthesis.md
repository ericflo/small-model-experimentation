# Cross-Program Synthesis

This synthesis is deliberately evidence-linked. It should be updated whenever a new result changes a research program, opens a new program, or retires an old hypothesis.

For the machine-checkable claim ledger, use [claims/index.md](claims/index.md).

## Executive Read

1. `Confirmed`: the imported corpus is a prototype for a broader research operating system. Its main value is not its original source structure; it is the set of reusable patterns it exposes for structured execution, candidate selection, active evidence, memory, posttraining, tool control, diagnostics, and infrastructure.
2. `Confirmed`: executable or structured intermediates are a strong recurring mechanism, but future work should compare representations and supervision sources instead of only producing more local wins.
3. `Confirmed`: candidate generation is often easier than deployable selection. Evidence-conditioned selection should be treated as a first-class research program.
4. `Promising`: memory, retrieval, operator banks, and active evidence can add coverage, but only when tied to verification, constraints, or selection. Plain prompt context is not enough.
5. `Open`: the most valuable next work is portfolio growth: new programs, new substrates, new diagnostics, and new evidence loops that make future experiments less likely to repeat old mistakes.
6. `Promising`: the corpus's biggest self-imposed blind spot was running the model only in **no-think** mode. Turning native thinking on is a real deployable lever (MBPP greedy +15pp) and, notably, here it moves the *deployable* line more than the oracle ceiling — so the central C2 selection bottleneck does **not** hold for the thinking axis. But thinking is a *budget* with an overthinking cost; and content controls (foreign/filler/shuffle across budgets 512/1024/2048) show the model uses thinking as *content* — irrelevant thinking collapses accuracy to ~4%, pure compute (contentless filler) ≈ no-think, and coherent reasoning is the entire gain, growing with budget (real−shuffle +0.105→+0.150). So the thinking accuracy gain is genuine coherent reasoning content at *every* budget; the "thinking ≈ compute" reading only appeared through a greedy-metric lens. See [test_time_reasoning_budget](../research_programs/test_time_reasoning_budget/charter.md) and claim C9.

## How To Read Prior Results

Do not read the imported tracks as a closed agenda. Read them as seed data for research-program design:

- A successful mechanism becomes a program hypothesis.
- A failed control becomes a warning label.
- A repeated bottleneck becomes a new program or backlog item.
- A useful artifact pattern becomes infrastructure.

## Program-Level Claims

### Structured Execution Is A High-Value Mechanism

Seed evidence includes [qwen_structural_latent_compiler_expansion](../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md), [qwen35_4b_foofah_selective_program_fallback](../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md), and [qwen35_4b_operator_inventory_search_pilot](../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md). The program-level lesson is not merely "use programs." It is that explicit execution surfaces give small models something checkable.

### Selection Under Visible Evidence Is A Core Bottleneck

[qwen35_4b_retrieval_adapt_verify_scale](../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md) found additional hidden-correct candidates, but deployable selectors still made too many wrong commits. Future work should treat selection as its own research object, with precision, recall, abstention, and hidden-oracle ceilings separated.

### Memory Must Be Mechanistic

[qwen_verified_skill_memory_rag](../experiments/qwen_verified_skill_memory_rag/reports/qwen_verified_skill_memory_rag_report.md) is a negative seed result for naive skill-card prompting, while retrieval-adaptation experiments show candidate coverage can improve. The useful memory program should test memory as candidates, constraints, tests, invariants, and evidence, not just prompt examples.

### Active Evidence Needs Downstream Coupling

[qwen_active_example_acquisition](../experiments/qwen_active_example_acquisition/reports/qwen_active_example_acquisition_report.md) shows modest lift from extra examples, but the larger opportunity is to couple acquisition to selection and verification.

### Infrastructure Is A Research Program

The repository itself needs to improve as experiments accumulate. Program charters, generated indexes, claim ledgers, artifact rules, and validation gates are not bureaucracy; they are what let many future agents build on shared memory instead of restarting.

### Native Thinking Was The Corpus's Blind Spot

[qwen35_4b_thinking_budget_scaling](../experiments/qwen35_4b_thinking_budget_scaling/reports/report.md)
turned on the native reasoning mode the corpus universally disabled and swept the thinking-token
budget on MBPP. Native thinking lifts deployable greedy pass@1 0.76→0.91; the deployable gain
exceeds the oracle-ceiling gain and narrows the selection gap, so the corpus's central C2 pattern
does not hold for this axis. Two cautions travel with it: accuracy is non-monotonic in the budget
(overthinking hurts; `unbudgeted` is a poor default), and a shuffled-thinking control reproduces
much of the gain (so a large share is compute + scaffold + token-presence, not coherent reasoning).
A **foreign-task-thinking control**
([qwen35_4b_thinking_content_vs_compute](../experiments/qwen35_4b_thinking_content_vs_compute/reports/report.md))
then **corrected** that caveat: the model *uses thinking as content* — splicing a different task's
thinking collapses accuracy to ~4% (it solves the wrong problem), a contentless filler arm ≈ no-think
(pure compute buys ~0), scrambled relevant thinking ≈ no-think, and coherent thinking is the entire
gain. A budget sweep (512/1024/2048) then showed the coherence advantage **grows** with budget
(+0.105→+0.150), refuting the overthinking-washout idea and exposing the scaling run's "2048 shuffle ≈
real" as a shuffle-protocol artifact. So **the behavioral gain is genuine coherent reasoning content at
every budget**; the "mostly compute/scaffold" read only appeared through a greedy-metric lens (the
separability/representational slice is separate and noisy). Strategic implications: re-baseline
CoT-substitute results against a fair, budgeted native-thinking baseline; and judge "is it reasoning?"
with explicit content controls — a single behavioral or greedy number misled here.

## Portfolio Implications

- Start with a program question, not an isolated run idea.
- Preserve self-contained experiments, but connect every result upward into program evidence.
- Prefer experiments that distinguish between mechanisms.
- Report oracle-only and deployable evidence separately.
- Add new programs when a line has durable uncertainty and multiple plausible experiments.
- Retire or demote lines when controls repeatedly contradict the mechanism.
