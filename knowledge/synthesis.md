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
7. `Promising`: the central C2 selection wall (coverage ≫ deployable selection) is **plumbing, not a capability limit** — and thinking is the plumbing. A frozen 4B's intrinsic self-verification (black-box, no execution) is weak/yes-biased with no-think (AUROC 0.77) but strong with thinking (AUROC 0.93); its own zero-training thinking-verifier selects best-of-8 to close **75%** of the pass@1→oracle gap. So the corpus's biggest program (selection) has real headroom, and C9's thinking and C2's selection meet in *thinking-augmented verification* (which inverts C9: thinking helps recognizing correct answers ≥ producing them). But a matched-cost selector showdown (`qwen35_4b_verifier_selector_showdown`) tempers this: when a cheap visible/execution signal exists, thinking-verification is Pareto-dominated — the deployable sweet spot is **visible test + a free no-think verifier** (0.870, 83% of the oracle gap); expensive thinking-verification only earns its cost in verifier-only settings. So the C2 wall is fixable, but with *cheap* plumbing. See `qwen35_4b_generator_verifier_gap`, `qwen35_4b_verifier_selector_showdown`, and claim C10.
8. `Promising`: on a **fresh, contamination-free** program-synthesis substrate (built to test elicitation of the FIXED 4B — no teacher, no scaling), the lever that unearths latent capability is **self-training, not test-time feedback**. A neurosymbolic multi-turn REPL loop (draft→execute→real feedback→refine) does **not** beat matched-compute sampling (repl_real 0.287 vs sample_more 0.338; feedback *content* +0.024 over a paired control) — the frozen model's test-time ceiling is its sampling distribution. But QLoRA-SFT on the 4B's **own** verified solutions (no teacher) banks capability into deployable single-shot: held-out fresh greedy@1 0.224→0.319 (+0.095, ~2.2 SE, +42%), pass@5 up (no diversity collapse), two seeds. Crucially this self-training **works here but regressed on contaminated MBPP** — so contamination/substrate, not the method, likely explains the corpus's earlier self-improvement failures. Scaled into an **expert-iteration flywheel** (3 rounds) the gain **compounds** (0.267→0.356→0.385→0.393, +47%; coverage grows each round, no collapse) but with **diminishing returns** and it is **coverage-bounded** — it widens the deployable footprint of the model's own distribution without extending the frontier (depth-3 never cracks). So for a fixed small model, self-training on verified self-solutions is the working capability lever, but it saturates at the sampling frontier — extending *that* is the open problem. See `qwen35_4b_neurosymbolic_repl_substrate` and claim C11.
9. `Promising`: **the frontier IS extendable without a teacher — via tool-augmented search + banking** (answering C11's open problem). A decompose-and-compose search (the 4B ranks the next primitive, the interpreter executes it, recurse over the 23 primitives) solves compositions monolithic sampling can't. Held to the brute-force bar, the model's *guidance* buys efficiency (~2.5× fewer interpreter calls, wins low-budget) not coverage (it plateaus at the planner-wall; brute-force enumeration matches/beats it) — the crack is the composition-structure + interpreter. **Banking** the search-found solutions (QLoRA-SFT, no teacher, replicated across harvest seeds) extends the frontier *into the weights*: monolithic held-out pass@5 0.125→0.237/0.263 — the exact bound M4 could not break. **Retro-audit correction:** a behavioral min-depth audit found 40% of nominal depth-3 tasks are shallower-equivalent; re-sliced, decompose solved only **17% of TRUE depth-3** (16/16 of the collapsed ones), monolithic true-d3 = **0 corpus-wide**, and all prior "depth-3" figures were inflated by this artifact. The extension is real but modest, and *nominal composition depth is not a valid difficulty axis without a min-depth check*. See `qwen35_4b_decompose_compose_frontier` and claim C12.
10. `Promising`: **the compositional wall is hypothesis identification, not execution — the fixed 4B is a reliable compiler starved of search.** A pre-registered anatomy (`qwen35_4b_depth_wall_anatomy`) found: the destruction hypothesis died on a verified factorial grid (op type irrelevant); solve odds fall **~30× per composed op** — identification beyond the first primitive runs at only **~2× better than chance** against the 63-op space, walling at depth 2; and the three-condition discriminator is unambiguous — **plan-given execution is 0.90–1.00 through depth 4** while bare identification is ~0, and even *showing every intermediate state* barely helps (the model cannot segment chains into the depth-1 identifications it does at 0.88). The same ~2× constant independently appears as C12's guided-vs-brute search efficiency. This one mechanism retro-explains C10 (verify ≫ generate), C11 (banking coverage-bounded: SFT teaches production, not identification), C12 (decompose works by externalizing segmentation+search), and M2 (feedback fails because identification binds). Phase-3 probes refine it: pre-segmentation only partially rescues (per-step identification degrades in composite context) and no-think 2AFC discrimination is weak (~0.73) despite perfect execution — the full ladder reads execution 1.00 > discrimination ~0.73 > segmented 0.50→0 > bare ~0. And the capstone probe (P12): thinking-mode 2AFC scores 0.50 — chance, *below* the no-think read (0.73) — deliberate simulation is systematically wrong and destroys the surface signal. Final form: **the wall is broken multi-step mental simulation**; what's intact is single-step recognition (0.88) and program→code *transcription* (~1.0 — the plan-given ceiling was transcription, not semantic execution). Deployment: **tools simulate and identify; the model transcribes** — and C9 refines to: thinking helps when its content is coherent reasoning, but *hurts* when the required content is simulation the model can't generate. See claim C13.

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
