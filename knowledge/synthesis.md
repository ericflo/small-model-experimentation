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
11. `Promising`: **repairing a broken primitive does not propagate — capability is format-local** (`qwen35_4b_simulation_keystone_repair`, pre-registered intervention test of C13). SFT on interpreter-generated state chains *fully repaired* the simulator (0.8+ through depth 5, length-generalizing, held-out-primitive transfer) — yet the inverse ladder did not move (bare 0.08→0.09). The only transfer observed follows **format adjacency** (I/O→code training tripled segmented identification, same output format), and narrow-format SFT causes **format capture** (adapters answer unrelated questions in their trained format, crashing 2AFC and taxing transcription). So: capability in the fixed 4B is organized by input→output format mappings, not shared internal primitives; mechanism diagnoses do not license training-transfer predictions; banking (C11/C12) teaches mappings, not components. Bonus: thinking *helps* single-pipeline simulation (length-fragile, not globally wrong — refines P12). See claim C14.
12. `Promising`: **the installation-mechanism triptych is complete — deployable capability = module × interface × procedure** (`qwen35_4b_context_composition`, pre-registered, same tasks/decoys as the keystone). CONTEXT composes: an explicit simulate-both-compare procedure lifts base thinking-2AFC to 0.83 (flat through depth 4; ICL demos only +0.04), and under that procedure the SFT-installed simulator reaches **0.95 parse-conditional** (+12pp over base) — the module was never sealed; **format capture is an interface failure** (parse 0.53 → deployable 0.51). But **hypothesis generation is un-composable**: no context strategy moves bare identification (0.08–0.13) — only external enumeration crosses that wall. Retro-correction: C13's P12 ('thinking hurts discrimination') was a budget+parser artifact — thinking merely fails to beat the surface heuristic *without a procedure*. Final elicitation recipe: **tools generate, context orchestrates, the model simulates and transcribes** — and SFT-installed modules need interface repair (mixed formats) to be deployable. See claim C15.
13. `Stopped`: **oracle-searchability does not imply model-readability** (`qwen35_4b_partial_structure_search`, unclaimed pending ledger re-grade). A type-only live-prefix oracle compresses exact depth-5 search 262,144x and preserves 12/12 development paths, but thinking P(viable) is chance within task on 7,200 children (AUROC 0.506, recall@4 0.251), below no-think (0.556/0.303), and wrong-task evidence is no worse. Pooled AUROC 0.557 is task difficulty, not sibling guidance. The Recognize -> Search -> Bank line correctly stops before search/banking. Exact visible-only depth-5 brute also finishes 60 tasks in 112 seconds and covers 60/60, so first locate the depth-6 resource crossover; if guidance becomes necessary, expose parameter domains and residual state rather than retrying type names with more thinking.

13. `Promising`: **the first blackbox-arbitrated install — breadth defeats locality at the emission seam** (`qwen35_4b_gauntlet_breadth_round1`, claims C49/C50, 2026-07-10). A firewall-clean 12-family agentic gym (invented content aligned only to menagerie's public axis descriptions; machine-checkable verifiers) + fast expert-iteration loop moved **menagerie quick from 0.140→0.363 and 0.152→0.446 on two fresh paired seeds** (HF backend, deterministic) — the first method in the corpus ever measured to move the held-out instrument, at ~7-9× the pre-registered bar. Gym-internal mean rose 0.184→0.701 **including two never-trained held-out families (+0.54/+0.61)** and a family with zero training examples (+0.40): the C43/C45/C48 substrate-locality laws do not extend to this regime. Two mechanism lessons: (a) full-weight SFT on the model's own verified naturally-closed chains installs ~nothing (near-self-distillation); the working recipe canonicalizes targets to the terse deployable answer, includes forced-close **recovery** examples (truncated chain as context, loss on the commit), and concentrates loss on the answer/action emission seam (think 0.2/answer 1.0) — **signal placement beat dose**; (b) the binding deployed constraint at these difficulty levels is the truncation cascade (consume any budget → force-close → verbose restart → no parseable answer), and repairing commit-from-partial-reasoning transfers substrate-generally. **Instrument hazard (C49, Confirmed): vLLM 0.24 runtime LoRA silently does not apply Qwen3.5-4B PEFT adapters** (name-tree mismatch, no error) — every vLLM adapter arm ever run measured the base model; gate adapter arms with an on-vs-off diff, deploy merged composite checkpoints, and use menagerie's HF backend for adapter events. See `research_programs/agentic_breadth_installation`.

14. `Stopped`: **canonical-answer likelihood after a cap-bound thought is a real but non-actionable selector** (`qwen35_4b_answer_potential_trace_sft`, claim C51, 2026-07-10). The RL-free idea passed meaningful mechanism controls—real thoughts beat token-shuffled (+0.555 nats) and foreign (+4.791) thoughts, and answer-format ranks were stable (tau 0.830)—but failed its preregistered outcome gate: within-task AUROC 0.617 < 0.65, top-choice uplifts +0.073/+0.058 < +0.10, and only 56.9% passed before answer mention. The key diagnosis is deployment mismatch: 99.37% of 2,048 thoughts hit the 512-token cap; fresh forced-close answers parsed only 13.2%, although parsed answers were 86.9% correct. Teacher forcing measured a useful counterfactual answer state after an injected close, not one the model reliably reached. G0 correctly refused N=128 and SFT. This sharpens C50 at the selection seat: **the close/commit seam must be part of the scored event**, and more samples cannot repair a cap-bound interface.

15. `Negative`: **entropy-routed pull-up finds real thought-direction signal but does not unlock capability when the weight edit is non-local** (`qwen35_4b_think_ftpo_round2`, claim C52, 2026-07-11). Round 1's near-parity pivot FTPO harm motivated the strongest rescue: retain only failed argmax tokens with a ≥0.5-logit lead, focused entropy, and non-degenerate varentropy; compare conventional demotion, bounded +0.5 successful-token uplift, and shuffled uplift on 155 matched forks. Pull-up cut non-target drift 36.6% and true labels beat shuffled labels on the parent gym (+6.25pp) and fresh repository agent (+13.89pp, CI touching zero), so the pivot directions were not empty. But every LoRA arm failed the exact-logit locality ceiling; uplift lost to deep base on hidden-tested coding (39/72 vs 43/72) and was flat/negative on fresh whitebox (+0.3pp/−3.1pp). The design stopped before menagerie. The durable rule is **token-local loss is not a context-local model edit**: entropy/varentropy are routing diagnostics, not monotone correctness or pressure scores, and a larger harvest is unjustified until the intervention itself clears locality.
16. `Negative`: **correct live-state actions plus better closure can still erase the interactive policy** (`qwen35_4b_interactive_policy_curriculum`, unclaimed; no claim ID allocated, 2026-07-11/12). Full-sequence DAgger on 1,386 unique model-visited states, 203 expert rows, and 681 C53 replay rows dropped trained-family episode macro 0.6048→0.3517 (−25.3pp) and untouched-family macro 0.6850→0.3519 (−33.3pp), with paired intervals wholly below zero. Clean parsing, atom retention (−2.2pp), and +10–13pp natural closure localize the failure to semantic pivots: only 55/2,270 targets were `VERIFY`, and the updated model almost entirely replaced verify/commit actions with the preceding revise operator on both trained loomfix and untouched patchwheel. The gate stopped RL and Menagerie. The sharper lesson is **state-distribution correction is not update locality**: a looping curriculum must preserve operator transition frequencies and neighboring policy behavior, especially scarce verification pivots, before consequence training is licensed.

17. `Negative`: **a minimal successful tool trace is not a general agent policy** (`qwen35_4b_repo_search_compress_bank`, unclaimed; no claim ID allocated, 2026-07-12). Search found 129 replay-valid repository repairs and compressed them into exactly balanced `INSPECT/PATCH/VERIFY/COMMIT` rows. Compact training then made all 48 trained-family tasks perfect on the identical four-step path (+16.7pp), but family-disjoint success collapsed 49/72→25/72 (−33.3pp, CI wholly below zero), lost to matched sampling by 18.1pp, and failed locality (0.386 vs 0.15). The decisive transition audit shows why: after a failed test apex patched again 24/26 times, while compact patched again 0/48; all 18 rejected recursive-overlay patches were repeated unchanged. Commit after pass remained perfect. **Operator-marginal balance preserves the happy-path vocabulary, not the verifier-conditioned contingency policy.** Success-only minimization deletes the changed-revision examples a looping agent needs. Future banks must balance state-conditioned recovery transitions or remain external retrieval/execution assets; action-only attribution is intentionally unresolved because the necessary gate cancelled that arm. Menagerie stayed sealed.

18. `Stopped`: **endpoint headroom is not enough for a multi-teacher experiment—every mandatory teacher needs a feasible absolute gain bar before production** (`qwen35_4b_specialist_policy_integration`, C53 follow-up, 2026-07-12). The regenerated C53 incumbent, 7/7 behavioral installation canaries, and disjoint compound-headroom gate all passed (compound macro 0.135). But the complete paired baseline found the sole tools family `ferrier` already at 0.994, so the preregistered `S0 + 0.10` qualification target was 1.094 on a score capped at 1.0. Because all four specialists were required, the run correctly stopped before best-of-8, DAgger, GRPO, teacher audit, MOPD, or benchmark exposure. This is **not evidence for or against OPSD/MOPD**; it is a reusable design failure. Future specialist-integration work must gate both downstream endpoint headroom and per-teacher theoretical headroom on disjoint calibration cells, then use a new harder tools/provenance split rather than weakening the observed bar.

19. `Stopped`: **removing the arbitrary gain bar exposed the real prerequisite: an external Pareto label is not automatically a same-prefix teacher advantage** (`qwen35_4b_pareto_policy_integration`, 2026-07-12). The clean successor independently regenerated and behavior-gated C54's `blend` and `apex` policies, then evaluated two frozen contamination-safe blocks under a rule that accepted any credible `delta > 0`. `blend - apex` on quick capability was negative in both blocks (`-0.00693`, `-0.03789`), pooling to `-0.02241` with a one-sided 95% lower bound of `-0.04897`; broad raw scores also favored `apex` on both strata in both blocks. `apex - blend` did show replicated deep capability (`+0.04563`, lower bound `+0.03401`) but exceeded the 0.02 allowance on six retention cells. All protocol checks passed, and the run stopped before teacher audit, locality, MOPD, controls, confirmation, or benchmark exposure. This **does not refute C54 on its menagerie instrument and does not test MOPD**. It narrows the next policy-space bet: teacher choice must be estimated on the student's actual state distribution. A fresh successor should freeze a disjoint verifier-backed continuation-advantage router, require replicated selected-teacher superiority, and only then test whether one checkpoint beats both teachers, a visible router, and matched-compute sampling.

20. `Stopped`: **same-prefix outcomes find a real deep teacher, but four-branch statewise argmax is not a stable two-teacher labeler** (`qwen35_4b_same_prefix_advantage_routing`, unclaimed; no claim ID allocated, 2026-07-12). The preregistered successor spent 12.7M sampled tokens on 384 fresh soup states and 9,216 quick/deep/student continuations. Deep replicated against both soup (`+0.1216`, `+0.0655`; pooled one-sided LCB `+0.0657`) and quick; the combined router also passed. Quick beat the soup by `+0.2009` in block 0 but lost by `-0.0253` in block 1, so the required two-teacher gate correctly stopped locality, MOPD, controls, confirmation, and all benchmark exposure. The diagnosis is winner conditioning, not a weak absolute score or a missing `+0.10` bar: component values correlated `0.79`--`0.86` across disjoint halves, yet only 6/26 block-1 quick selections remained strict audit winners; apparent selection advantage was `+0.319`, and posthoc `+0.10`/`+0.25` margins remained negative. Route evidence was 101 atoms versus only 10 episodes. **MOPD was untested at this stop; item 21 records its deep-only successor.** The highest-information next split was (a) requalify deep on fresh states and test deep-only routed MOPD from the joint soup, while still demanding one checkpoint beat both sources/router/sample-more; and (b) admit quick again only through cross-fitted direct `teacher - student` prediction, uncertainty-aware branch allocation, and a third untouched block—not another statewise argmax or tuned margin.

21. `Negative`: **same-prefix advantage is a real routing signal, but this NF4-trained routed MOPD operator does not install capability beyond its source or inference baselines** (`qwen35_4b_deep_advantage_mopd`, unclaimed; no claim ID allocated, 2026-07-15). Deep first replicated against soup (`+0.1650`/`+0.1220`) and quick, and a five-update pilot passed exact locality (`0.02760` drift; `3.11%` entropy drop). Three seeds completed four rounds and every control/protocol gate passed. On two sealed blocks, primary seed 42 was `−0.006845` pooled joint versus deep (one-sided 95% LCB `−0.012839`), `−0.001300` versus its soup initialization, `−0.003706` versus ordinary soup75, and `−0.169239` versus soup best-of-eight (LCB `−0.175468`); seeds 43/44 also trailed deep. Retention passed and untouched transfer improved. Correct-teacher pressure did beat wrong-teacher by `+0.005312` (LCB `+0.000099`) and matched non-advantage MOPD by `+0.005619` (LCB `+0.000582`), so the verifier-backed label is not empty—it moves the checkpoint in the intended direction without crossing the source frontier. The paired NF4/bf16 diagnostic localizes the next prerequisite: mean trainer objective gain `+0.02191` became merged bf16 gain `−0.000224` with correlation `−0.152`. Before more MOPD, prove a direct-bf16 deployment-parity micro-update that survives merge and beats deep, interpolation, and sample-more. Cross-fitted direct advantages, adaptive allocation (including zero quick), and a third untouched block remain necessary but not sufficient for later two-teacher integration. The frozen stop forbade benchmark exposure.

22. `In progress`: **the clean three-seed LoRA joint adjudication also misses state, so rank-causal controls are now compulsory** (`qwen35_4b_state_carry_vs_state_bag`, `qwen35_4b_state_carry_vs_state_bag_fullrank_delta`, and `qwen35_4b_state_formation_capacity_adjudication`; unclaimed; no claim ID allocated, 2026-07-14). The first exact-compute rank-32 LoRA pilot formed almost no joint node+phase+checksum state (0.00459 versus 0.40), while its direct-full-shape successor reached 0.00277 but was `PILOT_PROMOTION_BLOCKED` by simultaneous non-capacity failures and unmatched initialization/dropout RNG. The fresh adjudication removes those confounds: bit-identical shared state initialization, seed-matched data/order/dropout, exact K=1 bypass, positive state-path controls, three fixed-final 1,500-step seeds, and an immutable source-bound analyzer. Its LoRA joint result is again near chance: **0/57 required seed×split×depth cells pass 0.40**, maximum intact accuracy 0.0234375, all trained/deep/joint-shift categories miss, and adaptation contrast is uncertain. This validly establishes that the registered LoRA joint recipe does not form the state, but **still does not establish LoRA rank as the cause**. The frozen next branch mandates three LoRA state-only controls plus three matched 892M-parameter direct-full-shape joint runs; sealed contrast, representation pivots, and rank conclusions remain prohibited until that receipt.

## How To Read Prior Results

Do not read the imported tracks as a closed agenda. Read them as seed data for research-program design:

- A successful mechanism becomes a program hypothesis.
- A failed control becomes a warning label.
- A repeated bottleneck becomes a new program or backlog item.
- A useful artifact pattern becomes infrastructure.

## Program-Level Claims

### Structured Execution Is A High-Value Mechanism

Seed evidence includes [qwen_structural_latent_compiler_expansion](../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md), [qwen35_4b_foofah_selective_program_fallback](../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md), and [qwen35_4b_operator_inventory_search_pilot](../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md). The program-level lesson is not merely "use programs." It is that explicit execution surfaces give small models something checkable.

**Partial structure needs a readable state, not just an oracle label (2026-07-10, unclaimed,
[qwen35_4b_partial_structure_search](../experiments/qwen35_4b_partial_structure_search/reports/report.md)).**
An exact semantic live-prefix signal is highly useful to search, but frozen Qwen3.5-4B cannot recover it from
operation-type names plus visible I/O: thinking is chance within task and worse than no-think, despite a
promising pooled score. This narrows C10/C47's recognizer strength to completed/readable candidates and
reinforces C13's inverse-computation wall. It also corrects the resource premise behind C35's projection:
depth-5 list-DSL brute is measured, exact, and operationally cheap here. Retire type-only P(viable), measure
the next brute crossover, and make feasible-domain/residual state explicit before reopening learned guidance.

**GENERAL induction-via-reasoning IS installable: a general hypothesize-and-verify procedure transfers to a HELD-OUT rule family (2026-07-07, claim C45, [qwen35_4b_meta_induction](../experiments/qwen35_4b_meta_induction/reports/report_general.md)).** Endorsed follow-up to C44 (which found the shift-CoT shift-specific, OOF affine 0.13). Rule families = affine over positions keyed by multiplier a in {1,3,7,9}; a UNIFORM enumerate-and-verify CoT (try each candidate a, verify on examples, apply). Train on {a=1,3,9}, HOLD OUT a=7. **Held-out a=7 induction = 0.905, AS HIGH as in-family (0.875-0.955)** -- the model GENERALIZES to a rule family never seen as the answer; it learned the general hypothesize-verify-apply PROCEDURE. Combined with C44 (reasoning 1.00 vs forward-pass 0.01): **the fixed 4B CAN be taught GENERAL induction -- infer a novel rule and apply it -- but ONLY as a SERIAL reasoning procedure living in the chain-of-thought tokens, never in the weights.** The MOST CONSTRUCTIVE resolution of the arc's central limitation (executor-not-inducer): the induction wall is a serial-compute limit that a GENERAL reasoning procedure overcomes generally. For the mission: teach a general serial hypothesize-and-verify STRATEGY across diverse instances + always deploy with chain-of-thought; never compress into a forward pass (answer-only SFT fails). Caveats: all families affine (hypothesis class given in the CoT); a=7 arithmetic seen as a rejected candidate (what generalizes is the induction LOGIC within the affine class); the general CoT needs >=400 gen tokens (256 truncates -> false 0.00); single seed; batch-2 training (GPU-corruption workaround).

**The forward-pass induction wall is a SERIAL-COMPUTE limit, not a knowledge limit (2026-07-07, claim C44, [qwen35_4b_meta_induction](../experiments/qwen35_4b_meta_induction/reports/report_reasoning.md)).** Resolves the deepest open question (follow-up to C43). Train the same base on plain-words CoT demonstrating the induction procedure. **THE DISSOCIATION**: the reasoning-SFT model induces held-out shifts PERFECTLY via generation (1.00) but at CHANCE in a single forward pass (forced-digit 0.01) -- **the CoT is ~100% load-bearing**, so induction lives in the serial tokens, not the weights. The model literally cannot do the induction computation in one forward pass, even after SFT; it can only unroll it. This **resolves C43**: answer-only SFT crammed induction into the forward pass -> 0.40 + catastrophic forgetting (execute 0.09); reasoning-SFT unrolls it serially -> 1.00 + preserved execution (0.57). Unifies the arc's induction findings (C38 thinking rescues induction to 0.50; C13 broken mental simulation; C42 multi-step): the model's core limit is running MULTI-STEP COMPUTATION in a forward pass -- give it serial tokens and it works. For the mission: always give the fixed 4B serial compute (chain-of-thought) to elicit induction/multi-step reasoning; answer-only SFT is the wrong vehicle. Caveats: the CoT hand-codes the shift algorithm (so in-family 1.00 = executing a TAUGHT serial procedure, not general induction -- OOF affine 0.13); base+strategy-hint 0.00 is confounded (position-arithmetic too hard for the untrained base); single seed; general-induction (multi-family leave-one-out) owed.

**Can SFT install the SKILL of induction? Partially -- the wall is neither a hard bound nor cleanly liftable (2026-07-07, claim C43, [qwen35_4b_meta_induction](../experiments/qwen35_4b_meta_induction/reports/report.md)).** The mission-core 'lift the wall' test. Each episode = a random scrambled digit order (stated) + a hidden rule + examples + query; the model must infer + apply. Base fails at chance (shift-induce 0.087) though it CAN execute a stated rule (0.72 -- the review's mandatory gate: the wall is induction, not execution). Answer-only QLoRA on random shift episodes. **SFT PARTIALLY lifts the wall, data-limited**: shift induction 0.087 -> 0.35 (4k) -> 0.40 (8k), ~4.6x chance and still rising -- **but plateaus well below the execute ceiling (0.72)**, so only partly installed. **Out-of-family affine barely moves** (0.21 -> 0.30) -- the model learned a shift-SPECIFIC procedure, not general induction. **Two costs**: catastrophic FORGETTING (answer-only SFT crashed shift-execute 0.72 -> 0.09) and a default-fallback digit bias shrinking with data (0.37 -> 0.20). So the induction wall is **neither a hard architectural bound** (SFT lifts it several-fold, scaling with data) **nor cleanly liftable** (partial, procedure-specific, forgets execution) -- consistent with 'executor, not inducer': trained to induce, the fixed 4B learns a specific procedure, not the general skill. Relates to C22/C23 (execution wall is data-limited/crossable) -- induction is HARDER to install than execution. Owed: reasoning-SFT arm, multi-family leave-one-out for GENERAL induction, mix-execute to prevent forgetting; single seed.

**The model can localize its own errors: per-step confidence dips at the first slip (2026-07-07, claim C42, [qwen35_4b_error_localization](../experiments/qwen35_4b_error_localization/reports/report.md)).** Extends C40 (single-step metacognition) to MULTI-step. The model advances k steps in a cyclic order over depth-4-7 chains via scaffolded decoding (force 'Step i: <digit>', read the digit distribution at each step). Ground truth = local correctness (m_i == successor of the model's OWN previous step); familiar order -> genuine arithmetic slips ~31%/step. **Make-or-break control = de-trending** (confidence RISES with position 0.66->0.96). **Per-step error prediction survives de-trending** (AUROC 0.75 vs 0.73 raw). **The confidence dip is EXACTLY at the origin**: mean de-trended confidence by offset-from-first-error is minimized at offset 0 (-0.15), high just before (+0.23 at -2), recovering after. **Single-slip localization** (n=137): de-trended-residual argmin hits the slip 0.56 (raw 0.64) vs position-prior 0.36 vs uniform 0.19. **Targeted repair** (redo from located step) fixes 0.56 at avg 3.8 steps vs redo-all 5.6. So C40's implicit metacognition is **step-resolved** -- the per-step confidence carries WHERE the model slipped, not just THAT it did -- deployable targeted repair (strongest when the model slips once). Caveat: multi-slip chains (n=224) -- argmin finds AN error 0.76 but the FIRST only 0.27; single seed; execute-mode arithmetic slips.

**The confidence toolkit transfers to real code, but the winning signal is P(True), not sequence mean-logprob (2026-07-07/08, claim C46, [MBPP](../experiments/qwen35_4b_code_confidence/reports/report.md), [HumanEval](../experiments/qwen35_4b_humaneval_code_confidence/reports/report.md)).** The C41 owed generalization test is now run as two standalone experiments. On MBPP (244 problems, greedy+k=8, one visible test), P(True)-select picks a correct sample at **0.762** vs public-output majority **0.721** (p=0.014), mean-logprob 0.730, random 0.696, oracle 0.844; within-problem AUROC on mixed problems is P(True) 0.738 and mean-logprob 0.693 vs length 0.548. On all 164 HumanEval tasks with no public probes, the strict verifier-free comparison is P(True) **0.835** vs mean-logprob 0.787 vs random 0.766 (oracle 0.872), with greedy-solvability AUROC **0.862**. The rule refines C40/C41: calibrated uncertainty lives in concentrated single-token readouts (answer digit on the toy, A/B judgment token on code), while sequence averages dilute it. Visible execution still dominates when available, so confidence is the verifier-free select/abstain/route lever, not a replacement for tests.

**Beat sample-more with the model's own uncertainty: confidence-select beats majority vote, verification-free (2026-07-07, claim C41, [qwen35_4b_confidence_guided_compute](../experiments/qwen35_4b_confidence_guided_compute/reports/report.md)).** Turns C40's calibrated implicit confidence into a compute tool (mission: beat sample-more). Mix of C40 successor problems spanning coverage-limited (familiar-induce, greedy 0.21 -> pass@12 0.90) and capability-limited (novel-induce, greedy 0.07 -> pass@12 0.59, below pure-luck). Sample k=12, read each sample's P(answer). **Self-consistency (majority vote, the standard verifier-free method) is FLAT (~0.48) across all budgets** -- sample-more is wasted, because the model's mode is often confidently wrong. **Confidence-select (argmax per-sample P(answer), verification-FREE -- no execution) RISES 0.47 -> 0.62**, beating majority at every budget (oracle 0.83), because P(answer) is calibrated (C40): the most-confident sample beats the most-common one. **Max P(answer) predicts per-problem solvability at AUROC 0.83** -> abstain on low-confidence gives ~1.0 accuracy on the confident top third. (Confidence-guided ALLOCATION is ~tied with uniform confidence-select; the win is SELECTION + ABSTENTION.) Refines C17 (selection works with enough examples via execution-consensus): C41 shows selection works VERIFICATION-FREE via the calibrated logit and beats self-consistency. C46 resolves the real-code generalization caveat: on programs, use the concentrated P(True) judgment-token readout rather than sequence mean-logprob. (Single toy substrate + seed; review agent died on API error, self-vetted vs C10/C17.)

**The model knows when it will fail IMPLICITLY (answer-token probability) but NOT EXPLICITLY (self-report) (2026-07-07, claim C40, [qwen35_4b_metacognitive_boundary](../experiments/qwen35_4b_metacognitive_boundary/reports/report.md)).** First metacognition claim. Uses C39's VERIFIED competence boundary as crisp ground truth. Format-equalized single-value task; two non-degenerate logit confidence signals (verbalized 0-100 is a constant 100): IMPLICIT P(answer) = the model's probability on the digit it emits (softmax over the 10 digit tokens, one forward pass); EXPLICIT P(True) = Kadavath self-verification. **Implicit P(answer) tracks accuracy near-perfectly** (1.00/0.44/0.29/0.15 vs acc 1.00/0.40/0.19/0.10); **explicit P(True) is flat (~0.4)**, even underconfident on the perfect execute cell. **Within the surface-matched familiar-induce cell, P(answer) predicts per-item correctness at AUROC 0.95 (CI 0.90-0.99), crushing the external surface baseline (0.61) and explicit P(True) (0.46 = chance).** Selective prediction by low P(answer) lifts accuracy-on-attempted from 0.23 to ~1.0. So **the model knows when it will fail -- but only in its output distribution, not in anything it can say.** Deployable latent capability: read the answer-token probability (logits) as a confidence/abstain/route signal; never trust the explicit self-report. (Caveat: reversal_induce was intended as an easy-but-scrambled-looking dissociation but is genuinely hard; the surface-vs-competence contrast rests on the external baseline; single seed.)

**In-context learning is RETRIEVAL of familiar structure, not INDUCTION of novel structure (2026-07-07, claim C39, [qwen35_4b_icl_retrieval_vs_induction](../experiments/qwen35_4b_icl_retrieval_vs_induction/reports/report.md)).** Capstone unifying the arc: if the model can't induce a novel rule (C38), what is ICL doing? Execution-safe single-value task ('advance k in a cyclic order'), FAMILIAR structure (natural order 0-9) vs NOVEL structure (a stated random order) at matched complexity, x EXECUTE (rule stated) vs INDUCE (few-shot). **Execution is near-perfect and familiarity-independent** (familiar 1.00, novel 0.97 -- the model applies the novel scrambled-order rule almost perfectly when told it); **induction is familiarity-bound and collapses** (familiar 0.45 vs novel 0.12 = chance): the model cannot induce the novel rule from examples even though it executes that exact rule at 0.97, and more examples do not rescue it (0.15->0.05). So **in-context 'learning' surfaces/retrieves familiar (pretrained) structure; it does not create/induce novel structure.** Unifies C13-C38: the model is an EXECUTOR/RETRIEVER of pretrained structure (C37 execution intact), not an INDUCER of novel structure (C38, C32/C36) -- ICL is the retrieval half of reasoning, not the induction half. For the mission: 'unearthing latent capability' = surfacing structure the model already has; the fixed 4B cannot acquire genuinely novel structure in-context. (Methodological: the first vehicle, letter-ciphers, floored on char-assembly, application-only 0.20 -- pivoted to single-value.)

**The structure-PROPOSAL wall persists in language: the model executes a given rule but cannot induce one — proposal/induction is modality-general, while simulation is formal-specific (2026-07-07, claim C38, [qwen35_4b_language_proposal_wall](../experiments/qwen35_4b_language_proposal_wall/reports/report.md)).** Complement to C37. Tests whether the C32/C36 structure-proposal wall persists in language: relational-composition INDUCTION (R=4 made-up relations = random bijections over made-up entities, hidden depth-D rule; infer which relations compose from examples, apply to a new query). A review-mandated application-only control (rule given) establishes the ceiling. **Clean forward-pass dissociation at depth-1** (where application is easy): the model *executes* a given rule (0.86) but *cannot infer* one from examples (0.00 = chance) in a single pass. Induction is at chance no-think at all depths; thinking only *partially* rescues it (0.50 at depth-1, budget 4096, no truncation — reasoning correct but error-prone), still below application (0.75) and far below C37's simulation (0.99). So the model is an **executor, not an inducer, in language as in formal domains** — corroborating C32/C36 as a cross-modality law. **C37 + C38 together: the compositional wall has two components that dissociate by modality — SIMULATION/execution is modality-dependent (formal walls at depth-3, language does not), but PROPOSAL/INDUCTION is modality-general (hard in both formal and language) — the deeper, more fundamental limit.** The model reasons multi-step in language, but it does not induce rules; the structure-proposal wall is the one part of the whole arc that holds even in the model's native domain. (Caveat: this multi-relation substrate's application degrades at depth 2+, so induction is cleanly isolable only at depth-1; think depths 2–4 owed.)

**The compositional wall does NOT exist in language: the model chains depth-3+ multi-step SIMULATION in natural language near-perfectly — the wall is formal-modality-specific (2026-07-07, claim C37, [qwen35_4b_language_reasoning_wall](../experiments/qwen35_4b_language_reasoning_wall/reports/report.md)).** The first step OUT of the formal-composition arc (all 36 prior claims were formal/procedural). Tests C13's mental-SIMULATION wall (NOT the C32/C36 structure-proposal wall) in the model's native domain: contamination-free successor-chain traversal over made-up entities, same chain rendered linguistically vs as a Python dict, shortcut-hardened, no-think primary (mental simulation). **Result: NO depth-3 wall in language.** Linguistic-semantic no-think is near-perfect through depth-4 (0.99/1.00/0.99/0.94), and the made-up-relation control ('X gorps Y') is *also* perfect through depth-3 (1.00) — a genuine **modality** effect, not a semantic prior. Both degrade only at depth 5–6 (semantic gracefully to 0.76; the made-up relation collapses to 0.00 — semantics aids *deep* chaining). This is in **stark contrast to the formal-composition wall (depth-3, C13–C36)**: the model chains 3–4 reasoning steps in its native linguistic domain, so the 'compositional wall' is **NOT a general multi-step limit — it is specific to formal/procedural composition.** It relocates C13's 'broken mental simulation' to the formal modality. **Secondary (a striking surface-form effect): the formal-DICT rendering triggers CODE-MODE** — the model echoes the dict as a code block instead of simulating (d1 = 0.03) — so the surface *presentation* determines whether the model *reasons or codes*. (Scope: tests SIMULATION, C13; does NOT touch the C32/C36 proposal wall — a linguistic proposal task is the owed follow-up. Think conditions truncation-confounded; no-think is the clean primary.)

**The recent structure findings are model-level laws: wall-is-structure (C32) and brute-search-dominates (C34) replicate on string + register substrates (2026-07-06, claim C36, [qwen35_4b_crosssubstrate_structure](../experiments/qwen35_4b_crosssubstrate_structure/reports/report.md)).** C16 cross-substrate-tested the *early* ladder (C13–15); the recent sharper findings were never generalized. Family-generic replication on STRING (char edits), REGISTER (3-register machine), and LIST (anchor), depth-3, n=100. **C32 + C34 are model-level laws — essentially identical on all three:** the wall is STRUCTURE (base structure-cov = concrete-cov, value tax ≈ 0 everywhere), values are trivially searchable given structure (oracle-skeletonfill = 1.000), structure genuinely matters (random-skeletonfill low: 0.17 string, 0.09 list, 0.32 register), and **brute-force structure-search + value-fill + execution-consensus dominates the model at deploy (brute-deploy ~1.0 vs base ~0)**. So the fixed 4B is a **value-computer, not a deep-structure-proposer, across genuinely different substrates**; the compositional wall is structure-proposal everywhere; and with an interpreter, brute-force structure-search dominates the weights outright everywhere. Combined with C16 (early ladder) and C33/C35 (banking installs structure but collapses with depth), the **entire compositional arc is established as model-level, not an artifact of one hand-built DSL**. (Nuance: register is more value-fungible — a substrate property, not a break in the pattern.)

**Brute-force structure search dominates the tested banked models through depth 4 on the list DSL; depth 5 remains open (2026-07-06, claim C35 re-graded, [qwen35_4b_structure_search_scaling](../experiments/qwen35_4b_structure_search_scaling/reports/report.md)).** Brute structure-search + value-fill + execution-consensus stays near-perfect from depth 3 to 4 (0.975→0.967). The compared banked structure coverage is 0.51→0.10, but those points come from different, non-dose-matched models, so they do not establish a causal depth-collapse law. The measured conclusion is narrower: exhaustive search wins through depth 4 where it was run. The roughly one-million-skeleton depth-5 space was projected but never evaluated, so neither “the model takes over” nor “neither works” is established there. Partial-structure recognition-guided pruning is the direct unresolved test.

**With the interpreter, brute-force structure-search + value-fill + execution-consensus near-solves depth-3 (0.975), dominating the model; banking's structure is a forward-pass-only asset (2026-07-06, claim C34, [qwen35_4b_banking_installs_structure](../experiments/qwen35_4b_banking_installs_structure/reports/report.md)).** End-to-end test of the bank+value-fill recipe (confirming C33's inferred ~0.51), with the decisive brute-force control. The banked model emits Python, so recover its proposed structure from behavior (infer the op-type skeleton each of k=8 samples implements), value-fill against the true outputs, deploy via execution-consensus (plurality output-vector on fresh probe inputs). **Result (held-out depth-3, n=80): bank-fill deploys at 0.463 ≈ the banked model's structure-coverage 0.475 (confirms C33) — but brute-force structure enumeration (all 4096 skeletons, structure *not* from the model) + value-fill + consensus deploys at 0.975, near-solving depth-3 *without the model*.** After the 8-visible filter only ~2 skeletons survive (DSL not value-fungible, C32) and consensus picks the right one 97.5% of the time. Using the model's structure is *worse* than ignoring it — bank-fill is capped at the model's ~48% structure-coverage while brute enumeration always contains the true skeleton. **Net: with the interpreter available (free selection per C17), free structure-search dominates the model at deploy (0.975 vs 0.46 vs 0.20 forward-pass); banking's installed structure (C33) is a forward-pass-only asset (matters only when you must deploy in a single pass with no interpreter).** This is the arc's cleanest 'beat sample-more' deployable result — but the lever is the TOOL (structure-search + interpreter), not the weights. Scope: brute-force wins *because* the depth-3 structure space (4096) is enumerable; the model's structure-pruning would only become a deployable lever when the space is too large to brute-force (larger op-inventory or deeper compositions) — where model-guided-vs-brute (C25) lives.

**Banking installs STRUCTURE: base op-sequence structure-coverage 0.00 → banked 0.51 (held-out), converting the wall from structure-bound to value-bound (2026-07-06, claim C33, [qwen35_4b_banking_installs_structure](../experiments/qwen35_4b_banking_installs_structure/reports/report.md)).** Mechanistic follow-up to C32. Ran C32's format-immune structure-coverage (does the model program's *behavior* match the true op-type skeleton with any params?) on **base vs banked_1280** (C24), on held-out depth-3 (banked's frozen eval, disjoint from training), n=80. **Base has no structure** (structure-cov 0.000 = concrete-cov 0.000 — can't propose the depth-3 op-sequence, exactly C32). **Banking lifts structure-coverage to 0.512 on held-out tasks** → banking installs *generalizable* op-sequence structure, not memorized skeletons. And it **converts the wall from structure-bound to value-bound**: the base has no skeletons (struct = concrete = 0), but the banked model proposes the *right skeleton* 51% of the time while nailing the *full concrete program* only 36% — a **value tax of +0.15** (right-skeleton-wrong-param failures the base never had). Since oracle-skeletonfill = 1.0 (C32), value-filling the banked model's proposed skeletons would deploy at ~0.51 vs 0.36 alone (bank installs structure; value-fill recovers the value tax). **Mechanistic closure of the arc: the compositional wall is structure-proposal, and banking's entire lever is installing that structure** — unifying C22–24 (banking crosses depth-3), C32 (the wall is structure), C31 (values surface/searchable): **banking = structure-installation.** Explains why value-side interventions (C31 param-hint, C29 DPO) never moved the *base* wall — the base's problem is structure, not values; only after banking installs structure does a (fillable) value gap even exist.

**The compositional wall is STRUCTURE, not values: the model can't propose the op-sequence, but once structure is known values are trivially searchable (2026-07-06, claim C32, [qwen35_4b_structure_or_values](../experiments/qwen35_4b_structure_or_values/reports/report.md)).** C31 showed the model computes the op-type (structure) but reads the parameter off surface I/O (values). That raised a never-tested question about the *wall itself*: when the model fails depth-3, is it a STRUCTURE error (wrong op-type sequence) or a VALUE-binding error (right skeleton, wrong constants)? (The natural design — ask the model for the op-sequence — *failed*: op-seq generation solves 0.00 even at depth-1, a format handicap; and the review flagged "skeletonfill ≫ direct ⇒ values" as a false dichotomy. So the structure signal is *format-immune*: run each model program and check whether its *behavior* matches the true op-type skeleton with any params.) **Result (min-depth-verified, n=120/depth): the wall is STRUCTURE, decisively.** (1) **No value tax** — the model's STRUCTURE-coverage (right op-type sequence, any param) *equals* its concrete coverage (depth-3: 0.017 = 0.017; depth-2: +0.017): its failures are wrong-skeleton, not right-skeleton-wrong-param; there is no hidden pool of "right structure, wrong values" solutions. (2) **Values are trivial given structure** — oracle-skeletonfill = 1.000: knowing the op-type sequence, cheap value-search *always* finishes (consistent with C31: param surface-readable). (3) **The DSL is not value-fungible** — random structure barely works (R200 = 0.108 at depth-3): structure genuinely matters. **Net: the compositional wall is a STRUCTURE-PROPOSAL problem — the model can't propose which operations in which order; once structure is known, values are free.** This unifies the arc — C19 (depth-3 first-op is a representational "thread"), C25 (no step-1 lookahead), C31 (param surface-readable) all point to the same dichotomy: **the model reads/computes VALUES easily but cannot propose deep STRUCTURE.** It explains why tool-enumerated *structure* seeds (C22) and banking (installs structure) were necessary, and why value-side interventions (C31 param-hint, DPO) don't move the wall. (Refuted the initial "wall is values" hypothesis. Deployable recipe = structure-search + cheap value-fill = tool-augmented search, not a forward-pass gain.)

**The op-TYPE is model-computed (latent, elicitable) but the PARAMETER is only read off surface I/O — sharp localization of C30's deployable bottleneck (2026-07-06, claim C31, [qwen35_4b_probe_the_parameter](../experiments/qwen35_4b_probe_the_parameter/reports/report.md)).** C30 found the deployable bottleneck is the concrete first op's *parameter*, not the op-type C19 decodes. Is the parameter model-*latent* or surface-readable? A critical review fix: the layer-0 probe is a *degenerate* surface control (RoPE makes the fixed-template last-token embedding constant across tasks), so the real control is an **external classifier on raw I/O features** (list lengths, sums, min/max, elementwise diffs) with **no 4B forward pass**. **Decodability (depth-2, fsig-disjoint): the op-TYPE is genuinely MODEL-LATENT** — the residual probe (0.413) beats the surface classifier (0.272) — **but the PARAMETER given the type is SURFACE-READABLE** — a trivial I/O classifier (0.529) decodes it as well as / better than the model residual (0.493), both above chance (0.303). **Deployability on param-first-op tasks:** the parameter *is* the deployable bottleneck (oracle-full 0.095 ≫ oracle-type 0.007 — confirms C30, isolated to param tasks), but the model probe barely delivers (probe-full 0.014) and the *cheap surface pipeline delivers more* (surface-full 0.027) — you don't need the 4B for the parameter; wrong-param 0.000 (content-causal). The two-term check is textbook clean: probe-full deploys *exactly* like the oracle on tasks it decodes correctly (0.091 = 0.091) and like no-hint on those it gets wrong (0.0 = 0.0) — a faithful readout bounded by 26% concrete accuracy. **Net — sharp localization of the wall: the forward pass genuinely COMPUTES the op-type (a real latent capability, elicitable training-free via C30's externalization) but has no privileged representation of the parameter — it just reads it off surface I/O, which any trivial classifier does equally.** The training-free latent-elicitation ceiling is the op-type; the parameter is not a model-latent thing to unearth. (Also retroactively strengthens the C19/C30 methodology: "layer-0 at chance" was a weak surface control; the external-I/O baseline is the correct one, and it confirms the op-type is model-latent.)

**Externalizing the latent readout (decode→prompt) elicits deployable depth-2 where steering (C20) failed — but the decodable op-TYPE only narrows sampling; the PARAMETER is the deployable bottleneck (2026-07-06, claim C30, [qwen35_4b_probe_to_prompt](../experiments/qwen35_4b_probe_to_prompt/reports/report.md)).** The untried seam between C17 (selection is free but adds no coverage), C19 (the composition's first-op is linearly *decodable* from the base residual far above behavior at depth-2, 0.42, but a thread at depth-3), and C20 (that direction is *not steerable* — ActAdd inert). This **externalizes** the readout: refit C19's probe (replicated — d2 0.45@L21), decode the first-op from the base model's own activation on fresh fsig-disjoint tasks, and inject it as a **prompt hint** (shift the proposal, the only lever C17 allows) instead of steering the residual. **Externalization ELICITS deployable depth-2 where steering failed:** oracle-full (true op+param in the prompt) lifts depth-2 greedy@1 **6×** (0.030→0.190) and coverage 6× (0.050→0.310) — the *first test-time intervention in the whole arc to move deployable capability*. **But the deployable bottleneck is the PARAMETER, not the op-TYPE C19 decodes:** oracle-TYPE lifts *coverage* (0.050→0.190, narrows sampling) but not *greedy* (0.020); only the full op deploys single-shot. So the C19 type-only probe (0.32 eval acc) nets to ~zero — though the effect is genuine self-elicitation (benefit concentrated on probe-correct tasks). **Graded by depth exactly as C19 predicts:** real at depth-2 (headroom), ≈0 at depth-3 (thread; even oracle-full only 0.010). Controls clean: neutral placebo ≈ no-hint (format), wrong-hint hurts (content-causal), layer-0 probe at chance 0.05 (model-computed, not surface-readable). **Net: the latent readout IS usable at test time — by externalizing it (decode→prompt), not by steering (C20) — but the part C19 can decode (op type) isn't the deployment-limiting part (the parameter is).** Reconciles C20's decodable-but-not-steerable puzzle and adds the first working test-time lever, bounded by the representation.

**Preference training on the model's own failures does NOT close the coverage→deployable gap — DPO collapses generation; the gap closes with MORE SFT instead (2026-07-06, claim C29, [qwen35_4b_learn_from_failures](../experiments/qwen35_4b_learn_from_failures/reports/report.md)).** Every prior training in the arc was SFT-on-positives, which raises depth-3 coverage@16 but not deployable greedy@1. Does DPO/contrastive training on the model's OWN (correct, incorrect) samples — learning from its failures — raise greedy@1? Harvested 174 same-task (chosen=verified-correct, rejected=verified-wrong) pairs from banked_1280's own no-think samples (identical median length, no length heuristic). **The SFT model is already a strong verifier of its own samples (pre-DPO 2AFC = 0.810; matches C13's ~0.73) — but preference-optimizing that discrimination COLLAPSES generation:** DPO greedy@1 bumps to 0.050 at 0.25 epochs (within noise of SFT's 0.037; coverage flat), then craters — 0.000 by 0.5 ep, 0.013 by 3 ep (both greedy@1 *and* coverage crash; classic over-optimization, margin logp_c−logp_r blew to ~61). DPO never beats the compute control. **The effective lever is just MORE SFT:** SFT_2x (6 vs 3 epochs) triples greedy@1 (0.037 → 0.113) and doubles coverage (0.113 → 0.212) — the "gap" was partly *undertraining*. The shuffled control (0.037) confirms it's not pure loss-shape, and real DPO (0.013) is even worse. **Net: you cannot close the coverage→deployable gap by teaching the model to PREFER its correct over its wrong samples (that destroys generation) — just train longer on the correct ones. The strong latent sample-discrimination (2AFC 0.81) is a "read-only" verifier ability that does not transfer to a "write"/generation gain via preference training.** Extends the prior MBPP DPO work here (constrained DPO didn't beat sample-more) to the controlled depth-3 substrate, adding: DPO is fragile/collapses. (Limits: my DPO wasn't heavily constrained — a more careful recipe might avoid collapse — but across 0.25–3 epochs it never beat SFT_2x; single seed, n=80.)

**Banking correct decomposition PLANS beats banking ANSWERS on deployable depth-3 — content-causally, via the thinking channel (2026-07-05, claim C28, [qwen35_4b_bank_the_thoughts](../experiments/qwen35_4b_bank_the_thoughts/reports/report.md)).** The clean resolution of C26/C27's "never taught to reason" confound. Three fresh QLoRA from base on **matched** data (identical prompt+code; only the trace differs): A = `prompt→code`, T = `prompt→⟨correct decomposition plan⟩→code`, T_corrupt = same code with a *mismatched* plan. On frozen held-out depth-3 (n=80): **banking the PLAN beats banking the ANSWER and stacks with multi-sampling** — T coverage@16 **0.325** vs A **0.200** (greedy@1 0.050 vs 0.025). **Content-causal:** T_corrupt (same code, same thinking channel, *wrong* plan) collapses to **0.113 — below even A** — so it is the correct-reasoning *content*, not the think-format or extra test-compute; teaching wrong reasoning actively *hurts*. **Test-time channel:** T deployed no-think ≈ 0 (0.013) — banking plans installs a *reason-then-solve* skill that needs thinking to cash out. **This reconciles C26/C27** (test-time thinking on a *no-think*-trained model added nothing): once the reasoning is *banked*, thinking helps a lot — the earlier null was "never taught to reason," exactly the confound the user flagged. (Limits: Phase 1 uses *synthetic* plans; the model's-own rejection-sampled thoughts are Phase 2. T's step-1 lookahead-ranking eval didn't complete, so coverage-via-reasoning vs step-1-lookahead is still open; single seed; A/T token-matching deferred.)

**TEST-TIME thinking does NOT breach the lookahead wall — it amplifies recognition, not planning (2026-07-05, claim C26, [qwen35_4b_thinking_lookahead](../experiments/qwen35_4b_thinking_lookahead/reports/report.md)).** *(Scope caveat, added 2026-07-05: this tests **test-time** thinking on a model **never trained to reason about this substrate**. It does not show thinking is fundamentally useless for planning; it leaves wide open whether **banking successful reasoning traces** (`prompt → ⟨thinking⟩ → code`) would install planning-via-thinking — the clean untested version.)* C25 found the fixed 4B can't plan the first of 3 ops in a single forward pass (a lookahead wall). Thinking is serial test-time compute — the natural lookahead mechanism and the dormant C9 lever. Does a thinking budget breach the wall with **no training**? Channel-matched test (think→RANK vs no-think→RANK: think B tokens, close `</think>`, then the *same* 32-way likelihood ranking as C25 — parse-immune), headlined on **step 1** (goal 3 ops away, no intermediate state materialized — the only clean lookahead test; steps 2/3 are handed the true intermediate list, so a lift there is state-materialization not planning). **Result (n=40, chance 0.031): step-1 stays at chance across budgets — 0.025 → 0.050 → 0.075 at B=0/1024/2048, Wilson CIs all overlapping. Thinking does NOT breach the wall.** But the benefit pattern is diagnostic: thinking's lift scales *inversely* with lookahead distance — step-3 recognition (1 op away) 0.275 → **0.600**, step-2 (2 away) 0.000 → 0.325, step-1 (3 away, real planning) ~flat. So **thinking amplifies RECOGNITION, not PLANNING**, and only where the interpreter materializes the true intermediate state. This also *refutes* the internal-brute-force alternative (if the model could simulate the depth-3 path in its scratchpad, step-1 would rise; it doesn't — traces show confused meta-reasoning, not enumerate-and-test). **The killer juxtaposition with C25:** banking lifted this same step-1 lookahead metric (0.013 → 0.138, dose-dependent) while thinking does not — so **for the multi-step planning/lookahead gap, TRAINING (banking on execution-verified solutions) is required; test-time serial compute cannot elicit it.** For recognition, thinking is a powerful amplifier. This reconciles with C23 (base think single-shot depth-3 = 0: thinking can't do the whole composition because it can't plan the first steps) and sharpens the mission read: "elicit latent capability without training / beat sample-more" works for recognition but fails for multi-step planning. (Limits: closed-set ranking easier than free generation; n=40, one seed/budget, budgets ≤ 2048.)

**"Be your own tool-search" finds a first-move ranking wall, while banking improves closed-menu step-wise guidance (2026-07-05, claim C25 re-graded, [qwen35_4b_latent_decomposition](../experiments/qwen35_4b_latent_decomposition/reports/report.md)).** Base top-1 next-op ranking is 0.013/0.062/0.237 at goal distances 3/2/1 (chance 0.031): only the first move is at/below chance, while step 2 is weakly above chance. Base and random guidance solve 1/80 and 2/80 tasks, a one-task difference that supports “no better than random,” not a directional harm claim. Banking improves the ranking channel and low-budget search: banked1280 solves 18/80 versus random 2/80, base 1/80, and brute 23/80. The dose trend is clear at steps 2–3; the crucial step-1 increment from banked640 to banked1280 is only 10/80→11/80. Treat this as a closed-menu behavioral guidance lift with one beam and single adapter seeds, not proof of reusable internal planning machinery.

**Depth scaling & controls — no saturation, the gain is data-DIVERSITY, and the recipe repeats one rung deeper (2026-07-05, claim C24, [qwen35_4b_depth_scaling_controls](../experiments/qwen35_4b_depth_scaling_controls/reports/report.md)).** Three follow-ups stress-testing C23, each hardened by an adversarial workflow review. **(1) No saturation through 1280 tool-pairs (1156 distinct functions):** depth-3 think cov@16 climbs 0.00→0.087→0.212→0.375→**0.537** at N=0/40/160/640/1280, deployable greedy@1 rises to 0.188, distinct functions grow near-linearly (so it's real capacity, not harvest exhaustion). **(2) The gain is data-DIVERSITY, not compute:** a 2×2 at matched steps/mixture — holding compute fixed and adding diversity (up40 0.163 → train_640 0.375) is cleanly significant; holding diversity fixed and adding compute (N=40 0.087 → up40 0.163) is within noise. So C23's "data-limited" sharpens to *data-diversity-limited*: banking more **distinct** explorer-found verified solutions is what drives the gain, not more gradient steps. **(3) The tool-search+banking recipe repeats one compositional rung deeper, weakly:** on depth-4, raw base 0.00, the depth-3 scaffold already transfers to 0.067, and adding 320 depth-4 tool-pairs nearly triples coverage to **0.183** — but test-time-only (greedy flat 0.033) and marginally significant at n=60, the same weak stage depth-3 showed at low doses; no depth-3 forgetting (guardrail 0.425). Together these tighten the C13→C24 ladder recipe: **it is diversity-driven and rung-repeatable** — self-training installs in proportion to the distinct verified solutions the explorer banks, and the whole loop repeats one rung deeper at the same weak-then-scales efficiency. (Limits: single seed; n=60–80 underpowers adjacent-dose and depth-4 significance — point estimates clear, adjacent CIs overlap; depth-4 single dose; 2560 dropped, training 2× slower than budgeted.)

**The depth-3 install is DATA-LIMITED, not a representational cap — and it scales into deployable single-shot (2026-07-04, claim C23, [qwen35_4b_depth3_dose_response](../experiments/qwen35_4b_depth3_dose_response/reports/report.md)).** C22 crossed the depth-3 wall with tool-seeded banking but only weakly and test-time-only, leaving open whether that ceiling was a data limit or a hard representational cap. This dose-response settles it: banking N tool-found depth-3 solutions (interpreter search over the substrate's own DSL, no external model) installs depth-3 coverage that rises **monotonically** with N — think coverage@16 **0.00 → 0.087 → 0.212 → 0.375** at N ∈ {0,40,160,640}, with the top-dose Wilson lower CI (0.28) above the low-dose upper CI (0.17), and *no plateau*. Crucially the **deployable** install scales too: no-think coverage 0.00→0.338 and no-think single-shot greedy@1 0.00→**0.10** at N=640 (≈0 at C22's N=130). The held-out set has **0 leakage** (deduped by function-signature AND op-composition), so this is genuine generalization to novel depth-3 rules, not enumerating a finite DSL. So C22's "weak, test-time-dominated" install was insufficient *data*, not a hard bottleneck: the thin depth-3 representation (C19) thickens smoothly with more explorer-found training data and converts into deployable single-shot. This completes and quantifies the C13→C23 recipe: **an explorer the base lacks reaches the rung and yields verified solutions; banking installs them; and the amount installed scales with the number of explorer-found solutions, into deployable single-shot — self-training is the installer, external search the explorer, data the throttle.** This is "extend capability by a lot without a larger model" demonstrated and *scaling* — the only external ingredient is an interpreter-backed search (a tool). (Limits: single training seed, fixed epochs so data is confounded with gradient exposure, search-easy harvest bias — a cap could still appear past N=640, untested.)

**Tool-seeded banking crosses the depth-3 wall — but the installer weakens with depth (2026-07-04, claim C22, [qwen35_4b_tool_seeded_banking](../experiments/qwen35_4b_tool_seeded_banking/reports/report.md)).** The positive control C21 predicted, with the design hardened by an adversarial multi-agent workflow review (frozen paired eval, behavioral function-signature dedup, brute-vs-guided framing, a real significance bar). C21 said self-banking gives *zero* depth-3 coverage because the base samples ≈0 depth-3 solutions to bank; the fix is to seed banking with depth-3 solutions found by an **explorer** the base lacks. An interpreter-backed brute search over the substrate's *own* 16-op DSL (CPU-only, no external model) solved **130/130** depth-3 tasks — what monolithic sampling gets ≈0 of — and those pairs were added to C21's exact depth-1+2 set (the only delta). Result: **depth-3 think coverage@16 rose from a hard 0/40 (0.00, identical to C21) to 5/40 (0.125)** on frozen behaviorally-deduped held-out tasks — a *significant* unlock (above base's 95% CI, 5 distinct novel rules). So the recipe is validated: **tools explore, banking installs, both required.** But **CROSSED-BUT-WEAK**: the depth-3 install is test-time-dominated (no-think 0.025, deployable greedy@1 0.00), whereas the *same* banking installs depth-2 strongly and deployably (greedy@1 0.15). The new nuance the arc lacked: **the installer's efficacy decays with depth** — even *perfect* training data lands only a thread of depth-3 capability in one QLoRA round, consistent with C19 (the depth-3 inverse is barely represented) and C20 (not steerable). The deep wall resists installation too. And there is no free next rung (depth-4 stayed 0). The precise, complete recipe the whole C13→C22 arc yields: to extend the frontier one depth, an *explorer the base lacks* reaches the rung, *banking installs it* (weakly, mostly test-time), and *each rung must be seeded* — extendable, but with diminishing installation efficiency the deeper you go.

**Self-banking can't climb the wall — it is coverage-seed-bounded (2026-07-03, claim C21, [qwen35_4b_wall_climbing](../experiments/qwen35_4b_wall_climbing/reports/report.md)).** The apex bootstrapping test of the arc, and the mission's "extend capability by a lot" hope tested head-on: bank ONLY depth-1+2 self-solutions (130 verified pairs, 83 at depth 2, zero depth-3 examples), and ask whether the banked model now *samples* depth-3 compositions the base never could. **No — DEPTH-LOCAL.** The depth-2 install works and generalizes to held-out tasks (coverage 0.12 → 0.36, tripled — a clean replication of C18's within-depth expansion), but depth-3 coverage stays at exactly **0.00** (base is 0.00 too). A strong depth-2 composition skill does *not* length-generalize up to make even one depth-3 task samplable. So banking installs only depths the base can *already sample*; it cannot bootstrap the frontier upward. This completes the mechanistic picture of the wall — depth-3 composition is **not represented** (C19), **not steerable** (C20), and **not reachable by banking-shallow** (C21); all three self-training / test-time shortcuts fail at the deep wall by the same underlying fact. The only way up is to **seed each rung externally**: tool-augmented harvest (C12 decompose-search reaches depth-3 that sampling can't) → execution-verify → bank. The precise recipe: *tools reach the next rung, banking installs it, then the base samples it — repeat.* Self-training is the installer, not the explorer.

**Decodability ≠ steerability (2026-07-03, claim C20, [qwen35_4b_activation_steering](../experiments/qwen35_4b_activation_steering/reports/report.md)).** The causal follow-up to C19: if the first op is linearly *decodable*, can we *steer it out* — add the decoded direction back to the residual stream and make the model use it (training-free elicitation, the mission's dream)? **No.** Mean-difference (ActAdd) steering at the C19 probe-best layer is INERT: at depth 1 (the cleanest direction, probe 0.99) steering toward the true op never beats the no-steer baseline and only degrades fluency at higher strength; at depth 2 there is a faint predicted-direction whiff (steer-true +0.05 over baseline, steer-wrong below) but it is within noise of the random control and below the pre-registered +0.10 bar. The null holds at earlier layers (8, 12) and on identification (0.03→0.03); every pre-registered prediction was refuted. So the "unexpressed" information C19 found is *readable but not writable into behavior*. This strengthens the arc's throughline from a new angle: **every test-time intervention has now failed to move the wall** — sampling+selection is free but adds no coverage (C17), and steering cannot elicit the latent signal (C20). The only levers that move deployable capability remain **weight edits (banking, C18)** and **externalization (tools, C12)** — installing the capability, not reading it out. Honest limit: a clean negative for *standard* ActAdd; activation patching / optimized steering vectors are untested.

**Inside the wall: latent at shallow depth, absent at the deep wall (2026-07-03, claim C19, [qwen35_4b_latent_composition_probe](../experiments/qwen35_4b_latent_composition_probe/reports/report.md)).** The first mechanistic probe of the generation wall — the whole C13–C18 arc had only measured it behaviorally. Linear probes on residual-stream activations (last identification-prompt token, all 33 layers, 1500 verified tasks) decode the composition's *first operation*. **The wall's character changes with depth.** At depth 1 the first op is decoded at **0.99** (computed in early-mid layers, plateauing by layer 15) while the model names it only 0.44 — representation ≫ expression, i.e. genuine *latent* capability. At depth 2 the probe reads 0.42 vs behavior ~0.13. At depth 3 (the wall) the probe reads 0.27 but the shuffled-label floor is 0.14, so the *real* decodable signal (~0.13) has fallen to the model's behavioral level — the representation itself has thinned to a thread. So the generation wall is an **expression** failure when shallow (the inverse is computed but not routed to output) and a **representation** failure when deep (the inverse is not computed). Layer-0/embedding stays at chance, so the signal is the model's computation, not surface I/O. **Consequence:** activation steering has real headroom at depth 1–2 but almost nothing to steer *toward* at the true wall — no readout conjures information the forward pass never computed. This locks together with C18: banking *works* because it **installs the representation the base lacks** (base depth-2 first-op only 0.42, depth-3 a thread). Only proposal-installation (banking / tools; C18 / C12), not test-time readout or selection, crosses the deep wall.

**Banking shifts the proposal distribution — concentration AND expansion (2026-07-03, claim C18, [qwen35_4b_coverage_banking](../experiments/qwen35_4b_coverage_banking/reports/report.md)).** The correctly-aimed follow-through to C17 (only shifting the *proposal* distribution can beat sample-more): harvest the fixed 4B's OWN execution-verified solutions (80 SFT pairs, no teacher), QLoRA-SFT single-shot, evaluate on **disjoint held-out** tasks. Banking does two things, cleanly split by depth. **Depth 1 (base already covers): CONCENTRATION** — think greedy@1 rises 0.60→0.80 while the coverage ceiling stays flat; coverage is pulled into the deployable single-shot. **Depth 2 (base barely covers): EXPANSION** — the coverage ceiling rises 0.15→**0.45 (3×) on held-out tasks**, i.e. the banked model proposes correct compositions the base never sampled. Crucially the unique-program count *drops* (11.5→10.85), so this is not more diversity — banking moved the proposal *mass* onto correct compositions, which is exactly the lever C17 named. **Depths 3–4 do not move** (7 / 0 training examples; the wall holds). Bounded, honest: banking does not beat think-mode sample-more at k=1 (banked greedy 0.80 < base sample-more 1.00), but *banking + sample-more* beats *base + sample-more* (0.45 vs 0.15 at depth 2). The path to push the wall deeper (depth 3+, where plain sampling harvests ≈0) is to seed the training set with **tool-augmented harvest** (C12 decompose-search) — the concrete loop back from C18 to C12.

**The generation wall is COVERAGE, not selection (2026-07-03, claim C17, [qwen35_4b_coverage_vs_selection](../experiments/qwen35_4b_coverage_vs_selection/reports/report.md)).** A pre-registered decomposition drew K=32 identification samples per task (list + register, depths 1–4, 8 visible + 8 hidden examples) and compared selectors to the coverage ceiling. **Selection is free** — max(coverage − vfilter) = 0.00 in every cell; 90% of visible-passers also pass the hidden set, so an 8-example execution-filter, the model's own C10-style verifier, and even a *random* pick among visible-consistent candidates all recover the coverage ceiling identically. Single-shot undersells accessible capability 2–5× (first@1→coverage@32: list d2 0.10→0.30, register d2 0.15→0.60, d3 0.05→0.25), and sample+filter recovers it — but that *is* sample-more. The coverage wall's depth is set by hypothesis-space size (list collapses at depth 3; register survives to depth 4 via a smaller 12-op menu), mechanistically explaining C16's register floor as coverage-driven. **Consequence for the mission:** you cannot beat "sample more" by better test-time *selection* — selection is already plumbing (confirms C10 strongly). The only lever that beats sample-more is shifting the *proposal* distribution: tool-enumeration (C12) or banking verified solutions into the weights (C11/C12) so the right program is proposed at k=1. The experiment refuted its own selection-centric predictions (P3: no selection gap; P4: the verifier equals random because there is no hard choice). Residual: overfit traps (visible-pass, hidden-fail) false-deploy at deep register — an abstention gap no example-filter catches.

**Two of these are now cross-substrate LAWS, one is not (2026-07-02, claim C16, [qwen35_4b_crossfamily_laws](../experiments/qwen35_4b_crossfamily_laws/reports/report.md)).** The C13–C15 ladder was re-run on two genuinely different fresh, execution-verified families (string char-edits, a 3-register integer machine) beside the list anchor. **Model-level and substrate-invariant:** (1) the *compiler* — plan-given execution ≈ 1.00 at every depth in all three families (one collapsed flat line); (2) the *generation wall* — bare identification of novel compositions decays toward chance everywhere (transcription−identification gap ≥ 0.84 at depth ≥ 3). So "tools identify, the model compiles" is a property of the weights, not of lists. **Substrate-dependent:** simulation fidelity — C15's decay "constant" was list-specific. Mental-simulation accuracy is set by the cost of tracking the *state representation*: a compact 3-int register state is simulated robustly to depth 4 (0.92→0.72, ~flat), a variable int list decays (1.00→0.56), a mutating character string is floored near-zero even at depth 1 (0.24→0.00). Deployment corollary: externalize simulation to a tool only where the representation is expensive to track; for compact integer state a tool call is wasted. New sub-law — the wall's *floor* ≈ f(hypothesis-space size, simulability): register alone is small-enough-to-search *and* simulable, and alone keeps a nonzero deep-identification floor.

### Universal-Feature Curriculum Search

The first three contamination-controlled designed-curriculum experiments separate
local installability from broad transfer. A truth-audited 13-skill curriculum can
raise fresh synthetic accuracy, but an 800-row sequential continuation specialized
and regressed three public families. Mixing 400 designed rows into 1,120 replay rows
at lower rate still passed the local gate yet lost to both the mature policy and a
matched replay-only continuation. The control is the productive surprise:
replay-only reached 0.4851 aggregate, +0.0441 over the mature policy, with eight
strict family gains and no regressions. Thus broad replay had not saturated, and it
must be treated as an active capability baseline. The exact-token follow-up removed
the prior 17.3% exposure gap, but 40- and 80-row designed doses both failed the fresh
local screen; even the stronger 80-row arm reached only 0.538 accuracy, 0.615 parse,
and 10 cap contacts. Its benchmark stayed sealed. The open search is therefore not
"make the designed dose tiny": it must bridge the 80-to-400-row install/retention
gap or isolate concise answer commitment while preserving an exact-token replay
control. No universal-feature claim exists until one arm passes fresh local gates,
beats replay continuation on every family, replicates on same-backend events, and
beats matched-compute sampling.

### Selection Under Visible Evidence Is A Core Bottleneck

[qwen35_4b_retrieval_adapt_verify_scale](../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md) found additional hidden-correct candidates, but deployable selectors still made too many wrong commits. Future work should treat selection as its own research object, with precision, recall, abstention, and hidden-oracle ceilings separated.

[qwen35_4b_answer_potential_trace_sft](../experiments/qwen35_4b_answer_potential_trace_sft/reports/report.md)
adds a distinct failure mode: a dense oracle-side score can pass relevance controls and still rank a
counterfactual state rather than a deployable one. Answer gain modestly improved top-trace outcomes, but
missed its frozen actionability margins while nearly every thought was force-closed. Trace selection now
needs three gates before banking: within-task discrimination, practical top-choice lift, and autonomous
termination/parseability at the exact scored seam.

[qwen35_4b_balanced_core_answer_potential_sft](../experiments/qwen35_4b_balanced_core_answer_potential_sft/reports/report.md)
shows what happens after repairing that interface, but has no capability verdict yet. Its uncapped bank has
22,681 naturally closed candidates and 360-task potential selections with thoughts up to 14,325 tokens. Two
new design gates appear before SFT: exact token dose (the six-arm matrix is 34,446,994 two-epoch forward
tokens) and control support (success-RFT has only 97 unique traces from 58 tasks in four of nine cells, then
repeats them seven or eight times). Equal row counts are not enough; trace-banking comparisons must disclose
forward-token dose, unique source exposure, and task/cell coverage before training.

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

C51 adds a termination corollary. A thinking budget is not just how many tokens are available: if almost
every trace contacts the cap, injecting `</think>` and teacher-forcing an answer evaluates a state the model
did not autonomously reach. For trace-valuing or trace-SFT work, calibrate on the actual workload, gate
natural closure and parseability before scaling, and score the close/commit event rather than assuming it.

The uncapped continuation adds a resource/support corollary. Once long reasoning is allowed to finish, a few
hundred examples can contain tens of millions of training forward tokens, and correctness-filtered baselines
can have sharply narrower task support than dense selectors. Pre-register token budgets and support-matched
estimands; do not use global oversampling to describe a sparse success control as balanced.

C52 adds an intervention-locality corollary. Entropy/varentropy can identify focused, conflicted thought
forks, and outcome labels at those forks can beat shuffled labels, without producing a capability gain.
Positive-only pull-up is safer than pairwise demotion, but shared LoRA weights still move neighboring logits
enough to erase transfer. Require an exact-logit locality preflight—then an absolute base and matched-compute
agentic win—before treating thought-token steering as capability elicitation. Higher varentropy is not a
license to push harder; its measured safety relationship was non-monotone.

The unclaimed Jacobian transport result adds a causal-representation corollary. An averaged J coordinate at
layer 24 redirected 18/24 direct concept reports, but the same edit changed 0/24 separately mapped
consequences at every tested layer; the consequence margin was flat even while the direct margin crossed
zero. **Decodability, local writability, and downstream transport are distinct gates.** A token-aligned
direction that controls an imminent word is not yet a reusable reasoning variable. Future J-space work must
use a true context-local set clamp, exact realized-delta-norm controls, and a consequence firewall before it
can license thought-prefix valuation or capability claims. See
[qwen35_4b_jacobian_value_transport](../experiments/qwen35_4b_jacobian_value_transport/reports/report.md).

Its first context-local follow-up produced the opposite mechanistic signature: at the earlier selected-key
token, all-24 J clamping changed 48/48 direct keys **and 48/48 separately mapped digits**; pair-only J reached
47/48 consequences, wrong-donor J produced its own digit 48/48, and logit-lens/random controls produced
0/48. Full-state donors localized transport to early/middle bands and became inert after layer 20. But that
experiment correctly remained `INVALID_CONTROL`: one of 96 random rows missed the realized-norm bar and
bf16 rounding introduced up to 5.7% realized J-span projection. See
[qwen35_4b_context_local_jacobian_clamp](../experiments/qwen35_4b_context_local_jacobian_clamp/reports/report.md).

A fresh quantization-aware replication now resolves that invalidity. With the lens, band, scale, grammar,
and model revision frozen, all-24 J again changed 48/48 direct keys and 48/48 mapped consequences; pair J
reached 46/48, wrong-donor J produced its own consequence 48/48 and the target 0/48, while two independent
random arms and the concept logit lens remained 0/48. All 480 calibration and 960 confirmation control-layer
deltas met post-bf16 norm and <=1% J-span constraints; both paired bootstrap intervals were [1,1]. This is
strong evidence for a compact **causally consumed context-local concept state**, not merely an output motor.
The scope boundary is equally important: donor identity and coordinates are supplied, so this is an oracle
mechanism—not a capability gain. It unlocks native-thought value transport and a learned non-oracle
controller, which must still beat frozen and matched-compute sampling. See
[qwen35_4b_jacobian_transport_control_replication](../experiments/qwen35_4b_jacobian_transport_control_replication/reports/report.md).

The first native-thought transfer then failed one step earlier than value. On 16 fresh,
first-operation-identifiable tasks, every one of 48 natural traces contacted the frozen 160-token thought
cap without `</think>`; natural close, parse, success, and mixed-task counts were all zero. Model smoke also
measured up to 0.0625 historical-token activation drift when only suffix length changed, exposing Qwen hybrid
kernel geometry as a nuisance for fixed-delta patching. The frozen `NO_NATURAL_SEAM` decision correctly
canceled value fitting and causal outcomes. **This does not falsify J-space certainty**: it shows that prefix
value cannot be evaluated until the model naturally reaches the answer seam, and that later controls must be
matched dynamically per live sequence length. The warranted successor is a selection/confirmation budget
ladder, not an in-place cap increase. See
[qwen35_4b_native_thought_jacobian_value_transport](../experiments/qwen35_4b_native_thought_jacobian_value_transport/reports/report.md).

The separate natural-close repair then exhausted its full registered ladder.
On 16 new tasks, all 48 paired traces reached 1,024 thought tokens without
`</think>`, so close, parse, and usable-prefix rates were 0% at 256, 512, and
1,024; no cap was selected and the 24-task confirmation remained sealed. Every
row passed the cached-forward audit, and a post-decision scan found no exact
short-period pattern across any final 256-token tail. This rules out neither
coherent ongoing reasoning nor J-space certainty—it rules out **autonomous
termination as the measurement interface** at the deployed budget scale for
this workload. Do not keep extending the natural cap. A forced close can now be
studied only as an explicit test-time commit policy: calibrate and deploy the
same action, preserve C51's counterfactual-state label, require fresh
parse/headroom evidence, and use live-prefix/per-length controls before any
causal claim. See
[qwen35_4b_native_thought_seam_budget_ladder](../experiments/qwen35_4b_native_thought_seam_budget_ladder/reports/report.md).

Making close injection the explicit deployment action did not repair the seam.
On another 48 fresh traces, forced-only parse was 12.5%/16.7%/18.8% at
256/512/1024, exact success was 1/48 at every cap, only one task mixed outcomes,
and 85%--96% of post-close generations exhausted all 16 answer tokens. Decoded
only after the automatic failure, many answers simply restarted analysis. An
EOS-tolerant parser raised parse at best to 22.9% and correctness to 2/48, so
parser strictness cannot explain the stop. **A close delimiter is not an answer
slot.** This independently strengthens C51 and keeps J-space value unopened.
The warranted next interface supplies fixed syntax (`First:`) but never answer
identity, reads the next alias choice, and retains close-only free-form output as
a matched control; it must earn semantic headroom on fresh tasks before any
internal-value result. See
[qwen35_4b_forced_commit_jacobian_value_transport](../experiments/qwen35_4b_forced_commit_jacobian_value_transport/reports/report.md).

Supplying the fixed answer slot repaired that emission boundary but did not yet
earn semantic stability. In
[qwen35_4b_commit_slot_jacobian_value_transport](../experiments/qwen35_4b_commit_slot_jacobian_value_transport/reports/report.md),
an alias was already the unmasked full-vocabulary top token on 41/48 cap-1,024
rows and the 12 aliases held 68.5% total probability mass, whereas matched
close-only free-form parse was 2/48. Thus `First:` established answer mode; the
alias mask itself was rarely the deciding repair. Ordered thought at 1,024
scored 15/48 versus the equivalent 12/48 no-thought and 11/48 exact-token-
multiset shuffle, passing both frozen pooled gap gates. But only five tasks mixed
correct and incorrect traces versus six required, task-bootstrap intervals for
both gains crossed zero, and the effect was concentrated in a few task/alias
cells. Correct-alias verbalization was negatively rather than positively
associated with success, and three label-free logit-residual diagnostics all
underperformed the original slot. The terminal `COMMIT_SLOT_SEAM_FAIL` therefore
keeps confirmation and J value unopened. **Formatting is now separated from
semantic resolution:** power a fresh fixed-1,024 task-level replication before
raising the cap, changing the decoder, or fitting any certainty coordinate.

That powered repair now succeeds independently twice. In
[qwen35_4b_commit_slot_semantic_power_replication](../experiments/qwen35_4b_commit_slot_semantic_power_replication/reports/report.md),
qualification on 113 fresh tasks scored 92/339 ordered versus 46/339 exact
token-multiset shuffle and 11/113 no-thought; confirmation on another 113 tasks
scored 98/339 versus 47/339 and 8/113. One-sided task-bootstrap lower bounds for
ordered-minus-shuffle were +8.85pp and +9.44pp; mixed tasks were 32 and 31;
unrestricted top-is-alias was 88.2% and 87.6%. Each stage independently passed
every frozen gate. A post-decision audit found ordered-only versus shuffled-only
paired wins of 60:14 and 64:13, and correct-answer mentions did not explain the
gain. **The fixed cap-1,024 semantic commit seam is therefore replicated:**
coherent ordered thought carries task-general answer information beyond syntax
and an identical token bag.

This finally licenses native prefix-value/J measurement, but does not itself
establish J-space certainty or capability. Alias identity remains sharply
heterogeneous—one confirmation target had zero ordered successes, and shuffle
favored two targets—so a raw gold-logit/J coordinate can be a label or prior
readout rather than value. The next stage must use task-held-out folds and prove
incremental signal beyond correct-alias activity, ordinary slot margin, and
alias identity before sealed causal work. Every control must be constructed at
the live prefix length after bf16. Even a positive oracle value/causal stage
would still precede the deployable bar: a label-free controller must beat frozen
inference and matched-compute sampling on new contamination-free tasks.

**A repaired recovery interface reveals complementary local policies, not a
global winner (2026-07-12, Negative,
[qwen35_4b_recovery_payload_budget_harness](../experiments/qwen35_4b_recovery_payload_budget_harness/reports/report.md)).**
Giving every arm a 512-token JSON payload and measuring valid two-turn recovery
removed the predecessor's truncation/proxy artifacts: the locality-safe λ=.18
policy passed development transfer at 71.25%, +12.5pp over base and +21.25pp
over equal-reservation sampling, with perfect controlled transitions and exact
ordinary-task retention. Independent confirmation was still decisive: λ=.18
and action-only tied at 68.75%, failing the frozen +3pp incumbent margin while
all other gates passed; Menagerie stayed sealed. Their hidden-success union was
78.75% on both fresh blocks, with disjoint wins concentrated in different
algorithm/state cells. This is oracle evidence for complementarity, not a
deployable ensemble claim. The next safe capability producer is bounded
branching selected by public verifier evidence, followed only conditionally by
state→action-balanced winner banking; do not tune another scalar reason dose or
route on family identity.

**Policy complementarity did not transfer; the residual is conjunctive semantic
proposal, not recovery control (2026-07-12, Negative,
[qwen35_4b_recovery_verifier_branch_tournament](../experiments/qwen35_4b_recovery_verifier_branch_tournament/reports/report.md)).**
The action-only and reason-mixture recovery policies each scored 73.75% on four
new procedural families, but their deterministic union reached only 75.0%,
exactly tying equal-reservation action pass-if-either sampling. Frozen
feasibility correctly stopped the public selector before confirmation,
curriculum production, or Menagerie. Paired outcomes isolate the boundary:
58/80 were solved by both, one by each alone, and all 20 shared failures were
atomic reservations. Both policies still changed patches within two turns on
every controlled state; traces instead oscillated between whole-request
validation and input immutability, composing both only once across 40
stochastic action trajectories. Public routing cannot create a proposal neither
source makes. The warranted curriculum is executable supervision for the
transactional validate-copy-commit invariant across diverse families, mixed
with the existing conditional recovery replay—not another selector, branch, or
global reason dose.

**Transaction structure installs locally, but verifier-specific validation
policy is a separate unit (2026-07-13, Negative,
[qwen35_4b_transaction_invariant_recovery_curriculum](../experiments/qwen35_4b_transaction_invariant_recovery_curriculum/reports/report.md)).**
The action-seam curriculum passed apex-relative locality (0.119 drift), perfect
two-turn recovery, and trained-family calibration at 81.7%, versus parent 51.7%
and matched replay-only 38.3%. On unseen transaction dev it reached only 71.9%,
versus parent and equal-compute sample-more 70.3%; frozen +10/+5 and bootstrap
gates failed, so confirmation, broad retention, and Menagerie stayed sealed.
Yet every one of 16 target first patches newly proposed copied state,
whole-request resource validation, and atomic per-request commit. All omitted
the distinct negative-amount exception, then overcorrected after visible
failure by raising on every insufficient request. **Proposal structure and
validation-policy fidelity are empirically separable.** The next useful
curriculum should start from near-correct failed-test states and teach minimal
policy-preserving revisions across raise/False/None/reject semantics, rather
than add generic transactional examples or dose.

**Failure forensics do not guarantee curriculum headroom (2026-07-13,
Negative,
[qwen35_4b_validation_policy_counterexample_curriculum](../experiments/qwen35_4b_validation_policy_counterexample_curriculum/reports/report.md)).**
The residual-seam candidate passed direct C54 locality at 0.109 drift, but its
controls-first trained-family instrument was saturated before candidate
behavior: transaction parent 48/48 and matched extra-training control 48/48,
with every first patch already expressing negative handling, copy, and ordinary
false rejection. Explicitly stating the exception contract and supplying an
otherwise-correct partial converted the predecessor's implicit-policy failure
into a solved edit. Frozen +15/+10 bars were impossible, so transfer and
Menagerie stayed sealed. **Curriculum substrate qualification must precede
capability-production training:** demonstrate replicated parent headroom on
the exact prompt/verifier state, not merely a related historical failure, then
reserve disjoint skins for training and transfer.

**The semantic-recovery frontier moves from revision to evidence acquisition
(2026-07-13, Instrument failure with diagnostic routing value,
[qwen35_4b_semantic_policy_headroom_tournament](../experiments/qwen35_4b_semantic_policy_headroom_tournament/reports/report.md)).**
The exact learned transaction parent ran on two disjoint 72-case blocks, but
answer-cap contacts reached 12.08%/12.67% versus a frozen 5% ceiling, so the
formal verdict is `INSTRUMENT_FAIL` and no curriculum is licensed. No axis
qualified independently: negative and non-integer failed-test recovery were
perfect across both blocks; blank was 8/9 and 7/9 with only one in-band shape
per block and a different shape each time.

The opened trajectories still expose a useful state distinction without
rescuing the formal result. Every failed-test case reached a correct patch, and
four endpoint misses were later regressions. In rejected states with no test
output, inferred-contract first-patch correctness was 0/54, visible-test reads
before first patch were 0/72, and 64/72 eventually became correct after later
evidence. **The next capability strategy is active specification acquisition:**
counterfactually pair nearly identical issue/source states with discriminating
public evidence, teach `inspect evidence → evidence-faithful first patch`, and
replay the full recovery/verify/commit loop. Repair payload measurement in a
fresh frozen preflight; do not turn answer closure, entropy, or varentropy into
the capability label.

**A curriculum cannot repair inherited comparator distance (2026-07-13,
Stopped, unclaimed,
[qwen35_4b_counterfactual_evidence_acquisition_curriculum](../experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/reports/report.md)).**
Before any interface behavior or training, the exact transaction-replay parent
missed the frozen direct C54-apex locality prerequisite: median centered
non-target logit drift was 0.110735 on 48 fresh contexts versus 0.10, with all
48 row estimates above the ceiling. Entropy was retained (+0.013636),
varentropy was flat (+0.000297, diagnostic), and prompt equivalence held. The
formal verdict is `LINEAGE_LOCALITY_INFEASIBLE`; qualification, training,
transfer, retention, and Menagerie stayed sealed.

This neither tests nor disfavors active specification acquisition. The
reusable rule is narrower: direct locality eligibility must be established for
the exact parent, anchor, context block, and ceiling; it cannot be inherited
from lineage or a looser prior gate. Reopen the capability question only in a
fresh, apex-rooted or prospectively parent-qualified successor—never by raising
the observed ceiling or repairing this directory. No claim ID is allocated.

The licensed J-value measurement is now a clean negative for the shared
coordinate. All 144 value traces and 288 midpoint/endpoint states completed,
but task-held-out shared J AUC was 0.5021 with one-sided task-bootstrap lower
0.4417. Ordinary slot margin reached 0.5448 and a layer/dimension-matched non-J
residual readout reached 0.5292; all paired incremental uncertainty bounds
crossed below zero. The shuffled-within-task null averaged 0.5061, and every
rank/context/cardinality/finite control passed, so this is `NO_PREFIX_J_VALUE`,
not an invalid run. Causal confirmation stayed sealed.

The diagnostic lead is a **phase reversal**: the shared model's midpoint states
ranked later full-cap value at 0.6083, while endpoint states ranked it at 0.3958.
This means the replicated semantic seam does not imply a phase-invariant scalar
J certainty coordinate. It also means “no value anywhere” is too strong: the
midpoint point estimate warrants outcome-labeled diagnostics on already-open
rows. It cannot rescue the experiment. Any midpoint-specific axis must be frozen
and replicated on fresh tasks, beat equal-width non-J and margin controls with
task uncertainty, and pass cross-phase tests before causal data can open.

That allowed audit has now falsified the midpoint-specific lead on the opened
rows. A direct midpoint-only task-held-out fit reached 0.5375 AUC with lower
0.4417, below matched non-J state at 0.6000 and effectively tied with slot
margin at 0.5396; endpoint-only J was 0.4292. Midpoint/end coordinate
correlations averaged -0.0386 and coefficient cosine was -0.0681. This is
post-decision evidence, not a claim-grade result, but it changes routing: do not
spend a fresh split on the scalar midpoint J hypothesis. Preserve the replicated
coherent-content seam and seek label-free semantic interventions with matched-
compute controls.

The first non-scalar attempt is also gated negative. On 113 tasks, choosing the
alias with largest mean `P(ordered)-P(exact shuffle)` reached 0.3805 accuracy,
well above hard majority at 0.2920. It even recovered eight correct aliases not
chosen by any of the three paths. But it was only +0.0177 over minimum entropy
and +0.0265 over max confidence, with one-sided paired bounds below zero, while
an explicitly oracle-balanced task-mismatched shuffle reached 0.3894. Terminal
`NO_ORDER_SUPPORT_SELECTOR` left confirmation absent. The combined lesson is
now sharper: coherent thought transports answer-relevant semantics, but neither
a shared J scalar nor raw terminal counterfactual attribution reliably values
individual tasks. Stop re-ranking this commit state; move interventions earlier
to create different continuations/proposals, then demand a matched-compute win.

Moving the edit earlier is not sufficient. The balanced native branching
successor first spent four outcome-blind receipts making all post-bf16 controls
valid, then evaluated 144 supplied-target writes without loading gold. At every
norm-anchored alpha, J selected its supplied alias on 4/48 branches—exactly
1/12 chance and identical to non-J. Mean target-probability lift peaked at only
0.00566. Terminal `NO_NATIVE_J_BRANCH_CONTROL` therefore stops before any
continuation. This does not contradict the replicated donor-coordinate clamp:
it distinguishes **coordinate replacement at an explicit semantic token** from
**additive directions at an arbitrary last-thought token**. The only warranted
native J branch left is a final context-local anchor test with donor coordinates
and plain-text/full-activation controls; larger alpha/layer sweeps are retired.

That final late-anchor test is now terminal `INVALID_MECHANICS_CONTROL`, and its
adversarial audit prevents an overbroad negative. All 880 live numeric and 2,240
intervention rows reproduce calibration after canonical identity sorting.
Within the constrained alias set, text/full donor wrote 43/44 supplied aliases,
donor J wrote 42/44, and wrong-donor J wrote its own alias 42/44. But global
next-token parse was only 56/880, no consequence row parsed, and constrained
consequence stayed at 6/44 text/full versus 5/44 donor J and 4/44 non-J/source;
donor-J probability lift was only `+0.00170`. Worse, both component mappings
advanced by the same cyclic task index, so the purportedly randomized alias ->
operation -> label composition was constant across tasks. **Direct semantic
writing is not consumed computation, and independently changing component maps
does not prove their composition changes.** This exact late opaque interface is
retired, but the parse failure and composition confound mean it cannot establish
a general negative about native J-state transport. The warranted pivot is a
fresh deployable experiment that supplies concrete text hypotheses before
reasoning, generates full continuations, tests composed relations directly,
and beats matched-compute sampling with a visible-only selector before any
installation claim. See
[qwen35_4b_semantic_anchor_coordinate_branching](../experiments/qwen35_4b_semantic_anchor_coordinate_branching/reports/report.md).

**Early concrete text is local control, not yet composition (2026-07-13,
Promising diagnostic inside terminal instrument failure).** The fresh
early-text pivot supplies a sharper split, but still stops before a capability
test. All 392 mechanics rows authenticated in
[qwen35_4b_early_text_hypothesis_forking](../experiments/qwen35_4b_early_text_hypothesis_forking/reports/report.md).
Correct and deranged early bound hypotheses each drove their own supplied
operation on 84/96 direct-execution rows, while deranged produced 0/96 of the
registered target; the effect covered all 24 operations and all four contexts.
So semantic text present before thought can be **consumed local control**, not
merely a late writable name. But every diagnostic arm exceeded the frozen 5%
answer-cap ceiling, duplicate/placebo also failed parse, and the independent
full-program ceiling was only 3/8. Terminal `INVALID_INTERFACE_PARSE` kept all
qualification and matched-sampling arms sealed. **The remaining bottleneck is
composition, not local addressability:** materialize each candidate operation's
public consequences and ask the model for the residual relation on a fresh
depth-three substrate; do not spend another run on opaque names, timing, parser
repair, or larger budgets.

The first materialized-residual implementation has no capability result. Its
model-free 264-task construction passed, but one live attempt stopped before an
experimental request and the repaired attempt returned 52 rows only in memory:
a receipt expected tokenizer EOS `248044` although the pinned tokenizer uses
`<|im_end|>` ID `248046`, and semantic authentication preceded durable writes.
The terminal `STARTED` transaction cannot be replayed. This is an infrastructure
incident, not evidence for or against residualization. Resume only with fresh
task/record identities and seed domain, while durably quarantining returned
bytes before authentication. See
[qwen35_4b_materialized_residual_sibling_search](../experiments/qwen35_4b_materialized_residual_sibling_search/reports/report.md).

The fresh successor supplies the missing durable test, but splits cleanly by
interface. All nine transactions and 1,984 rows authenticated in
[qwen35_4b_materialized_residual_sibling_search_fresh_replication](../experiments/qwen35_4b_materialized_residual_sibling_search_fresh_replication/reports/report.md).
Free generation is terminal `MECHANICS_INTERFACE_INVALID`: every thought hit
cap, the materialized/name/shuffled/echo suffix arms parsed only 12/7/12/20 of
52, and materialized/name/shuffled/direct solved zero. That is not a broad
residualization refutation because the registered ABI failed. The parse-immune
cheap ranker is a clean negative: materialized recall@4 0.257 lost to every
structured comparator (name 0.281, shuffled 0.323, listwise 0.271, surface
0.375) and cleared random by only 0.149 versus a +0.15 gate while missing all
absolute floors. Retire the cheap viability/top-four path. A future residual
generator must first pass a disjoint, known-answer echo calibration at >=90%
parse/exact echo and <=5% cap contact; more cap, parser relaxation, and post-hoc
threshold changes are not evidence.

That calibration has now failed cleanly while revealing a narrower interface
hypothesis. All 240 outputs authenticated in
[qwen35_4b_materialized_residual_answer_seam_factorial](../experiments/qwen35_4b_materialized_residual_answer_seam_factorial/reports/report.md),
but every think/no-think x freeform/`PROGRAM:` arm was 0/48 strict parse and
exact echo under the registered HF-model-EOS boundary, so mechanics stayed
sealed. Post-decision, removing only an exact `<|im_end|>\n` suffix recovered
48/48 frozen-parser exact outputs in both no-think arms. Thinking retained extra
close-boundary failures: 38/48 think/`PROGRAM:` and 24/48 think/freeform after
suffix removal, although their final expected-answer segments matched 48/48 and
29/48. The sampled boundary was tokenizer EOS 248046, newline 198, then HF EOS
248044. This is not a license to repair the parser after seeing the result. It
is evidence that **termination identity is part of the interface**: run one
fresh successor that registers first tokenizer EOS as the answer-stage commit
event, retains HF EOS and malformed terminators as controls, and must
independently qualify before residual mechanics. If it fails, retire this
residual-generation branch.

That fresh successor has now qualified the interface cleanly in
[qwen35_4b_tokenizer_eos_answer_commit_factorial](../experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/reports/report.md).
Across 48 fresh known-answer rows, both tokenizer-EOS no-think cells were 48/48
strict exact/parse with zero cap contacts, while all matched HF-model-EOS cells
were 0/48; all 192 paired prefixes authenticated. Thinking was worse at this
short exact-output interface (38/48 structured, 30/48 freeform, with 16
freeform cap contacts), so only the frozen no-think `PROGRAM:` cell advances.
This is strong causal evidence that **the termination token is part of the
deployed interface**, not evidence that a capability was installed or
magnified. Residual mechanics and hidden labels stayed sealed. The next result
must come from the frozen winner behind a second committed-green lock and must
beat every structured and taskwise matched-compute direct control.

The conditional run later confirmed transport but did not reach a capability
comparison. Under a separately reviewed and committed-green lock, the selected
interface achieved 24/24 exact echo and parse with zero caps, and all five
transactions durably preserved 4,056 outputs. Visible analysis then failed
because post-chain replay of the stored transport decision reused the initial
authorization primitive, which requires later invocations to be absent. No
visible selection or hidden read occurred. This is a terminal instrument
failure: it strengthens the local interface result but leaves residual
capability unadjudicated. A fresh successor must distinguish one-time temporal
authorization from later immutable-decision replay.

The replay-hardened fresh successor now closes that uncertainty in
[qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay](../experiments/qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay/reports/report.md).
It replicated calibration (48/48 in both no-think tokenizer-EOS cells; all
HF-EOS controls 0/48), passed transport 24/24, authenticated all 4,056 outputs,
and cleared the generation ABI with 98.78--99.83% parse and zero cap contacts.
The result is nevertheless terminal negative: materialized, name-only,
shuffled, sampled-token-matched direct, and logical-model-token-matched direct
all achieved 0/24 selected success and 0/24 oracle proposal coverage. Exhaustive
CPU search found 88 visible-consistent, hidden-correct programs and covered all
24 tasks, so this is proposal-distribution failure rather than interface,
selector, or task failure. **A reliable answer-commit seam does not expose a
residual synthesis capability.** Retire this semantic-materialization prompt.
Any J-space continuation must earn a task-held-out correctness readout beyond
ordinary and equal-width non-J controls and then a forward causal increase in
correct-proposal coverage over matched sampling; a readable coordinate alone
does not install or elicit capability.

**Exact-token universal dose interpolation exposes a narrow answer-commit seam
(2026-07-13, local mechanism evidence).** The contamination-free
[qwen35_4b_universal_mid_density_token_match](../experiments/qwen35_4b_universal_mid_density_token_match/reports/report.md)
bridge fixes row count, slot order, optimizer steps, parent, and forward-token
exposure while replacing zero, 160, or 240 replay rows with truth-audited abstract
skills. Fresh local seed 88005 gives a sharp nonmonotonic result: replay is 17/26
accurate, 18/26 parsed, with nine cap contacts; 160 rows reach 19/26, 23/26, and
three; 240 rows fall to 17/26, 22/26, and five. The 160-row arm misses the frozen
parse and cap bars by one case each, so benchmark promotion is correctly empty.
**More generic curriculum is not the next lever:** keep the 160-row capability mix
and target the remaining commit/termination seam under a fresh exact-token control.
This is local mechanism evidence only; broad retention and universality remain
unmeasured.

**Higher autonomous-close loss does not cross that seam (2026-07-14, local
mechanism negative).** The result-separated
[qwen35_4b_universal_close_weight_token_match](../experiments/qwen35_4b_universal_close_weight_token_match/reports/report.md)
starts from the authenticated 160-row near-miss and compares replay, ordinary fresh
execute/induct training, and byte-identical training that changes only the natural
`</think>` span from weight 0.2 to 1.0. Fresh seed 88006 gives parent
16/26 accuracy, 20/26 parse, and six caps; replay 14/26, 18/26, and eight;
ordinary target 15/26, 23/26, and three; close-weighted target 16/26, 23/26,
and three. Promotion is empty and seed 78136 remains sealed. The paired treatment
contrast is decisive at this dose: **fresh target data improves emission, but close
weighting adds no parse or cap benefit**, leaves execute/induct at 0/4, and only
redistributes parent task wins. Do not tune close weight again. The next mechanism
must couple bounded computation with canonical answer commitment under a fresh,
unchanged gate; broad transfer remains unmeasured.

**Separately scored canonical search substates still do not transfer to the deployed
execution interface (2026-07-14, local mechanism negative).** The result-separated
[qwen35_4b_universal_search_scaffold_token_match](../experiments/qwen35_4b_universal_search_scaffold_token_match/reports/report.md)
replaces 80 exact-token-matched replay rows with 16 each of apply, fit, reject,
execute, and two-branch search lessons. Fresh seed 88007 gives parent/replay/scaffold
18/16/16 correct, with all three at 23/26 parse and three caps. The scaffold is 0/2
execute, 0/2 induct, and 0/2 probe; promotion is empty and seed 78137 stays sealed.
The mechanism anatomy is sharper than the aggregate null: both execute traces reach
the correct final state but run to cap without answering, while probe accuracy falls
from 2/2 in both controls to 0/2. **Canonical two-operation decomposition neither
installs commitment nor preserves independent simulation/scoring at the
natural-language variable-depth interface.** The next trial must change that
interface—explicit variable-depth state tables, separately verified hypothesis
scores, and answer-only commit—not add another dose of the failed scaffold.

**Truth-audited natural-language state tables still remain off-policy at deployment
(2026-07-14, local mechanism negative).** The result-separated
[qwen35_4b_universal_state_table_compiler_token_match](../experiments/qwen35_4b_universal_state_table_compiler_token_match/reports/report.md)
replaces 80 exact-token-matched replay rows with 20 each of variable-depth execution
tables, independent hypothesis score tables, first-error repair, and verified commit.
Fresh seed 88008 gives parent/replay/candidate 19/16/16 correct, 23/21/22 parsed, and
3/5/5 caps; execute+induct+probe totals are 4/6, 2/6, and 1/6. Promotion is empty and
seed 78138 stays sealed. The anatomy separates narrow computation gains from a failed
procedure install: candidate fixes one trace and one optimization case, and computes
one state exactly before losing only on whitespace, but treats a cycle declaration as
an operation, repeats both induction cases to cap, miscounts probe distinctness, and
fails to commit one correct execute result. **Executable truth in an idealized trace
is insufficient when the trace is not conditioned on the model's actual failure
prefix.** Retire another hand-authored surface. The next controlled test should train
fresh on-policy failure-prefix corrections with executable oracle continuations and
exact serialization, while preserving same-parent exact-token replay, fresh seeds,
and the unchanged local gate.

**On-policy failure collection still fails when correction begins after a long
realized prefix (2026-07-14, local mechanism negative).** The result-separated
[qwen35_4b_universal_on_policy_prefix_repair_token_match](../experiments/qwen35_4b_universal_on_policy_prefix_repair_token_match/reports/report.md)
mines 230 reachable failures from 288 fresh parent rollouts, balances six failure
classes, and replaces replay targets with 60 masked-prefix executable-oracle
corrections. Candidate and replay each train 320 rows for 40 updates and exactly
304,313 forward tokens, with 200 aligned replay positions. Fresh seed 88009 gives
parent/replay/candidate 16/18/15 correct, 24/23/23 parsed, 2/3/3 caps, and 2/1/0 of
six on execute+induct+probe. Candidate is 0/2 on each target kind, fails every
relative check, and has one paired win versus four losses against replay. Promotion
is empty and aggregate seed 78139 stays sealed. **On-policy substrate alone is not
enough: teacher-forcing the model's long realized failure state does not install the
earlier decision policy needed on a fresh trajectory.** Because masking also removes
33,421 supervised target tokens and selected prefixes are cap-heavy, the supported
negative is the complete matched-forward-compute recipe, not all on-policy learning.
Retire long masked failure-prefix continuation; move the intervention to short
pre-failure decisions and match target exposure in the next result-separated test.

**Clean on-policy restarts improve termination but still do not install target
competence (2026-07-14, local mechanism negative).** The result-separated
[qwen35_4b_universal_failure_selected_restart_target_match](../experiments/qwen35_4b_universal_failure_selected_restart_target_match/reports/report.md)
removes the failed trajectory entirely and closes the predecessor's exposure
confound: candidate and replay each have 320 rows, 297,731 forward tokens, 126,796
loss-bearing targets, absolute loss mass 27,632.8, 40 updates, zero skips, and 200
aligned shared rows. The candidate teaches four fresh selected failures per each of
13 skills from the original prompt. Fresh seed 88010 nevertheless gives
parent/replay/candidate 17/16/15 correct, 21/22/25 parsed, 5/4/1 caps, and 2/2/0 of
six on execute+induct+probe. Candidate fails its accuracy and every target floor plus
all strict relative checks; promotion is empty and aggregate seed 78140 stays sealed.
**The intervention learned bounded answer emission, not the semantic policy:** it
cut parent cap contacts by four and mean output by 34 tokens while losing two correct
tasks and both probe successes. Hand-authored executable truth remains off-policy
even when it starts before the error. The next controlled mechanism should distill
short verifier-correct sibling trajectories that the same model actually samples on
greedy-failure tasks, with an availability gate, exact-exposure replay, and a
matched-compute sample-more ceiling.

**Balanced failure-only sibling mining can be infeasible precisely because some skills are already saturated (2026-07-14, prerequisite stop).** The result-separated [qwen35_4b_universal_successful_sibling_target_match](../experiments/qwen35_4b_universal_successful_sibling_target_match/reports/report.md) froze 624 fresh tasks and one authenticated parent event before any sibling sampling. It found 227 hard failures overall, but the mandatory four-per-skill gate failed: count and route had zero hard failures and select had two. No sibling input, training, local event, or benchmark event ran; aggregate seed 78141 stays sealed. This does not test successful-sibling distillation. It corrects the intervention unit: repair data should target the ten skills with a live residual, while active replay and the unchanged all-skill gate preserve already-mastered skills. A successor may reuse the immutable published collection only in a new result directory with a prospective residual policy.

**The axis-line closure arc: installs replicate, the repair axis is dead, and dose-stacking saturates one adapter (2026-07-15, three preregistered events).** The stack trial ([qwen35_4b_axis_replay_stack_medium_target_match](../experiments/qwen35_4b_axis_replay_stack_medium_target_match/reports/report.md)) replicated the axis install across parents (+6 axis total; hygiene 9/10 twice; best-in-event termination) and measured replay round-two LOCAL drift; a training-free re-adjudication with a detectability-corrected bar ([qwen35_4b_axis_stack_readjudication_medium_pilot](../experiments/qwen35_4b_axis_stack_readjudication_medium_pilot/reports/report.md)) confirmed the map on a third fresh instrument (hygiene/explore/termination install; tracefix trends to chance; protocol is redundant dose) and preserved the three-event failure forensics (1,296 completions) that specified v2; and v2 itself ([qwen35_4b_axis_corpus_v2_staged_repair](../experiments/qwen35_4b_axis_corpus_v2_staged_repair/reports/report.md)) fired its preregistered kill rule — DEMONSTRATED bounded search installed no better than asserted search (bugfind 3/0/3 tie, bugmend 3/4/2 loss) — while exposing THIRD-DOSE INTERFERENCE: the third designed dose continued in place on one rank-32 lineage tied its parent on the axis total (19/50), lost the installed explore edge, and dropped retention five points as the third replay round won the holdout outright (25/50). Standing laws: (1) trace-repair is not installable in this model by small designed doses regardless of pedagogy — closed by kill rule, reopenable only with a new mechanism argument; (2) one adapter lineage saturates after ~two designed doses; fresh adapters from clean parents are required for further content, and replay continuation remains the strongest single broad-instrument move; (3) honest kill rules and preregistered readings converted seven consecutive non-promotions into a precise, cumulative mechanism map rather than noise.

**Axis atoms install without forgetting but under-convert to family scores; replay compounds a third time; the all-families goal is now blocked by exactly two frozen constants (2026-07-15, first local promotion + aggregate pilot negative).** [qwen35_4b_goal_gap_axis_curriculum_target_match](../experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/reports/report.md) designed 160 single-turn atoms for the four empirically stuck benchmark families (public axis descriptions only, fresh vocabulary, executable truth, unique-repair/unique-route/parseable-wrong-decoy constructions) and trained them against a three-axis exact-exposure replay control from the `designed_fresh` parent. It became the FIRST universal-line experiment to pass its local gate — axis holdout 28/40 vs parent 22 / replay 18, retention byte-equal to the parent — and then consumed sealed seed 78,144: base 0.1085, axis 0.4223, parent 0.4644, replay_repeat 0.5081. The candidate beat base +0.3138 with 7 strictly positive families, 3 ties, 0 negatives and flipped warren; the replay control flipped rites and posted the line's best recorded aggregate; the pilot gate failed on the aggregate comparisons and the experiment closed per contract. Portfolio laws sharpened: (1) task-level installation and family-level scoring are separated by more than surface — sirens stayed at exactly 0.500 despite the hygiene kind nearly doubling locally, menders at 0 despite the tracefix win; (2) replay continuation has now compounded aggregate three consecutive times (0.4410→0.4851→0.5081 across lineages) and is the strongest single measured intervention; (3) menders = 0 and sirens = 0.500 for EVERY arm at EVERY seed at quick/tb1024 — the goal's remaining wall is two families wide, and the next believable step is instrument-forensics on those two constants, not another same-shape dose.

**The designed dose is surface-general and think-economical; the budget lesson is not; and the local gate's induct floor is the wall made exact (2026-07-15, terminal local negative with a positive preregistered mechanism reading).** [qwen35_4b_universal_fresh_surface_budget_commit_target_match](../experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/reports/report.md) re-rendered the near-miss designed160 distribution on six entirely fresh surfaces and trained it (plus a 40-row bounded-scan budget-commit ablation) against a three-axis exact-exposure replay control from the authenticated parent, then judged all arms on one frozen 104-task ORIGINAL-surface gate — a built-in surface-transfer test. designed_fresh scored 69/97/7 (correct/parsed/caps of 104) versus parent 63/87/18 and replay 62/91/13, winning all four preregistered strict comparisons on surfaces it never trained on while shortening generation 31% — the designed dose binds to structure, not vocabulary, and repairs termination as a side effect. The budget-commit substitution was at-or-below replay everywhere (16 caps): taught stop-on-contract behavior stayed in its format and the lost designed rows cost semantics. Decisive negative: induct was 0/8 for EVERY arm including the parent, so the gate's induct ≥ 4/8 floor is structurally unpassable for this lineage — the quadrupled screen turned the C38/C39 induction wall from occasional 26-task luck into an exact zero. No promotion; aggregate seed 78143 sealed unconsumed. Portfolio consequence: the generic-dose line closes on a mechanism win it cannot convert under its own gate; the next attack targets the benchmark's empirically stuck families (menders/warren/sirens/rites) with axis-designed content and preregisters floors achievable under the known wall.

**Same-parent successful-sibling mining is closed: policy support is empty exactly at the wall skill (2026-07-14, terminal availability stop).** The residual successor [qwen35_4b_universal_residual_successful_sibling_target_match](../experiments/qwen35_4b_universal_residual_successful_sibling_target_match/reports/report.md) fixed the predecessor's intervention unit — ten live-residual treatment skills, select/count/route replay-protected — and completed its single authenticated `n=16` event (225 prompts, 3,600 outputs, no rerun). The frozen model-free selection qualified 855/3,600 siblings, and nine of ten skills met the four-task quota easily, but induct supplied qualified siblings on only 2 of 46 failure tasks from 736 samples: outcome `STOP_INSUFFICIENT_SUCCESSFUL_SIBLINGS`, zero rows selected, training/local/aggregate seeds 50/88012/78142 never consumed. Read with C38/C39: what greedy decoding cannot induce, short-budget temperature-0.6 re-sampling cannot supply either, so harvesting the parent's own successes structurally cannot cover the skill that most needs repair. Curriculum signal for the wall must be designed, not harvested.

**The retention-measurement arc: the forgetting tax was mostly measurement noise, and the instrument is now calibrated (2026-07-15, five preregistered events ending in an eval-only calibration).** The de-stack recovery ([qwen35_4b_hygiene_explore_destack_medium](../experiments/qwen35_4b_hygiene_explore_destack_medium/reports/report.md)), the interleaving refutation ([qwen35_4b_interleaved_replay_dose_medium](../experiments/qwen35_4b_interleaved_replay_dose_medium/reports/report.md)), and the diversity refutation ([qwen35_4b_dose_diversity_mechanism_cell](../experiments/qwen35_4b_dose_diversity_mechanism_cell/reports/report.md)) had priced an apparently intrinsic ~5–10-point retention tax per designed dose; the rank-capacity cell ([qwen35_4b_rank_capacity_vehicle_cell](../experiments/qwen35_4b_rank_capacity_vehicle_cell/reports/report.md)) then tripped its preregistered SCREEN_INSTABILITY guard when the known −9 re-measured at −5 on a fresh screen. The funded calibration study ([qwen35_4b_retention_screen_calibration](../experiments/qwen35_4b_retention_screen_calibration/reports/report.md)) re-measured all five published composites on four fresh screens (20 authenticated eval runs, zero training) and closed the question: the same-screen delta-vs-parent noise is SD 4.27, so the ±5 single-screen band every gate had used was ~1.2 SD wide; ALL five historical tax readings (−9, −10, −10, −7, −5) sit inside measured noise; and the pooled taxes are only 0.75–3.75 points. Standing laws: (1) the per-dose retention tax is real but 1–4 points, several times smaller than priced — the install/retention trade is cheaper than the program believed; (2) every future retention adjudication follows the frozen `pooled_k3` protocol — three fresh screens, ±5 band on their mean (2 × 4.27/√3 = 4.9 ≈ the historical band, correctly resized); (3) an adversarial design review caught the calibration's own draft measuring the wrong estimand (level SD instead of delta SD — screen difficulty cancels in same-screen deltas), the second time in the line a review corrected an instrument before it could mint a false law; (4) the rank-64 capacity question is now cheaply adjudicable (both arms published; descriptive read +3.0 favoring rank 64, within noise).

**The two-family wall was the measuring tier: menders/sirens are quick-instrument artifacts, and the goal gate's venue is medium (2026-07-15, zero-GPU receipt forensics).** [qwen35_4b_menders_sirens_tier_forensics](../experiments/qwen35_4b_menders_sirens_tier_forensics/reports/report.md) swept all 2,278 committed gateway receipts (356 cleaned family-score rows across two tiers and five experiments) and adjudicated the goal-gap pilot's standing claim that menders = 0 and sirens = 0.500 for every arm at every quick/tb1024 seed. The claim has three committed counterexamples at the line's own instrument (base sirens 0.375; candidate menders 0.021; replay_refresh menders 0.125), and the paired within-event strict-win analysis shows the all-ten-families goal gate passed 9 of 94 historical medium arm-events versus 1 of 84 at quick — at medium, base never sits at a family ceiling (0/95), sirens spreads to 0.2–0.6 (exactly-0.5 in only 14/95 base events vs 49/82 at quick), and menders stays beatable (treated arms reached 0.4). Standing consequences: (1) the constants are item-draw artifacts of the quick tier's 1/8-step granularity, so another same-shape menders/sirens treatment judged at quick is dead a priori; (2) the goal gate's realistic venue is the medium tier, where strict wins were arithmetically available on all ten families in every historical event; (3) the honest limit — all nine medium passers came from the gym-trained line, so instrument feasibility is established but transfer for the contamination-free universal arms is not; the funded successor is that line's first medium-tier paired measurement (base + best published composites, one fresh sealed seed, tb1024, goal gate recorded).

**The universal line at medium: eight strict wins, zero losses, and a two-tie goal gate (2026-07-15, the line's first medium-tier paired event).** [qwen35_4b_universal_medium_tier_measurement](../experiments/qwen35_4b_universal_medium_tier_measurement/reports/report.md) measured the contamination-free portfolio in the venue the tier forensics identified: four published composites, sealed medium seed 78,150, tb 1,024, write-ahead one-seed ledger, base inside the historical envelope on every family. Three durable findings: (1) the quick aggregate ordering INVERTS at medium — replay_repeat (0.5081, best-ever at quick) drops to last of the treated (0.2981) while the install carrier hygiene_explore leads (0.3379 vs base 0.0567) — the non-convex tier-Pareto frontier (C54) is now replicated inside the universal line, so quick-tier aggregate compounding must never again be read as general improvement; (2) all three treated arms took 8/10 strict family wins versus base (the historical mode from the forensics), and hygiene_explore/replay_repeat lost NOTHING — ties only at menders and rites, both 0.0 — so the recorded all-families goal gate is exactly two tie-flips wide, the closest position in program history; (3) the two remaining families dissociate: sirens resolved to a strict win at medium granularity exactly as predicted (instrument artifact), rites is elicitable in-lineage (designed_fresh scored 0.1 in the same event), but menders stayed 0 for every clean arm — a genuine marginal-capability gap (gym-trained arms historically reached 0.3–0.4 there; the clean line's only nonzero ever is one quick item), and the same-shape trace-repair dose aimed at it is closed by kill rule. The program's binding question is now singular: a genuinely new mechanism argument for menders, carried with rites, from the hygiene_explore parent, under pooled_k3 retention.

**The episode-protocol dissociation: state-chains install, feedback-repair fails a third time (2026-07-15, the two-tie install's split verdict).** [qwen35_4b_feedback_loop_state_chain_install](../experiments/qwen35_4b_feedback_loop_state_chain_install/reports/report.md) took the program's closest-ever goal position (8/10 with ties at menders/rites) and dosed the two missing episode protocols on invented legality-bounded formalisms from the hygiene_explore parent. The gate split exactly along the program's deepest fault line: narrated hidden-state tracking (u_statechain) INSTALLED — 11/20 on fresh instances, strictly above both controls, extending C14's state-chain law from formal simulation to the episode protocol — while repair-with-rerun-feedback (u_feedloop) installed NOTHING: 0/20 on fresh instances of its own training formalisms, below untrained controls, making episode-feedback the third failed pedagogy at the menders-shaped skill after asserted repair and demonstrated bounded search. The repair kill rule extends accordingly: no small designed SFT dose of any tested pedagogy installs the repair-shaped skill in this model; only different mechanism classes remain believable. Secondary laws held precisely: the retention tax versus the parent was −3.0 (inside the revised 1–4-point law); the replay control gained +2.67 retention and 10/20 statechain untrained (replay compounding again); and the calibrated pooled_k3 instrument's first live adjudication measured delta SD 4.08 against the calibration's 4.27, catching the candidate 0.67 outside the replay band — the instrument the program built this same day worked exactly as designed. Sealed seed 78,151 was never opened.

**The thinking-budget lever is closed: two preregistered stops at two seeds, zero scores exposed (2026-07-15, the budget-probe pair).** [qwen35_4b_medium_budget_probe_measurement](../experiments/qwen35_4b_medium_budget_probe_measurement/reports/report.md) and [qwen35_4b_medium_intermediate_budget_probe](../experiments/qwen35_4b_medium_intermediate_budget_probe/reports/report.md) tested the one non-training lever on the binding menders constraint — thinking budget at the event (C44 serial-compute; the truncation-cascade history; the budgets-maxed directive) — and the trusted gateway's per-arm wall budget refused base at BOTH tb8192 and tb4096 before any treated arm ran (base fits at tb1024, 157 s). Per the frozen consequence the lever is closed entirely for paired medium events. The pair is also a template for cheap decisive stops: base-first frozen order, write-ahead ledgers, and preregistered stop outcomes bought a complete answer for two seeds with nothing exposed. Standing position: menders has now defeated three small-dose SFT pedagogies and the deployment-budget lever; the reachable ceiling for believable training paths is 9/10 families; what remains believable for menders are different mechanism classes — dose scale (C43: partial installs were data-limited, and every failed menders attempt was 80–160 rows) and on-policy episode training — each requiring its own intake and kill rules.

**The conversion and the recorded sweep: a taught skill moved its benchmark family, and the parent passed all ten (2026-07-15, the statechain-only dose's sealed event).** [qwen35_4b_statechain_only_dose](../experiments/qwen35_4b_statechain_only_dose/reports/report.md) delivered the program's first pooled_k3 promotion (axis 21/40 strictly over both controls; retention −2.0/−2.67, inside the calibrated bands — the revised tax law and the new instrument both priced it correctly), then opened sealed seed 78,154. Three readings: (1) the candidate beat base and its exposure-matched replay control but not its parent (−0.017 aggregate) — the dose trades; (2) candidate rites 0.300 versus 0.100 for BOTH matched controls — the FIRST local-install→family conversion in program history, giving the episode-protocol mechanism an end-to-end causal chain from designed synthetic data that looks nothing like the eval to a held-out benchmark family; (3) hygiene_explore_parent recorded the first 10/10 all-families goal-gate pass ever (aggregate 0.3663 vs base 0.0800; menders 0.017, warren 0.150 vs 0.100; zero ties, zero losses) — the "9/10 ceiling" was a draw-dependent floor-tie exactly as the tier forensics predicted. Frozen scope: single-item margins on one seed; the confirmation law (independent seeds + matched-compute sample-more) governs before any claim, and the confirmation cell is the immediate funded successor.

**The confirmation: aggregate transfer unconditional, the sweep repeats once, and the goal narrows to a single family (2026-07-15, the three-seed replication).** [qwen35_4b_goal_gate_confirmation](../experiments/qwen35_4b_goal_gate_confirmation/reports/report.md) — the program's first standalone-compliant cell under the owner's reproducibility directive (full six-stage lineage package with vendored root and fixed-seed rebuild) — replicated the recorded 10/10 on three fresh sealed medium seeds under a frozen ordered verdict, with the readout provenance-anchored end-to-end after the adversarial review caught unanchored verdict inputs pre-freeze. Result: AGGREGATE_ONLY. The aggregate win is now beyond dispute (strict on all three seeds, 4/4 all-time, margins ~3–6×); the all-families sweep repeated on seed 78,157 (two full 10/10s across four independent sealed seeds); and the two non-passing seeds carried ZERO strict losses — 9/10 and 8/10 blocked purely by menders 0-margin ties (both) and one warren tie, while warren itself WON +0.267 elsewhere. The program's exact position: the goal's primary condition is demonstrated but not confirmed at the preregistered 2/3 bar, and the entire remaining distance is one family — menders — where every small-dose pedagogy and the budget lever are closed and the dose-scale mechanism class (C43: partial installs were data-limited) is the funded bet with a binary success criterion: any reliable nonzero yield completes the gate.

**Scale does not overcome a zero: the menders wall is dose-independent, and every SFT route to the last family is closed (2026-07-16, the dose-scale null).** [qwen35_4b_menders_dose_scale](../experiments/qwen35_4b_menders_dose_scale/reports/report.md) ran the one mechanism class the kill rules still permitted — 800 episode-feedback rows (10× the failed dose) across eight legality-bounded formalisms, exposure-matched to the row (the control's solver-proven-minimal repetition disclosed with direction-of-bias stated) — and the candidate's fresh-instance transfer came back at exactly the untrained controls' guess floor (1/40 = 1/40 = 1/40), with retention outside the parent band. The dose curve is flat at zero across an order of magnitude, which separates this wall from C43's data-limited installs (scale amplified a 0.087 there; here there is no signal to amplify) and hardens C38/C48 into a dose-independent law for the eliminative-inference class: the model cannot learn two-round eliminative repair from demonstrations at any tested dose, pedagogy, or surface diversity. Program position, stated exactly: the all-families goal stands DEMONSTRATED (two 10/10 sweeps across four independent sealed seeds; the aggregate transfer unconditional at 4/4, margins 3–6×) and NOT CONFIRMED at the preregistered 2/3 bar, with menders the sole gating family and no believable SFT path remaining to it. The believable frontier is now a different mechanism class entirely — on-policy episode training with live feedback — or consolidating the program's map, which this session has made unusually complete: calibrated instruments, priced taxes, a proven conversion mechanism (statechain→rites), a first-ever recorded sweep with its replication rate measured, and every closed door closed by a preregistered rule rather than fatigue.

## Portfolio Implications

- Start with a program question, not an isolated run idea.
- Preserve self-contained experiments, but connect every result upward into program evidence.
- Prefer experiments that distinguish between mechanisms.
- Report oracle-only and deployable evidence separately.
- Add new programs when a line has durable uncertainty and multiple plausible experiments.
- Retire or demote lines when controls repeatedly contradict the mechanism.
