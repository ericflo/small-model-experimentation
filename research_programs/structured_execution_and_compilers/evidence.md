# Evidence

## Seed Experiments

- [qwen_structural_latent_compiler_expansion](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [qwen_compiler_multiseed_reattribution](../../experiments/qwen_compiler_multiseed_reattribution/reports/qwen_compiler_multiseed_reattribution_report.md)
- [qwen_typed_bytecode_expert_iteration](../../experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.md)
- [latent_executor](../../experiments/latent_executor/README.md)
- [structured_slot_initializer_ladder](../../experiments/structured_slot_initializer_ladder/README.md)

## Key Result

- [qwen35_4b_commit_slot_jacobian_value_transport](../../experiments/qwen35_4b_commit_slot_jacobian_value_transport/reports/report.md)
  (unclaimed; terminal `COMMIT_SLOT_SEAM_FAIL`): a fixed latent answer interface
  repaired formatting—an alias was already the unmasked next token on 41/48 at
  cap 1,024—but semantic correctness remained task/alias concentrated. Real
  ordered thought was 15/48 versus 11/48 exact-length shuffled, with five mixed
  tasks versus six required. A structured slot can expose a decision without
  making that decision reliable; task-level confirmation remains mandatory.

- [qwen35_4b_partial_structure_search](../../experiments/qwen35_4b_partial_structure_search/reports/report.md)
  (unclaimed while ledger re-grade is open): **type-only partial viability is oracle-useful but model-unreadable.**
  A width-4 exact live-prefix beam retained a hidden solver on 12/12 dedicated depth-5 development tasks at
  262,144x completed-leaf compression. Frozen Qwen3.5-4B thinking P(viable), however, was chance within task
  on 7,200 depth-4 children (AUROC 0.506, CI 0.470--0.543; recall@4 0.251) and significantly below no-think
  AUROC (delta -0.049, CI -0.090 to -0.010). Pooled AUROC 0.557 was a task-difficulty mirage; wrong-task
  visible examples were no worse. The gate correctly stopped depth-5 model search and banking. Separately,
  exact visible-only depth-5 brute covered 60/60 and selected 56/60 in 112 seconds on eight CPU workers, so
  the next question is the real depth-6 resource crossover, then a residualized rather than type-only state.

- [qwen35_4b_crosssubstrate_structure](../../experiments/qwen35_4b_crosssubstrate_structure/reports/report.md) (claim C36): the recent structure findings are MODEL-LEVEL LAWS. C32 (wall-is-structure) + C34 (brute-search dominates) replicate on STRING (char edits) + REGISTER (int machine) + LIST: base ~0, structure-cov = concrete-cov, oracle-skelfill 1.0, random low, brute-deploy ~1.0 on all three. The fixed 4B is a value-computer not a deep-structure-proposer, across substrates.

- [qwen35_4b_structure_search_scaling](../../experiments/qwen35_4b_structure_search_scaling/reports/report.md) (claim C35, re-graded): brute-full deploy stays 0.967 at depth 4 (vs 0.975 at depth 3), while the tested banked models' structure coverage is 0.10 and 0.51. The banked comparison crosses non-dose-matched models, so it is not a causal depth curve. Brute dominates the measured list-DSL cells through depth 4; depth 5 was projected, not tested, and is the open model-guided-search regime.

- qwen35_4b_banking_installs_structure phase 2 (claim C34): end-to-end bank+value-fill deploy. bank-fill deploys 0.463 (= banked structure-cov, confirms C33) BUT brute-force structure enumeration + value-fill + execution-consensus deploys 0.975 (near-solves depth-3) WITHOUT the model. With the interpreter, free structure-search dominates; banking's structure is forward-pass-only. Extends C17 (selection free) to structure-search. Scope: brute wins because the 4096-skeleton space is enumerable.

- [qwen35_4b_banking_installs_structure](../../experiments/qwen35_4b_banking_installs_structure/reports/report.md) (claim C33): banking installs STRUCTURE -- base op-sequence structure-coverage 0.00 -> banked 0.51 (held-out depth-3, generalizable). Banking converts the wall from structure-bound (base) to value-bound (banked struct 0.51 > concrete 0.36, value tax +0.15, fillable). Mechanistic closure of C32: banking = structure-installation. Unifies C22-24/C31/C32.

- [qwen35_4b_structure_or_values](../../experiments/qwen35_4b_structure_or_values/reports/report.md) (claim C32): the compositional wall is STRUCTURE, not values. The model's STRUCTURE-coverage (right op-type sequence, any param) = its concrete coverage (value tax +0.000 at depth-3) -> failures are wrong-skeleton; oracle-skeletonfill=1.0 (values trivial given structure); random-skeletonfill low (DSL not value-fungible). Unifies C19/C25/C31; explains why tool-structure-seeds (C22)+banking were necessary. (op-seq generation fails at 0.00 = separate format handicap.)

- [qwen35_4b_thinking_lookahead](../../experiments/qwen35_4b_thinking_lookahead/reports/report.md)
  (claim C26): **TEST-TIME thinking (on a model never trained to reason about this task) does NOT breach the
  lookahead wall — it amplifies recognition, not planning.** *(Scope: leaves open whether banking successful
  reasoning traces would install planning-via-thinking — the clean untested version.)* C25
  found the fixed 4B can't plan the first of 3 ops in one forward pass. Does *thinking* (serial test-time
  compute, the dormant C9 lever) breach it with no training? Channel-matched test (think→RANK vs no-think→RANK,
  parse-immune), headlined on STEP 1 (the only clean lookahead test). **Step-1 stays at chance across budgets**
  (0.025 → 0.050 → 0.075 at B=0/1024/2048; Wilson CIs overlap). But thinking's benefit scales *inversely* with
  lookahead distance: step-3 recognition (1 op away) 0.275 → **0.600**, step-2 0 → 0.325, step-1 (3 away, real
  planning) ~flat. So thinking **amplifies recognition, not planning** — and internal-brute-force is refuted
  (step-1 would rise if the model could simulate the path; it doesn't). **The juxtaposition with C25:** banking
  lifted step-1 lookahead (0.013 → 0.138) while thinking does not — so *for the planning gap, training is
  required; test-time compute alone can't elicit it*. Reconciles with C23 (base think single-shot depth-3 = 0).
  Design hardened by an adversarial review. Limits: closed-set ranking, n=40, budgets ≤ 2048.
- [qwen35_4b_latent_decomposition](../../experiments/qwen35_4b_latent_decomposition/reports/report.md)
  (claim C25, re-graded): the base next-op ranker is at/below chance only for the first move three operations
  from the goal; step 2 is weakly above chance and terminal recognition is stronger. Base-guided versus random
  search solved 1/80 versus 2/80 tasks, so the defensible conclusion is “no better than random,” not “worse.”
  Banking improved step-wise rankings and low-budget search (18/80 banked versus 2/80 random and 1/80 base),
  roughly matching brute's 23/80. The dose trend is supported at steps 2–3, not at step 1 (10/80 versus 11/80
  for the two banked doses). This is a closed-menu behavioral guidance lift, not demonstrated internal planning;
  one beam and single adapter seeds bound it.
- [qwen35_4b_depth_scaling_controls](../../experiments/qwen35_4b_depth_scaling_controls/reports/report.md)
  (claim C24): **three follow-ups to C23 — no saturation, the gain is data-diversity, and the recipe repeats
  one rung deeper.** (1) The depth-3 dose curve does NOT saturate through 1280 tool-pairs (1156 distinct
  functions): cov@16 climbs 0.00/0.087/0.212/0.375/**0.537**, deployable greedy@1 → 0.188; distinct functions
  grow near-linearly so it's real capacity. (2) A 2×2 at matched steps/mixture splits the gain: diversity
  (up40 0.163 → train_640 0.375, same compute) is **cleanly significant**; the pure compute effect (N=40 0.087
  → up40 0.163, same 40 functions) is **within noise** — so C23's "data-limited" is data-DIVERSITY-limited.
  (3) The tool-search+banking recipe repeats one rung deeper, weakly: depth-4 cov@16 base 0.00 → scaffold
  transfer 0.067 → banked_d4 **0.183** (~3×), but test-time-only (greedy flat 0.033) and marginally
  significant at n=60; no depth-3 forgetting (guardrail 0.425). Design hardened by an adversarial workflow
  review (scaffold-only baseline, distinct-fn counts, 0-leak d3 0/2305 & d4 0/318, true-depth-4). Limits:
  single seed, n=60–80 underpowers adjacent-dose/depth-4 significance, depth-4 single dose, 2560 dose dropped.
- [qwen35_4b_depth3_dose_response](../../experiments/qwen35_4b_depth3_dose_response/reports/report.md)
  (claim C23): **the depth-3 install is DATA-LIMITED, not a representational cap — and it scales into
  deployable single-shot.** C22 left open whether its weak depth-3 install was data-limited or capped. Bank N
  tool-found depth-3 pairs (N ∈ {40,160,640} nested, interpreter search over the 16-op DSL); eval on a frozen
  paired held-out set with **0 leakage** (function-sig AND op-composition dedup → novel rules only). Depth-3
  think coverage@16 rises MONOTONICALLY **0.00 → 0.087 → 0.212 → 0.375**, no plateau; top-dose Wilson lower CI
  (0.28) > low-dose upper CI (0.17). The DEPLOYABLE install scales too: no-think coverage 0.00→0.338, no-think
  single-shot greedy@1 0.00→**0.10** at N=640 (≈0 at C22's N=130). Depth-2 guardrail rose (scaffold intact). So
  the deep wall is a DATA bottleneck, not a hard cap: the thin depth-3 thread (C19) thickens with more
  explorer-found data and converts to deployable single-shot. Design hardened by an adversarial workflow
  review. Limits: single seed, fixed epochs (data~gradient confound), search-easy bias (untested past 640).
- [qwen35_4b_tool_seeded_banking](../../experiments/qwen35_4b_tool_seeded_banking/reports/report.md)
  (claim C22): **the C21 positive control — tool-seeded banking crosses the depth-3 wall self-banking couldn't,
  but weakly.** Harvest depth-3 via an interpreter-backed explorer (CPU brute-search over the substrate's own
  16-op DSL, no external model, 130/130 solved — what sampling gets ≈0 of), add to C21's exact depth-1+2 pairs,
  bank. On a frozen paired held-out set (behavioral dedup; design hardened by an adversarial multi-agent
  review): depth-3 think coverage@16 **0.00 (0/40) → 0.125 (5/40 distinct novel tasks)** — a significant unlock
  vs the hard 0/40 floor where C21 self-banking gave exactly 0. Validates the recipe: **tools explore, banking
  installs.** But CROSSED-BUT-WEAK — the install is test-time-dominated (no-think depth-3 0.025, greedy@1 0.00)
  vs depth-2 which installs deployably (greedy@1 0.15). New nuance: **the installer's efficacy decays with
  depth** (echoes C19). No free next rung (depth-4 stayed 0). Each rung must be seeded by the explorer.
- [qwen35_4b_wall_climbing](../../experiments/qwen35_4b_wall_climbing/reports/report.md)
  (claim C21): **self-banking is coverage-seed-bounded — it can't climb the wall.** Apex bootstrapping test:
  bank ONLY depth-1+2 self-solutions (130 pairs, 83 at depth 2, no depth-3 examples), does the banked model
  now sample depth-3? **DEPTH-LOCAL.** Depth-2 install works and generalizes to held-out tasks (coverage
  0.12→0.36, tripled — clean C18 replication) but depth-3 coverage stays at exactly **0.00** (base 0.00 too) —
  a strong depth-2 composition skill does NOT length-generalize up. Banking installs only depths the base can
  already sample; it cannot bootstrap the frontier. Completes the wall picture: depth-3 is not represented
  (C19), not steerable (C20), not reachable by banking-shallow (C21). **The only way up is to seed each rung
  externally** — tool-augmented harvest (C12 decompose-search) → verify → bank. Sharpens C11-M4 into a hard
  cross-depth wall. Pre-registered P2 (unlock) refuted; P1/P3 held.
- [qwen35_4b_activation_steering](../../experiments/qwen35_4b_activation_steering/reports/report.md)
  (claim C20): **decodability ≠ steerability.** Causal follow-up to C19: build mean-difference (ActAdd)
  directions for the first op from C19's cached activations and add them back to the residual stream during
  generation (forward hook). **INERT** — at depth 1 (cleanest direction, probe 0.99) steering toward the true
  op never beats baseline and only degrades at high strength; at depth 2 a faint predicted-direction whiff
  (+0.05, within noise of the random control, below the +0.10 pre-reg bar); null at earlier layers (8, 12) and
  on identification (0.03→0.03). All pre-registered predictions refuted. The latent signal C19 found is
  *readable but not writable into behavior*. Strengthens the throughline: test-time interventions (selection
  C17, steering C20) don't move the wall; only weight edits (banking C18) and tools (C12) do. Honest limit: a
  clean negative for standard ActAdd — patching / optimized vectors untested.
- [qwen35_4b_latent_composition_probe](../../experiments/qwen35_4b_latent_composition_probe/reports/report.md)
  (claim C19): **first look INSIDE the wall.** Linear probes on residual-stream activations (last
  identification-prompt token, all 33 layers, 1500 verified tasks) decode the composition's first operation.
  **The wall's nature changes with depth:** depth-1 first-op is decoded at **0.99** (rises to ~0.99 by layer 15)
  while the model names it 0.44 / generates it 0.68 → representation ≫ expression = latent capability; depth-2
  probe 0.42 vs behavior ~0.13; depth-3 (the wall) probe 0.27 but the shuffled floor is 0.14, so the real
  signal (~0.13) ≈ behavior — the representation itself has thinned to a thread. So the wall is an EXPRESSION
  failure when shallow (info present, unexpressed) and a REPRESENTATION failure when deep (info not computed).
  Layer-0 stays at chance (signal is computed, not surface). **Implication:** steering has headroom at depth
  1–2 but almost nothing to steer toward at the deep wall; explains why banking (C18) was necessary — it
  *installs* the representation the base lacks. Only proposal-installation, not test-time readout, crosses the
  deep wall.
- [qwen35_4b_coverage_banking](../../experiments/qwen35_4b_coverage_banking/reports/report.md)
  (claim C18): **banking self-verified solutions does BOTH — concentrates AND expands.** The correctly-aimed
  follow-through to C17 (only shifting the proposal distribution can beat sample-more). Harvest the fixed 4B's
  OWN execution-verified identification solutions (80 SFT pairs, no teacher), QLoRA-SFT single-shot, eval on
  DISJOINT held-out tasks (4 arms, base/banked × no-think/think). **Depth 1: CONCENTRATION** — think greedy@1
  0.60→0.80, ceiling flat. **Depth 2: EXPANSION** — banked coverage@16 0.15→**0.45 (3×)** on held-out tasks:
  proposes correct compositions the base never sampled (unique-program count even drops — the proposal *mass*
  moved onto correct programs, C17's lever working). **Depth 3–4: no move** (7 / 0 training examples; wall
  holds). Bounded: doesn't beat think sample-more at k=1, but banking+sample-more > base+sample-more. To push
  the wall deeper you need verified deep examples plain sampling can't harvest → seed with tool-search (C12).
  Refuted its own concentration-only prediction (P3) in the optimistic direction.
- [qwen35_4b_coverage_vs_selection](../../experiments/qwen35_4b_coverage_vs_selection/reports/report.md)
  (claim C17): **the generation wall is COVERAGE, not selection.** Pre-registered decomposition — draw K=32
  identification samples/task (list + register, depths 1–4, 8 visible + 8 hidden examples), grade vs
  visible+hidden, compare selectors to the coverage ceiling. **Selection is free:** max(coverage − vfilter)
  = 0.00 in every cell; 90% of visible-passers also pass hidden, so an 8-example execution-filter, the
  model's own C10-verifier, and even a random pick among visible-consistent candidates all recover the full
  coverage ceiling identically. Single-shot undersells 2–5× (first@1→cov@32: list d2 0.10→0.30, register d2
  0.15→0.60, d3 0.05→0.25) and sample+filter recovers it — but that IS sample-more. The coverage wall's
  depth is set by hypothesis-space size (list collapses at d3; register survives to d4 via a smaller op
  menu), mechanistically explaining C16's register floor as coverage-driven. **Implication:** you cannot
  beat sample-more by better selection — the lever is shifting the PROPOSAL distribution (C12 tool-search /
  C11-C12 banking). Refuted its own selection-centric predictions (P3, P4). Residual: overfit traps
  (visible-pass, hidden-fail) false-deploy at deep register — an abstention gap no example-filter catches.
- [qwen35_4b_crossfamily_laws](../../experiments/qwen35_4b_crossfamily_laws/reports/report.md)
  (claim C16): **cross-substrate generality test** of the C13–C15 ladder on two genuinely different fresh,
  execution-verified, collapse-rejected families (STRING char-edits, REGISTER 3-int machine) vs the LIST
  anchor, 100 verified tasks/family. Verdict SCOPED, and the split is the finding: **two rungs are
  model-level LAWS** — transcription/compiler (plan-given execution ~1.00 at every depth in all three
  families; the curves collapse to one line) and the **generation wall** (bare identification collapses with
  depth everywhere; trans−ident gap ≥0.84 at depth≥3) — so *tools identify, the model compiles* is
  substrate-general. But **simulation fidelity is substrate-dependent** (C15's decay constant was
  list-specific): register (compact state) is robust ~flat (0.92→0.72), list decays (1.00→0.56), string is
  floored near-zero (0.24→0.00). New sub-law: the wall's *floor* ≈ f(hypothesis-space size, simulability) —
  register alone (small op-menu + simulable) keeps a nonzero deep-ident floor (0.16/0.08). Promotes C13,
  narrows C15. (Caught a spurious string-sim-0.00 "law" — a quote-blind parser — before any scored run.)
- [qwen35_4b_depth_wall_anatomy](../../experiments/qwen35_4b_depth_wall_anatomy/reports/report.md)
  (claim C13): pre-registered anatomy of the compositional wall. It is **identification, not execution** —
  plan-given execution 0.90–1.00 through depth 4 (zero execution deficit) while bare identification runs at
  ~2× over chance per composed op (odds fall ~30×/op; wall at depth 2), insensitive to op type, and barely
  helped by shown intermediates (segmentation deficit). Retro-explains C10/C11/C12 with one mechanism:
  the fixed 4B is a reliable compiler starved of hypothesis search — tools identify, the model compiles.
  Also: 40% of nominal depth-3 tasks were shallower-equivalent (min-depth audit; C12 corrected).

- [qwen35_4b_decompose_compose_frontier](../../experiments/qwen35_4b_decompose_compose_frontier/reports/report.md)
  (claim C12): **the frontier is extendable without a teacher.** A decompose-and-compose search (4B ranks
  next primitive → interpreter executes → recurse) cracks depth-3 monolithic sampling can't (0.125→0.40+,
  3.4×); against the brute-force bar the model's guidance buys efficiency not coverage (planner-wall). And
  BANKING the search-found solutions (QLoRA-SFT, no teacher) extends the frontier into the weights (monolithic
  pass@5 0.125→0.237, depth-3 4×) — the bound M4 couldn't break. Answer to C11's open problem: tool-augmented
  search harvests frontier-exceeding solutions, banking pulls them into the model's distribution.
- [qwen35_4b_neurosymbolic_repl_substrate](../../experiments/qwen35_4b_neurosymbolic_repl_substrate/reports/report.md)
  (claim C11): a **fresh, contamination-free** procedural program-synthesis substrate (random primitive
  compositions, held-out-execution graded, oracle-solvable 100%) — a reusable asset for elicitation claims
  with no memorization confound. On it, a neurosymbolic execution-feedback REPL loop did NOT beat
  matched-compute sampling (M2), but self-training on the 4B's own verified solutions banked capability into
  held-out single-shot (M3, +0.095). Extends C1 (executable intermediates) toward execution *feedback* and
  self-correction *training*.

## Current Read

Structured execution is one of the strongest imported signals. The next useful work is not another isolated
positive run; it is controlled comparison of representations and supervision sources — and, per C11, on a
CONTAMINATION-FREE substrate so that gains (and non-gains) are honestly measurable.

## Replicated semantic commit interface (2026-07-12)

[qwen35_4b_commit_slot_semantic_power_replication](../../experiments/qwen35_4b_commit_slot_semantic_power_replication/reports/report.md)
establishes a stable constrained compiler output seam on a fresh procedural
depth-two substrate. At fixed cap 1,024, ordered thought independently beat an
identical token-multiset shuffle by +13.57pp and +15.04pp across two disjoint
113-task stages, while the unrestricted next token was already an alias on
88.2% and 87.6% of rows. Syntax therefore exposes rather than manufactures most
of the answer-mode state. The effect remains target heterogeneous and free-form
close-only output remains poor. The subsequent task-held-out shared J-value
measurement failed at chance (0.5021), below slot margin (0.5448) and equal-
width non-J residual state (0.5292); midpoint and endpoint J rankings had
opposite signs (0.6083 versus 0.3958). The interface is a stable output seam,
not evidence for one phase-invariant scalar compiler state. Causal work stayed
sealed.
