# Evidence

## Seed Experiments

- [qwen_structural_latent_compiler_expansion](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [qwen_compiler_multiseed_reattribution](../../experiments/qwen_compiler_multiseed_reattribution/reports/qwen_compiler_multiseed_reattribution_report.md)
- [qwen_typed_bytecode_expert_iteration](../../experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.md)
- [latent_executor](../../experiments/latent_executor/README.md)
- [structured_slot_initializer_ladder](../../experiments/structured_slot_initializer_ladder/README.md)

## Key Result

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
