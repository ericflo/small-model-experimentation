# Evidence

## Seed Experiments

- [qwen_structural_latent_compiler_expansion](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [qwen_compiler_multiseed_reattribution](../../experiments/qwen_compiler_multiseed_reattribution/reports/qwen_compiler_multiseed_reattribution_report.md)
- [qwen_typed_bytecode_expert_iteration](../../experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.md)
- [latent_executor](../../experiments/latent_executor/README.md)
- [structured_slot_initializer_ladder](../../experiments/structured_slot_initializer_ladder/README.md)

## Key Result

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
