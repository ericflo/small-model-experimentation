# Pre-registration: Cross-Family Laws

Logged 2026-07-02, before any new-family data. C13–C15 (transcription intact, simulation length-fragile,
identification ~chance-multiple, format-locality) rest on ONE substrate (integer-list pipelines). This
tests whether they are model-level LAWS or list artifacts, on two genuinely different fresh families —
STRING (char edits) and REGISTER (3-register machine) — with the LIST family (measured in
qwen35_4b_depth_wall_anatomy) as the anchor. Verified-depth generation (collapse-rejected) in all families.

## Anchor (list, from depth_wall_anatomy, verified tasks)

transcription (plan-given) ~1.00 through depth 4; simulation (thinking) 0.96/0.88/0.58/0.30 by d1–4;
bare identification ~0.0 at depth ≥3 (only depth-1/2 register any solves).

## Predictions (locked, per new family = string, register)

- **P-L1 (transcription law)**: plan-given → code ≥ 0.85 at every depth 1–4 in BOTH families. Refuted if
  any family's transcription falls below 0.6 at depth ≤3 (would mean transcription is list-specific).
- **P-L2 (simulation-decay law)**: simulation output-accuracy decays monotonically with depth in both
  families; d4 ≤ 0.5 in both. STRONG form: after normalizing by each family's single-step (d1) accuracy,
  the per-op retention ratios are within ±0.15 across families (a family-invariant decay constant).
- **P-L3 (generation wall)**: bare identification ≈ 0 at depth ≥3 in both families (< 0.15), and the
  transcription−identification gap (the compiler-vs-searcher signature) is ≥ 0.6 in both.
- **P-L4 (ordering law)**: within every family, transcription > simulation > identification at depth 3.

## Decision mapping

- LAWS: P-L1 ∧ P-L3 ∧ P-L4 hold in both families (transcription intact, generation wall, ordering) ⇒
  C13–C15 upgrade to model-level laws; the arc is about the MODEL, not the list substrate. P-L2 strong
  form additionally ⇒ a quantitative family-invariant simulation-decay constant.
- SCOPED: any of P-L1/P-L3/P-L4 fails in a family ⇒ the failing claim is list-specific; report which
  parts of C13–C15 are model-level vs substrate-dependent (still a durable, honest result).

Families are Python-expressible (identification graded by executing the model's `transform` against hidden
I/O, exactly as for lists). Register has a genuinely different state type (fixed-size tuple, not a
variable-length list), stressing whether "length" or "steps" drives the decay.
