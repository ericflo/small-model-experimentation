# Decision: Positive-Delta Pareto Policy Integration

- Date: 2026-07-12
- Programs: `agentic_breadth_installation`, `posttraining_and_adaptation`,
  `test_time_reasoning_budget`, `benchmark_generalization`
- Experiment: `qwen35_4b_pareto_policy_integration`
- Status: executed; stopped at teacher qualification

## Context

`qwen35_4b_specialist_policy_integration` stopped because its fixed absolute
specialist rule demanded 1.094 from a score capped at 1.0. That preserved the
preregistration but failed the scientific objective: saturation should have
made the tools lane a retention anchor, not vetoed unrelated capability
production or integration.

C54 then supplied a stronger test than the original hypothetical four-teacher
design. Two independently trained, same-origin Qwen3.5-4B policies occupy a
measured non-convex frontier: `blend` is quick/short-optimal, while `apex` is
medium/deep-optimal. Data-dose interpolation did not combine them.

## Decision

Create a clean successor that regenerates this Pareto pair and tests one-model
consolidation with corrected same-prefix on-policy MOPD. Teacher qualification
uses no arbitrary effect-size floor: paired delta must be positive, both frozen
replicate blocks must be positive, and the one-sided stratified-bootstrap lower
bound must exceed zero.

Saturated cells are retention anchors. The final integrated checkpoint, not
each intermediate teacher, must beat matched-compute sampling.

## Why this path

- It tests integration on capabilities already evidenced as complementary.
- It directly attacks C54's one-checkpoint capacity tradeoff.
- Correct routing, wrong routing, off-policy distillation, parameter merging,
  union SFT, and sample-more cleanly distinguish policy-space consolidation
  from dose or inference routing.
- It keeps all training and procedural gates contamination-free and opens the
  held-out instrument only after mechanistic success.

## Stop rule

Stop only for uninstalled artifacts, absent reproducible complementarity,
failed correct-route/locality evidence, unsafe update drift, or integration
failure against controls. A small positive effect is not a stop condition.

## Execution outcome

Both specialists were independently regenerated, explicitly merged, and
behavior-gated. Two disjoint qualification blocks then tested the corrected
rule on the contamination-safe procedural proxy. The quick `blend` policy's
capability delta was negative in both blocks (`-0.00693`, `-0.03789`), pooling
to `-0.02241` with a one-sided 95% lower bound of `-0.04897`. The deep `apex`
policy had a replicated capability advantage (`+0.04563`, lower bound
`+0.03401`) but regressed beyond 0.02 on six retention cells.

The registered stop rule therefore fired for absent reproducible
complementarity. No teacher audit, locality probe, MOPD update, integration
control, confirmation, or benchmark invocation ran. This is not an MOPD
negative; it shows that an external aggregate Pareto label does not by itself
identify a better teacher on the state distribution where distillation occurs.

## Successor decision

Do not retry coarse quick/deep routing with more updates or a weaker gate. If
the policy-space line continues, create a fresh experiment that treats teacher
choice as an estimand: on disjoint same-prefix states, sample both same-origin
teachers, estimate verified continuation advantage, and authorize distillation
only where the selected teacher has a replicated positive advantage over the
alternative. Freeze that state-routing rule before training and compare the
final one-checkpoint policy against both teachers, a visible router, and
matched-compute sampling.
