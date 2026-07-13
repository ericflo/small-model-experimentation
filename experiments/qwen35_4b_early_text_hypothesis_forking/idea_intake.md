# Idea Intake: Early Text Hypothesis Forking

## Rough idea

Use the late-anchor Jacobian result as a timing clue rather than another reason
to edit activations. Enumerate all 24 bound first-operation hypotheses before
reasoning begins, let Qwen3.5-4B complete each program, and select with visible
executions only.

## Closest prior work

1. `qwen35_4b_semantic_anchor_coordinate_branching` is the immediate parent. It
   placed opaque aliases after 512 thought tokens and is terminal invalid at
   unrestricted parse plus a fixed composed label map. It did not generate or
   select full candidate continuations.
2. `qwen35_4b_hypothesize_verify_wall` used one generic enumerate-and-verify
   scaffold and depth-local SFT. It did not allocate one generation trajectory
   per bound first-operation hypothesis or compare that systematic bank with a
   duplicate bank and token/forward-matched sample-more.
3. `qwen35_4b_partial_structure_search` scored partial skeletons to prune
   external search. It did not condition full native trajectories on an
   exhaustive hypothesis bank; its partial-state judge failed.
4. `qwen35_4b_coverage_vs_selection` establishes the burden: selection is often
   plumbing, so a new method must shift proposal coverage beyond sample-more.

## Novelty decision

Proceed as a distinct experiment. The novel causal contrast is candidate timing
and systematic allocation: early bound hypotheses versus the identical token
sequence after 512 thought tokens, duplicate/placebo hypotheses, and clean
sampling. The endpoint is a deployable full-program pool under visible-only
selection, not a token-writing or oracle-donor diagnostic.

## Failure modes carried forward

- Test composed semantic mappings directly; independently changing components
  is insufficient.
- A type-only bank leaves parameter binding unsolved. Enumerate the 24 legal
  bound operations and balance task/branch positions independently.
- Exhaustively enumerate all 24² programs as the public-data scope ceiling;
  beating sampling cannot be described as beating symbolic search.
- Generate a full answer and require unrestricted parse before interpreting
  constrained or oracle metrics.
- Count every independent late prefix, resumed prefill, injected token, forced
  close, and branch token.
- Do not credit exhaustive enumeration alone; require selected and oracle
  coverage gains over matched sampling.
- Do not train or open confirmation after a failed mechanics/qualification gate.
