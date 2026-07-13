# Idea intake: counterfactual evidence-acquisition curriculum

## Program fit

- Primary program: `agentic_breadth_installation`.
- Supporting programs: `active_evidence_acquisition`,
  `process_control_and_tool_use`, `posttraining_and_adaptation`, and
  `benchmark_generalization`.
- Closest program scorecards reviewed: Agentic Breadth Installation and Active
  Evidence Acquisition.
- Related queue items: `acquisition_policy_comparison_pool` and
  `family_aware_evidence_policy`. This experiment does not consume either broad
  queue item; it tests one narrower coding-policy installation strategy with a
  fixed evidence-search action.

## Prior evidence

1. `qwen35_4b_semantic_policy_headroom_tournament` is the direct predecessor.
   Its formal result was an interface failure, but its opened trajectories
   showed a sharp state contrast: failed-test cases could use supplied evidence,
   while inferred rejected states had 0/54 correct first patches and 0/72
   visible-test reads before the first patch.
2. `qwen35_4b_verifier_conditioned_recovery_bank` showed that balancing
   conditional state-to-action transitions can install useful recovery locally,
   while its reason-supervised arm failed locality. It motivates action-only
   transition banking rather than think-token pressure.
3. `qwen35_4b_transaction_invariant_recovery_curriculum` showed that a semantic
   program can install strongly on trained families yet miss the required
   held-out transfer margin. It supplies the exact training parent and the
   complete-loop replay bank used here.
4. Active-example and active-trace experiments show that discriminating public
   evidence can improve downstream decisions, but learned acquisition has been
   mixed and sometimes ties simple fixed or random policies. Those experiments
   selected examples or DSL probes from a provided pool; they did not train a
   coding agent to locate repository evidence before its first edit.

### Closest newly landed near-neighbor

`qwen35_4b_early_text_hypothesis_forking` is the closest conceptual
near-neighbor. It preregisters external enumeration of all 24 bound first-step
hypotheses before native thinking on an exhaustively searchable depth-two list
DSL, followed by visible-only selection. It does not train weights, acquire
evidence with tools, edit repositories, or test conditional recovery. This
experiment therefore makes no broad novelty claim about early counterfactual
proposal shaping. Its distinct uncertainty is whether a single trained policy
can *autonomously seek* discriminating public repository evidence and bind it
to a correct first patch across family, path, and query shifts.

## Unresolved uncertainty

Can zero-think-loss, transition-balanced action supervision install the
conditional policy

```text
ambiguous implementation -> search public discriminator ->
evidence-faithful first patch
```

without turning into unconditional file reading, losing the already learned
rejected/failed/verify/commit loop, or merely teaching lexical repository
templates?

## Mechanism

Each inferred counterfactual dyad holds the issue, source, file tree, evidence
location, and every non-discriminating public byte constant. Exactly one public
evidence file changes, and its two values require opposed patches. Training
both members at the action seam should make the search result causally relevant
to the next patch. Complete prior recovery blocks preserve the rest of the
loop, while equal weighted action mass at every conditional transition prevents
global operator totals from hiding a missing rare transition.

The mechanism is false if aligned training does not beat within-dyad shuffled
patch labels, an equal-dose explicit-contract control, the unchanged parent,
the incumbent, matched-operator nondiscriminating search, or
actual-compute-matched sampling. It is also false for the claimed transfer scope if gains disappear on
unseen families, disjoint evidence-path skins, or the held-out signature-query
skin.

## Control plan

- Baselines: exact start checkpoint, C54 apex incumbent, and dual-overmatched
  sample-more prefixes from both checkpoints.
- Mechanism controls: within-dyad shuffled evidence-to-patch targets;
  explicit-contract redundant acquisition at equal dose; a deterministic
  matched-operator search whose output is asserted to exclude the evidence path
  and marker; and host-injected correct evidence as a non-deployable
  reachability ceiling.
- Robustness: two qualification blocks, family-held-out development and
  confirmation, three public evidence channels, disjoint path regimes, an
  unseen query skin, explicit no-search retention, and two legacy coding-loop
  retention suites.
- Locality: direct candidate-to-apex next-token audit, with start-to-apex
  feasibility before training and incremental parent drift reported separately.
- Hidden-label boundary: hidden executable tests, branch labels, and oracle
  patches are host-only. They score fixed actions and never enter prompts,
  stopping, selection, or tool output.

## Evidence output

- A two-block qualification receipt distinguishing interface failure, missing
  acquisition headroom, missing evidence-utilization reachability, and a
  trainable substrate.
- Exact transition-balanced bank, within-dyad shuffle proof, tokenizer/action
  mass receipt, training/merge lineage, and locality receipt.
- Dyad-level trained-family, transfer, sample-more, explicit, normal-loop, and
  legacy-recovery receipts with actual sampled and logical token costs.
- Entropy/varentropy stratification by semantic next action for diagnosis only;
  neither quantity selects labels or changes loss.
- Program evidence and synthesis updates after a result. No claim ID is
  reserved in advance; a claim is considered only if replicated transfer or
  Menagerie evidence changes the durable corpus read.

## Decision

- Run the experiment: **yes**, only through the staged preregistered gates.
- Create a new program: **no**; the direction already belongs to Agentic
  Breadth Installation with Active Evidence Acquisition as a supporting line.
- Write synthesis only: **no**; the autonomous acquisition-and-binding policy
  remains untested.
- Defer: training, transfer, and Menagerie remain sealed until their preceding
  gates pass.
