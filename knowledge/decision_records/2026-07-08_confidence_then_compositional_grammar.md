# Decision Record: Confidence First, Compositional Grammar Next

- Date: 2026-07-08
- Status: accepted
- Programs: benchmark_generalization, evidence_conditioned_selection, posttraining_and_adaptation
- Experiments: qwen35_4b_code_confidence, qwen35_4b_meta_induction
- Claims: C41, C45, C46

## Decision

Finish the confidence-on-real-code follow-through before starting the next training run, then use the next training slot for a sharpened compositional-grammar induction experiment.

The confidence line remains the immediate priority because C46 is cheap, verifier-free, and directly tests whether the fixed model's own logits can beat sample-more baselines on real code. The next training line is not a flat "non-affine held-out family" test. It should train a serial hypothesize-and-verify procedure over a compositional grammar and evaluate three endpoints separately: held-out combinations, held-out productions, and held-out composition depth.

## Context

C41 showed confidence-select beating self-consistency on a toy single-token substrate. C46 resolves the owed real-code generalization: on MBPP and HumanEval, the transferable confidence readout is P(True), a concentrated judgment-token logit, not sequence mean-logprob. Visible execution still wins when available, so P(True) is the verifier-free select/abstain/route lever.

C45 showed that reasoning-SFT can install a general hypothesize-and-verify procedure within an affine menu, but its own caveats leave open whether this is true grammar search or menu verification. A flat held-out non-affine family is low-information: if the family appears in the enumerated candidate set, success can be menu verification; if it is outside the taught hypothesis class, failure mostly repeats the known proposal wall.

## Alternatives Considered

- Option: start the non-affine induction training run immediately. Rejected for sequencing and design reasons: C46 had a cheap decisive follow-up ready, and the proposed non-affine version did not isolate grammar composition.
- Option: stop at C46 MBPP only. Rejected because HumanEval replication is cheap relative to training and directly tests the cross-substrate claim.
- Option: run held-out combinations only for the next training slot. Rejected as the primary endpoint because condition x action combinations can still be an implicit menu; it is useful as an intermediate rung, not the strongest test.

## Consequences

- Future confidence experiments should test thinking-judge P(True), a select + abstain + route policy against matched-compute sample-more, and step-resolved P(True) for repair.
- Future induction experiments should report held-out combinations, held-out productions, and held-out composition depth separately.
- Every compositional-grammar induction run needs execute-given-rule ceilings, token-budget/truncation checks, mixed execute examples to prevent forgetting, and automated example-set sufficiency checks so examples determine the rule while not handing out a lookup table.
- Updated knowledgebase pages: C46 in the claim ledger, shared synthesis, benchmark/evidence selection program evidence, scorecards, and program backlogs.

## Reversal Criteria

Reverse the sequencing if a matched-compute sample-more policy beats P(True)-select + abstain on real code, or if thinking-judge P(True) proves too costly relative to its lift. Reverse the training design if a smaller pilot shows held-out combinations alone fail despite high execute ceilings and adequate token budgets; in that world, production/depth tests should wait until the base cross-product search is established.
