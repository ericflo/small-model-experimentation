# Semantic-policy headroom tournament

**Status:** finished

This no-training tournament found no replicated post-failure semantic axis and
stopped on its answer-cap instrument gate. Its trajectory contrast moves the
capability target earlier, to evidence acquisition before the first patch.

## Research program

- Program: `agentic_breadth_installation`.
- Direct predecessor: `qwen35_4b_validation_policy_counterexample_curriculum`.
- Parent: its unchanged learned transaction checkpoint, exact weight SHA-256
  `1cf5fb...41ba3`.

## Question

Which validation-policy conflicts still produce replicated failed-test headroom
in the transaction-trained model, under the exact looping coding harness, and
are therefore legitimate substrates for a later counterexample curriculum?

## Design

This experiment trains nothing and cannot invoke Menagerie. It crosses three
conflicts with three public representations:

- negative quantity: malformed `ValueError` versus ordinary insufficiency
  `False`;
- non-integer quantity: malformed `TypeError` versus ordinary insufficiency
  `False`;
- blank resource: malformed `ValueError` versus ordinary unknown resource
  `False`;
- bundle mappings, record dictionaries, and tuple sequences.

Nine inferred-contract families state the valid input domain and ordinary
rejection policy but require the agent to infer malformed behavior from visible
tests/failure output. Three explicit-contract controls state the exception
verbatim. Every partial implementation is otherwise correct and fails visible
and hidden tests only at the semantic conflict; every oracle passes.

Two content-disjoint blocks each contain 36 unique repositories (12 families ×
three tasks) and 72 controlled recovery cases. An inferred axis qualifies only
if failed-test success is 15–80% in both blocks, at least two of three shapes
are individually inside that band in each block, explicit-control success is
≥85%, and invalid/cap contacts remain ≤5%. At least one replicated axis must
qualify.

## Why this is different

The predecessor trained before proving that its rewritten substrate retained
the historical failure. Parent and control were already 48/48, so the
treatment effect was unidentifiable. Here, exact-substrate parent headroom is
the only outcome. Eligible axes and families are emitted mechanically by frozen
rules for use in a separate future experiment; no update, threshold change, or
benchmark escalation occurs here.

## Firewall and compute

All repositories are fresh procedural fixtures. Hidden tests and repair objects
stay host-side; only public issue/source/test/tool output reaches the model.
Both blocks use the same merged checkpoint, copied vLLM 0.24 runner, 512 think
+ 512 answer tokens, one greedy trajectory, and six turns. Nothing under
`benchmarks/` is read/imported, and Menagerie authorization is hard-coded false.

## Run

```bash
python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --lock-design <commit>
.venv/bin/python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --gpu-smoke
.venv/bin/python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --full
```

## Result

**Verdict: `INSTRUMENT_FAIL`.** The preregistered answer-cap gate failed in
both parent blocks, so no semantic axis is licensed for training. The runner
stopped with its registered gate code; no checkpoint was trained and Menagerie
remained sealed.

| Inferred axis | Headroom A failed-test success | Shapes in band | Headroom B failed-test success | Shapes in band |
| --- | ---: | ---: | ---: | ---: |
| negative quantity | 9/9 | 0/3 | 9/9 | 0/3 |
| non-integer quantity | 9/9 | 0/3 | 9/9 | 0/3 |
| blank resource | 8/9 | 1/3 | 7/9 | 1/3 |

Negative and non-integer handling were saturated after direct failed-test
evidence in both blocks. Blank-resource repair was uneven: record was the only
shape inside the 15–80% band in A, while tuple was the only one in B. Thus no
axis met the frozen requirement for two supported shapes in both blocks even
apart from the interface failure. Explicit controls passed at 9/9 and 8/9, and
invalid actions stayed at 1.69% and 1.38% of turns.

The cap gate failed at 43/356 turns (12.08%) in A and 46/363 (12.67%) in B,
against a 5% ceiling. Forensics localize the problem: 78/89 capped answers
contained a valid first tool call, and 77 of those continued with post-call
run-on. All capped cases still retained the targeted recovery transition, but
all 12 end-to-end failures contacted the cap. The formal stop therefore stands;
the association cannot be dismissed or interpreted as the semantic mechanism.

## Interpretation

This qualification does not support another post-failure policy curriculum.
The parent usually converts explicit verifier evidence into the correct
semantic revision already, and the remaining blank-resource misses do not
replicate across representations. The more promising frontier is earlier in
the loop: acquiring ambiguous public evidence and binding it to the *initial*
proposal before a failed test supplies the answer.

The trajectory contrast makes that pivot concrete. Every one of 72 failed-test
cases reached a fully correct patch; the four terminal misses were destructive
regressions after correctness. In the rejected-patch condition, by contrast,
none of 54 inferred-contract cases produced a fully correct first patch, and
zero of 72 rejected trajectories inspected visible tests before first patching.
The model can use decisive evidence once handed to it, but does not acquire
that evidence before committing to an ambiguous proposal.

A successor should use counterfactual pairs whose issue and source are held
constant while visible evidence flips the required policy, then balance the
`inspect→patch`, `rejected_patch→changed_patch`, and
`failed_test→diagnose/revise` transitions. It should repair the measurement
with a payload-safe, parse-aware answer allowance, while keeping response
closure diagnostic rather than turning slop suppression into the capability
objective. Exact metrics and hashes are in
[`reports/result_receipt.json`](reports/result_receipt.json).

## Knowledgebase update

Program evidence, backlog, scorecard, and shared synthesis record the formal
instrument failure and the earlier-loop pivot. The claim ledger is unchanged:
this no-training qualification produced no checkpoint or benchmark result.

## Artifacts

Committed design and compact receipts live here. Detailed parent trajectories
will live under `large_artifacts/qwen35_4b_semantic_policy_headroom_tournament`
per [`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).
