# Natural-Language State-Table Universal Curriculum

**Status:** finished

Fresh local negative on 2026-07-14; aggregate sealed.

This result-separated successor tests whether truth-audited, variable-depth
natural-language state tables plus independent hypothesis scoring and a short
verified commit install a reusable reasoning procedure better than an exact-token
replay continuation from the same parent.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can one cleanly installed procedure improve every held-out
  benchmark family rather than redistribute wins?
- Prior anchors: `qwen35_4b_universal_search_scaffold_token_match`,
  `qwen_trace_procedure_depth_stress`, `qwen_constrained_abi_parser`, C37, and C38.

## Question

Does matching the training interface to variable-depth natural-language execution
teach the model to maintain explicit state, compare independently simulated
hypotheses, and stop with a concise answer—without sacrificing broad replay behavior?

## Hypothesis

The failed predecessor often reached the correct final state but did not commit, and
it regressed hypothesis selection. A truth-audited table that records each natural-
language transition should make execution inspectable; separate rows that score each
hypothesis on every probe should preserve discrimination; an answer-only commit after
verification should train the missing emission seam. The mechanism is false if the
candidate cannot beat both its parent and an exact-token replay control on a fresh
unchanged local gate.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated `close_xi` adapter from
  `qwen35_4b_universal_close_weight_token_match`; the failed scaffold adapter is not
  inherited.
- Dataset/task source: fresh deterministic procedural synthesis owned by this
  experiment. No benchmark source, item, transcript, or result detail may be read.
- Candidate: variable-depth natural-language state execution, independent hypothesis
  scoring, verification/repair, and concise commit lessons.
- Mechanism-falsifying control: same-parent replay continuation with identical forward
  tokens, optimizer steps, backend, seed, and position-aligned shared replay.
- Frozen arms: 320 rows and exactly 286,814 forward tokens each, zero skips, 40
  optimizer steps, and 200 byte-identical replay rows at the same positions. Candidate
  contains 80 curriculum rows plus 40 replay filler; control contains 120 replay rows.
- Primary admission: the inherited absolute local capability gate, a new explicit
  probe ≥0.50 check, and strict paired wins over parent and active replay both overall
  and on execute/induct/probe combined.
- Conditional broad admission: aggregate-only same-backend evaluation only after the
  sole candidate passes every local check; all reported families must improve before
  higher-tier confirmation or matched-compute sample-more.
- Reserved seeds: construction `77112`, training `46`, fresh local `88008`, and
  conditional aggregate `78138`.

## Run

Frozen smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_state_table_compiler_token_match/scripts/run.py --smoke
```

The adversarial review passed and the harness now exposes exactly one expensive stage
per invocation: `train-control`, `train-candidate`, `local`, `merge`, or `benchmark`.
Each stage requires a clean worktree; every predecessor receipt must already be
committed at `HEAD`. Follow `reports/preregistration.md` and publish/CI-verify every
stage before starting the next.

## Results

CPU construction produced 80 truth-audited rows: 20 each execute, score, repair, and
commit. All 80 answers recompute from executable state; all score rows evaluate three
hypotheses on five probes; correct hypothesis position is balanced 7/7/6. Exact-token
materialization succeeded at 320 rows, 286,814 tokens, zero skips, and 200 aligned
replay positions per arm. The frozen smoke passes 48 tests. The active replay control
then trained from the authenticated parent for all 40 steps over all 320 rows with
zero skips and final loss 0.4226. Its adapter weights/config hashes are
`83a741e4...409a` / `13838f2e...843`; receipt/log hashes are
`b05dc72e...e99a` / `5f4d1fe3...60ba`. The candidate independently restarted from
the same parent and completed the same 320 rows, zero skips, and 40 steps with final
loss 1.059. Its adapter weights/config hashes are `36e54804...5d0f` /
`7101cc87...4b34`; receipt/log hashes are `6aab42b3...2be2` /
`26907944...c059`. Training losses are operational evidence only.

Fresh paired local seed 88,008 rejected the candidate. Parent, replay, and candidate
scored 19/26, 16/26, and 16/26 correct; parsed 23/26, 21/26, and 22/26; and contacted
the 1,024-token cap 3, 5, and 5 times. Candidate execute/induct/probe was 0/2, 0/2,
and 1/2, for 1/6 target cases versus replay 2/6 and parent 4/6. It failed accuracy,
parse, cap, execute, induction, and every strict relative check. Promotion is empty,
so no merge or benchmark event ran and conditional aggregate seed 78,138 remains
sealed.

## Interpretation

Truth-audited natural-language tables did not install a reusable deployed procedure.
The candidate sometimes improved isolated computation: it solved one parent/replay
trace miss, fixed one optimization case, and computed a state exactly before losing
only on spaces. But it also treated a reference cycle declaration as an operation,
repeated both induction cases to the cap, miscounted a probe score, and reached the
correct execute result without committing before the cap. The idealized training
interface therefore remained off-policy relative to the model's actual failure
prefixes. Retire another hand-authored trace surface; the next result-separated test
should use fresh on-policy failure-prefix correction with executable oracle
continuations and exact serialization, under the same controls and gates.

## Knowledgebase Update

- Program evidence: records the fresh exact-token local negative and failure anatomy.
- Program backlog: retires idealized state-table surfaces and queues on-policy
  failure-prefix correction.
- Shared synthesis: adds the off-policy-interface boundary.
- Claim ledger: unchanged.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `data/design_receipt.json`
- `data/stream_token_receipt.json`
- `runs/training/replay_after_close.json`
- `runs/training/replay_after_close.log`
- `runs/training/state_table_after_close.json`
- `runs/training/state_table_after_close.log`
- `runs/local/seed88008.json`
- `runs/local/seed88008_promotion.json`
- `analysis/local_failure_forensics.md`
- `scripts/run.py`
- `reports/design_review.md`
- `reports/preregistration.md`
- `reports/report.md`
- `reports/artifact_manifest.yaml`
