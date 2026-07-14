# Close-Weighted Universal Commit Seam

**Status:** in-progress · since 2026-07-13 · replay and standard arms trained; close treatment and paired local evaluation remain

## Research program

- Program: `agentic_breadth_installation`
- Parent: `qwen35_4b_universal_mid_density_token_match`
- Mechanism anchors: C50 (`qwen35_4b_gauntlet_breadth_round1`), C51 (`qwen35_4b_answer_potential_trace_sft`), and `qwen35_4b_tokenizer_eos_answer_commit_factorial`

## Question

Can a short synthetic-curriculum adapter continuation that explicitly trains the
model's autonomous `</think>` transition turn the designed160 local near-miss into
a clean install, or is any improvement explained by targeted execute/induct data or
replay alone?

## Hypothesis

The parent already solves enough local cases but fails to stop and emit: all three
of its unparsed designed160 cases hit the 1,024-token cap, and the misses are
confined to execute/induct tasks. C50 says successful broad installation depends on
where loss is placed near the emission seam, while C51 says answer likelihood after
an injected close is non-actionable unless autonomous closing is part of the trained
event. Raising the natural close-span weight from the ordinary thought weight 0.2 to
the answer weight 1.0 on fresh execute/induct rows should improve closure more than
byte-identical ordinary SFT.

## Setup

- Only model: `Qwen/Qwen3.5-4B`, revision `851bf6e...`.
- Warm start: authenticated published `designed160` adapter, weights
  `f05c13ae...94654`, config `0cd3ca7c...91e58`.
- Fresh targeted block: 40 `u_execute` and 40 `u_induct` rows selected without
  outcomes from source rows absent from the parent's designed160 stream.
- Shared block: 200 replay rows. Targeted arms add the 80 designed rows plus 40
  replay fillers; replay repeat uses 120 replay rows with the exact same 87,454
  forward-token sum.

| Arm | Variable 120-row block | Close loss |
| --- | --- | --- |
| `replay_repeat` | replay only | 0.2 everywhere |
| `standard_xi` | 80 fresh target + 40 replay | 0.2 everywhere |
| `close_xi` | byte-identical to `standard_xi` | 1.0 only on target closes; 0.2 otherwise |

Every arm has 320 rows, 286,814 forward tokens, zero tokenizer skips, batch size 1,
gradient accumulation 8, and 40 optimizer steps. `standard_xi` and `close_xi` use
the same bytes, order, shuffle seed, prompts, thoughts, answers, and optimizer
schedule. Their sole assigned-weight contrast is the two-token autonomous close span
on the 80 target rows. The treatment changes that span's weight from 0.2 to 1.0;
all other assigned token weights remain unchanged.

Training is one epoch at learning rate `1e-5`, rank 32 / alpha 64,
`w_think=0.2`, max length 4,096, and seed 44. The active replay arm uses the same
parent, rows, tokens, steps, and optimizer settings.

The local screen uses fresh procedural seed 88,006, greedy generation, 1,024 tokens,
and the unchanged absolute gate: accuracy ≥0.65, parse ≥0.90, cap contacts ≤2, and
no repeated feasible-route abstention. Only `standard_xi` and `close_xi` are
promotion candidates; replay repeat and the immediate parent are controls.

If either candidate passes, one aggregate-only quick@1,024 event at fresh seed
78,136 compares base, `blend`, replay refresh, the immediate parent, active replay,
and every eligible candidate as explicitly merged checkpoints on `qwen_vllm`. The
benchmark firewall forbids reading or importing benchmark items, sources,
transcripts, or private outputs.

## Run

```bash
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --stage train-control
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --stage train-standard
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --stage train-close
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --stage local
```

Merge and benchmark stages are conditional on a treatment arm passing locally:

```bash
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --stage merge
.venv/bin/python experiments/qwen35_4b_universal_close_weight_token_match/scripts/run.py --stage benchmark
```

## Results

The replay control and ordinary-weight `standard_xi` arm each completed their
registered 40/40 updates over 320 rows and 286,814 forward tokens with zero skips.
Replay train loss/wall time were 0.4477 / 303.44 seconds; standard were 0.6882 /
302.15 seconds. Their adapter weights hashes are `ca5601cd...59d78` and
`271569fd...3569c`. The close-weighted arm and all model evaluation remain pending.

The outcome-free feasibility audit still passes: target rows have zero overlap with
the parent designed160 rows, both unique data streams encode without skips, and the
replay variable block exactly matches the targeted block.

## Interpretation

None yet. A `standard_xi` improvement would support targeted-data continuation but
not close weighting. A `close_xi` advantage over standard would isolate autonomous
close loss placement. Failure of both while replay is stable would retire this small
close-weight intervention and motivate a different termination mechanism rather than
another generic dose.

## Knowledgebase update

- Program backlog: this is the active result-separated termination successor.
- Program evidence and synthesis: unchanged until a result exists.
- Claim ledger: unchanged; no result or universal-feature claim exists.

## Artifacts

- `idea_intake.md`: novelty, near-duplicates, and falsifier.
- `data/stream_manifest.json`: exact source exclusions, selections, slots, and sums.
- `data/stream_token_receipt.json`: zero-skip and exact-exposure proof.
- `scripts/train_think_close.py`: separately weighted autonomous close span.
- `reports/design_review.md` and `reports/preregistration.md`: frozen threats and gates.
- `reports/artifact_manifest.yaml`: external parent and planned trained artifacts.
