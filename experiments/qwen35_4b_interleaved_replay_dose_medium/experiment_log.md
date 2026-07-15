# Interleaved-Replay Dose Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened after the de-stack test confirmed recovery (both flags TRUE; axis
  15/20 vs 11/8) while the direct dose broke the retention bands, isolating
  the replay-interleaving law. This trial reproduces the retention-safe recipe
  exactly: the same verified corpus, warm-started from the already-receipted
  interleaving replay round.
- Seeds `55122/56/88019/78149` reserved; the escalation rule (mechanism study
  if retention breaks despite interleaving) is frozen.
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-15 — Model-free pipeline run (inherit → measure → materialize → validate → design → gate)

- Adapted the full pipeline from the de-stack predecessor
  (build/measure/materialize/validate/train/merge/gate/eval/benchmark/harness)
  with exactly ONE design change: the parent is the predecessor's own
  `replay_clean` arm (the interleaving replay round). Every fail-closed
  convention kept (hash pins, `--check` byte-identity, TODO-PIN fail-closed,
  encoder binding, merge self-pin in the gate receipt, shared
  `finalize_promotion` writer, full benchmark CLI, weight recomputation).
  `gen_axis_v2.py`, `gen_curriculum.py`, `train_think.py`,
  `src/vllm_runner.py` copied byte-identical from the predecessor.
- Inherited the treatment corpus byte-identically
  (`data/sft_hygiene_explore.jsonl` sha256 `8b3e9791…c22cf1`, manifest
  `cbc9ae6d…03903`, replay `25a9595f…abf0c2`); `build_corpus.py` authenticates
  the donor pins AND re-derives the corpus from the copied generator at seed
  77,119, requiring exact byte reproduction (verified: byte-identical).
- Measured exact spans (`source_token_lengths.json`
  `1aef3dd5b9020ebfb672efcb8904da0406514984701415a58479e4ac08d3ecf5`;
  treatment vector forward 19,582 / nonzero 5,793 / mass×5 9,665 — identical
  to the donor, as the bytes demand); the three-axis MILP at slot seed 55,122
  solved optimally in 4.6 s: both 240-row variable blocks at forward 147,792 /
  nonzero 63,001 / mass×5 71,525; arm totals 1,373,106 / 579,624 / 633,716;
  1,280 position-aligned shared rows; zero skips. Streams:
  `replay_interleaved2.jsonl` `0918144f…3f8d75`, `dose_after_replay.jsonl`
  `9c91383b…2e77ee`; manifest `cb617490…790138`; independent validation
  receipt `stream_token_receipt.json`
  `0e2197fda5957e9bb260d9e93bd782d343256b9828e9f5a3aa09cce26d2db1b9`.
- Froze the design receipt (`data/design_receipt.json`
  `646f5ee2c7a28eafa42625db5a4cc4dcc0ac431b89980a2f07d481496f345ec6`) binding
  the interleaved-parent identity (tracked merge receipt `24367084…b90332`,
  tree `19759e12…8fc67`, weights `2cef3e5e…187b04`, adapter `f6f910ed…6fb8` /
  `015bb135…4d961`), the corpus inheritance, and the lifecycle substring
  contracts (seeds 56 / 88,019 / 78,149).
- Froze the 124-task gate at seed 88,019: `local_tasks_seed88019.jsonl`
  `6e927f591f9ae9d2edad6e263be3f7c0262b39de4854314a64802e656b98c15b`, runner
  input `dcf482cb…c9d568`, design receipt `d4aeb7fd…a5624`; zero
  canonical-message overlap proven against both inherited corpora, the donor's
  corpora AND materialized streams, both fresh training streams, regenerated
  construction rows, prior local seeds 88,000–88,018, and all six predecessor
  frozen gates (88,013–88,018).
- Filled the stream pins (materialize/validate/train_trial exposure
  constants); left fail-closed as TODO-PIN: `PUBLISHED_ARM_HASHES` (both arms,
  filled after each training stage publishes) and `EXPECTED_TREE_SHA256` (both
  arms, filled after merge-arms publishes); verified each aborts with the pin
  message.
- `run.py --smoke` green end to end (every `--check` byte-identical, 50 unit
  tests, py_compile); training remains sealed behind the pushed-checkpoint
  gates and the PASS_CONTROL_TRAINING / PASS_CONTROL_MERGE verdicts.
- No model, GPU, training, local, or benchmark event has run.
