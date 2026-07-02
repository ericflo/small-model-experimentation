# Qwen3.5-4B Simulation Keystone Repair

## Research Program

- Program: `structured_execution_and_compilers` (+ `posttraining_and_adaptation`). Insight-first,
  **pre-registered** (`reports/prereg.md`, predictions + decision rules locked before each phase).
- The intervention test of C13's causal claim: C13 *diagnosed* one broken primitive — multi-step mental
  simulation — under every inverse capability (identification, segmentation, discrimination,
  feedback-use). This experiment **repairs the primitive and watches whether the untrained capabilities
  move.** Executes C13's next_tests #1 and #2.

## Why this is the highest-stakes question in the arc

Unlike C11's banking (coverage-bounded by what the model can already solve), **simulation training data is
unlimited and teacher-free** — the interpreter emits verified state-chain traces for any pipeline. If
repairing the keystone transfers up the ladder, "train broken primitives, not end tasks" becomes the
unearthing strategy, breaking the coverage bound. If simulation repairs but the ladder stays frozen,
capabilities are separately represented in a fixed model — SFT is task-local, and mechanistic diagnoses do
not license training-transfer predictions. If simulation cannot be trained at all, the wall is
architectural serial compute. **Every branch is a durable law.**

## Design

- **Phase 0 (falsification gate)**: frozen-model simulator microbenchmark — given a STATED pipeline + one
  input, write the full state chain (no code). d 1–5 × k {0,2}, n=25/cell, no-think + thinking arms.
  Kill condition: simulation NOT broken in isolation (d4 ≥ 0.8).
- **Phase 1**: two QLoRA arms from base at **matched training tokens** (~230k): **SIM** (pipeline+input →
  state chain; depths 1–3 only; 3 primitives held out) vs **PROD** (I/O examples → reference code — direct
  end-task training, same unlimited-ground-truth regime; the only difference is the supervised *content*).
- **Phase 2**: all three models on (a) simulation — in-distribution, length-generalization d4–5, held-out
  primitives; (b) the full C13 ladder on fresh verified tasks — bare identification, plan-given
  transcription, segmented identification, no-think 2AFC, thinking 2AFC.

## Phase 0 results (gate PASSED; P-K0b refuted — a C13 refinement)

| depth (better arm = thinking) | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| output exact-match | 0.96 | 0.88 | 0.58 | 0.30 | 0.36 |

- **P-K0a confirmed**: simulation is broken in isolation, decaying with length (d4 ~0.3 ≪ the 0.8 kill
  threshold) → keystone experiment proceeds.
- **P-K0b refuted**: thinking *helps* single-pipeline simulation at every depth (no-think d3: 0.46 vs
  think 0.58). Refines C13/P12: deliberate simulation is not globally wrong — structured single-pipeline
  simulation works at short lengths (0.88 at d2) and is **length-fragile**; P12's chance-level 2AFC failure
  reflects the double-simulation + comparison load.

## Run

```bash
../../.venv/bin/python scripts/run_simbench.py --n-per-cell 25            # Phase 0
../../.venv/bin/python scripts/make_training_data.py 500                  # Phase 1 data (CPU)
bash scripts/phase12_chain.sh                                             # train arms + all retests
```

## Results

Phase 1–2 pending — `runs/simbench_{sim,prod}.json`, `runs/ladder_{base,sim,prod}.json`.
