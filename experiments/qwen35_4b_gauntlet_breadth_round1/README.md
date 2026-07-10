# Gauntlet round 1: breadth-first agentic expert iteration

Build a 12-family firewall-clean agentic gym (atoms + multi-turn episodes,
machine-checkable verifiers), harvest Qwen3.5-4B's own verified thinking
episodes, think-channel QLoRA SFT, and test whether substrate breadth installs
general capability — measured blackbox on menagerie quick/medium/slow fresh
seeds against the 0.112/0.146/0.138 baselines.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: does breadth-first self-training on many diverse,
  verifier-gated agentic substrates install capability that the corpus's
  single-substrate installs never did — capability that transfers to held-out
  families and to the blackbox menagerie instrument?
- Prior anchors: C11 (banking verified self-solutions works), C28/C43/C48
  (think-channel recipe; answer-only SFT forgets), C43/C45/C48 (every install
  measured so far is shift/substrate/depth-LOCAL — all from single-substrate
  training), C47 (execution verifier at the training seat), C9 (think is the
  deployment default).

## Question

Every locality law in the corpus was derived from training on ONE narrow
substrate at a time. Menagerie's baselines (aggregate 0.112 quick / 0.146
medium / 0.138 slow / 0.168 deep, with six families at or near zero on quick)
have never been targeted by any install. This experiment resolves: does one
round of breadth-first expert iteration (10 simultaneous, format-diverse
agentic families) move (a) held-out items of trained families, (b) two
never-trained gym families, and (c) the blackbox menagerie aggregate — or does
locality survive breadth?

## Hypothesis

Breadth defeats a specific component of locality: the generic agentic
protocol competencies (state-ledger discipline, terse action/answer emission,
finishing thinking within budget, horizon persistence) are shared across all
families, so a broad mixture should install them where any single substrate
could not, lifting menagerie quick above base + noise (≥ +0.03 aggregate).
Axis-specific competence (induction, repair) may remain local; the gym-internal
transfer ladder (trained-family held-out items vs held-out families) separates
the two.

## Setup

- Model: Qwen/Qwen3.5-4B (pinned repo revision), thinking mode, QLoRA r32/α64
  think-channel SFT (C48 recipe; never answer-only).
- Dataset/task source: `src/gym/` — 12 procedurally generated families
  (10 trained + 2 held out), invented content, machine-checkable verifiers;
  see `reports/gym_design.md`.
- Train/eval split: harvest on generation seed 11001 (atoms) / 21000+
  (episodes); gym-internal eval on disjoint seed 90001; menagerie on fresh
  seeds per event (logged in `runs/menagerie_log.jsonl`, never reused).
- Baseline: base model, same seeds, same greedy + think-budget decode.
- Controls: held-out families (near-transfer), per-family deltas,
  parse/forced-close/horizon diagnostics to separate protocol-shape gains
  from axis gains.
- Primary metric: paired adapter−base menagerie delta with a pre-registered
  three-way decision rule (positive / negative / inconclusive) over two fresh
  quick seeds + one medium event, grounded in a base-vs-base null calibration;
  see reports/gym_design.md "Success criteria".
- Oracle-only metrics: gym oracle policies validate instruments (never train
  on oracle outputs — provenance is the model's own verified samples).
- Hidden-label boundary: verifiers/golds never enter prompts; menagerie
  contents never read (CLI + aggregate scores only).

## Run

Smoke (CPU-only: config + all family selftests):

```bash
python3 scripts/run.py --smoke
```

Full pipeline (single-tenant GPU — one stage at a time):

```bash
../../.venv-vllm/bin/python scripts/harvest.py --stage both        # ~2-4 h
python3 scripts/build_sft.py                                        # CPU
../../.venv/bin/python scripts/train_think.py \
    --out ../../large_artifacts/qwen35_4b_gauntlet_breadth_round1/adapters/round1
../../.venv-vllm/bin/python scripts/eval_gym.py --tag base
../../.venv-vllm/bin/python scripts/eval_gym.py --tag round1 \
    --adapter ../../large_artifacts/qwen35_4b_gauntlet_breadth_round1/adapters/round1
python3 scripts/bench.py --seed <fresh> --tier quick --arms base adapter \
    --adapter large_artifacts/qwen35_4b_gauntlet_breadth_round1/adapters/round1
```

## Results

Fill this after the run. Separate deployable evidence from oracle/hidden evaluation.

## Interpretation

What changed after this result? What is now more likely, less likely, or still unknown?

## Knowledgebase Update

- Program evidence updated:
- Program backlog updated:
- Claim ledger updated:

## Artifacts

- `src/gym/` — the 12-family gym (generators, verifiers, selftests)
- `src/harness.py`, `src/vllm_runner.py` — batched generation + episode driver
- `scripts/` — pipeline stages (selftest, harvest, build_sft, train, eval, bench)
- `configs/default.yaml`
- `runs/` — harvest yields, eval tables, menagerie event log (large row files
  gzipped; adapters external under `large_artifacts/`, see manifest)
- `reports/gym_design.md`, `reports/design_review.md`,
  `reports/artifact_manifest.yaml`
