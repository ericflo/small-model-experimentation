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

Deployable evidence (greedy, think-mode, deployed budgets, bare model — no
scaffolding anywhere):

- **Menagerie, six paired events, two backends** (each event = base and
  install on the same fresh seed; HF backend is deterministic, the vLLM
  event uses the merged checkpoint): quick 0.140→0.363 (+0.223, seed 52004),
  0.152→0.446 (+0.294, 52005), 0.115→0.400 (+0.285, 52007), 0.152→0.379
  (+0.227, 52008), 0.150→0.424 (+0.274, 52010, vLLM/merged); medium
  0.122→0.445 (+0.324, 52006) and 0.161→0.424 (+0.264, 52009) with EVERY
  family positive at medium. Pre-registered decision rule met on all legs:
  verdict POSITIVE.
- **Gym-internal** (held-out item seeds): mean 0.184→0.701 (+0.518),
  including the two never-trained held-out families (brinework +0.540,
  spindle +0.608) and zero-training-data stallwright (+0.395). Parse
  failures collapsed (caravan 98→8 of 100).
- **Round 3 (expert iteration)**: re-harvest with the round-2 model opened
  the starved frontiers at the data level (stallwright 0/160→48/60 correct)
  but blackbox deltas did NOT compound (quick +0.285/+0.227 vs round-2's
  +0.223/+0.294; medium +0.264 vs +0.324) — the install is a one-time recipe
  step change; same-recipe iteration re-saturates.
- **Round-1 null (mechanism)**: full-weight SFT on the model's own verified
  naturally-closed chains installed nothing (near-self-distillation); the
  working round-2 recipe added terse-target canonicalization, forced-close
  recovery examples, and emission-seam loss weighting.

Oracle/hidden evidence is confined to instrument validation: gym oracle
policies certify the graders; no oracle output, benchmark content, or
external model enters training. Instrument finding C49 (Confirmed): vLLM
runtime LoRA silently no-ops on Qwen3.5-4B PEFT adapters — all valid
comparisons above are paired within-backend, and the harness now gates
adapter application on-vs-off. Full tables: `reports/report.md`.

## Interpretation

The binding deployed constraint at these difficulty levels was the
truncation cascade at the answer-emission seam (consume any think budget →
force-close → verbose restart → no parseable answer). Training the model on
its own verified outputs to conclude and to commit from a truncated chain
removes that constraint substrate-generally — across the gym families it
never saw and across the blackbox instrument. More likely now: breadth +
strict verifiers + emission-seam gradient placement installs general agentic
competence (the C43/C45/C48 locality laws do not extend to this regime).
Less likely: dose or same-recipe iteration as further levers (round 3
re-saturated). Still unknown: how much of the delta is protocol-emission
repair vs axis competence (recovery-arm-only ablation queued), whether
breadth is causal for held-out transfer (breadth-vs-dose ablation queued),
and whether difficulty escalation reopens the frontier.

## Knowledgebase Update

- Program evidence updated: `research_programs/agentic_breadth_installation/evidence.md`
- Program backlog updated: `research_programs/agentic_breadth_installation/backlog.md`
  (+ queue proposal `gauntlet_round3_expert_iteration`, executed)
- Claim ledger updated: C49 (Confirmed — vLLM LoRA silent no-op + shipped
  instrument gate) and C50 (Promising — breadth install moves the blackbox;
  round-3 re-saturation scoped); `knowledge/synthesis.md` executive read #13

## Artifacts

- `src/gym/` — the 12-family gym (generators, verifiers, selftests)
- `src/harness.py`, `src/vllm_runner.py` — batched generation + episode driver
- `scripts/` — pipeline stages (selftest, harvest, build_sft, train, eval, bench)
- `configs/default.yaml`
- `runs/` — harvest yields, eval tables, menagerie event log (large row files
  gzipped; adapters external under `large_artifacts/`, see manifest)
- `reports/gym_design.md`, `reports/design_review.md`,
  `reports/artifact_manifest.yaml`
