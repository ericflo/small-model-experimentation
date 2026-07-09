# Menagerie

Menagerie is a held-out, procedurally generated agentic capability suite for
offline evaluation during Qwen3.5-4B training loops. It contains 10 bespoke
families, all fictional content, zero overlap with public benchmarks, and zero
overlap with the repository's training substrates. It is the blackbox
counterweight to the corpus's whitebox experiments: train on the corpus, but run
Menagerie only as an external instrument. See [`../README.md`](../README.md) for
the benchmark firewall: run-only, never train on items, transcripts, family
source, or scores-as-labels.

> **Benchmark firewall:** Agents must not read Menagerie contents: family
> sources, generated items, transcripts, or result details. Only invoke
> `run.py` or `validate_suite.py` and read aggregate scores; family directory
> names are public metadata, but contents are read-forbidden.

## Families

| Family | Capability axis | Paradigm |
| --- | --- | --- |
| `chronicle` | State tracking: simulate an evolving fictional world from an ordered event stream and report one final-state fact. | single-turn |
| `lockpick` | Active rule induction and exploitation: choose informative probes against a hidden symbol machine under a tight probe budget, induce the input->output rule, then invert it to construct an opening input for a target output. | multi-turn |
| `menders` | Debugging and program repair: localize and repair a small broken program from failing execution evidence, then use rerun feedback across a bounded multi-turn episode. | multi-turn |
| `mirage` | Calibrated abstention: answer when the prompted constraint system forces a unique target value, and reply `IMPOSSIBLE` when the constraint system is provably unsatisfiable. | single-turn |
| `rites` | Protocol/state-machine compliance: execute a compact, freshly generated procedure while mentally tracking hidden-but-documented state and flags. | multi-turn |
| `siftstack` | Information triage over a noisy fictional ledger: resolve aliases, ignore near-miss distractors, apply later-record supersession, and aggregate a count, sum, or current/latest value from document-bound facts. | single-turn |
| `sirens` | Goal fidelity under prompt injection: complete a fictional document-retrieval task while untrusted documents contain embedded adversarial directives. | multi-turn |
| `stockade` | Bounded optimization under explicit constraints. | single-turn |
| `toolsmith` | Orchestrating dependent tool calls when later calls require opaque values returned by earlier calls. | multi-turn |
| `warren` | Partially observable exploration with spatial memory: navigate an unseen, text-described chamber graph to a named target under a strict move budget. | multi-turn |

## Tiers

Tier budgets are for a resident model with thinking enabled and greedy decoding
(`do_sample=False`). Thinking is the default deployment mode; `--no-think` is
an explicit opt-out. Think budgets floor at 1024 tokens and escalate by tier.
`--think-budget N` overrides both atom and episode budgets for explicit
compute-response studies.

| Tier | Budget | Think budget | Atoms | Episodes | Total items |
| --- | ---: | ---: | --- | --- | ---: |
| `quick` | 60 s | 1024 | L1-L2, 4/level (80 items) | none | 80 |
| `medium` | 300 s | 1024 | L1-L3, 5/level (150 items) | L1-L2, 3/level, max 4 turns (60 items) | 210 |
| `slow` | 1200 s | 2048 | L1-L4, 5/level (200 items) | L1-L3, 2/level, max 10 turns (60 items) | 260 |
| `deep` | 3600 s | 4096 (episodes capped at 2048/turn) | L1-L4, 6/level (240 items) | L1-L4, 3/level, max 14 turns (120 items) | 360 |

Speed comes from the harness and contract:

- Atoms run as one batched pass.
- Episodes are lockstep-batched, so wall clock scales with horizon and batch
  count, not with serial item count.
- The action protocol is terse: atoms end with `ANSWER: <value>`; episodes use
  one-line actions and a 96-token action budget.
- The HF backend chunks work at `--max-batch`: 48 items with thinking or 96
  with `--no-think` by default.
- The default vLLM backend schedules internally with `max_num_seqs=64` and is
  approximately four times faster than HF.

Current token-math projections from
`python3 run.py --estimate --tier all`:

```text
families: 10 (assumed)
model load time excluded (~35 s once, vLLM)
tier     atom_think ep_think    worst_s  expected_s no_think_worst_s no_think_expected_s ctx_worst  budget_s     flag
quick          1024     1024       58.0        42.5              3.4                 2.6      1888        60   WITHIN
medium         1024     1024      288.0       191.6             21.8                15.0      2920       300   WITHIN
slow           2048     2048     1139.2       740.1             46.9                31.7      5444      1200   WITHIN
deep           4096     2048     3066.9      1981.9            117.8                78.2      6444      3600   WITHIN
```

Measured seed-31337 baselines and walls live in
[`results/BASELINES.md`](results/BASELINES.md).

## Tier Predictiveness

All tiers use the same generators. The tiers scale difficulty, horizon, item
count, and thinking budget; they do not swap in a different task distribution.

The instrument was validated with a noisy-oracle ladder from
`results/instrument_validation.json` at seed `0`. `eps=0.0` is the oracle;
`eps=1.0` is fully random.

| Tier | eps=0.0 | eps=0.25 | eps=0.5 | eps=0.75 | eps=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quick` | 1.000 | 0.675 | 0.463 | 0.250 | 0.013 |
| `medium` | 1.000 | 0.762 | 0.513 | 0.278 | 0.027 |
| `slow` | 1.000 | 0.747 | 0.510 | 0.263 | 0.016 |
| `deep` | 1.000 | 0.747 | 0.467 | 0.247 | 0.016 |

Pairwise Spearman rank stability across the eps ladder, rounded to the printed
validation precision:

| Pair | Spearman |
| --- | ---: |
| `deep\|medium` | 1.000 |
| `deep\|quick` | 1.000 |
| `deep\|slow` | 1.000 |
| `medium\|quick` | 1.000 |
| `medium\|slow` | 1.000 |
| `quick\|slow` | 1.000 |

This validates the instrument as a tiered measurement device: when policies are
synthetically degraded by increasing `eps`, every tier ranks the policies
consistently. It validates the instrument, not real-model cross-tier agreement;
see [`results/BASELINES.md`](results/BASELINES.md) for the measured real-model
correlations.

## Usage

Run commands from `benchmarks/menagerie` unless noted.

CPU policy smoke run:

```bash
python3 run.py --tier quick --backend oracle --seed 1001 --out results/quick_oracle_seed1001.json
```

Qwen run using the default `qwen_vllm` backend and the required repository
virtualenv interpreter:

```bash
/home/ericflo/Development/small-model-experimentation/.venv/bin/python /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/run.py --tier quick --seed 1001 --out /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/results/quick_qwen_seed1001.json
```

The harness process itself, including default vLLM runs, must run under that
required HF-virtualenv interpreter. The vLLM engine lives in `.venv-vllm` and
is spawned as a subprocess; if that environment is missing, the runner reports
an actionable error with the commands from
[`docs/vllm_inference.md`](../../docs/vllm_inference.md) needed to build it. Use
`--backend qwen` for the approximately four-times-slower HF deterministic parity
oracle.

Qwen runs use tier thinking budgets by default. Add `--no-think` only for an
explicit no-thinking reference run, or `--think-budget N` to override both atom
and episode budgets.

Checkpoint run:

```bash
/home/ericflo/Development/small-model-experimentation/.venv/bin/python /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/run.py --tier quick --model-id /path/to/checkpoint --device cuda:0 --seed 1001 --out /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/results/quick_qwen_seed1001.json
```

Instrument validation:

```bash
python3 validate_suite.py --seed 0 --out results/instrument_validation.json
```

From the repository root:

```bash
make bench TIER=quick
```

Use a fresh `--seed` for each real evaluation so generated items remain
unexposed across training loops. Compare runs only with matched seed, tier,
backend, model, decoding setup, and engine configuration. The HF parity oracle
is deterministic; vLLM continuous batching can introduce small item-level
variation.

## Interpreting Results

The runner writes one JSON file per run. The important fields are:

- `per_family`: family-level mean score, item count, and family wall time.
- `aggregate`: mean of the family means.
- `per_item`: score-only per-item records by default (`id`, `family`, `level`,
  `mode`, `score`, `turns`, and `wall_s`). Full transcripts and score details
  require `--debug-artifacts` with an output filename containing
  `DO_NOT_TRAIN`; these artifacts must never enter the repository.
- `think_budget`: when thinking is enabled, an `{atom, episode}` budget object.
- `within_budget`: whether observed wall time stayed inside the tier budget.

Compare runs only when seed, tier, backend, engine configuration, and decoding
setup match. For model selection during a training loop, use a fresh seed for
each evaluation event, then compare checkpoints evaluated on that same fresh
seed and tier.

## Extending

New families must implement [`CONTRACT.md`](CONTRACT.md): deterministic fresh
generation, stdlib-only CPU code, machine-checkable scoring, oracle and random
policies, terse action formats, and the full `META` declaration. Each family
must pass `python3 -m families.<name>.selftest`, including the oracle, random,
degenerate, noisy-oracle, budget, purity, and any adversarial battery checks
needed for that family, and the suite-level `python3 validate_suite.py` run must
remain green before the family is added to a tier.
