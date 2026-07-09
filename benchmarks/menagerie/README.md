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

Tier budgets are for a resident model with thinking enabled, greedy decoding
(`do_sample=False`). Thinking is the default deployment mode; `--no-think` is
an explicit opt-out. The thinking budget is part of the tier ladder and can be
overridden with `--think-budget`.

| Tier | Target | Config budget_s | Think budget | Atoms | Episodes | Max episode turns |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `quick` | <60s | 60 | 256 | L1-L2, 4 per level | none | n/a |
| `medium` | <300s | 300 | 512 | L1-L3, 2 per level | L1-L2, 2 per level | 4 |
| `slow` | <1200s | 1200 | 1024 | L1-L4, 3 per level | L1-L3, 1 per level | 10 |
| `deep` | <3600s | 3600 | 2048 | L1-L4, 4 per level | L1-L4, 1 per level | 14 |

Speed comes from three choices in the harness and contract:

- Atoms run as one batched pass.
- Episodes are lockstep-batched, so wall clock scales with horizon and batch
  count, not with serial item count.
- The action protocol is terse: atoms end with `ANSWER: <value>`; episodes use
  one-line actions and a 96-token action budget.
- Thinking uses batch size 48 by default. `--no-think` uses batch size 96 unless
  `--max-batch` overrides it.

Current token-math projections from
`python3 run.py --tier all --estimate`:

```text
families: 10 (discovered)
model load time excluded (~60 s once)
tier     think_budget batch    worst_s  expected_s no_think_worst_s no_think_expected_s  budget_s     flag
quick             256    48       54.2        28.6              6.6                 4.1        60   WITHIN
medium            512    48      295.7       152.4             43.3                25.4       300   WITHIN
slow             1024    48     1176.6       598.1            105.0                61.5      1200   WITHIN
deep             2048    48     3104.1      1565.6            141.8                82.9      3600   WITHIN
```

Real-model timing is pending the first GPU-idle window.

## Tier Predictiveness

All tiers use the same generators. The tiers scale difficulty, horizon, item
count, and thinking budget; they do not swap in a different task distribution.

The instrument was validated with a noisy-oracle ladder from
`results/instrument_validation.json` at seed `0`. `eps=0.0` is the oracle;
`eps=1.0` is fully random.

| Tier | eps=0.0 | eps=0.25 | eps=0.5 | eps=0.75 | eps=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deep` | 1.000000 | 0.718315 | 0.490457 | 0.254583 | 0.012480 |
| `medium` | 1.000000 | 0.776111 | 0.494881 | 0.305944 | 0.037960 |
| `quick` | 1.000000 | 0.675000 | 0.462500 | 0.250000 | 0.012500 |
| `slow` | 1.000000 | 0.725278 | 0.492276 | 0.279444 | 0.016639 |

Pairwise Spearman rank stability across the eps ladder, rounded to the printed
validation precision:

| Pair | Spearman |
| --- | ---: |
| `deep|medium` | 1.000 |
| `deep|quick` | 1.000 |
| `deep|slow` | 1.000 |
| `medium|quick` | 1.000 |
| `medium|slow` | 1.000 |
| `quick|slow` | 1.000 |

This validates the instrument as a tiered measurement device: when policies are
synthetically degraded by increasing `eps`, every tier ranks the policies
consistently. It does not validate that real-model scores on quick, medium,
slow, and deep will correlate; that remains to confirm once checkpoints exist.

## Usage

Run commands from `benchmarks/menagerie` unless noted.

CPU policy smoke run:

```bash
python3 run.py --tier quick --backend oracle --seed 1001 --out results/quick_oracle_seed1001.json
```

Qwen backend run, using the required repository virtualenv interpreter:

```bash
/home/ericflo/Development/small-model-experimentation/.venv/bin/python /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/run.py --tier quick --backend qwen --seed 1001 --out /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/results/quick_qwen_seed1001.json
```

Qwen runs use tier thinking budgets by default. Add `--no-think` only for an
explicit no-thinking reference run, or `--think-budget N` to override the tier
budget.

Checkpoint run:

```bash
/home/ericflo/Development/small-model-experimentation/.venv/bin/python /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/run.py --tier quick --backend qwen --model-id /path/to/checkpoint --device cuda:0 --seed 1001 --out /home/ericflo/Development/smx-menagerie/benchmarks/menagerie/results/quick_qwen_seed1001.json
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
unexposed across training loops. Runs are deterministic for a fixed seed, tier,
backend, model, and decoding configuration.

## Interpreting Results

The runner writes one JSON file per run. The important fields are:

- `per_family`: family-level mean score, item count, and family wall time.
- `aggregate`: mean of the family means.
- `per_item`: item-level records, including transcripts unless
  `--no-transcripts` was used.
- `within_budget`: whether observed wall time stayed inside the tier budget.

Compare runs only when `seed`, `tier`, and decoding setup match. For model
selection during a training loop, use a fresh seed for each evaluation event,
then compare checkpoints evaluated on that same fresh seed and tier.

## Extending

New families must implement [`CONTRACT.md`](CONTRACT.md): deterministic fresh
generation, stdlib-only CPU code, machine-checkable scoring, oracle and random
policies, terse action formats, and the full `META` declaration. Each family
must pass `python3 -m families.<name>.selftest`, including the oracle, random,
degenerate, noisy-oracle, budget, purity, and any adversarial battery checks
needed for that family, and the suite-level `python3 validate_suite.py` run must
remain green before the family is added to a tier.
