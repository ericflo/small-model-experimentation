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

Tier budgets are for a resident model, no thinking mode, greedy decoding
(`do_sample=False`). They assume the harness defaults unless overridden.

| Tier | Target | Config budget_s | Atoms | Episodes | Max episode turns |
| --- | ---: | ---: | --- | --- | ---: |
| `quick` | <60s | 60 | L1-L2, 4 per level | none | n/a |
| `medium` | <300s | 300 | L1-L3, 6 per level | L1-L2, 3 per level | 4 |
| `slow` | <1200s | 1200 | L1-L4, 8 per level | L1-L3, 6 per level | 10 |
| `deep` | <3600s | 3600 | L1-L4, 16 per level | L1-L4, 10 per level | 14 |

Speed comes from three choices in the harness and contract:

- Atoms run as one batched pass.
- Episodes are lockstep-batched, so wall clock scales with horizon and batch
  count, not with serial item count.
- The action protocol is terse: atoms end with `ANSWER: <value>`; episodes use
  one-line actions and a 96-token action budget.

Current token-math projections from
`python3 run.py --tier all --estimate`:

```text
families: 10 (discovered)
model load time excluded (~60 s once)
tier     mode       batch    worst_s  expected_s  budget_s     flag
quick    no-think      96        6.6         4.1        60   WITHIN
quick    think512      48       95.2        49.1        60     OVER
medium   no-think      96       50.0        29.5       300   WITHIN
medium   think512      48      591.4       304.7       300     OVER
slow     no-think      96      210.1       123.0      1200   WITHIN
slow     think512      48     2338.7      1204.6      1200     OVER
deep     no-think      96      688.9       402.2      3600   WITHIN
deep     think512      48     6983.8      3596.9      3600     OVER
```

Real-model timing is pending the first GPU-idle window.

## Tier Predictiveness

All tiers use the same generators. The tiers scale difficulty, horizon, and
item count; they do not swap in a different task distribution.

The instrument was validated with a noisy-oracle ladder from
`results/instrument_validation.json` at seed `0`. `eps=0.0` is the oracle;
`eps=1.0` is fully random.

| Tier | eps=0.0 | eps=0.25 | eps=0.5 | eps=0.75 | eps=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deep` | 1.000000 | 0.739360 | 0.477786 | 0.255211 | 0.019029 |
| `medium` | 1.000000 | 0.758406 | 0.485615 | 0.263288 | 0.023373 |
| `quick` | 1.000000 | 0.675000 | 0.462500 | 0.250000 | 0.012500 |
| `slow` | 1.000000 | 0.739309 | 0.488960 | 0.252655 | 0.025030 |

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
