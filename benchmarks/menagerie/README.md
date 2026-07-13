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
an explicit opt-out. Every named tier now thinks at a fixed, generous **8192**
tokens so the think budget no longer caps measured capability; the tiers stay
ordered by coverage and wall-clock cost, not by think budget. The `huge` tier
adds fully **uncapped** thinking — its budget (65536, == `max_model_len`) is
bounded only by the context window via the runner's per-prompt guard.
`--think-budget N` overrides both atom and episode budgets for explicit
compute-response studies.

| Tier | Budget | Think budget | Atoms | Episodes | Total items |
| --- | ---: | ---: | --- | --- | ---: |
| `quick` | 60 s | 8192 | L1-L2, 4/level (80 items) | none | 80 |
| `medium` | 300 s | 8192 | L1-L4, 2/level (80 items) | L2, 2/level, max 6 turns (20 items) | 100 |
| `slow` | 1200 s | 8192 | L1-L4, 5/level (200 items) | L1-L3, 2/level, max 10 turns (60 items) | 260 |
| `deep` | 3600 s | 8192 | L1-L4, 6/level (240 items) | L1-L4, 3/level, max 14 turns (120 items) | 360 |
| `huge` | 129600 s | 65536 (uncapped, context-bound) | L1-L4, 8/level (320 items) | L1-L4, 4/level, max 14 turns (160 items) | 480 |

Serial compute now dominates the wall-clock: lifting the think budget means the
named tiers' theoretical worst case (every generation binding its full 8192
budget) no longer fits inside their unchanged `budget_s`. This is deliberate —
`budget_s` and coverage are left as-is to keep the tiers ordered by cost, and the
estimator therefore flags the named tiers `OVER` on worst-case wall (measured
walls run well under worst case because base-model episodes terminate early).

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
quick          8192     8192      440.3       321.9              3.4                 2.6      9056        60     OVER
medium         8192     8192     1103.4       734.0             11.1                 7.6     10588       300     OVER
slow           8192     8192     4416.0      2865.3             46.9                31.7     11588      1200     OVER
deep           8192     8192    10603.5      6735.1            117.8                78.2     12588      3600     OVER
huge          65536    65536   105228.8     66881.9            157.0               104.2     65520    129600   WITHIN
```

The `OVER` flags on the named tiers are expected (see above): their `budget_s`
is intentionally unchanged while think budgets were lifted to 8192, so
worst-case wall exceeds `budget_s`. No tier is `CTX-OVER`: `ctx_worst` stays
under `max_model_len` (65536) for every tier, including `huge`, because the
estimator mirrors the runner's per-prompt guard that clamps think to the context
window. `huge` shows `WITHIN` because its `budget_s` is sized to its
context-bounded worst case.

Measured seed-31337 baselines and walls live in
[`results/BASELINES.md`](results/BASELINES.md).

## Tier Predictiveness

All tiers use the same generators. The named tiers scale difficulty, horizon,
and item count at a common fixed 8192 think budget (the `huge` tier additionally
uncaps thinking); they do not swap in a different task distribution.

The instrument was validated with a noisy-oracle ladder from
`results/instrument_validation.json` at seed `0`. `eps=0.0` is the oracle;
`eps=1.0` is fully random.

| Tier | eps=0.0 | eps=0.25 | eps=0.5 | eps=0.75 | eps=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quick` | 1.000 | 0.675 | 0.463 | 0.250 | 0.013 |
| `medium` | 1.000 | 0.738 | 0.492 | 0.274 | 0.024 |
| `slow` | 1.000 | 0.747 | 0.510 | 0.263 | 0.016 |
| `deep` | 1.000 | 0.747 | 0.467 | 0.247 | 0.016 |
| `huge` | 1.000 | 0.740 | 0.477 | 0.246 | 0.016 |

Pairwise Spearman rank stability across the eps ladder, rounded to the printed
validation precision:

| Pair | Spearman |
| --- | ---: |
| `deep\|huge` | 1.000 |
| `deep\|medium` | 1.000 |
| `deep\|quick` | 1.000 |
| `deep\|slow` | 1.000 |
| `huge\|medium` | 1.000 |
| `huge\|quick` | 1.000 |
| `huge\|slow` | 1.000 |
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

Adapter runs are gated: the vLLM backend refuses to score an adapter whose
LoRA has no effect on outputs (an on-vs-off greedy probe at engine start;
claim C49 documents the silent no-op this catches). Prefer merged full
checkpoints via `--model-id` (config-verified against the pinned Qwen3.5-4B
architecture), or `--backend qwen` for HF/PEFT adapter application.

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
