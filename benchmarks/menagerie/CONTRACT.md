# Menagerie family contract

Every task family in `families/<name>/` implements this contract exactly.
The harness, the tier configs, and the instrument-validation suite all assume it.
Deviations break the suite; the selftest is the gate.

## What a family is

One diverse, bespoke **agentic capability axis**, evaluated through procedurally
generated tasks with machine-checkable scoring. Families must be:

1. **Novel by construction.** All task content (entity names, operation names,
   mini-language keywords, world vocabulary) is procedurally generated from
   fictional token stock — never real-world nouns, never a known puzzle format
   verbatim. It must be impossible for any pretraining corpus or public
   benchmark to contain these items. The family README documents the 3 nearest
   public benchmarks and states concretely how this family differs.
2. **Knowledge-free.** No real-world trivia required. Everything needed to act
   is in the prompt/observations. We measure capability, not recall.
3. **Fresh forever.** Same `(seed, level, n)` → byte-identical items
   (determinism); different seed → disjoint items. No wall-clock, no global RNG
   (use `random.Random(seed)` locally).
4. **Cheap.** Stdlib-only, pure CPU. Generation + scoring < 50 ms per item.
   No I/O, no network, no files.

## Module layout

```
families/<name>/
  __init__.py        # empty
  family.py          # everything below lives here
  selftest.py        # runnable gate: python3 -m families.<name>.selftest
  README.md          # capability axis, task description, novelty statement,
                     # level ladder, example item + oracle transcript
```

## Required API (`family.py`)

```python
META = {
    "name": "<name>",                  # directory name
    "capability": "<one line: the capability axis>",
    "paradigm": "single-turn" | "multi-turn",
    "action_format": "<one line telling the model how to answer/act>",
}

def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    # mode == "atom":    single-shot items (max_turns == 1)
    # mode == "episode": full multi-turn items (single-turn families: same as atom
    #                    but MAY be a harder/composited variant)
    # level in {1,2,3,4}: difficulty ladder (see below)
    # Every item dict contains at least:
    #   {"id": str, "level": int, "mode": str, "max_turns": int, ...private fields}
    # Private fields (targets, hidden state) are NEVER shown to the model.

class Env:
    def __init__(self, item: dict): ...
    def reset(self) -> str                      # initial observation (includes task instructions
                                                #  and the required action/answer format)
    def step(self, action: str) -> tuple[str, bool]   # (next observation, done)
    # Deterministic. Malformed actions get a curt corrective observation, never a crash.

def score(item: dict, transcript: list[dict]) -> dict:
    # transcript: [{"obs": str, "action": str}, ...] in order.
    # Returns {"score": float in [0,1], ...any diagnostics}.
    # Machine-checkable only — no LLM judging, no fuzzy semantics.

def oracle_policy(item: dict, history: list[dict]) -> str    # perfect play
def random_policy(item: dict, history: list[dict], rng) -> str
```

## Difficulty ladder

- **L1**: easy — a competent human solves it in <30 s; designed so a weak model
  scores well above 0. (An instrument that floors at 0 measures nothing.)
- **L2**: moderate; **L3**: hard; **L4**: frontier — should challenge much
  stronger models than a 4B. Difficulty must be *monotone*: the noisy-oracle
  score must decline with level (selftest checks this at ε=0.5).

## Answer / action protocol

- Atoms: the observation ends with an explicit instruction to reply with a
  final line `ANSWER: <value>`. Scorer extracts the LAST line matching
  `ANSWER:` (case-insensitive, whitespace-tolerant) and compares canonically.
- Episodes: actions are a single terse line (family defines the verb grammar in
  the observation, e.g. `MOVE N` / `CALL tool(args)` / `SAY x`). The scorer and
  env must tolerate surrounding whitespace and case where unambiguous.
- Model output budget: atoms parse within 64 new tokens; episode actions within
  96. Observations ≤ 800 chars per turn; atom prompts ≤ 1200 chars. Episodes:
  `max_turns` ≤ 4 (L1-2) / 10 (L3) / 14 (L4).

## Scoring discipline

- `score` ∈ [0,1]; partial credit is encouraged where objective (e.g., ratio to
  brute-force optimum, fraction of constraints held), but must not reward
  degenerate play (see gates).
- Where an optimum is claimed, the generator must verify it by brute force at
  generation time (keep instances small enough for that).

## Selftest gates (`selftest.py` must assert ALL, exit non-zero on failure)

1. Determinism: `generate(7, L, 6, mode)` twice → identical JSON, all levels/modes.
2. Seed disjointness: seeds 7 vs 8 share no item content.
3. Oracle perfection: oracle_policy scores **1.0** on every item (all levels, both modes).
4. Random floor: mean random_policy score ≤ 0.05 (or ≤ 0.15 with justification
   in README for ratio-scored families).
5. Degenerate resistance: empty-string policy, constant most-plausible-answer
   policy, and echo-the-observation policy each score ≤ 0.1 mean.
6. Monotone difficulty: noisy-oracle (ε=0.5: each turn, 50% oracle action / 50%
   random) mean score is non-increasing L1→L4 (tolerance 0.05).
7. Budgets: prompt/observation char limits, max_turns caps, generation+scoring
   < 50 ms/item.
8. Purity: module imports nothing beyond the stdlib.

## What the harness guarantees (so families don't)

- Batched lockstep execution across families (speed comes from the harness).
- Chat templating, generation, token budgets, think-mode toggles.
- Aggregation, per-item logging, timing.

Families never import torch/transformers and never print.
