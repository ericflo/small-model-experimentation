# The Gauntlet — gym design (round 1)

A firewall-clean, procedurally generated training gym of 12 task families for
breadth-first expert iteration on Qwen3.5-4B. The gym is the *training*
counterpart to the corpus's blackbox measurement doctrine: train here, measure
on menagerie (fresh seeds, aggregate scores only).

## Provenance and firewall statement

- Authored in a session that read ONLY the permitted benchmark surface:
  `benchmarks/README.md`, `benchmarks/menagerie/README.md`,
  `benchmarks/menagerie/CONTRACT.md`, `benchmarks/menagerie/results/BASELINES.md`.
  No family sources, generated items, transcripts, or per-item results were
  ever read. Family alignment is to the ten *public one-line capability-axis
  descriptions* and the generic protocol conventions (final-line
  `ANSWER: <value>` atoms; one-line verb-grammar episode actions; bounded
  horizons; corrective observations) — all deliberately-public metadata.
- All gym content is invented here: family names, vocabularies, mechanics,
  and surface framings share nothing with the public menagerie family *names*,
  and cannot share content with their items (never seen, by construction).
- Menagerie is consumed only via `run.py` CLI with a fresh seed per evaluation
  event; only `aggregate` and `per_family` numbers are read. No benchmark
  score, item, or transcript is ever used as a training signal.
- Training data provenance: the model's OWN sampled episodes/answers, filtered
  by gym-internal machine verifiers. No other model is involved anywhere.

## Scientific question

The corpus's install laws are all *local* (C43 shift-specific, C45/C48
substrate-local, C14 format-local) — but every one was derived from training
on a single narrow substrate. Round 1 tests the untried variable: does
**breadth** (10 simultaneous, format-diverse families spanning 10 capability
axes) install capability that generalizes (a) to held-out items of trained
families, (b) to held-out gym families, and (c) to the blackbox menagerie
instrument?

## The 12 families

Trained (10). Paradigm: A = atoms (single-turn), E = episodes (multi-turn).

| # | family | capability axis | modes | verifier |
|---|--------|-----------------|-------|----------|
| 1 | `caravan` | state tracking over a natural-language event stream (goods, wagons, crew; gains/losses/transfers/spoilage/conditionals), final-state question | A | event simulator |
| 2 | `foundry_ledger` | information triage: aliased patrons, near-miss distractors, later-record supersession; count/sum/latest queries over a bell-foundry commission ledger | A | canonical-record aggregator |
| 3 | `stallwright` | bounded optimization: choose a subset/assignment of market stalls under capacity/adjacency/quota constraints maximizing fees; partial credit = achieved/optimum | A | brute-force optimum |
| 4 | `runeward` | calibrated abstention: small rune-valued constraint systems; unique solution → value, provably unsatisfiable → `IMPOSSIBLE` (~50/50, no surface cue) | A | constraint solver |
| 5 | `kilnrite` | protocol/state-machine compliance: freshly generated kiln-firing procedure with documented flags and conditions; execute or answer state queries | A+E | state machine |
| 6 | `glyphgate` | active rule induction: hidden glyph-string transformation; probe under budget, then construct an opening input for a target output | A+E | hidden-rule application |
| 7 | `loomfix` | program repair: one-bug programs in SPOOL (invented 4-register mini-language), failing tests shown; patch and rerun | A+E | SPOOL interpreter + tests |
| 8 | `ferrier` | dependent tool chaining: wharf-operations tool registry returning opaque handles consumed by later calls (3–7 call DAG) | A+E | DAG/handle checker |
| 9 | `burrowmaze` | partially observable exploration with spatial memory: hidden chamber graph, current-chamber view only, strict move budget | A+E | graph simulator |
| 10 | `gatepost` | goal fidelity under prompt injection: retrieval over documents with embedded adversarial directives; answer the original question | A+E | gold-answer match (+ injection-compliance diagnostic) |

Held out (2) — never trained, gym-internal transfer probes:

| # | family | probes | modes |
|---|--------|--------|-------|
| 11 | `brinework` | near-transfer: state tracking + triage hybrid on a fresh surface (salt-evaporation works) | A |
| 12 | `spindle` | execute a freshly stated multi-step symbolic procedure on a token tape (formal execution + compliance hybrid) | A+E |

## Shared protocol shape

- **Atoms**: one user message; the prompt ends with an explicit instruction to
  finish with one final line `ANSWER: <value>`. Scoring extracts the LAST
  `ANSWER:` line (case-insensitive, whitespace-tolerant), canonicalizes per
  family, and compares. Binary score, except objective partial credit
  (`stallwright`: achieved/optimum ratio; `burrowmaze` episodes: found=1).
- **Episodes**: system message states the rules and a terse one-line action
  grammar (`DO X` / `PROBE X` / `CALL t(a,b)` / `GO exit` / `PATCH k op` /
  `READ d` / final `ANSWER: v` or family-specific commit verb). Each turn the
  model replies with exactly one action line (first non-empty line of the
  answer channel is taken). Malformed actions get a curt corrective
  observation and cost the turn. Horizons: L1 4, L2 6, L3 10, L4 14 turns.
- **Levels** L1–L4 scale entity counts, stream lengths, constraint counts,
  probe budgets, graph sizes, and horizons. L1 is sized so the base model has
  real pass@6 yield; L4 is stretch.
- **Format diversity**: families deliberately vary surface style (prose,
  bullet records, terse logs) and vocabulary; nothing shares invented proper
  nouns across families.

## Family module contract (`src/gym/families/<name>.py`)

Every family module exposes exactly:

```python
FAMILY: str                      # module name
LEVELS: tuple[int, ...] = (1, 2, 3, 4)
HAS_EPISODES: bool

def gen_atoms(seed: int, level: int, n: int) -> list[dict]
    # deterministic; returns dicts with keys:
    #   id, family, level, prompt (str), gold (JSON-safe verification payload)

def score_atom(item: dict, reply_text: str) -> float
    # extracts the final ANSWER: line itself (via gym.base helpers) and
    # returns a score in [0,1]

def oracle_atom(item: dict) -> str
    # a correct full reply (may be just the ANSWER line)

class Episode:                   # only if HAS_EPISODES
    def __init__(self, seed: int, level: int): ...
    spec: dict                   # JSON-safe episode description (for logging)
    max_turns: int
    def system_prompt(self) -> str
    def initial_observation(self) -> str
    def step(self, action_line: str) -> tuple[str, bool]   # (observation, done)
    def score(self) -> float                                # in [0,1], any time after done or budget exhaustion

class OraclePolicy:              # only if HAS_EPISODES
    def __init__(self, episode: Episode): ...
    def act(self, observation_history: list[str]) -> str

def selftest() -> None
    # asserts: generator determinism (same seed → identical items);
    # oracle atoms score 1.0 on >=95% of a 40-item sample per level;
    # random/degenerate replies score <=0.1 mean; episodes: OraclePolicy
    # reaches score >=0.95 mean, random-action policy <=0.1 mean;
    # all items JSON-serializable; prompts <= 1400 chars (atoms),
    # observations <= 800 chars (episodes).
```

Determinism: every generator derives all randomness from
`random.Random((seed, level, index))`-style local instances. No wall-clock, no
global RNG.

Hard content rules for implementers:
- Do NOT read anything under `benchmarks/` — the ten one-line axis
  descriptions quoted in this document are the only benchmark-derived input.
- Do NOT use the words chronicle/lockpick/menders/mirage/rites/siftstack/
  sirens/stockade/toolsmith/warren anywhere in family code or content.
- Invent all vocabulary; keep items solvable purely from the prompt (no
  world-knowledge dependence), machine-checkable, and free of real-world
  named entities.

## Iteration doctrine (user directive, 2026-07-09)

Iteration speed is the research budget: menagerie quick is a 44-second
instrument and must be consulted EVERY round, not after multi-hour harvests.
The loop therefore runs in two profiles:

- **fast** (`configs/fast.yaml`, the default): small harvest (K=2, L1–L2,
  family-adaptive think budgets), ~15-min train, menagerie quick
  base-vs-adapter on one fresh seed each round. Target cycle ≤ ~2 h. Fast
  rounds are exploration — their quick deltas guide the next round and are
  never claim evidence on their own.
- **full** (`configs/default.yaml`): the registered decision regime (bigger
  dose, K=4, L3, guess-gate active), run once fast rounds identify a recipe
  worth confirming; claims follow the pre-registered decision rule below
  (two quick seeds + medium + null calibration).

## Harvest → filter → train (round 1)

Registered operative regime (revised after the pre-run smoke,
`runs/harvest_smoke`, which showed (a) terminal-marker pollution masking real
competence — fixed in `gym/base.py` — and (b) at think 2048 the model solves
hard-family items in-think but force-closes while double-checking, so the
budget moved to 4096; both revisions predate any full harvest or training run):

- **Harvest** (vLLM, `src/vllm_runner.py`, budget thinking): atoms L1–L3,
  100/100/40 items per family-level, K=4 samples, temperature 0.8,
  top_p 0.95, top_k 20, atom think budget 4096, answer cap 192 (training
  filter keeps only ≤96-token answer shapes). Episodes L1–L2 for the six
  episode families, 40 per family-level, K=4 rollouts, per-turn think budget
  2048, lockstep-batched.
- **Filter** (the verifier is the training seat, per C47): keep samples with
  verified score == 1.0 (stallwright: == optimum), naturally-closed thinking
  (no forced close — bakes think-economy into the data), clean parse, think
  ≤2000 tokens (quick deploys at 1024, but medium/slow/deep deploy at
  2048–4096; prefer-shortest biases the kept mix low), answer ≤96 tokens.
  Lucky-guess gate (C28): items with answer_domain < 5 need ≥3/K correct
  samples before any is kept. At most 2 samples per atom item (shortest
  thinking first); at most 2 rollouts per episode instance; episode turns
  require action_ok (no refused/malformed actions as targets).
  **Atom target canonicalization** (added pre-training after the fast-harvest
  filter audit): the base model precedes its ANSWER line with a verbose
  re-explanation, which a ≤96-token answer filter rejected wholesale (e.g.
  68/104 caravan keepers). Atom targets keep the model's own think chain
  verbatim and replace the answer region with exactly `ANSWER: <value>` —
  the model's own verified value in the terse deployable shape (deployment
  scorers read a short answer window). Episode targets already use the
  extracted one-line action. Per-family cap
  900 trimmed by (level, kind) round-robin, shortest-think-first within cell.
- **Yield gate (pre-registered fallback)**: if a trained family lands < 100
  SFT examples, top-up harvest that family at K=8; if still < 100, train with
  what exists and flag the family as harvest-starved in the report; a family
  at 0 is documented as unharvestable-at-round-1, and the breadth claim is
  scoped to the families actually represented.
- **Train**: QLoRA r32/α64 think-channel SFT — targets are the model's own
  `<think>…</think>` + answer/action tokens; loss on assistant tokens only;
  episodes contribute one example per assistant turn with the context rendered
  exactly as at generation time (prior-turn thinking stripped by the chat
  template, matching deployment). Never answer-only (C43).
- **Eval ladder** per round: (i) gym-internal greedy@1 on held-out seeds of
  trained families + the two held-out families, base vs adapter (whitebox);
  (ii) menagerie quick, fresh seed, base vs adapter same seed (blackbox);
  (iii) medium at milestones, slow before any claim.

## Success criteria and noise discipline (pre-registered decision rule)

- **Null calibration first**: before any adapter eval, run base-vs-base twice
  on the same fresh quick seed (and once more on a second seed) to measure the
  actual same-seed decode-nondeterminism spread of the instrument as deployed.
  The published ~0.011 figure is a cross-backend single realization, not our
  H0; the calibration replaces it.
- **Primary decision metric**: paired adapter−base delta on menagerie quick,
  averaged over TWO fresh seeds, plus one medium event (episodes + 100 items)
  as confirmation. POSITIVE (breadth moves the blackbox instrument): quick
  mean delta ≥ +0.03 with both seeds positive, AND medium delta ≥ +0.02.
  NEGATIVE ("locality survives breadth"): quick mean delta ≤ +0.01 AND medium
  delta ≤ +0.01 AND flat gym-internal held-out-family transfer. Anything in
  between is INCONCLUSIVE → iterate (more seeds, more rounds) rather than
  claim. A quick-only detectable effect is honestly ~≥ +0.06 per event; the
  two-seed mean plus medium confirmation is what buys power at +0.03.
- Gym-internal transfer ladder gives the mechanism read: trained-family
  held-out items (distribution fit — note small latent spaces at L1 make this
  rung partly rule-memorization; interpret accordingly) vs held-out families
  (cross-substrate transfer) vs menagerie (blackbox generality).
- Diagnostics tracked: parse-failure rate, forced-close rate, episode horizon
  use, per-family deltas. Protocol-shape vs axis-competence inference rule
  (pre-registered): a broad uniform small lift across most menagerie families
  accompanied by gym parse-fail collapse reads as protocol-shape installation;
  lifts concentrated in families matching trained axes with stable parse rates
  read as axis competence. Both are real installs; they are reported as
  separate components, not conflated.
- Eval strata never harvested (L4 atoms, L3 episodes) are labeled
  extrapolation rungs in all tables.
- Negative results are codified with equal weight: "locality survives
  breadth" would be the strongest available hardening of C43/C45/C48.
