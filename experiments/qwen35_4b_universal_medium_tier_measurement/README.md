# Universal-Line Medium-Tier Measurement

The contamination-free universal line's first medium-tier paired benchmark event: four published composites, one fresh sealed seed, the goal gate recorded at the granularity where history says it is winnable.

**Status:** finished · 2026-07-15 · verdict MEASUREMENT_READ_COMPLETE — hygiene_explore leads at medium (0.3379); all three treated arms hit 8/10 strict family wins vs base (the historical mode) with hygiene_explore and replay_repeat carrying zero losses; the goal gate is two tie-flips wide (menders and rites, both at 0)

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the tier forensics (goal gate 9/94 at medium vs 1/84 at quick; base never at a family ceiling at medium; menders/sirens constants are quick artifacts); the goal-gap pilot (7 families up, 0 down at quick, gate failed on the two artifacts); replay_repeat 0.5081 best-ever quick aggregate.

## Question

Where does the line's Pareto set actually stand at medium: does the quick ordering hold, how close is each arm to the recorded all-ten-families goal gate, and which families block the next dose?

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms (explicit merges, tree-hash bound at event time): `base` (tree 26d8ee48…, weights b654e033…), `designed_fresh` (93433aa2…), `replay_repeat` (4c4f3561…), `hygiene_explore` (9eb653d7…).
- Event: tier medium, think budget 1,024, sealed fresh seed 78,150, trusted gateway only, one-seed ledger, sequential same-seed runs in frozen order.
- Readings (no promotion bars): medium aggregates + ordering vs quick; recorded goal gate (strict wins vs base per family); base sanity envelope vs the forensics' historical distribution; blocking families per arm.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_medium_tier_measurement/scripts/run.py --smoke
.venv/bin/python -B experiments/qwen35_4b_universal_medium_tier_measurement/scripts/run.py --stage benchmark
```

## Results

The single event ran clean (all four arms authenticated, within budget, base inside the historical envelope on every family):

| arm | medium aggregate | strict wins vs base | ties | losses |
|---|---|---|---|---|
| base | 0.0567 | — | — | — |
| hygiene_explore | **0.3379** | 8/10 | menders, rites | none |
| designed_fresh | 0.3197 | 8/10 | menders | warren (0.050 vs 0.067) |
| replay_repeat | 0.2981 | 8/10 | menders, rites | none |

Per-family (base → hygiene_explore): chronicle 0→0.5, lockpick 0→0.2, menders 0→0, mirage 0→0.6, rites 0→0, siftstack 0→0.5, sirens 0.4→0.6, stockade 0→0.113, toolsmith 0.1→0.7, warren 0.067→0.167.

- The quick aggregate ordering INVERTED at medium: replay_repeat (0.5081 quick, best-ever) ranks last of the treated at 0.2981, while the install carrier hygiene_explore leads — the non-convex tier-Pareto frontier reappears inside the universal line.
- Sirens resolved exactly as the forensics predicted: base 0.4, every treated arm 0.6 — a strict win at medium granularity where quick manufactured ties.
- Menders stayed at 0 for ALL four arms — for the contamination-free line it is a genuine capability gap, not purely an instrument artifact (though replay_refresh once scored 0.125 at quick, so it is a marginal-capability-plus-item-draw gap, not an absolute wall).

## Interpretation

The program has never been closer: hygiene_explore beats base strictly on eight families, loses none, and needs exactly two ties flipped — menders > 0 and rites > 0 — to pass the recorded goal gate. rites is demonstrably elicitable in this lineage (designed_fresh scored 0.1 in this same event; replay flipped it at quick). menders is the binding constraint: two designed same-shape attempts already failed (the closed tracefix axis), gym-trained arms historically reached 0.3–0.4 there, and the clean line's only nonzero was one quick item. Successor design must attack menders with a genuinely new mechanism argument, with rites carried alongside, from the hygiene_explore parent, under the pooled_k3 retention protocol.

## Knowledgebase Update

- Program evidence updated: the medium map, the ordering inversion, and the two-tie goal-gate position recorded.
- Program backlog updated: forensics successor closed; the two-family install (menders + rites) from the hygiene_explore parent is the funded next branch.
- Claim ledger updated: no new claim; the tier-frontier law gains a universal-line replication.

## Artifacts

- `data/design_receipt.json`: seed/tier/budget/model/gateway/forensics pins.
- `reports/preregistration.md`, `reports/benchmark_design_review.md`: contract and authorization.
