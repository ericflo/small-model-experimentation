# Menders/Sirens + Tier Forensics

Adjudicate the two "frozen constants" blocking the all-families goal gate (menders = 0, sirens = 0.500 at quick/tb1024) from committed gateway receipts only: zero GPU, zero seeds, `benchmarks/` never read.

**Status:** finished · 2026-07-15 · verdict CONSTANTS_ARE_INSTRUMENT_ARTIFACTS — three committed quick/tb1024 counterexamples exist; the goal gate passed 9/94 historical medium arm-events versus 1/84 at quick; base never ceilings at medium; the funded successor is the universal line's first medium-tier paired measurement

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the goal-gap pilot (7 families up, 0 down, gate failed on menders/sirens ties); the backlog's queued menders/sirens forensics prerequisite; the earlier gym line's medium-tier receipt corpus.

## Question

Are menders = 0 and sirens = 0.500 structural walls of the 4B, or artifacts of the quick instrument's granularity and item draws — and at which tier is the all-ten-families goal gate actually passable?

## Method

Sweep every committed gateway receipt (`experiments/*/runs/**/*.json`, 2,278 files), extract every ten-family score block with tier/seed/budget provenance (380 raw rows, 356 cleaned), then: constant check with exact counterexamples; per-tier base family profiles (floor/ceiling frequencies); paired within-event strict-win adjudication of the goal gate. Cleaning rules and their one honest post-first-look refinement are recorded in the preregistration; the harness re-derives the analysis byte-identically from the committed table.

## Results

- **The constants were never universal.** At the line's own quick/tb1024 instrument: base sirens 0.375 (seed 78,131), candidate menders 0.021 (78,131), replay_refresh menders 0.125 (78,133) — all in committed receipts. The goal-gap event (78,144) drew base at exactly menders 0 / sirens 0.500.
- **The goal gate is ~8x more passable at medium.** Strict-win-all-ten passed 9/94 medium arm-events vs 1/84 quick. Strict-win distribution (medium): mode 8/10 (46 events), 20 at 9/10, 9 at 10/10.
- **Base never ceilings at medium** (0/95 events have any family at 1.0; quick had 2/82), and sirens un-sticks: base exactly-0.5 in 14/95 medium events vs 49/82 quick; base sirens spans 0.2–0.6 at medium. Menders at medium: base zero in 54/95, max 0.3; treated arms reached 0.4.
- **Near-miss blockers:** menders/sirens/warren at quick; menders/rites/warren at medium.

## Interpretation

The two constants are quick-instrument artifacts — coarse 1/8-step granularity plus item draws — not model walls. The all-families goal gate's realistic venue is the medium tier, where every family has strict-win headroom against base in every historical event. Honest limit: all nine medium passers came from the old gym line, whose arms trained ON menagerie-family data; the contamination-free universal arms (best aggregate 0.5081 at quick) have never been measured at medium. The funded successor is that measurement: base + the line's best published composites, one fresh sealed medium seed, tb1024, paired same-backend, goal gate recorded.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_menders_sirens_tier_forensics/scripts/run.py --smoke   # analysis reproduces byte-identically
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_menders_sirens_tier_forensics/scripts/run.py --full    # sweep + analysis both reproduce
```

## Artifacts

- `runs/receipt_table.json`: the raw 380-row sweep with provenance sha256 per receipt.
- `runs/constants_analysis.json`: cleaned analysis — constant check, base profiles, paired goal-gate adjudication.
- `reports/preregistration.md`: frozen questions, cleaning rules, and the honest ordering note.
