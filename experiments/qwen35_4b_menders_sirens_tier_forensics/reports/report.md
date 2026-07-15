# Menders/Sirens + Tier Forensics Report

## Summary

Analysis-only forensics over every committed gateway receipt (2,278 files, 356 cleaned family-score rows, zero GPU): the two "frozen constants" blocking the all-families goal gate — menders = 0 and sirens = 0.500 for every arm at quick/tb1024 — are instrument artifacts, not model walls. Three committed counterexamples exist at the line's own instrument, and the paired within-event strict-win analysis shows the goal gate (all ten families strictly above base) passed 9 of 94 historical medium-tier arm-events versus 1 of 84 at quick. At medium the base never sits at a family ceiling and sirens leaves its 0.500 sticking point (exactly 0.5 in 14/95 base events vs 49/82 at quick).

## Research Program Fit

The backlog's queued prerequisite before any new menders/sirens treatment; it converts "the wall is two families wide" into "the venue was one tier too coarse," and de-risks the goal-gate path's next benchmark event.

## Method

See the preregistration: frozen sweep + cleaning rules (one honest post-first-look refinement recorded), constant check with exact counterexamples, per-tier base family profiles, paired strict-win adjudication.

## Results

Quick: 84 paired arm-events, 1 goal-gate pass, 8 near-misses at 9/10 (blockers menders/sirens/warren); base at a ceiling in 2/82 events. Medium: 94 paired arm-events, 9 passes, 20 near-misses at 9/10 (blockers menders/rites/warren); base ceilings 0/95; base menders zero in 54/95 (max 0.3), sirens 0.2–0.6. Counterexamples at quick/tb1024: base sirens 0.375 (78,131), candidate menders 0.021 (78,131), replay_refresh menders 0.125 (78,133).

## Controls

Raw and cleaned tables both committed; every row carries its source receipt sha256; the harness re-derives the analysis (smoke) and the sweep (full) byte-identically.

## Oracle Versus Deployable Evidence

Receipt analysis only; `benchmarks/` never read; no new model evidence of either kind.

## Next Stage

Closed. Funded successor: the universal line's first medium-tier paired measurement — base + best published composites, one fresh sealed medium seed, tb1024, goal gate recorded. Caveat carried: the nine historical medium passers were gym-trained arms; instrument feasibility is established, line transfer is not.

## Artifact Manifest

Everything in-repo (two JSON artifacts under `runs/`); no external artifacts.
