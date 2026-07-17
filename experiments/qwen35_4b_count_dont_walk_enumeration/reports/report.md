# Qwen35 4b Count Dont Walk Enumeration Report

## Summary

Design frozen; no GPU stage has run. This cell is the evidence-backed
successor to `qwen35_4b_enumerative_repair_protocol` (lifecycle 26),
changing ONLY the expression pedagogy of the enumeration discipline.
The predecessor's post-closure forensics showed 20 of its 21
unparseable gate rows were 1,024-token cap truncations caught
mid-CORRECT canonical walk: the discipline installed (9/40 vs 0/40 on
both controls) but expresses as a verbose linear walk whose token cost
grows with the number of prior attempts. This cell teaches the same
discipline as a constant-cost computation — count the k tried entries,
target candidate k+1, locate it by index arithmetic over rendered
per-step candidate ranges — with every training think target verified
under a frozen 120-token cap (measured max 105, mean 95.8, constant in
k).

## Research Program Fit

Menders is the last benchmark family never moved by any documented
intervention. The predecessor produced the starkest install-vs-convert
contrast on record; its failure decomposed to expression cost, not
ordering logic. A constant-cost expresser is the direct test of whether
that leak was the conversion blocker.

## Method

Byte-equivalent clone of the reference cell (simulators, legality
bounding, canonical order rule, K_CYCLE, uniqueness invariants, exact
zero-delta MILP, single-kind 160-row full-concentration dose,
single-kind promotion bar, pooled_k3 retention bands, frozen
two-direction menders consequence rule with the 0.50 canonical-next
fidelity precondition, six-slot normalized-pin hardened benchmark
runner) except three designed deltas: (1) fixed-shape five-line compact
think targets (count → k+1 → range lookup → slot → answer), byte-checked
per row against a pure re-derivation and capped at 120 real-tokenizer
tokens; (2) the order statement additionally renders per-step candidate
counts/ranges, verified against the exhaustive enumeration; (3) a new
non-gating `expression_cost` gate reading (per-arm axis think-token
distribution + truncation count). Arms: `replay_ctl7` control and
`count_walk` candidate, both trained from the zero-root composite via
the standard recipe; standalone lineage package vendored (stage 7 =
count_walk). Seeds: construction 77191, namespace 55171, training 85,
gate 88056, screens 88057/88058/88059, sealed benchmark 78163.

## Results

Pending. Fail-closed TODO pins hold the six benchmark slots; the
normalized runner pin is frozen at `bc2b4129f782…` in pre-fill state.

## Controls

Pending (replay_ctl7 trains first; promotion requires strict totals
over BOTH the parent composite and the replay control, inside pooled_k3
retention bands across three fresh screens).

## Oracle Versus Deployable Evidence

The predecessor's truncation forensics are the oracle-side evidence:
the walk was correct where visible, so fidelity was budget-limited.
This cell's deployable question is whether a sub-120-token expression
converts installed discipline into menders episodes.

## Next Experiments

If canonical-next fidelity clears the frozen 0.50 precondition and
menders still does not move, the expression-cost hypothesis for the
conversion blocker is dead and the family closes to this whole pedagogy
axis. If fidelity stays below 0.50 despite constant-cost targets, the
leak is not expression cost and the forensics reading was wrong.

## Artifact Manifest

See `artifact_manifest.yaml` (corpus `21e6f5cb70…`, design receipt
`d835fc5c38…`; adapters and receipts land at their gated stages).
