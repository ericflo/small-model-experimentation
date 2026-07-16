# Goal-Gate Confirmation Report

## Summary

The three-seed replication closed AGGREGATE_ONLY. The aggregate transfer is unconditional — hygiene_explore strictly beat base on all three fresh sealed seeds (0.3287/0.3737/0.3837 vs 0.0586/0.1122/0.0982; with the discovery, 4/4 all-time, never close). The all-families sweep replicated once: seed 78,157 passed 10/10, making two full sweeps across four independent sealed seeds; the frozen 2/3 majority bar failed because 78,155 (9/10) and 78,156 (8/10) were blocked entirely by ties — menders at a 0.0 margin on both, warren once (while warren WON +0.267 on the other) — with zero strict losses anywhere. The verdict localizes the goal to a single family: any reliable nonzero menders yield completes the gate. Every reading is provenance-anchored (receipt shas pinned in closed ledger records; the readout refuses any break in the sealed chain).

## Research Program Fit

The program goal demands demonstration and confirmation; the confirmation law demands independent seeds. This cell is both, at the gateway's highest supported tier.

## Method

See the preregistration.

## Results

`runs/benchmark/confirmation_readout.json`: verdict AGGREGATE_ONLY; per-seed goal gates 9/10, 8/10, 10/10 (PASS); aggregate strict wins 3/3; fragility — menders margin 0.0 on both failing seeds, warren +0.267/tie/win; discovery seed reported (0.3663 vs 0.0800, 10/10) and never counted.

## Controls

Both arms published and tree-authenticated once per runner invocation, before any gateway call; per-seed write-ahead ledger whose closed records sha-pin the sealed summary and both per-arm receipts (the readout reads verdict inputs only through those pins); implementation-signature equality across all six runs and against the pinned discovery summary; fragility margins preregistered.

## Oracle Versus Deployable Evidence

Gateway aggregates and public family scores only; `benchmarks/` never read.

## Next Stage

Closed. The menders dose-scale intake (the one permitted mechanism class) is the funded successor with a precisely-known target; the zero-root lineage rebuild stays queued.

## Artifact Manifest

Two composite pins external with committed receipts; everything else in-repo.


## Erratum (2026-07-16)

The sweep-rate framing in this document ("two full sweeps across four
independent sealed seeds", ~50%) reflects the 78,154–78,157 window and
omits the earlier 78,150 reading (8/10, menders+rites ties). Over ALL six
recorded goal-gate readings the rate is 2/6 (exact 95% CI [0.04, 0.78]),
with menders blocking every miss. See
`experiments/qwen35_4b_sweep_rate_consolidation` for the consolidated
record; the per-seed facts in this document are unchanged.
