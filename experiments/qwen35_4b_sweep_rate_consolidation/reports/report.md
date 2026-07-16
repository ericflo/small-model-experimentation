# Qwen35 4b Sweep Rate Consolidation Report

## Summary

The program's informal "~50% sweep rate" was computed over a favorable four-seed window. The full committed record holds SIX goal-gate readings of the headline hygiene_explore composite against base at sealed medium/tb1024 seeds, and the honest all-events rate is **2/6 = 0.333** — exact Clopper-Pearson 95% CI [0.043272, 0.777222]; under a Beta(1,1) prior the posterior is Beta(3,5) with mean 0.375 and 95% credible interval [0.098988, 0.709579]. Every reading was re-derived from its sha256-pinned committed summary; the one recorded goal-gate block (78154) agrees with the recomputation. No verdict changes — AGGREGATE_ONLY already recorded the frozen confirmation bar as failed — but the quotable rate is corrected, and the carrier documents are listed for amendment.

## Research Program Fit

`agentic_breadth_installation`, lifecycle 24: terminal bookkeeping. The program's map closes with calibrated instruments and closed doors; this cell makes the headline replication number honest before the record is quoted onward.

## Method

Analysis-only (no model, zero GPU, zero seeds, `benchmarks/` never read). The six summaries are byte-copied into `data/source_summaries/` and hard-pinned by sha256; `collect_readings.py` fails closed on any drift (local copy or a present-but-different original), recomputes strict wins/ties/losses per family (FAMILIES byte-identical to the tier-forensics analyzer, enforced by test), and cross-checks recorded goal-gate blocks. `analyze_sweep_rate.py` computes the rate, exact CI (integer-shape beta quantiles via the binomial-tail identity and deterministic bisection — stdlib only, externally cross-checked against scipy to six decimals), the posterior, the blocker table, the base draw distribution, and the erratum; `--verify` and the harness require byte-identical re-derivation of both artifacts.

## Results

Six verified readings (all: aggregate win vs base; zero strict family losses):

- 78150 (universal_medium_tier_measurement, hygiene_explore): 8/10 — menders + rites ties. MISS.
- 78154 (statechain_only_dose, hygiene_explore_parent): 10/10. PASS (recorded goal-gate block agrees).
- 78155 (goal_gate_confirmation, hygiene_explore): 9/10 — menders tie; warren WON +0.267. MISS.
- 78156 (goal_gate_confirmation, hygiene_explore): 8/10 — menders + warren ties. MISS.
- 78157 (goal_gate_confirmation, hygiene_explore): 10/10. PASS.
- 78159 (zero_root_lineage_rebuild, hygiene_explore_original): 9/10 — menders tie. MISS.

Blockers: menders in ALL FOUR misses (0-margin draws — both arms at zero); rites once; warren once. Base draws (computed): zero on all six seeds for lockpick/menders/mirage/rites/siftstack; chronicle above zero on exactly the two passing seeds (78154, 78157); stockade and warren on four; sirens and toolsmith on all six. The intake's guess "base rites>0 on 2 seeds" was wrong — chronicle is that family.

Erratum: the "~50%" / "two of four sealed seeds" figure is a window artifact — 2/4 over 78154–78157, a window that starts at the first pass and omits the earlier 78150 miss; the zero-root docs extended it to a "fifth data point" (2/5 = 0.4) while keeping the "~50%" label. Corrected: 2/6. Carriers: `experiments/qwen35_4b_goal_gate_confirmation/README.md` + `reports/report.md`, `knowledge/synthesis.md` (confirmation, dose-scale, and zero-root entries), `knowledge/experiment_brief.json` (menders_dose_scale and repair_verifier_signal_probe briefs), `knowledge/experiment_viz.json` (zero-root note), `experiments/qwen35_4b_zero_root_lineage_rebuild/README.md` + `experiment_log.md` + `reports/preregistration.md`.

## Controls

Fail-closed provenance (pinned sha256s on both the local copies and the originals when present); recomputation from raw per-family scores rather than trusting derived blocks, with mandatory agreement where a block exists; FAMILIES byte-identity to the forensics analyzer; byte-identical re-derivation of both artifacts; negative tests for every drift path.

## Oracle Versus Deployable Evidence

Not applicable — no model event. All inputs are committed receipts from prior sealed events; this cell adds no new measurement, only honest arithmetic over the existing record.

## Interpretation

The correction is small in mechanism and large in honesty: the composite's aggregate transfer remains unconditional (6/6, margins 3–6×, zero strict losses in sixty comparisons), and the gate remains draw-gated at exactly menders — but "roughly half of sealed seeds sweep" overstates the record. The right sentence going forward: two full sweeps in six all-time readings (2/6, CI [0.04, 0.78]), every miss a menders 0-margin draw.

## Next Experiments

None funded from here — this is terminal bookkeeping. The carrier documents listed in the erratum should be amended (or their windows scoped explicitly) whenever they are next touched.

## Artifact Manifest

No external artifacts; everything is committed in-repo. `reports/artifact_manifest.yaml` records the smoke/full commands.
