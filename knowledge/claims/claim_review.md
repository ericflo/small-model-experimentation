# Adversarial Review of the Claim Ledger

Generated 2026-07-06 by a 155-agent adversarial workflow: one skeptic per claim
(instructed to *refute* it against the cited experiments' actual reports/data),
every serious finding independently re-verified (default-reject), plus
cross-cutting sweeps for contradictions, synthesis fidelity, and statistical
hygiene.

- **41** claims reviewed
- **102** verified per-claim findings (12 high, 83 medium, 7 low) + **17** cross-cutting
- Dominant issues: **overclaim (33)**, weak_evidence (27), stat_hygiene (23),
  unsupported (7), contradiction (5), number_error (4), wrong_status (3)

Spot-checked against source: C22's depth-3 deployable (no-think greedy@1) is
0.00→0.00 while the "5/40 (0.125)" headline is think-mode search — the
experiment's own design_review.md flags this. The review is corroborated by the
corpus's own honesty notes.

---

# Adversarial Review — Claim Ledger, Prioritized & Deduplicated

Every item below survived independent verification. Grouped by severity; duplicates across the per-claim and cross-cutting passes are merged.

---

## HIGH SEVERITY

### H1 — C1 "structured beats direct" is Confirmed on one un-replicated experiment
**Claim:** C1 (Confirmed). **Merges 3 findings.**
Only one of the three cited experiments (`foofah_selective_program_fallback`, 55.2% direct vs 62.4% program) actually contains a direct-answer baseline — single seed, greedy, fixed candidate pool, no CIs. `latent_compiler_expansion` has no direct arm (measures the compiler's own executor accuracy); `operator_inventory_search_pilot` is a no-model "search-side ceiling" comparing two structured arms. This is exactly what C1's own `avoid` field warns against.
**Fix:** Downgrade Confirmed → Promising. Drop the two non-baseline experiments as comparative support (reclassify the compiler one as latent-compiler feasibility). Do not restore Confirmed until the structured-vs-direct contrast is replicated across seeds and ≥2 substrates.

### H2 — C8 "process artifacts determine compounding" is Confirmed with n=0
**Claim:** C8 (Confirmed). **Merges 3 findings.**
A causal claim ("charters/ledgers/CI *determine whether future work compounds*") whose defining test has never been run. All four evidence items are `kind:doc` (the artifacts themselves — circular), and C8's own `next_tests` says the effect is still to be measured. The program scorecard treats the benefit as an open goal ("the next lift is faster research navigation").
**Fix:** Downgrade to Promising/Aspirational (process-note). Soften "determine whether future work compounds" → "are intended to help future work compound." Reframe as a design principle until an experiment measures duplicate-reduction/compounding.

### H3 — C5 is stale-Open; the comparison it awaits was already run (cross-cutting)
**Claim:** C5 (Open).
C5 still says adaptation's value is "unclear unless compared against frozen sampling," citing only 3 old experiments, with next_test "run one update method against frozen alternatives." That comparison has since been run repeatedly: C29 (SFT_2x triples greedy@1, DPO collapses vs a matched-compute control), C11/C18/C23/C24 (QLoRA-SFT banking extends the coverage ceiling that frozen sample+select cannot). C5 was never updated.
**Fix:** Promote to Promising (Confirmed for SFT-banking specifically). Rewrite summary to record that self-training-on-verified-self-solutions beats frozen inference-time alternatives for coverage-extension while preference/DPO collapses. Add C11/C18/C23/C24/C29 to evidence.

### H4 — C22 depth-3 "unlock" significance is overstated by ~an order of magnitude, plus a 2× number error
**Claim:** C22 (Promising). **Merges 2 per-claim + 1 cross-cutting finding.**
- "Highly significant vs the 0/40 floor (p<0.01)" is false: 5/40 vs 0/40 gives Fisher two-sided p≈0.055 (one-sided 0.027) — not significant at 0.05. `significant_unlock=true` was derived by comparing the banked point estimate to base's *rule-of-three* upper CI (0.075), ignoring the banked arm's own uncertainty; Wilson [0.055,0.261] overlaps base's Wilson upper (0.088).
- **Number error:** "deployable greedy@1 0.15" is the *think*-mode number. Deployable = no-think = 3/40 = **0.075**. The load-bearing contrast is 0.075 vs 0.00, not 0.15 vs 0.00.
**Fix:** Replace "p<0.01" with Fisher p≈0.055 two-sided / 0.027 one-sided, flag CI overlap, retitle away from "crosses the depth-3 wall" to "weak/marginal, test-time-only (deployable greedy@1 = 0)." Correct 0.15 → 0.075 in summary/implication/report.

### H5 — C18 headline "3× expansion" comes from a harness the pre-registration excluded
**Claim:** C18 (Promising).
The depth-2 "EXPANSION 3×" (0.15→0.45) — the novel half of the claim vs C17 — is on the *think* harness, but the prereg locks scoring to **no-think**. On the pre-registered no-think harness the expansion is only 0.05→0.15 (+0.10, Fisher p=0.605, null). (Depth-1 "concentration" 0.60→0.80 is also n.s., p=0.30; depth-2 think itself is p=0.082.)
**Fix:** Lead with the pre-registered no-think result (0.05→0.15, non-significant); label the 3× as a post-hoc think-harness figure. Note the directional effect is independently corroborated by C21, but C18's own headline numbers are a single non-significant run.

### H6 — C9 "coherence advantage GROWS with budget" rests on one anomalous point
**Claim:** C9 (Promising).
+0.105/+0.108/+0.150 at 512/1024/2048: 512→1024 is flat (+0.003); the only movement is the 2048 point, which the report documents as an anomalous slow-recovery arm after a mid-run CUDA fault. No CIs, n=100, single seed. Two independent 512-runs disagree by 0.017 — larger than the 512→1024 gradient.
**Fix:** Downgrade "grows with budget" → "roughly constant across budgets (noisy; elevated 2048 point is an anomalous recovery arm, within run-to-run variation)." Do not present the +0.045 spread as a trend without seeds/CIs.

### H7 — C10 matched-cost ranking is sub-noise; title credits the wrong lever (cross-cutting merge)
**Claim:** C10 (Promising). **Merges per-claim stat_hygiene + cross-cutting weak_evidence + cross-cutting title overclaim.**
The 0.850/0.860/0.870 selector ordering (visible-only / thinking-verifier / visible+no-think) is 1–2 tasks at n=100, <1 SE, no CIs — the report itself says "within per-condition noise." Yet the ledger asserts "Pareto-DOMINATED," a "deployable sweet spot 0.870," an 83%-vs-66% gap-closure, and the title credits an expensive thinking-verifier while the body reassigns the win to cheap plumbing.
**Fix:** State the ordering is within eval noise at n=100; drop "Pareto-dominated"/"sweet spot"/precise gap-closure figures; retitle to reflect the matched-cost conclusion (cheap visible test + free no-think verifier captures most of the gap; thinking-verification pays only in verifier-only settings). Require CIs or a second seed before asserting any ordering.

### H8 — C9/C10 capability & verification numbers are on likely-contaminated MBPP (cross-cutting)
**Claims:** C9, C10 (Promising).
+15pp MBPP greedy pass@1 (C9) and verifier balanced-acc 0.827 / AUROC 0.926 (C10) are single-seed n=100 on MBPP, which the experiments flag as "likely partly contaminated." For a self-verification claim this is load-bearing (recognizing memorized solutions inflates the verifier), so "the C2 wall is plumbing not capability" is not contamination-controlled. C10 additionally states this structural conclusion as settled fact rather than scoping to MBPP.
**Fix:** Move the contamination caveat into the C9/C10 *summaries*; scope both to "on likely-contaminated MBPP." Gate any "capability (vs plumbing)" framing on the owed contamination-controlled + think-generated-pool re-run.

### H9 — C15 "CONTEXT COMPOSES DISCRIMINATION" is not significant
**Claim:** C15 (Promising).
The 0.74→0.83 base lift is McNemar p=0.108 (paired n=120). On the metric that actually measures discrimination (parse-conditional on the 113 tasks parsed in both conditions) it is 0.788→0.823, p=0.60 — essentially null. Most of the raw gain is parse-rate compliance (0.94→1.00), not better discrimination.
**Fix:** Soften Finding #1 to "suggestive, not significant (paired p≈0.11; discrimination-quality gain null)"; state the raw gain is largely parse-rate compliance.

### H10 — C20 "INERT" has no working positive control for upward steering
**Claim:** C20 (Promising).
The pre-registered positive/sanity control (depth-1 steer_true up ≥+0.15) FAILED (+0.03). No arm raises any op's naming above baseline; the only demonstrated efficacy is *suppressive*. So "adding the direction back does NOT change behavior" is confounded with "this ActAdd rig cannot elevate any naming at all." (Related medium: the steered mean-difference vector is not the logistic direction that achieved C19's 0.99 decodability; and the P4 identification arm ran off-prereg no-think, flooring its baseline.)
**Fix:** Soften "test-time readout cannot elicit latent capability" → "ActAdd mean-difference addition failed to elevate the true op, and the intended positive control failed, so the null is method-limited." Scope the title to the ActAdd direction; downgrade the P4 refutation to "not testable as pre-registered (floored)."

### H11 — C31 "surface deploys better / you don't need the 4B" is a 2-task gap
**Claim:** C31 (Promising).
"surface_full 0.027 > probe_full 0.014" is 4 vs 2 raw greedy successes out of 148 param tasks (cov@6: 8 vs 5), driven by depth-2 (3 vs 1); depth-3 is 1 vs 1. Paired McNemar p=0.625, Fisher p≈0.68 — both arms at floor. (Related medium number error: the "26% concrete accuracy" bound is the decode-eval depth-2 figure; the deployment readout is ~15% (22/148), and only 15% reproduces the observed 0.014.)
**Fix:** Downgrade to "both near-zero on param tasks (2 and 4 of 148); difference within noise." Remove settled-fact framing from summary and `avoid`. Correct "26%" → ~15% in the deployability context.

### H12 — Depth-3 inflation (C12/C13 retro-audit) never propagated to C22/C23/C24 (cross-cutting)
**Claims:** C22, C23, C24 (Promising).
C12/C13's behavioral min-depth audit found ~40% of nominal depth-3 tasks are behaviorally depth≤2 and true monolithic depth-3 solves were 0 corpus-wide ("every prior depth-3 figure was inflated by this artifact"). C22/C23/C24 still headline nominal depth-3 coverage as clean generalization to "novel depth-3 rules" with no min-depth caveat; their dedup was by op-signature, which does not exclude behaviorally-collapsed tasks.
**Fix:** Add an explicit min-depth caveat to C22/C23/C24 (their depth-3 populations are ~30–40% behaviorally collapsed; true-depth-3 numbers are substantially lower) and cross-reference C12/C13.

### H13 — C2 "selection wall" reframed to coverage by later claims but still Confirmed (cross-cutting)
**Claim:** C2 (Confirmed).
C2's mechanism ("visible evidence still selects/commits incorrectly") is reattributed by C10 ("plumbing not capability") and C17 ("SELECTION IS FREE: max(coverage−vfilter)=0.00 across all 8 cells; the wall is COVERAGE"). Synthesis line 15 agrees. Only the narrow overfit-trap false-deploy residue survives as a selection effect.
**Fix:** Scope C2's "selection commits incorrectly" framing to the residual overfit-trap/low-example regime; record that C10/C17 reattribute the primary gap to coverage/proposal. Keep Confirmed only for the narrowed "coverage ≠ deployable accuracy" statement.
**Also (stat-hygiene, medium):** C2's three "corroborations" are really 2 substrates — `verify_scale` and `independent_retrieval_consensus` share the same 24-task MBPP residual pool and 364-algorithm library (near-identical covered ids). Add a caveat that corroboration spans MBPP-24-residual + Foofah with no within-substrate multi-seed replication.

### H14 — Synthesis Executive Read still asserts the retracted P12 "thinking hurts" result (cross-cutting, self-contradiction)
**Synthesis item 10 + item 12 + C13.**
Item 10 presents P12 as settled ("thinking-mode 2AFC 0.50 — below no-think 0.73 — deliberate simulation is systematically wrong … thinking hurts when the content is simulation"). C13 explicitly *retracts* this ("P12 RETRO-CORRECTED … at budget 1024, base thinking-2AFC 0.74–0.79 ≈ no-think"), C14 found thinking *helps* single-pipeline simulation, and synthesis item 12 two lines later calls P12 "a budget+parser artifact" — the document simultaneously asserts and retracts the same result. C13's own summary also still carries both the "thinking HURTS" clause and its retraction.
**Fix:** Rewrite item 10's P12 sentence to the retro-corrected value inline and drop "thinking hurts/systematically wrong"; anchor the broken-simulation mechanism to C14 Phase 0. Fold item 12's correction into item 10. Delete the "thinking HURTS" clause and the C9 "thinking hurts on simulation" refinement from C13.
**Also (unsupported, high, C13 evidence list):** C13's evidence array cites *only* `depth_wall_anatomy`, whose own P12 probe the claim disavows; the load-bearing support (C14 Phase 0; C15 retro-correction) lives in `simulation_keystone_repair` and `context_composition`, neither listed. Add both to C13's evidence array.

---

## MEDIUM SEVERITY

### C3 (Negative) — scope overreach + weak positive half
- Summary lists four validated levers ("candidates, tests, constraints, verifier inputs") but only *candidate* retrieval was tested; tests/constraints/verifier appear only as future work. Narrow to candidate solutions; move the rest to hypothesis/next_tests.
- The positive half rests on a single-seed n=24 "pilot gate"; semantic-specific lift is only 2 tasks (15, 25) after controls also recover task 20. Add that caveat; don't present as settled.

### C4 (Promising) — prescriptive "must" outruns the evidence
- The "must optimize the final decision" core is carried by `oracle_elimination`, an output-aware **oracle upper bound** (a cheat); the one deployable learned decision-coupled controller *loses* to the informativeness heuristic on the primary split (104 vs 109/120). Downgrade "must" → "appears to help"; add the oracle caveat.
- `active_example_acquisition` is null support: active1_plain is byte-identical to fixed-order order1_plain, lift is 1/30 tasks, no CIs/seeds. Demote or annotate as null.

### C5 (Open) — the DAgger evidence is non-discriminative
`live_tool_dagger` test split has oracle ceiling 2/12; every non-degenerate policy (frozen rules, LoRA, oracle) ties at 2/12. Don't treat the LoRA-vs-rule tie as informative. (See H3 for the status fix.)

### C6 (Confirmed) — narrow, partly single-seed base
Two of three cited experiments (`factor_recombination_ladder`, `feature_factorized_rule_diversity`) are design-replicas (same 240-record budget, same controls, overlapping factors), both near-null on recombination; the 23.3% "signal" is 12/14 from one family (`sorted_join_holdout`) scoring 100% regardless of the trace variable. The third (attribution ablation) is single seed (s750). Downgrade to Promising, or add a genuinely independent + multi-seed substrate; note the ladders are replicas.

### C7 (Confirmed) — universal principle on one small-n substrate
The oracle/deployable gap comes solely from `reliability_exec_opsd_audit` (single seed, 24 MBPP tasks); the other cited experiment is a near-zero-gap pilot. Scope Confirmed to the CI-enforced governance practice (which is genuinely enforced), or downgrade the empirical "reliability requires" framing.

### C9 (Promising) — remaining hygiene
- Uncontrolled coherence confound: "real" uses the original answer; filler/shuffle/foreign regenerate from a spliced prefix, and there is no real-regenerated control. Soften "the ENTIRE gain / 100% coherent content" → "consistent with coherent content being the dominant driver, pending a regeneration-matched control."
- (low) "at EVERY budget" generalizes from 3 caps that all emit only ~530–630 actual thinking tokens; scope to "tested budget caps on MBPP."

### C10 (Promising) — C9-inversion metric mismatch
"thinking helps verification at least as much as generation" compares +0.20 balanced-accuracy against C9's +15pp *greedy* pass@1 — incommensurable scales, and the matched sampled-generation baseline is ~flat, not +15pp. Downgrade to qualitative ("thinking also substantially improves verification"); drop the "at least as much" ranking.

### C11 (Promising) — two secondary overstatements
- "depth ≥3, all arms 0.0" is wrong: only depth 3 is all-zero; at depth 4 the feedback arm is the single best (0.10). Change to "depth 3."
- Contamination attribution for the earlier MBPP self-improvement failure is unconfirmed (new substrate differs in task family, quantity, difficulty, metric; the package never mentions contamination). Soften to "consistent with a substrate/contamination hypothesis (untested; many confounds differ)."

### C12 (Promising) — banking framing
- "Frontier extension into the weights" is depth-2-dominated (6 of 9 gained tasks are depth-2, inside the existing sampling frontier); the true-frontier depth-3 gain is 1→4/40 (~1.4 SE) presented as "4×." State the significant gain is non-frontier consolidation; drop "4×" for a 1→4-task change.
- "REPLICATED" varies only the harvest seed on the *same* eval set (shared noise) and the frontier metric differs ~1.75× (d3 pass@5 0.10 vs 0.175). Soften to "harvest-seed robustness check (same eval set)."
- Program-novelty leak: 9/40 eval depth-2 programs are exact op-sequence matches to training programs (22.5%); depth-3 is disjoint. Report train/eval target_ops disjointness; restrict the frontier claim to depth-3 or add a depth-2 memorization caveat.

### C13 (Promising) — remaining items
- (low) "ZERO execution deficit" mislabels plan-given=1.00, which the report itself calls transcription (interpreter executes). Reword to "zero program→code transcription deficit."
- The "~2× constant, two measurements, quantitatively retro-explains C10/C11/C12/M2" is a loose coincidence across different substrates/metrics (2.1 vs ~2.5), no test. Downgrade "quantitatively retro-explains" → "is consistent with"; drop the "one constant" identity.

### C14 (Promising) — pre-reg fidelity + scope
- (low) "VERDICT (locked decision rules): separable branch" omits that the primary manipulation check P-K1 (+30pp at d3) *missed* (thinking-arm +22pp); the separable branch rests on the d4 metric. Add a caveat that the locked d3 threshold was missed.
- Headline "capability is organized by FORMAT MAPPINGS, not shared primitives" is stated model-wide from one task family, QLoRA r32, single run per arm. Scope the summary to the tested family/setting pending the full-FT and cross-family tests.

### C15 (Promising) — pre-registration reversal + selection bias
- The "genuinely composes / sealed modules softens" conclusion reverses the prereg: raw A5−A2 = −0.32 (which the prereg maps to SEALED MODULES); it uses a non-preregistered parse-conditional metric, and even there +0.12 is below the pre-registered +0.15 bar. Soften: raw refutes composition; parse-conditional only hints repairability.
- The +12pp compares SIM's parse-conditional accuracy on its self-selected ~53% format-surviving subset (n=64) against base on the full set — an upper bound, not a clean delta. Move the conditioning caveat from `avoid` into the summary.
- P12 retro-correction is stated as established cause ("budget-512 + weak parser") but prompt+budget+parser changed jointly, not factorially; and the 0.73–0.78 no-think anchor is imported from experiments not in C15's evidence list. Downgrade to correlational; fix the citation.

### C16 (Promising) — "LAW" language on n=3 / single seed
- The "floor ~ f(hypothesis-space, simulability), both factors necessary" sub-law is n=3 confounded families; floor tracks simulability alone (register vs string, near-equal op-space, differ only in sim). Downgrade "NEW SUB-LAW" → untested hypothesis; note the op-menu-size test is unrun.
- (low) LAW #2 "holds in ALL families" contradicts the prereg predicate P-L3, which returned `false` for register (bare 0.16 > 0.15). Soften to "holds in string and list; partially in register."
- "model-level LAWS / universal reliable compiler" rests on single seed, n=25/cell, 3 families measured once. Soften "law/universal" → "candidate cross-substrate regularity (single seed)."
- (weak_evidence) The string "unsimulable even at depth 1" finding depends on a hand-tuned regex parser that previously misfired on this exact family, with no second-parser/manual audit. Add a caveat / independent grader.

### C17 (Promising) — over-general "every cell" + recall≠precision + unmeasured mechanism
- (low) "rndVP identical to coverage in every cell / verifier adds nothing" is false at register-d1 (0.871 vs 0.90); the pre-registered P4 check was hardcoded to the list family only. Soften to "7 of 8 cells"; re-run P4 on register.
- "selection is free" is a *recall* metric blind to precision: at deep register a majority of deployable tasks are guaranteed false-deploys (d4 3/5). Add that "free" means recall not precision.
- The "sampling prob ∝ 1/hypothesis-space" mechanism is never measured (identification runs use no op-menu). Downgrade to explicit hypothesis or measure op-frequencies.

### C19 (Promising) — banking-mechanism + probe hygiene
- "Banking INSTALLS the missing representation" is presented as C18's explanation but only the *base* model was probed; the banked-model probe is unrun. Downgrade to hypothesis.
- The depth-3 "crossover" compares a floor-subtracted probe signal (0.13) against raw behavioral naming (0.127); floor-adjusted symmetrically, representation still ~2–4× exceeds expression. Weaken "crossover" → "representation modestly exceeds expression even at depth 3."
- (low) Max-over-33-layers probe vs a single-permutation null at the already-selected best layer, single split/seed, no resampled CIs. Compute a max-selected null over multiple permutations; report bootstrap CIs.

### C20 (Promising) — see H10 for the high items; medium residue folded there.

### C21 (Promising) — untested positive control stated as recipe
- Implication states "the ONLY way up is tool-seeded harvest → verify → bank" as established, but only self-banking was tested; the tool-seeded round is the unrun positive control. Soften to a prediction.
- (contradiction) "P3 (no two-rung leap) HELD" contradicts the machine verdict (`stays_near_zero=false`, banked1 d4=0.04 > prereg 0.03). Report P3 as "held in Δ but the pre-registered absolute ≤0.03 not met (base was already 0.04)."
- Title generalizes "the wall is not climbable by pure self-training" from one substrate/transition/seed. Scope to "in the list substrate, banking depth-2 did not unlock depth-3."

### C22–C24 (Promising) — dose-arc hygiene (cross-cutting stat)
The entire depth/dose arc (C22–C25, C33–C35) is a **single QLoRA training seed** with eval-noise-only CIs; only C11/M3 replicated seeds (depth-1/2). Add a standing "single training seed; CIs eval-noise only" caveat; soften "decisively"/"no saturation"/"dose-dependent" to "in a single-seed dose run"; require ≥3-seed replication before promoting any dose/depth claim above Promising. Specifics:
- **C22:** installer-efficacy-"DECAYS with depth" rests on one rung + an unseeded depth-4 (0.00); the depth-2 anchor is confounded by a larger combined training set. Scope the "decays with depth" language.
- **C23:** "deployable install scales too / P3 held" — no-think deployable measured at only 2 doses (0 and 640); "scales" is unjustified for the deployable metric (only think has the 4-point curve). Also "DATA-LIMITED decisively" is confounded (epochs fixed → more data = more gradient steps). Reword to "reaches deployable at top dose (0→0.10)" and "not a hard cap through N=640; data-vs-compute unresolved."
- **C24:** Arm-3 "recipe repeats one rung deeper" — the experiment's own verdict is `false`, deployable greedy@1 is identical (0.033 vs 0.033), the ~3× is cov@16-only with overlapping CIs. Soften implication to "may repeat, weakly, test-time-only"; state deployable gain is zero.

### C25 (Promising) — search-guide overclaims
- "base-guided search is WORSE than random" is 1 vs 2 solved of 80 (Fisher p=1.0, CIs overlap). Soften to "no better than random (within noise)."
- "dose-dependent at EVERY step including lookahead" fails at step 1 (640→1280 is 10 vs 11/80). Qualify: dose-dependent at steps 2–3 only.
- (low) "NO lookahead / at-or-below chance step-1/2" — step-2 is above chance (2× on top-1). Restrict "at/below chance" to step 1.
- The "~17×" guide gain has a 1-task denominator; lead with absolute counts (18/80 vs 2/80 random vs 1/80 base).

### C26 (Promising) — recognition/planning mislabel
Step-2 in the harness (goal shown, only start-state advanced) requires genuine 2-op lookahead, and thinking lifts it 0.000→0.325; the claim buckets this as "recognition." Reframe the axis as "planning-from-materialized-start-state (lifted) vs planning-the-first-move-from-raw-input (flat)," not "recognition vs planning."

### C27 (Open) — additive-stacking is budget-selected + imprecise
- "interaction ~0.00" holds only at the 2048 budget; at 1024 the same cells give −0.075 (sub-additive) — the magnitude the claim calls "no stacking" elsewhere. Report both budgets; note the additive headline is budget-dependent.
- "almost EXACTLY additive" overstates precision at n=40 (interaction SE ~0.15). Apply the same n=40 wide-CI caveat the claim already applies to the step-1 null.

### C28 (Promising) — Phase-1 "plan beats answer" not significant
"BANKING THE PLAN BEATS BANKING THE ANSWER (0.325 vs 0.200)" is McNemar p=0.064 and does not replicate on greedy (verdict flags `T_beats_A_greedy=false`). Soften Phase-1 to a trend; anchor the plan-quality conclusion on Phase 2, where T_synth vs A_self *is* significant (p=0.007).

### C29 (Promising) — "more SFT" gains not significant + in-sample verifier
- "SFT_2x triples greedy@1 / doubles coverage" is 3/80 vs 9/80 (Fisher p≈0.13) and 9/80 vs 17/80 (p≈0.13) — CIs overlap. Soften "triples/doubles" to a directional trend with Wilson CIs, or replicate.
- "same-task signal made collapse worse" (DPO 0.013 vs shuffled 0.037) is 1/80 vs 3/80, pure noise. Drop it or anchor only on the one significant signal (coverage 9/80→1/80, p≈0.018).
- "strong latent verifier (2AFC 0.81)" is measured entirely in-sample (chosen strings = SFT training positives). Mark as in-sample; test on held-out samples before calling it a verifier ability.

### C30 (Promising) — oracle framed as deployable lever
- "the FIRST test-time lever that moves deployable capability" rests entirely on the oracle-full arm (given the TRUE op+param — not deployable); the actually-deployable probe nets to ~zero. Describe oracle-full as a ceiling establishing the correct *channel*; state no deployable lever was demonstrated.
- "genuine self-elicitation" is 5 vs 3 tasks on a post-hoc probe-correct subset. Downgrade to "suggestive, underpowered, post-hoc."
- "controls clean (wrong<no-hint content-causal)" is 0/100 vs 5/100, Fisher p≈0.06. Soften to "directionally consistent but not significant."

### C32 (Promising) — "no value tax" underpowered + metric artifacts
- The depth-3 no-value-tax rests on 2 struct-correct events of 120 — near-zero power. Soften "decisively/REFUTED" → "suggestive, underpowered (n=2)."
- "DSL is NOT value-fungible" cites only depth-3 random (0.108) and omits depth-2 random = 0.600; the metric is a search-budget-over-skeleton-space control (16^depth). Report both depths; relabel.
- oracle-skeletonfill=1.000 is a construction guarantee (params drawn from the enumerated support). Move the "by construction" caveat into the summary.
- structure-coverage misclassifies out-of-support wrong-value programs as *structure* failures, biasing toward "structure." Add that caveat.
- (low) "cannot propose deep structure" is measured only at k=8; soften to "at k=8 almost never proposes the depth-3 op-sequence."

### C33–C36 — brute-force / cross-substrate framing
- **C33:** "Deployable recipe bank+value-fill ~0.51" omits the same experiment's model-free brute-force control (0.975) that >2× dominates it; the report says "the model is unnecessary." Add the brute-force control; banking is a forward-pass asset, not deploy-with-interpreter. Also the "~0.512" is structure-*coverage*; the measured end-to-end deploy is **0.463** — correct the number and update the stale "not run end-to-end" next_test.
- **C34:** "brute-fill is the arc's cleanest beat-sample-more deployable result / the model is unnecessary" is a **model-free** enumerator over the known grammar — a tool/interpreter result, not weight-elicitation (violates the fixed-4B north-star framing). And "using the model's structure is WORSE than ignoring it" is an abstain-as-failure artifact: bank candidates are a strict subset of brute, so 42/80 abstentions (only 1 actual wrong answer) drag it to 0.463; bank+brute-fallback = brute. Reframe as a tool result; report the abstentions; say "subsumed/redundant," not "worse."
- **C35:** "banking's structure collapses with depth (0.51→0.10)" crosses **two different, non-dose-matched banked models** (banked_1280 at 1280 examples vs banked_d4 at 320); dose is uncontrolled and the claim's own next_test flags it. Soften to "the single, under-dosed depth-4 model shows far lower coverage." Also "NEVER beats brute-force" and the depth-5 "model never wins" are generalized from a two-point trend + an untested depth-5. Soften "NEVER" to "up to depth-4 on this substrate"; mark depth-5 speculative.
- **C36 (Confirmed):** two of four headline numbers (oracle-skeletonfill=1.000, brute_cov=1.000) are generator tautologies, not cross-substrate evidence. And "the ENTIRE compositional arc is model-level" contradicts the claim's own scope note (banking/depth-collapse/probing were list-only). Downgrade Confirmed → Promising (or narrow to "base-model wall-is-structure + brute-dominates, 3 substrates, single seed each"); drop the tautologies and the "ENTIRE arc" extrapolation; drop "law."

### C37 (Promising) — depth-4 intactness rests on the prior-carrying arm
- (contradiction) "Both collapse only at depth 5–6" is false: the contamination-clean symbolic control walls d3→d4 (1.00→0.55). Correct it.
- "mental simulation INTACT at depth 4–5" holds only in the linguistic-*semantic* arm (which carries a pretraining ordering prior); the clean control is 0.55/0.01. Confine the clean-modality claim to depth-3.
- "the wall is formal-modality-specific" lacks a clean matched formal baseline on the *same* successor-chase task (the only formal arm is code-mode-confounded and discarded; the formal wall is imported from the harder C13/C14 tasks). Soften; owe a matched formal-traversal baseline.

### C38 (Promising) — think-arm comparison + scope
- "induction think 0.50 still below application 0.75" is n=24, z=1.85, p=0.064, overlapping CIs. Add the n=24/not-significant caveat.
- The clean dissociation is single-seed, single-depth (d1 only; application degrades at d2+). State that explicitly; owe a second seed.
- Implication elevates one language substrate to a "cross-modality LAW / modality-general." Downgrade to "consistent with a cross-modality pattern; single language substrate so far."

### C39 (Promising) — "not data-limited" confounded + scope
- The "more examples make novel induction worse" argument omits that the *familiar* arm also degrades (0.45→0.25) — a general prompt-length/format effect that doesn't isolate familiarity. Add the familiar-arm drop; soften "NOT data-limited."
- Implication generalizes one cyclic-digit substrate, single seed, to all of ICL ("ICL = the retrieval half of reasoning / unifies C13–C38 / regardless of #examples"). Scope to the tested substrate pending the non-cyclic replication next_tests already call for.

### C40 (Promising) — item-level metacognition inverted where it matters
- "genuine item-level self-knowledge / abstains on low P(answer)" hides that within the novel_induce cell P(answer) AUROC is 0.359 (below chance) and *inverted* (correct items lower-confidence). Scope "excellent" to familiar/reversal cells; note novel routing works only at the condition level.
- "selective prediction lifts 0.23→~1.0" quotes the ~5%-coverage endpoint (discards ~95%). Report coverage with the number (~1.0 at ~5%, ~0.87 at 20%).
- (cross-cutting) C40's chance self-verification vs C10's strong thinking-verifier are opposite verdicts on self-verification via a logit read. Add a cross-reference stating the boundary (executable code + thinking vs cyclic-order induction) and flag the open task/format-vs-self question in both claims rather than a one-line nuance.

### C41 (Promising) — pooled aggregate masks a regime inversion; no CIs
- "confidence-select (0.62) beats self-consistency (0.48)" inverts on the capability-limited third (novel_induce: 0.025 vs 0.062 vs greedy 0.075) — confidence-select is the *worst* method there. The entire aggregate win is from familiar_induce; a saturated "execute" regime (all methods 1.0) inflates every absolute number by +0.33 and sets the reported gap. Report per-condition numbers with caveats.
- The stated mechanism ("its high-confidence samples are right") is contradicted on hard problems (highest-P sample correct 2.5%). Scope the mechanism to coverage-limited problems.
- No CIs / paired test, single seed, single toy substrate, self-vetted design. Add a McNemar/bootstrap test; soften "beats majority at every budget"; gate "beats sample-more" on the owed real-code replication.

---

## LOW SEVERITY (synthesis status-tag hygiene)

- **Synthesis item 9:** bolded "the frontier IS extendable without a teacher" states a ~1.5 SE, single-seed, 17%-of-true-depth-3 effect as fact (C12's own `avoid` warns against exactly this). Soften the lead to match Promising.
- **Synthesis item 4:** bucket tagged `Promising`, but the anchoring memory claim C3 is Negative; reserve Promising for the coupled variant C4.
- **Synthesis item 1:** `Confirmed` applied to an editorial framing ("prototype for a broader research operating system") with no backing claim. Drop the tag or relabel as a thesis/assumption.

---

## OVERALL READ

The ledger is **methodologically self-aware but status-inflated and slow to propagate its own corrections** — sound in its raw data, sloppy in its headlines and bookkeeping.

The good: nearly every experiment carries pre-registrations, `avoid` lists, honesty notes, and machine verdicts, and several of the sharpest criticisms here (P12 retro-correction, the min-depth audit, the contamination flags, the oracle-vs-deployable distinctions) were *first surfaced by the corpus itself*. The underlying numbers reproduce. This is not a corpus that fabricates results.

The bad, and it is systematic:
1. **"Confirmed" is over-issued.** C1, C6, C7, C8, C36 all wear Confirmed on single-seed, single-substrate, tautological, or literally-unrun evidence. C8 is the worst — a Confirmed causal claim with n=0.
2. **Significance language is routinely wrong.** Repeated "p<0.01 / highly significant / decisively" on differences that are 1–5 tasks at n≤100, single seed, no CIs (C10, C15, C18, C22, C25, C28, C29, C41). The C22 "p<0.01" is off by an order of magnitude.
3. **Deployable-vs-oracle-vs-think metrics get swapped to the flattering one** (C22 0.15-vs-0.075, C30/C34 oracle-as-lever, C24 cov@16-vs-greedy, C31 surface-deploys-better).
4. **Corrections don't propagate.** C5 is stale-Open after being answered five times over; the depth-3 inflation audit never reached C22/C24; C2 stays Confirmed after C10/C17 reattribute its mechanism; the synthesis simultaneously asserts and retracts the P12 result on adjacent lines.

Net: treat the **findings** as largely real and the **status tags, headline numbers, and superlatives as unreliable until re-graded**. The fixes are almost all downgrades, scope-narrowing, and number corrections — not retractions. A disciplined pass that (a) demotes the five over-Confirmed claims, (b) replaces every "significant/decisively" with the actual test or an eval-noise caveat, (c) standardizes on the deployable metric in headlines, and (d) reconciles the stale cross-references would turn a sloppy-looking ledger into a genuinely sound one, because the evidence underneath mostly supports the softened versions.