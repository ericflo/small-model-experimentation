# Does the installable hypothesize-and-verify skill move the structure wall? Report

## Summary

The wall holds — and the way it holds is the finding. Aimed the C45 hypothesize-and-verify serial
strategy at the C32/C36 structure-proposal wall via three routes (training-free prompt scaffold,
zero-shot C45-adapter transfer, DSL-native reasoning-SFT). **No arm clears the pre-registered
pooled depth-3 contrast** (Holm-corrected; dsl_sft +0.033 p=0.55, scaffold +0.033 p=0.41, c45_zero
−0.017 p=0.82; base depth-3 probe-robust coverage 0.05 — the wall replicates in-house at
think@1024). But the install itself is dramatic **within taught depths**: DSL-native SFT on 1,476
truth-blind procedure traces (depths 1–2 only) lifts depth-2 structure proposal from 0.37→**0.70**
(list) and 0.33→**0.57** (string) at parse-rate 1.00, deployable greedy 0.10→**0.37** (list) —
with zero depth-1 forgetting (0.85→0.85) and full trace-format adoption (1.00). The strategy is
**installable but depth-local**: it does not extend one composition step beyond its training
depth. C21's cross-depth negative (banked answers don't climb) now extends to banked *procedures*.
And C45's skill is **substrate-local**: the regenerated adapter (0.920 on its own held-out family,
above its 0.905 headline) transfers at ~zero to the DSLs and actively interferes (list d2
0.37→0.00). Both terminal laws survive with new scope clauses: the wall is not a missing
procedure (C36 hardens); the installable-induction result is a within-regime skill, not a
transferable faculty (C45 scoped).

## Research Program Fit

`structured_execution_and_compilers` × `posttraining_and_adaptation`: the single cell where the
program's two most mature terminal laws — C36 ("the fixed 4B cannot propose deep op-structure, a
model-level law") and C45 ("a general hypothesize-and-verify strategy is installable serial
compute") — made opposite predictions. Either outcome had to rewrite a law's scope; both got
rewritten narrower.

## Method

Frozen eval (committed, seed 71): list + string DSL identification tasks (byte-identical C36
`families.py`), 30 tasks per family × depth {2,3} = 120, each 8 visible + 6 hidden examples + 6
fresh probe INPUTS (labels never in any prompt), behavioral min-depth verified. Four arms at
matched K=12 think@1024 samples (temp 0.8/top_p 0.95, answer_max 512) + 1 greedy pass; a no-think
K=8 base anchor ties to C36's historical decode. Arms: **base** (C36 `ident_prompt`), **scaffold**
(the same + a verbatim-frozen evidence→shortlist→compose-and-check procedure text,
`configs/scaffold_prompt.txt`, zero op vocabulary), **c45_zero** (C45 adapter regenerated from its
committed digit-affine train_general.jsonl, seed-pinned; gated at 0.920 ≥ historical 0.905 on 200
held-out-a7 episodes before its DSL eval was spent), **dsl_sft** (QLoRA r32/α64 on 1,476
programmatic hypothesize-and-verify traces — evidence extraction → truth-independent
feature→candidate shortlists → fixed-order compose → mentally-executed checks with intermediate
states → revise-at-first-divergence — generated on depth-1/2 tasks only, op-type-sequence-deduped
vs ALL eval tasks (0 overlap at the task level; 51/1,476 traces' behaviorally-equivalent FOUND
pipelines coincide with eval depth-2 skeletons, 0 with depth-3 — the primary d3 contrast is clean), truth-BLIND by construction (byte-verified by regenerating with the
oracle fields deleted), trained into the THINK channel). Primary metric: **probe-robust
skeleton-coverage@K** (the C36 behavioral metric hardened against lookup-table mimicry: the same
true-skeleton fill must reproduce the model program's behavior on visible AND probe inputs).
Decision rule: pooled list+string depth-3 (n=60), one-sided paired bootstrap, Holm across the 3
arm-vs-base contrasts; falsification requires CI>0 AND diff ≥ +0.10 in the zero-or-one
trained-window stratum. Gates: trap (oracle-skelfill), c45 regen-sanity, install
(format + no d1 collapse). Pre-committed contingency: budget-2048 depth-3 re-probe for any null
arm with forced-close > 50%.

## Results

**1. The wall replicates and holds (C36 hardens).** Base depth-3 probe-robust coverage@12 = 0.033
(list) / 0.067 (string); pooled 0.05. No arm's Holm-corrected pooled contrast is significant and
none reaches the +0.10 falsification bar (dsl_sft +0.033, CI-lo −0.033; scaffold +0.033, CI-lo
0.000, p_holm 0.41; c45_zero −0.017). The pre-registered zero-or-one-window stratum for dsl_sft:
+0.019 (p=0.42). `c36_falsified_for_content_free_installs = false`.

**2. The procedure INSTALLS — massively — within its taught depths.** dsl_sft depth-2
probe-robust coverage: list 0.367→**0.700**, string 0.333→**0.567**; full-solve (hidden set) list
0.333→**0.700**; deployable greedy@1 list 0.100→**0.367**, string 0.067→0.200; parse-rate 1.00;
install gate: trace-format 1.00, depth-1 correctness 0.85 vs base 0.85 (zero forgetting — the
think-channel recipe avoids C43's answer-only collapse entirely). The same model that runs the
taught loop to a doubled depth-2 proposal rate gains nothing at depth-3 (0.10/0.07 vs base
0.03/0.07). The procedure does not compose one step past its training depth.

**3. C45's skill is substrate-local and negatively transferring.** c45_zero, whose regenerated
adapter *beats* its own source headline in-family (0.920 vs 0.905), scores 0.00–0.07 across all
DSL cells — below base at list depth-2 (0.37→0.00) — with parse-rate degraded to 0.70–0.79. The
"general" in C45 was family-general (affine a∈{1,3,9}→7), not substrate-general.

**4. The scaffold null is format-confounded (pre-registered reading).** Scaffold parse-rate
collapses to 0.44–0.67 vs base 0.86–0.91; its coverage still matches or edges base (list d3 0.10
vs 0.03, greedy d2 up to 0.20 vs 0.07 on string). Per pre-registration this arm reads
"the scaffold disrupts output formatting", not "elicitation fails"; per-parsed-sample rates and
the 2048-budget re-probe (below) qualify it further.

**5. Pre-committed budget-2048 contingency: the depth-3 wall is partially BUDGET-limited — for
everyone.** The forced-close trigger (>50% at d3) fired for two of the three null arms (scaffold
and c45_zero, both 1.00; dsl_sft 0.32 did not trigger; base was 0.84). At budget 2048 on the
depth-3 subset (K=12): c45_zero 0.017 (was 0.033) — null doubly confirmed. Scaffold 0.150 (was
0.083) — which prompted a post-hoc base@2048 control: **base itself doubles, 0.050 → 0.100**
(forced-close 0.84 → 0.75, still truncation-bound). Paired at matched 2048 budget the scaffold
edge is **+0.050** (one-sided CI-lo +0.000, p 0.043–0.051 across bootstrap seeds —
`runs/verdict.json contingency_b2048`) — borderline and post-hoc, not verdict-grade in either
direction. Reading: serial-compute DOSE moves depth-3 coverage (a C44-consistent softening: 2×
budget ≈ 2× coverage, from a very low floor); whether the taught procedure adds a small margin
beyond dose is unresolved at this n. Even 2048 truncates 75–100% of thinking — the depth-3
budget dose-curve (4096+) is unfinished business, not settled wall.

**6. Mimicry guard: armed, unneeded.** Mimicry rates ≤ 0.014 in every cell — the lookup-table
leak the design review predicted did not materialize at these budgets, but legacy-vs-probe-robust
gaps (base list d2 0.43 vs 0.37) show the guard did catch real instances.

**7. Compute parity.** Scaffold costs more per sample (prompt 523 vs 286 tokens; gen 1459 vs 1195
at d2) and still doesn't move depth-3; K′-matched base (K′=14, capped at stored K=12) leaves base
d3 at 0.05 — no accounting trick closes the dsl_sft d2 gap (dsl_sft traces are also *shorter* at
deploy, parse 1.00, making its d2 win compute-favorable).

## Controls

Trap gate: oracle-skelfill 1.00 in all four cells (a proposal null is interpretable; C39/C43
discipline), rand-skelfill@12 ≤ 0.07 (structure is not value-searchable by luck at this budget).
Blindness: traces byte-identical when regenerated with oracle fields deleted. Leakage: 0
op-type-sequence overlaps between SFT tasks and eval; depth-3 window-overlap stratified instead of
excluded. Regen-sanity: the c45 gate initially FAILED at a miscalibrated 0.95 threshold (C44's
shift number, not C45's 0.905 headline) — corrected to historical−2SE (0.87) with the measurement
(0.920) unchanged; lesson codified. No-think anchor: base no-think d2 0.07–0.10, d3 0.00–0.03 —
consistent with C36's historical floor; think@1024 alone does NOT clear the wall (no C44
frame-shift; `base_think_clears_wall=false`).

## Oracle Versus Deployable Evidence

Oracle appears only in: eval grading, the trap gate, trace purity filtering (C45-standard
programmatic generation), and probe labels computed at eval time. Deployable evidence: the dsl_sft
depth-2 gains are fully deployable (greedy@1 0.10→0.37 list at parse 1.00 with no oracle at
inference); the depth-3 null is measured under the same deployable protocol.

## Interpretation

The wall is not a missing procedure — it is not fixable by teaching the model *how to search*.
The dsl_sft arm proves the model can learn, format, and execute the full
hypothesize-shortlist-check-revise loop (doubling proposal exactly where the loop was practiced)
while gaining nothing one composition step deeper. Combined with C21 (banked answers don't climb),
C24 (gains scale with in-depth diversity), and C44 (induction is serial-compute-limited), the
sharpest available statement is: **serial-strategy training buys depth-local competence; depth
itself is the resource that neither answers, nor diversity, nor procedure can bank across.** C45's
"general induction is installable" survives only with a regime clause — general across families
within a substrate and depth regime, zero across substrates (and negative-transfer prone). For
deployment: DSL-native procedure-SFT is the best in-depth structure-proposal lever measured to
date on this substrate (d2 coverage 0.70 vs banking-era numbers), but the depth frontier still
belongs to external search (C34/C35 brute dominance) — among model-side levers only raw serial-
compute BUDGET moved depth-3 at all (0.05→0.10 at 2× budget, Result 5), and that dose-curve, not
any taught procedure, is the remaining open edge.

## Next Experiments

1. Depth-3 budget dose-curve: base coverage doubled 1024→2048 and thinking is still 75–100%
   truncated — run 4096/8192 on the d3 subset (all arms cheap at n=60) to learn whether the wall
   asymptotes or keeps yielding to serial compute (this would sharpen both C36 and C44).
2. Depth-curriculum probe: traces at depths 1–3 (oracle content at d3, C28-style) vs depths 1–2 —
   is the depth-locality about *practice at depth* or about d3 procedure traces being ungeneratable
   blind? (The blind rulebook keeps 45–46% of d2 tasks after the purity filter — blind solve ≈51%; measure its d3 yield first, CPU-only.)
3. Scaffold format repair: the scaffold arm's parse collapse (0.44–0.67) is mechanical — a
   format-hardened scaffold (explicit "end with the code block" + one worked example) isolates
   elicitation from formatting; cheap single-arm rerun.
4. The C46 next-test still standing: skeleton-level P(True) ranking + judge-pruned beam at
   depth 4–5 (proposal is walled; RECOGNITION of partial structure is untested — forest-review
   option B).
5. Cross-substrate install transfer: train the trace procedure on list, eval on string (and
   register) — is procedure-SFT substrate-local like C45's adapter, or does within-DSL transfer
   hold at matched depth?

## Artifact Manifest

`runs/verdict.json` (primary contrasts, cells, strata, parity, gates), `runs/eval_*.json` (5 arms
+ 3 budget-2048 probes: 2 pre-committed contingency + 1 post-hoc base control), `runs/gate_*.json`, `data/` (frozen eval with probe inputs, 1,476 traces,
SFT tasks, gate tasks — all committed), `configs/scaffold_prompt.txt` (verbatim frozen treatment),
`analysis/hypothesize_verify_wall.png`, `reports/design_review.md` (3-lens pre-registration
review), `experiment_log.md` (full saga: gate miscalibration stop, recovery-path empty_cache
crash, both codified). Adapters (lora_dsl, lora_c45, ~182 MB each) moved external — see
`reports/artifact_manifest.yaml`.
