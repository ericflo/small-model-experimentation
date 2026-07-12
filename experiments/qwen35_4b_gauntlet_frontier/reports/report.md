# Gauntlet frontier: difficulty escalation past the breadth-install plateau Report

## Summary

Nine paired menagerie decision events across six escalation levers all land
the treated model in the same 0.375–0.447 aggregate band (base 0.070–0.161):
the emission-policy install is a large one-time step to a robust ceiling,
and no variant of training on the model's own verified outputs moves it
further (claim C53). The attribution ablations revise C50: a single rich
verified family installs nearly the full effect and nearly full
cross-substrate transfer — the cargo is a behavioral policy, not breadth of
content.

## Research Program Fit

Direct successor to `qwen35_4b_gauntlet_breadth_round1` in
`agentic_breadth_installation`; executes the pre-registered ablations and
the difficulty-escalation lever named in the plateau verdict.

## Method

Ablations (matched dose, trained from base on rounds-2/3 slices, merged
deployment): recovery-only (553 trimmed forced-close examples), ferrier-only
(665), breadth-matched (665). Frontier: gym extended to L1–L6 (L1–L4
byte-identical; horizons to 22; +patchwheel, +packhouse), harvested with the
round-3 model (2,514 examples, L3–L6 mass), union-trained. Sharp1024:
think ≤900 subset so the quick tier's 1024 budget is on-policy. All events:
paired base-vs-merged, fresh seed, vLLM backend both arms.

## Results

quick: recovery-only +0.1216; ferrier-only +0.3141; breadth-matched +0.3302;
frontier +0.2716/+0.3135; sharp1024 +0.2934/+0.2376.
medium: frontier +0.3196; sharp1024 +0.3418.
Gym-internal (frontier adapter): L1–L4 mean 0.681, L5–L6 0.466 (base ~0.18 /
~0). Held-out transfer: ferrier-only ≈ breadth on spindle (0.748/0.752) and
runeward (0.944/0.933); breadth's edge is axis-aligned (menagerie chronicle
+0.750 vs +0.375; gym stallwright 0.517 vs 0.379).

## Controls

Same-seed paired arms, one backend, fresh seed per event, seeds logged and
reuse-blocked; matched-dose ablation arms; held-out gym families; the C49
adapter-application gate live in the instrument; encode-time skip counts
audited (source of the C50 recovery-arm correction — rounds 2–3 trained on
zero recovery examples; sharp1024 trains 1,018 trimmed ones and still lands
in-band).

## Oracle Versus Deployable Evidence

All headline numbers are deployable (greedy, think-mode, deployed budgets,
bare merged model). Oracle policies validate gym instruments only; no
benchmark content, oracle output, or external model enters training.

## Interpretation

The plateau is not difficulty coverage (in-gym frontier competence installs
while the band holds), not missing recovery supervision, not budget
mismatch, not insufficient breadth, and not dose. It is a second wall: the
capability residual cannot be taught by the model's own verified outputs.
Next mechanisms (queued): scaffold-distillation of tool-found solutions,
on-policy RL at residual failures, failure-forensics curricula.

## Oracle-trace distillation addendum

Hand-coded solver procedures for the seven residual families, narrated as
1,680 truth-blind think-channel traces (selftest-gated) and trained from
base with the sharp1024 mix: quick +0.282/+0.268, medium +0.289 — the
highest absolute levels observed (0.474/0.482) but paired deltas in-band.
Twelve paired events, seven lever classes, one band: the second wall holds
even against gold procedure supervision the model cannot self-generate.

## Clearance campaign addendum — tier dissociation

A curated concentrate mix decisively clears quick (paired +0.330/+0.417,
mean +0.374, absolute 0.505) but not medium (arm-mean +0.259 over three
events, pre-declared metric); the breadth-matched arm's momentary double
clearance regressed on replication. Across 19 paired events, tier
performance dissociates under mix composition (quick record: concentrate;
medium record: sharp1024 +0.342); no arm clears both decisively — the
conjunctive bar trades against itself at this model capacity.

## Blend addendum — convex mix composition breaks the quick tier

The blend arm (concentrate + L3+ episode mass) dominates both parents on
both tiers, holds both absolute records (0.522 quick / 0.521 medium), and
decisively breaks the quick plateau (arm-mean +0.409). Medium arm-mean
converges at +0.307 over five no-exclusion events. Final map (28 paired
events, 10 arms): mix composition is a convex optimization axis; the
quick-tier ceiling yields to it, the medium tier does not clear +0.32 at
arm-mean within this recipe family.

## Breakthrough addendum — novel serial-compute mechanisms clear medium; tier-Pareto frontier

First-principles diagnosis (not a recipe remix): the medium ceiling is a
serial-compute wall — the residual axes (search/induction) need long
derivations that exceed the deployed budget, so the emission policy commits
early to wrong answers. Three new mechanisms exploiting the verifier+generator
control: (1) a custom length-penalized COMPRESSION ADVANTAGE that banks the
model's own shortest correct trace per hard item (amortizing serial compute
into weights), (2) SKIN-SHUFFLING (fresh pseudo-vocabulary every row -> bind
mechanics not tokens), (3) GRPO-lite contrast (wrong pushed down, C29-guarded).

The APEX arm (blend recipe + these mechanisms, co-trained from base) is the
FIRST arm to decisively clear the +0.32 medium bar: **medium +0.345 (all three
events above)**, lifting exactly the reasoning-heavy axes prior arms left flat
(lockpick/induction +0.267, stockade/optimization +0.289). Its quick mean is
+0.308. The mirror arm blend clears quick (+0.409) not medium. The dose
interpolation apex60 is strictly dominated (quick +0.313 / medium +0.285),
proving the trade-off non-convex. Conclusion (claim C54): both tiers are
individually breakable past +0.32, but no single 4B adapter clears both — the
two tiers occupy a non-convex Pareto frontier and compete for the fixed model's
representational budget. Absolute levels 0.45-0.52 (quick) / 0.43-0.48 (medium)
from a ~0.12 base.

## Next Experiments

- Failure forensics on the residual axes at deployed budgets (gym proxies).
- Scaffold-distillation arm (tools find what the model cannot sample; bank
  the verified result — C22-24 precedent, now with a blackbox arbiter).
- On-policy RL (GRPO, execution rewards) at residual failure modes, guarded
  by C29.
- Slow/deep-tier ceiling confirmation; multi-seed band quantification.

## Artifact Manifest

`reports/artifact_manifest.yaml` — adapters and merged checkpoints external
under large_artifacts/; harvest/eval row files gzipped in runs/.
