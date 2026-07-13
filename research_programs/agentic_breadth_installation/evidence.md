# Evidence

## Seed Experiments

- Experiment: `qwen35_4b_gauntlet_breadth_round1` — gym built (12 families,
  10 trained + 2 held out), two fast training rounds run, first
  menagerie-arbitrated install in the corpus.

## Confirmed Claims

- Claim: C49 (Confirmed) — vLLM 0.24 runtime LoRA is a silent no-op for
  Qwen3.5-4B PEFT adapters; on-vs-off behavioral gate required; deploy via
  merged composite checkpoints or the HF backend.
- Claim: C50 (Promising) — breadth-first expert iteration with
  emission-seam-weighted loss installs substrate-general agentic competence:
  menagerie quick +0.223/+0.294 on two fresh paired seeds (HF backend,
  deterministic), gym +0.518 including never-trained held-out families.

## Negative Findings

- Finding: full-weight SFT on the model's own naturally-closed verified
  chains (round 1) installs nothing measurable — near-self-distillation; the
  deployment-critical post-force-close state must be in-distribution and the
  gradient concentrated on the answer/action emission seam.
- Finding: stallwright (bounded optimization) is unharvestable at round 1 —
  the base model never concludes its optimization deliberation even at a
  4096-token think budget; the axis moved only by transfer (+0.395 gym) and
  its menagerie analogue (stockade) did not move.

- Claim: C53 (Promising) — the second wall: nine paired events across six
  escalation levers land the treated model at 0.375-0.447 aggregate; the
  emission-policy install is a one-time step and no self-verified-output
  recipe variant moves the band; a single rich family installs nearly the
  full effect (breadth is an axis-aligned increment, not the mechanism).

- Claim: C54 (Promising) — novel serial-compute mechanisms (compression
  advantage + skin-shuffle, co-trained from base = the apex arm) DECISIVELY
  clear the +0.32 MEDIUM bar for the first time (+0.345, all events), breaking
  the wall gold-procedure supervision (C53) could not; but no single 4B adapter
  clears quick AND medium — a non-convex tier-Pareto frontier.

## Current Read

Breadth + strict verifiers + emission-seam supervision is the first recipe in
the corpus to move the blackbox instrument, and the locality laws
(C43/C45/C48) do not extend to it. The binding deployed constraint at
current difficulty is the truncation cascade (consume budget → force-close →
verbose restart → no parseable answer); repairing commit-from-partial-
reasoning transfers across substrates the model never trained on. Trust only
paired same-backend menagerie comparisons; never a vLLM adapter arm (C49).
Next: the residual is a capability core — scaffold-distillation of
tool-found solutions, on-policy RL at residual failures, and
failure-forensics curricula are the queued beyond-recipe mechanisms
(see C53 next tests). Same-recipe scaling is closed.

## Deep-Advantage MOPD Qualification and Locality (2026-07-13 — Passed prerequisites)

`qwen35_4b_deep_advantage_mopd` repeated same-prefix qualification on two new
192-state blocks from the immutable 40/60 joint soup. Deep was selected on
28/26 states and beat soup on disjoint audit branches by +0.1650/+0.1220
(pooled +0.1421, one-sided 95% lower bound +0.1230); it beat quick by
+0.2000/+0.1420 (lower bound +0.1534). Every support, block-sign, and
uncertainty gate passed, authorizing exact-logit locality.

The locality round needed three fixed candidate batches and found 90 deep
routes, from which it froze 60 deep units, 20 soup anchors, and 60 disjoint
matched non-advantage controls. The five-update 15-deep/5-soup pilot passed:
centered non-target drift was 0.02760, entropy drop was 3.11%, and exact target
loss improved 0.01293→0.01170. This authorizes four-round MOPD. The exact
measurement is one midpoint token per consumed unit, and no capability result
exists yet.

Quick also passed diagnostically on 29/18 routes in this fresh replication,
after failing one soup-relative block in the predecessor. This is not license
to add it to the locked deep-only treatment. It makes the later two-teacher
design more worthwhile, but that return still requires cross-fitted direct
advantage prediction, adaptive branch allocation, and a third untouched block.

## Specialist Integration Attempt (2026-07-12; stopped before training)

`qwen35_4b_specialist_policy_integration` attempted the accepted next mechanism:
four live-state DAgger/execution-RL capability producers followed conditionally
by same-origin MOPD integration. Its new compound environments, runtime,
regenerated incumbent, explicit merge, and 7/7 behavioral-installation gates
passed. The disjoint compound macro was 0.135 versus the `<0.60` headroom bar.

The full baseline exposed a terminal design error before training: the sole
tools family `ferrier` scored 0.994, making its frozen `S0 + 0.10` target 1.094
under a score ceiling of 1.0. Because all four specialists were mandatory, the
run stopped before best-of-8, DAgger, GRPO, teacher audit, integration, or
benchmark exposure. This does not test MOPD efficacy. It establishes a reusable
portfolio rule: endpoint headroom is insufficient; every mandatory teacher's
theoretical improvement bar must be feasible on a disjoint baseline before
capability-production spend.

## Pareto Policy Integration Qualification (2026-07-12 — Negative prerequisite)

`qwen35_4b_pareto_policy_integration` corrected the earlier gate rather than
lowering it. Independently regenerated and behavior-gated C54 `blend` and
`apex` policies were compared on two fresh contamination-safe procedural
blocks. `blend - apex` on quick capability was negative in both blocks
(`-0.00693`, `-0.03789`), pooling to `-0.02241` with a one-sided 95% lower
bound of `-0.04897`. `apex - blend` on deep capability was replicated
(`+0.04563`, lower bound `+0.03401`) but six retention cells regressed beyond
0.02. All protocol checks passed.

The external menagerie tier ranking therefore did not supply a clean
quick/deep teacher crossover on the distribution where distillation would run.
The experiment stopped before teacher audit or any MOPD update, so MOPD remains
untested. The strategic correction is to estimate teacher choice on same-prefix
states: a future run should freeze a disjoint verified continuation-advantage
router, not assume that an aggregate instrument label is a local teacher label.

## qwen35_4b_interactive_policy_curriculum (2026-07-11/12 — Negative)

The full-sequence live-state DAgger warm start failed its preregistered
mechanism gate before RL or Menagerie. Against the regenerated C53 incumbent,
train-family episode macro fell 0.6048→0.3517 (−0.2531; paired-bootstrap 95%
CI [−0.2954, −0.2103]) and three untouched families fell 0.6850→0.3519
(−0.3331; CI [−0.3804, −0.2869]). Atom retention stayed inside its −0.03
guard (−0.0215), parsing stayed perfect, and natural closure improved by
10–13pp, ruling out generic collapse.

The failure was semantic-operator capture: only 55/2,270 training targets were
`VERIFY`; after training, loomfix produced 600 `PATCH` and zero `RUN` actions,
while the untouched analogous patchwheel produced 599 `RULE` and one `RUN`.
Thus DAgger taught a fluent shared observe/revise trace but erased the scarce
verify/commit pivots that make the loop effective. Live-state correctness is
not enough: a shared update must preserve the incumbent operator distribution
and behavior outside corrected states. Entropy/outcome variance can route
state acquisition, but must not become token pressure. RL, matched controls,
and Menagerie were correctly cancelled; zero benchmark seeds were consumed.

## qwen35_4b_think_ftpo_round1 (2026-07-11, C52 — Negative)

The first different-mechanism recipe after C50's re-saturation: single-position
preference training (FTPO) on outcome-conditioned think-block pivot points
(prefix-tree divergence of n=16 verifier-scored rollouts). Preregistered
mechanism gate FAILED (−0.039/−0.076 vs a +0.05 bar on held-out band tasks);
the shuffled-label control degraded identically, so the harm is the training
regime, not the steering signal. Guards localize the channel: no C29-style
collapse (the two-tier logit tether works), no-think channel clean — the
damage is think-flow convergence (natural close halves at think@2048). Read:
FTPO's safety/efficacy requires the rejected token to be a CONFIDENT OUTLIER
(loop initiators, lexical attractors); near-parity pivot tokens violate the
precondition and the ε-margin objective's collateral dominates. Menagerie was
correctly never exposed (mechanism-gate rule). Census bonus: repetition loops
are ~0.1% at deployed budgets — the loop-FTPO variant belongs to 16k+ only.
That result queued confident-wrong-turn filtering (failing branch's token also
locally dominant); round 2 below has now tested it.

## qwen35_4b_think_ftpo_round2 (2026-07-11, C52 — Low-dose null)

The registered confident-wrong-turn rescue also failed, but isolated the next
bottleneck. A frozen entropy/varentropy selector retained 155 failed-argmax
pivots. Conventional demotion, bounded positive-only uplift, and shuffled
uplift all failed exact-logit locality (mean per-row median non-target drift
0.229/0.145/0.120 logits vs a 0.10 ceiling). Pull-up was materially safer than
push-down and true labels beat shuffled labels on the gym (+6.25pp) and fresh
repository agent (+13.89pp, paired-bootstrap CI touching zero), so the steering
directions contain some signal. They did not elicit breadth: repository hidden-
test pass was base 43/72, uplift 39/72, demote 34/72, shuffled 29/72; fresh
whitebox uplift was +0.26pp at think@1024 and −3.06pp at think@2048. Every
coarse C49/collapse/no-think/gym guard passed, but P1/P2/P3 did not, so
menagerie remained sealed (zero seeds consumed).

Read: confident-outlier geometry is necessary but not sufficient; the active
constraint is context locality of the parameter update. Entropy/varentropy can
route and diagnose pivots, but higher varentropy was not safer (lowest-V
quartile had the cleanest uplift drift). Do not scale this LoRA recipe. Require
a lower-dose or context-gated mechanism to clear P1 before another harvest or
agentic transfer run.

## qwen35_4b_repo_search_compress_bank (2026-07-12 — Negative)

Executable search and replay compression produced a superficially excellent
install on the six trained repository families: apex 40/48 versus compact
48/48, with every compact trajectory reduced to exactly
`INSPECT→PATCH→VERIFY→COMMIT`. That behavior did not transfer. On four wholly
held-out algorithm families, compact fell 49/72→25/72 (−33.3pp; paired 95% CI
[−44.4,−22.2]) and lost to matched-compute sampling by 18.1pp. Locality also
failed (0.386 centered-logit drift versus 0.15), invalid actions rose
9.3%→26.0%, and verification retention fell 1.00→0.88. Menagerie remained
sealed.

The transition audit resolves why exact operator balance was insufficient.
After a failed visible test, apex chose another patch on 24/26 next actions;
compact did so on 0/48. It still committed after every passed test. All 18
recursive-overlay trajectories repeated the same rejected patch. Success-only
minimization installed a family-specific happy path while deleting the
verifier-conditioned recovery policy. Marginal operator counts do not preserve
conditional transition structure. Because the necessary gate cancelled the
action-only arm, plan-gradient attribution remains open; the supported negative
is the complete compact plan-plus-action recipe.

## qwen35_4b_verifier_conditioned_recovery_bank (2026-07-12 — Locality-gated negative)

The direct C54 successor repaired the missing intervention unit. Fifty-seven
model-found repository repairs produced 399 rows/arm balanced at seven public
state→action transitions, including rejected-patch→changed-patch,
failed-test→diagnose/revise, and passed-test→commit. Every bank/replay/firewall
gate passed. On 60 fresh trained-family recovery cases, frozen apex scored
0.483, matched happy-action training 0.817, recovery action-only 0.850, and
recovery-plus-plan 0.917. The plan arm used 480 mean sampled tokens versus
2,340 for base and was selected by the frozen rule.

The headline recipe nevertheless stopped at exact-logit locality: selected
drift was 0.303 versus a 0.15 ceiling and unrelated entropy fell 0.106 nats, so
no transfer family or Menagerie seed was exposed. Exploratory controls localize
the damage. Happy action and recovery action-only passed locality at 0.083 and
0.098; action-only unrelated entropy was flat (+0.006). Nominal 5% plan-token
mass produced a 29.5% larger merge-delta norm than action-only and a step-10
pre-clip gradient of 42.1 versus 1.8. Seam entropy/varentropy explains why:
every JSON action-start token was already rank 1, while imposed ordinary-state
plan starts were ranks ~8,404 (inspect→patch), ~1,163 (patch→verify), ~135
(start→inspect), and 3 (pass→commit). Plan SFT drove all to rank 1 and near-zero
entropy.

Read: conditional action banking contains a strong, parameter-local recovery
signal; broad lexical plan imitation adds trained-family efficiency but violates
locality because token-mass weighting ignores realized surprisal/gradient. The
next experiment should interpolate the already-trained reason delta under a
locality-first gate, with full-dose action-only as the safe anchor. Do not expose
the untouched four-family transfer blocks until a scaled checkpoint passes.

## qwen35_4b_recovery_reason_locality_interpolation (2026-07-12 — Local but policy-gated)

The frozen action→reason path disproved the simplest non-separability reading.
All four action-anchored mixtures passed the parent's exact locality instrument:
drift was 0.100/0.104/0.111/0.121 for λ=.10/.18/.24/.30, with bounded entropy
and varentropy change, while full reason reproduced 0.303. The two independently
trained deltas strongly cancel in weight space: summed mixed-delta norm falls
from 29.17 at action to 23.47 at λ=.30. This is not a monotone dose path.

Behavior also has a sharp safe optimum. On the 60-case selection block λ=.18
scored 0.967, versus 0.483 base, 0.817 happy, 0.850 action, and 0.917 full
reason. It reached 0.933 failed-test and 1.000 rejected-patch success. However,
no candidate cleared the registered policy gates: λ=.18 had 0.104 invalid
actions/turn versus a 0.077 ceiling, and only 0.333 immediate rejected change
versus the 0.60 bar. Confirmation, transfer, and Menagerie stayed sealed.

Post-stop forensics distinguish the failures. Every one of λ=.18's 24 invalid
steps closed thinking and exhausted the exact 256-answer-token cap inside a
long exact-replacement JSON payload; this is a real harness payload bottleneck,
not repetition or free-form slop. Conversely, all 30 rejected cases changed the
patch within two turns and solved: 20 used sensible INSPECT→PATCH and 10 used
PATCH→VERIFY. The immediate-only proxy rejected retained conditional recovery.

Read: a locality-safe recovery policy exists on this weight path, but the
registered harness cannot deploy it cleanly. The next experiment should freeze
λ=.18, enlarge tool-answer capacity for every arm under matched compute, and
measure rejected→changed-patch within two turns. This result does not support a
transfer or benchmark claim.

## qwen35_4b_same_prefix_advantage_routing (2026-07-12 — Route-gated negative)

The clean successor to the two earlier specialist stops finally measured both
same-origin policies on exact soup states. Across 384 fresh states and 9,216
teacher/student continuations, deep independently passed both contrasts in
both blocks; the combined router also passed. Quick did not: selected quick
beat the soup by `+0.2009` in block 0 but lost by `-0.0253` in block 1. The
frozen rule stopped before locality, MOPD, controls, or Menagerie.

The mechanism audit rejects the tempting threshold repair. Quick block 1 had
an apparent `+0.319` selection margin, yet only 6/26 states remained strict
quick winners on independent audit. Requiring observed margins of `+0.10` or
`+0.25` left the soup-relative audit mean negative. Absolute policy scores were
reliable (`r=0.79`--`0.86`); statewise winner conditioning was not. Routing was
also atom-heavy (101/288 atoms, 10/96 episodes), so composable episode evidence
is especially thin.

Read: deep is the first source policy to clear the intended local-teacher
prerequisite, but the required two-teacher composition does not exist under
this labeler. A new deep-only routed-MOPD experiment is the shortest test of
the update kernel. Reintroducing quick requires cross-fitted direct advantage
prediction and a third untouched block; otherwise retire it rather than tune a
margin. This result is not evidence against MOPD.

## qwen35_4b_recovery_payload_budget_harness (2026-07-12 — Confirm-gated negative)

The matched interface repair validated the predecessor's post-stop diagnosis.
The fixed locality-safe λ=.18 checkpoint passed a third disjoint locality block
(0.114 centered-logit drift; entropy Δ −0.0059), while a 512-token answer slot
reduced candidate cap hits to 0.5%/7.8%/7.9% of turns across
calibration/dev/confirm. Valid rejected-patch and failed-test change-within-two
reached 100% on both transfer blocks. The old invalid-action and
immediate-transition stops were therefore harness/proxy failures.

The candidate passed every development gate at 0.7125 recovery, +0.125 versus
base and +0.2125 versus equal-reservation sample-more, with exact normal-task
retention. Independent confirmation stopped the claim: candidate and
action-only both scored 0.6875, missing the frozen candidate ≥ action +0.03
bar. All other confirm checks passed, including +0.0875 versus base, +0.225
versus sample-more, nonnegative deltas on all four held-out families, perfect
two-turn recovery, locality, and exact normal retention. Menagerie remained
sealed.

Paired forensics locate a new deployable opportunity. Candidate and action-only
had a 0.7875 hidden-success union on both dev and confirm, with 10/6 and 8/8
exclusive wins. Confirmation action-only wins clustered in `pattern_router`
rejected-patch states while candidate wins clustered in `rate_buckets`; neither
global policy dominates. Hidden union is oracle-only. The warranted successor
is bounded public-verifier branching between the two local policies, compared
to equal-compute sampling, followed only conditionally by transition-balanced
winner banking. Do not tune another scalar reason dose or route on family
identity.

## qwen35_4b_recovery_verifier_branch_tournament (2026-07-12 — Prospective infeasibility)

The predecessor's replicated 0.7875 action/reason union did not transfer to
four new procedural repository families. On 80 prospective-dev recovery cases,
C54 base scored 0.6125, λ=.18 and action-only each scored 0.7375, and their
deterministic hidden union reached only 0.7500. Equal-reservation pass-if-either
sample-more scored 0.7375 for λ=.18 and 0.7500 for action. The union therefore
failed every frozen +0.03 feasibility contrast before the public selector was
scored. Confirmation, winner banking, and Menagerie stayed sealed.

The paired failure anatomy is more useful than the aggregate null: 58 cases
were solved by both sources, one by candidate only, one by action only, and 20
by neither. Every shared failure was the new `atomic_reservations` family.
Both source policies retained 1.00 changed-patch-within-two on both controlled
states, so this is not another recovery-loop deletion. Traces repeatedly fixed
whole-request validation or input immutability separately and then regressed
the other; action sample-more assembled the full conjunction once in 20 cases.

Read: public selection cannot manufacture proposal coverage from globally local
policies with the same semantic core. Retire branch arbitration for this
recovery line. The next intervention should install the missing transactional
validate-copy-commit invariant from executable tool-found solutions across
diverse training families, mix the existing conditional recovery bank as
replay, and require transfer to structurally different transactional families
plus broad recovery retention before Menagerie.

## qwen35_4b_transaction_invariant_recovery_curriculum (2026-07-13 — Transfer-dev negative)

The fixed action-seam curriculum passed exact locality against C54 apex (0.119
centered non-target drift; entropy +0.011, varentropy −0.0002) and strongly
installed six trained transaction families: 0.817 versus recovery parent 0.517
and matched replay-only 0.383. Both changed-patch-within-two transitions were
1.00; invalid actions and answer-cap contacts improved. Thus executable
programmatic supervision can locally move semantic coding proposals without
deleting the loop or broad neighboring logits.

It did not meet unseen transfer. On four transaction families, primary was
0.719, parent 0.703, replay-only 0.641, and equal-reservation parent sample-more
0.703. Candidate-parent paired CI was [−0.031,+0.078], below the registered
+0.10 bar. All family and interface guards passed, but atomic reservations—the
only headroom family—remained 0/16 for every arm. Confirmation, broad
retention, and Menagerie stayed sealed.

The proposal audit advances the mechanism despite the score null. Every one of
the candidate's 16 first atomic patches contained copied state, all-resource
validation, and atomic per-request commit; none contained the separate negative
amount exception. After the visible test reported the omission, every
trajectory overcorrected by raising on all unavailable/insufficient requests,
destroying required `False` decisions. The missing unit is now
verifier-faithful validation-policy discrimination, not transaction structure
or recovery syntax. Next: near-correct counterexample states that isolate one
policy distinction at a time, with complete recovery replay and matched extra-
transaction controls.
