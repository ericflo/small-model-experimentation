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

## qwen35_4b_counterfactual_evidence_acquisition_curriculum (2026-07-13 — Lineage-locality infeasible)

The exact transaction-replay start checkpoint was compared directly with the
C54 apex anchor on the frozen 48-context block before interface behavior or
training. Median centered non-target logit drift was 0.110735 against the
registered 0.10 ceiling; all 48 row-level drift estimates exceeded 0.10. The
entropy guard passed (+0.013636 versus a -0.05 floor), varentropy was
essentially flat (+0.000297, diagnostic only), and rendered prompts matched.
The formal verdict is `LINEAGE_LOCALITY_INFEASIBLE`.

Per preregistration, interface selection, acquisition qualification, all three
training arms, behavior, transfer, retention, uncertainty analysis, and
Menagerie remained sealed. This result does not test counterfactual evidence
acquisition, transition-balanced replay, or capability installation. It
establishes only that this exact parent-anchor pair was ineligible under the
new direct 0.10 attribution contract.

Read: eligibility under a direct locality ceiling is not inherited from
lineage or from a prior looser or different-context gate. A successor must use
a new experiment and fresh locality contexts, begin from apex itself or
prospectively select a fixed apex-compatible parent using outcome-free locality
blocks, then independently re-establish complete-loop behavior and acquisition
headroom before training. Do not raise the observed ceiling, swap lineage
after the result, or rescue this directory.

## Deep-Advantage MOPD (2026-07-15 — Terminal capability negative)

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
loss improved 0.01293→0.01170. The exact measurement is one midpoint token per
consumed unit. Seeds 42, 43, and 44 subsequently completed all four registered
integration rounds, establishing replicated route supply and optimizer-gate
completion.

All matched controls are now built. Four-round full-prefix non-advantage MOPD,
wrong-teacher quick MOPD, and off-policy continuation SFT each completed every
consume-once update and passed their frozen training gates; the 25%/50%/75%
deep parameter soups have exhaustive model-byte receipts. Independent canonical
replay passes for aggregate controls receipt
`103ef4cc0b24d7c10666b6f0adfcd4dfae4720415c7fbbc76b681ab79162640b`.
The sealed two-block comparison is terminal negative. Primary seed 42's pooled
joint deltas are `−0.006845` versus deep (one-sided 95% LCB `−0.012839`),
`−0.001300` versus the soup initialization, `−0.001872` versus off-policy SFT,
`−0.003706` versus soup75, and `−0.169239` versus soup best-of-eight (LCB
`−0.175468`). Both block means versus deep are negative, every better-source
stratum cell fails, and seeds 43/44 also trail deep by `−0.003450`/`−0.005660`
pooled joint. Retention passes, while untouched `brinework` and `spindle`
transfer improve `+0.015625`/`+0.010590`.

There is a small real mechanism signal: primary seed 42 beats matched
non-advantage MOPD by `+0.005619` pooled joint (LCB `+0.000582`) and
wrong-teacher by `+0.005312` (LCB `+0.000099`). Advantage routing and teacher
identity therefore affect the update direction, but the operator does not
cross the source frontier and loses to simple interpolation and sample-more.
The terminal analyzer emitted `stop_before_benchmark_cli`; no benchmark was
authorized or opened.

Quick also passed diagnostically on 29/18 routes in this fresh replication,
after failing one soup-relative block in the predecessor. This is not license
to add it to the locked deep-only treatment. It makes the later two-teacher
design more worthwhile, but those estimator improvements are not sufficient.
First require a direct-bf16 deployment-parity microtrial whose causal update
survives merge and beats the source, interpolation, and sample-more. A later
two-teacher return still requires cross-fitted direct advantage prediction,
adaptive branch allocation (including zero quick allocation), and a third
untouched block.

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

## qwen35_4b_validation_policy_counterexample_curriculum (2026-07-13 — Calibration infeasible)

The one-transition residual update itself was local: candidate-to-C54 drift was
0.109, entropy changed +0.021, and varentropy −0.011. Candidate and matched
extra-transaction control each received 36 steps, 336 rows, zero think loss,
and identical transition/operator action mass from the same learned transaction
parent. The candidate changed only 24 post-diagnosis revision rows.

Controls-first calibration made the causal comparison impossible. The parent
and matched control each solved all 48 fresh train-skin recovery cases, with
perfect failed-test and rejected-patch changed-within-two and zero invalid
actions. All 48 parent first changed patches already included negative
handling, copied state, and ordinary `False` rejection. The theoretical
candidate ceiling could not clear +15/+10, so candidate scientific behavior,
transfer, retention, and Menagerie stayed sealed.

Read: making the rule explicit and the partial state otherwise correct removed
the predecessor's residual. The original atomic miss was conditional on its
more implicit contract and proposal dynamics, not evidence of a generic
inability to write the distinction. Add a substrate-headroom gate before
capability production: qualify multiple semantic conflicts under the exact
prompt/verifier distribution, require replicated non-saturation, and only then
bank/train on disjoint skins. Do not lower the current bars or expose the
trained candidate post-stop.

## qwen35_4b_semantic_policy_headroom_tournament (2026-07-13 — Instrument failure)

The no-training qualification ran the exact learned transaction parent on two
content-disjoint 72-case blocks. Its formal verdict is `INSTRUMENT_FAIL`:
answer-cap contacts were 43/356 turns (0.1208) and 46/363 (0.1267), above the
frozen 0.05 ceiling. Explicit controls passed at 9/9 and 8/9; invalid actions
were only 0.0169/0.0138. Most cap contacts contained a valid tool call followed
by run-on, but all 12 endpoint failures contacted the cap, so the registered
stop cannot be waived. No training or Menagerie event ran.

No semantic axis qualified even descriptively under the frozen rule. Negative
and non-integer failed-test recovery were 9/9 across all three representations
in both blocks. Blank-resource recovery was 8/9 and 7/9, but only record was
inside the 0.15–0.80 band in A and only tuple in B; replicated two-shape support
was absent.

The trajectory-state contrast changes strategy. Every one of 72 failed-test
cases reached a fully correct patch, and the four terminal misses were later
regressions. Without test output, inferred-contract rejected cases produced
0/54 fully correct first patches; none of all 72 rejected trajectories read the
visible tests before first patching, although 64/72 eventually reached correct
workspaces after later evidence. Read: the missing unit is active specification
acquisition before proposal, not post-failure semantic revision. The warranted
successor should counterfactually pair nearly identical issue/source states
with flipped public evidence, teach evidence inspection then evidence-faithful
first patches, replay the full conditional loop, and beat matched replay and
sample-more on held-out evidence channels before Menagerie.

## qwen35_4b_universal_curriculum (2026-07-13 — First pilot specialization negative)

The inherited designed-curriculum scaffold was not admissible: 16/600 induction
traces contradicted their answers, at least 33/600 nominal two-step rules
collapsed to one primitive, its smoke command was dead, its shell failed open,
and the advertised run had never started. A replacement deterministic 13-skill,
six-surface generator now enforces executable truth, induction
query-identifiability, genuine dead ends, behavioral depth, byte determinism,
and zero tokenizer skips.

The first frozen arm continued C53 `blend` for one epoch on 800 designed rows.
It installed locally: fresh synthetic accuracy rose 0.500->0.692, parse rate
0.615->0.962, and cap contacts fell 10/26->1/26. Firewall-clean paired
quick@1024 (seed 78131, merged qwen_vllm) rejected universality. Candidate was
0.3073 versus base 0.1667 (+0.1406) and blend 0.4458 (-0.1385); chronicle and
siftstack each gained +0.75 over base, but rites, stockade, and warren each
fell -0.125. Six families were positive, one zero, three negative.

The from-base factorial arm then co-trained all 800 designed and 2,240 replay
rows. Its effective-batch-8 recovery completed 3,040/3,040 rows with zero skips
and loss 1.366. On prospectively frozen local seed 88002 it reached 0.692
accuracy and recovered routing to 2/2, but parse rate was 0.846 (<0.90) and cap
contacts were 4/26 (>2); induction and execution were each 0/2. It failed two
local gates, so benchmark seed 78132 remained sealed.

Read: human-designed executable supervision has real held-out signal, but a
designed-only continuation specializes and displaces the mature broad policy,
while from-base replay co-training does not reliably install concise execution.
Local accuracy is not a retention or emission proxy. The result-separated
`qwen35_4b_universal_replay_anchor` successor now tests the more direct
integration repair: low-rate mature warm start with replay in every optimizer
window versus a matched replay-only refresh. No shared claim changes from this
negative parent factorial.

## qwen35_4b_universal_replay_anchor (2026-07-13 — Designed arm negative; replay anchor advances)

Both matched 1,520-row continuations started from C53 `blend`, used 190
effective-batch-8 steps at `1e-5`, and were explicitly merged. The candidate
substituted 400 truth-audited designed rows for 400 replay rows; the mechanism
control used replay throughout and received 17.3% more forward-token compute.
The candidate passed its frozen synthetic gate at 0.731 accuracy, 0.962 parse,
one cap contact, and zero feasible-route abstentions.

Firewall-clean paired quick@1024 seed 78133 rejected the designed candidate.
`warm_union` scored 0.4238: +0.2488 versus base, -0.0172 versus `blend`, and
-0.0613 versus replay refresh. It regressed `rites` -0.125 below base and
strictly improved only five families. Every universality rule except positive
aggregate failed.

The replay-only control is the strategic result. `replay_refresh` scored 0.4851,
+0.3101 versus base and +0.0441 versus `blend`. All ten family deltas were
nonnegative and eight were positive; `rites` and `sirens` tied base. This is not
a universal-feature claim, because strict all-family lift and replication are
absent. It does show that C53's broad replay policy was not saturated and that
replay is an active capability intervention, not a neutral retention control.

Read: the arm with a 26% designed substitution still failed broad retention even
under a low-rate replay-anchored warm start. Because replay refresh had 17.3%
more forward-token exposure despite matched steps, the failure rejects this arm
but does not isolate designed content as the cause of the whole gap. The next
result-separated test should start from the authenticated replay-refreshed policy,
reduce designed density by an order of magnitude, and match both optimizer steps
and forward-token exposure against replay continuation. Require retention of all
eight observed gains and strict lift of every family; do not tune on seed 78133
or reuse this directory.

## qwen35_4b_universal_low_density_token_match (2026-07-13 — Exact-token local negative)

Three 1,520-row continuations started from the authenticated replay-refresh
adapter and received exactly 190 effective-batch-8 updates and 1,429,053 forward
tokens. A common 1,440-row replay core occupied identical slots. The 0-, 40-, and
80-row designed doses replaced two, one, or zero independently token-matched
40-row replay blocks. All three trained with zero tokenizer skips.

Fresh local seed 88004 rejected every arm before merge or benchmark. Replay repeat
scored 0.500 accuracy, 0.538 parse, and 13 cap contacts; designed40 scored
0.500/0.538/12; designed80 scored 0.538/0.615/10. The inherited replay-refresh
anchor was 0.538/0.577/11. Every candidate passed the feasible-route abstention
check but failed the frozen accuracy ≥0.65, parse ≥0.90, and cap-contact ≤2
requirements. The promotion receipt contained no eligible arm, and benchmark seed
78134 remained unconsumed.

Read: exact forward-token parity removes the prior compute-dose ambiguity at these
low densities. Forty or 80 designed rows are insufficient to install concise local
execution from the strong replay anchor. The 80-row arm directionally improves
parseability and cap behavior over replay repeat, but only ties the inherited
anchor's accuracy and remains far outside the gate. This does not measure broad
retention and does not reject intermediate doses or termination-focused mechanisms.
A successor must use a new directory and fresh seeds, preserve exact-token replay
controls, and pass a prospectively frozen local gate before any benchmark event.

## qwen35_4b_universal_mid_density_token_match (2026-07-13 — 160-row near miss; 240-row reversal)

Three 1,520-row continuations started from the authenticated replay-refresh adapter
and each received 190 effective-batch-8 updates and exactly 1,405,510 forward tokens
with zero skips. A common 1,280-row replay core occupied identical slots. Replay
repeat retained three token-matched replay blocks; the designed arms replaced two
or three 80-row blocks, each covering all 13 truth-audited skills.

Fresh local seed 88005 rejected every arm before merge or benchmark. Anchor and
replay repeat each scored 17/26 accuracy, 18/26 parse, and 9 cap contacts.
`designed160` improved to 19/26 accuracy, 23/26 parse, and 3 cap contacts;
`designed240` fell back to 17/26, 22/26, and 5. Every candidate passed accuracy
≥0.65 and had zero feasible-route abstentions, but none met parse ≥0.90 and cap
contacts ≤2. The 160-row arm missed each remaining bar by one case. Promotion was
empty, and aggregate seed 78135 remained unconsumed.

Read: representative designed density has a nonmonotonic local optimum near 160
rows. Relative to exact-token replay, that arm adds two correct cases, five parsed
answers, removes six cap contacts, and shortens mean output by about 218 tokens.
Another 80 generic designed rows reverses the accuracy gain and worsens parse/cap
behavior, so further dose interpolation is not the warranted next move. A new
experiment should hold the 160-row capability mix fixed and isolate concise answer
commitment or termination with an exact-token active control and fresh local seed.
This result contains no broad-retention evidence and does not license a lower gate.

## qwen35_4b_universal_close_weight_token_match (2026-07-14 — Close-weight mechanism negative)

Three short exact-token continuations started from the authenticated `designed160`
adapter. All received 320 rows, 286,814 forward tokens, 40 effective-batch-8
updates, and zero skips. The active control replayed only incumbent data. The two
target arms shared byte-identical fresh execute/induct rows; their sole assigned-loss
contrast was weight 0.2 versus 1.0 on the natural `</think>` span.

Fresh local seed 88006 rejected both treatments before merge or benchmark. The
immediate parent scored 16/26 accuracy, 20/26 parse, and six cap contacts; replay
repeat scored 14/26, 18/26, and eight. Ordinary target training scored 15/26,
23/26, and three, while close-weighted training scored 16/26, 23/26, and three.
All arms passed the route-abstention check. Close weighting missed each frozen
numeric bar by one case/contact, promotion was empty, and aggregate seed 78136
remained sealed.

Read: fresh target data improves emission relative to parent and replay, but the
byte-identical contrast rejects higher close-span loss as the cause. Ordinary and
close-weighted arms have identical parse and cap metrics, both remain 0/4 on the
targeted execute/induct cases, and close only adds one non-target abstention win.
Its parent-relative accuracy tie is a three-kind-for-three-kind redistribution, not
a generalized install. Retire close-span dose tuning. A successor needs a different
bounded-computation/canonical-answer commitment mechanism, fresh seeds, an active
replay control, and the unchanged local gate before any broad evaluation.

## qwen35_4b_universal_search_scaffold_token_match (2026-07-14 — Staged-search mechanism negative)

Two short continuations started independently from the authenticated `close_xi`
near-miss. The candidate replaced 80 of 120 variable replay rows with 16 each of
executable apply, fit, reject, execute, and two-branch search lessons. The active
control used replay only. Both arms had 320 rows, exactly 286,814 forward tokens,
40 effective-batch-8 updates, ordinary thought/close weights, and zero skips; 200
replay slots were byte-identical.

Fresh local seed 88007 rejected the scaffold before merge or benchmark. Parent,
replay, and scaffold scored 18/26, 16/26, and 16/26 correct; every arm parsed 23/26
and contacted the cap three times. Scaffold was 0/2 execute, 0/2 induct, and 0/2
probe, versus parent 1/2, 1/2, and 2/2. It failed accuracy, parse, cap, execute, and
induct checks; promotion was empty and aggregate seed 78137 remained sealed.

Read: separately supervising canonical two-operation search substates does not make
them reusable at the deployment interface. Scaffold gains two cases and loses four
versus parent; mean output lengthens to 520.5 tokens from 434.2. On both execute
misses it computes the correct final state in visible thought but over-explains to
the cap, while both probe regressions show damaged independent simulation/scoring.
Do not add more canonical two-op/two-branch lessons. A successor must be
result-separated and prospectively test variable-depth natural-language state tables,
hypothesis scoring, and verified answer commitment under fresh seeds and exact-token
replay control.

## qwen35_4b_universal_state_table_compiler_token_match (2026-07-14 — State-table mechanism negative)

Two short continuations again started independently from authenticated `close_xi`.
The candidate replaced 80 of 120 variable replay rows with 20 each of variable-depth
natural-language execution tables, independently recomputed hypothesis scores,
first-error repair, and verified commit. Candidate and replay each used 320 rows,
exactly 286,814 forward tokens, 40 updates, zero skips, and 200 position-aligned
identical replay rows.

Fresh local seed 88008 rejected the candidate before merge or benchmark.
Parent/replay/candidate scored 19/16/16 correct, parsed 23/21/22, and contacted the
cap 3/5/5 times. Their execute+induct+probe subtotals were 4/6, 2/6, and 1/6.
Candidate was 0/2 execute, 0/2 induction, and 1/2 probe; it failed five absolute gates
and every strict relative check. Promotion was empty and aggregate seed 78138 remains
sealed.

Read: truth-audited natural-language tables can improve isolated computation without
installing the deployed procedure. The candidate gained one trace and one optimize
case versus both controls and computed one state semantically correctly before losing
only on whitespace. But it treated a cycle declaration as an extra operation,
repeated both induction cases to cap, miscounted probe distinctness, and reached one
correct execute result without committing. The idealized traces remained off-policy
relative to actual failure prefixes. Retire another hand-authored trace surface. A
successor should use fresh parent rollouts and executable-oracle corrections at the
first observable failure prefix, while retaining exact-token replay, fresh seeds, and
the unchanged strict local gate.

## qwen35_4b_universal_on_policy_prefix_repair_token_match (2026-07-14 — On-policy prefix mechanism negative)

The successor collected 288 fresh parent rollouts and found 230 reachable failures,
then selected exactly ten from each of six failure classes. Its candidate masked the
realized parent prefix and supervised executable-oracle correction from the first
machine-observable failure. Candidate and replay independently trained 320 rows,
40 updates, zero skips, and exactly 304,313 forward tokens; 200 replay positions were
identical. The candidate carried 33,421 fewer supervised target tokens, a registered
intervention caveat. Both adapters were explicitly merged and authenticated before
one same-vLLM local event.

Fresh seed 88009 rejected the candidate. Parent/replay/candidate scored 16/18/15
correct, 24/23/23 parsed, 2/3/3 cap contacts, and 2/1/0 of six on
execute+induct+probe. Candidate was 0/2 on every target kind, failed six absolute
checks and all four strict relative checks, and had only one paired win against four
losses versus replay. It improved no per-kind count; order, probe, and trace each
lost one. Local/promotion hashes are `b4b333ca...b8c8` /
`1e048e75...f5c`; all raw hashes revalidated, no benchmark data was read, and
aggregate seed 78139 remains sealed.

Read: collecting failures on-policy fixes substrate mismatch but conditioning loss on
long realized failure prefixes does not teach the earlier decisions needed to avoid
or repair analogous fresh trajectories. Cap-boundary selection and reduced target
exposure remain coupled, so this rejects the complete matched-forward-compute recipe,
not every on-policy objective. Retire long masked failure-prefix continuation. A
successor should supervise short pre-failure decision boundaries and match nonzero
target exposure (or include an exact target-token control) before another local gate.

## qwen35_4b_universal_failure_selected_restart_target_match (2026-07-14 — Clean-restart mechanism negative)

This successor removed both registered predecessor confounds. It selected four fresh
parent failures per each of 13 skills but discarded every failed trajectory, teaching
52 truth-audited solutions from the original prompt. Candidate and replay each used
320 rows, 297,731 forward tokens, 126,796 loss-bearing targets, absolute loss mass
27,632.8, 40 updates, zero skips, and 200 aligned byte-identical replay rows. Both
arms independently started from the same authenticated replay parent, then deployed
as complete authenticated composites through one same-vLLM local event.

Fresh seed 88010 rejected the candidate. Parent/replay/candidate scored 17/16/15
correct, 21/22/25 parsed, 5/4/1 cap contacts, and 2/2/0 of six on
execute+induct+probe. Candidate was 0/2 separately on all three target kinds, missed
the 17/26 accuracy floor, and failed all four strict total/target comparisons. Local
and empty-promotion hashes are `39fe68b9...de9e` / `4c381fbd...6759`; all 78 raw
requests and model-tree boundaries authenticated, no benchmark data was read, and
aggregate seed 78140 remains sealed.

Read: clean restarts reliably changed bounded emission without installing semantic
competence. Relative to parent, the candidate produced four more parses and four
fewer cap contacts with 34 fewer mean sampled tokens, yet lost two correct tasks and
erased both probe successes. Removing the wrong prefix and matching target exposure
therefore do not make hand-authored oracle traces policy-compatible. Retire this
balanced oracle-restart package. The warranted next test is policy-supported
successful-sibling distillation: on fresh procedural tasks, train only where greedy
fails but a prospectively sampled same-model sibling is short and verifier-correct,
then compare against exact-exposure replay and matched-compute sample-more.

## qwen35_4b_universal_successful_sibling_target_match (2026-07-14 — Prerequisite stop)

The registered same-parent trial materialized 624 fresh tasks, 48 per each of 13
skills, and collected one authenticated greedy event before opening sibling sampling.
The event completed 624/624 rows and 296,259 sampled tokens at 859.6 tok/s with no
recovery or rerun. Model-free grading found 227 hard failures overall.

The prospective four-failures-per-skill prerequisite was impossible: count and route
had zero hard failures and select had two. The experiment therefore stopped
`STOP_INSUFFICIENT_GREEDY_FAILURES`; no sibling input, sibling event, training arm,
local result, or benchmark result exists, and aggregate seed 78141 remains sealed.

Read: this is not evidence against successful-sibling distillation. It falsifies the
design assumption that every universal skill needs and can supply failure-only
repair data from the current parent. A successor should treat the ten skills with at
least four failures as the residual intervention set and preserve saturated skills
with active replay and an unchanged all-skill retention gate. Reuse of the published
immutable collection is legitimate only in a new result directory with a new sampling
seed and prospective residual policy.

## qwen35_4b_universal_residual_successful_sibling_target_match (2026-07-14 — Terminal availability stop)

The residual successor inherited the immutable 624-task source and 227-failure
inventory, prospectively treated the ten skills with at least four hard failures
(225 rows), and completed its single authenticated same-parent `n=16` event at seed
66117 from published-green commit `fc5a333b`: 3,600/3,600 outputs, 2,337,087 sampled
tokens at 739.2 tok/s, no recovery or rerun.

The frozen model-free selection ran from green checkpoint `915a7c62` and qualified
855/3,600 siblings (natural stop, closed canonical thinking, exact answer, ≤768
thinking tokens; dominant rejections: over the short budget 1,527, wrong answer
1,359). Per-task availability was execute 29, optimize/probe/repair 21,
state/trace/verify 12, order 11, abstain 6 — and induct 2, below the mandatory four.
The outcome is `STOP_INSUFFICIENT_SUCCESSFUL_SIBLINGS` with zero selected rows;
inventory/receipt hashes are `60c95b7a...083e` / `d3926daf...ad01`. No training
corpus, adapter, local result, or benchmark result exists; seeds 50/88012/78142
remain unconsumed and benchmark data was never read.

Read: this closes same-parent successful-sibling mining as a universal-curriculum
source. Nine of ten residual skills supplied quota easily, so the residual/retention
separation worked; the design failed only at induct, where 46 failure tasks and 736
samples yielded two supported tasks. The parent's policy support is empty exactly at
the program's wall skill (C38/C39): what greedy decoding cannot do, temperature-0.6
re-sampling within a short-thinking budget cannot supply either. Curriculum signal
for the wall must come from designed synthetic data that does not depend on parent
policy support — per the queued bounded-computation plus canonical-answer-commitment
successor spec — not from harvesting the parent's own successes.

## qwen35_4b_universal_fresh_surface_budget_commit_target_match (2026-07-15 — Terminal local negative; positive surface-generality reading)

The bounded-computation successor trained three exactly-matched arms from the
authenticated `replay_after_close` parent (three-axis MILP: forward 1,356,964,
nonzero targets 576,718, loss mass ×5 631,326 per arm; zero deltas, zero skips)
and evaluated them in one frozen 104-task original-surface gate at seed 88,013 —
training rendered only six fresh surfaces, so the gate doubled as a surface-
transfer test.

Totals (correct/parsed/caps of 104): parent 63/87/18; replay 62/91/13;
designed_fresh 69/97/7; budget_commit 62/88/16. Mean generated tokens
515.8/534.0/357.2/396.2. `designed_fresh` passed the correct/parse/caps/
abstention bars and won ALL FOUR preregistered strict comparisons (total and the
24-row execute+induct+probe subtotal versus both parent and replay) — the
preregistered surface-generality reading is POSITIVE: the designed dose binds to
structure, not surface vocabulary, and simultaneously repairs termination
(−11 caps, +10 parses, −31% generation length vs parent). But induct was 0/8 for
EVERY arm including the parent, so the induct ≥ 4/8 floor was structurally
unpassable; no candidate promoted, and aggregate seed 78,143 is permanently
sealed. `budget_commit` was at-or-below replay on every headline number: the
bounded-scan lesson did not generalize to termination and its 40-row
substitution cost semantics.

Read: (1) the first positive mechanism reading in the universal line — the
designed dose is surface-general and think-economical; (2) the budget-commit
content lever is retired; (3) the per-kind induct floor is now known to be
unpassable for this lineage at n=8 (the C38/C39 wall made exact), so the line's
local gate can never promote a continuation regardless of treatment quality.
Successors must either attack induction with a fundamentally different mechanism
or preregister floors that are achievable given the known wall — and the
program's goal-gap forensics (menders/warren/sirens/rites, recorded in the
successor intake) point the next attack at the benchmark's actual bottleneck
rather than the local gate's hardest kind.

## qwen35_4b_goal_gap_axis_curriculum_target_match (2026-07-15 — First local promotion; aggregate pilot negative)

The goal-gap successor trained a 160-row designed axis corpus (four stuck-family
axes, public descriptions only, fresh vocabulary, executable truth) against a
three-axis exact-exposure replay control from the `designed_fresh` parent, and
became the first universal-line experiment to PASS its local gate: axis holdout
28/40 versus parent 22 and replay 18 (hygiene 9 vs 5/5, explore 7 vs 6/3,
tracefix 4 vs 3/2, protocol tied at the control ceiling), with retention
byte-equal to the parent (71/95/9 of 104) while replay drifted (65/89/15).

The conditional aggregate pilot then consumed seed 78,144 (quick, tb 1,024,
four weight-authenticated composites): base 0.1085, axis_curriculum 0.4223,
parent 0.4644, replay_repeat 0.5081. The candidate beat base +0.3138 with 7
strictly positive families, 3 ties, 0 negatives — flipping warren — but lost
the aggregate to parent and replay, so the pilot gate fails and the experiment
closed per contract. The replay control flipped rites and posted the line's
highest recorded aggregate at any seed.

Read: (1) designed axis atoms INSTALL (first promotion ever; zero retention
cost) but under-convert to the corresponding quick-tier families — task-level
capability and family-level scoring are separated by more than surface
(sirens stayed exactly 0.500 despite hygiene nearly doubling locally; menders
stayed 0 despite the tracefix win). (2) Replay continuation compounds
aggregate a third consecutive time (0.4410→0.4851 at 78133; 0.4644→0.5081
here); it is the strongest single intervention this line has measured and the
presumptive parent for successors. (3) The all-families goal is now blocked by
exactly two families frozen for every arm at every seed at this tier
configuration: menders (0 everywhere) and sirens (0.500 everywhere). Successors
must either explain those two constants (instrument-level forensics from
public metadata and score behavior only) or find a mechanism that moves them;
another same-shape axis dose is not a believable next step for menders after
two failed transfer attempts (loomfix, tracefix).

## qwen35_4b_axis_replay_stack_medium_target_match (2026-07-15 — Local negative on the breadth bar; stack survival and replay-drift readings)

The stack trial retrained the inherited axis corpus from the 0.5081
replay-compounded parent against a replay-squared exact-exposure control. At
the frozen 144-task gate (seed 88,015): axis holdout candidate 24/40 vs parent
18 vs replay_squared 15 (hygiene 9/5/5, tracefix 2/1/0, protocol 8/8/3 —
tied at the parent ceiling for the second consecutive experiment — explore
5/4/7); retention candidate 64/98/6 vs 65/92/12 vs 64/86/18. Nine of ten
checks passed; the 3-of-4 kind-breadth bar alone failed, so seed 78,145 sealed
and the medium-tier pilot never ran.

Read: (1) STACK SURVIVAL — the axis install transfers across parents (+6 axis
total twice, hygiene 9/10 twice, best-in-event termination both times);
(2) REPLAY ROUND-TWO DRIFT — the second replay round degraded every local
quality number (parse 86, caps 18, axis 15/40 with wild kind variance),
so the aggregate compounding at seed 78,144 is aggregate-specific or
seed-fortunate, not a general quality gain; (3) INSTRUMENT FLAW — the protocol
holdout ties at the parent ceiling in two independent experiments, silently
tightening 3-of-4 into 3-of-3; successors must handle undetectable kinds
prospectively. Queued next (calibrated): a training-free fresh-instrument
re-adjudication of the published composites with a detectability-corrected
breadth bar and a conditional medium pilot — the mechanism evidence is
replicated, the blocker is gate noise, and the cost is merge/eval only.

## qwen35_4b_axis_stack_readjudication_medium_pilot (2026-07-15 — Corrected-bar negative; the three-replication mechanism map)

The training-free re-adjudication judged the published stack composites on a
third fresh instrument (seed 88,016) with the detectability-corrected bar. All
four kinds were detectable; required wins 3. Candidate 22/40 vs parent 15 and
replay_squared 18 — the axis-total win's third replication — with kind wins on
explore (7/3/6) and hygiene (7/5/5), a third consecutive protocol tie with the
parent (7/7/5), and a tracefix loss (1/0/2). Retention 65/98/5 vs 61/92/12 and
66/91/13. Two wins < 3: NOT_PROMOTED; seed 78,146 sealed; the medium pilot
never ran.

Read — the mechanism map across three preregistered fresh instruments:
INSTALLED: hygiene (won all three events), explore (two of three), and
think-economy/termination (caps halved in every event); the axis TOTAL won all
three events (+7, +6, +6). NOT INSTALLED: tracefix (4/10 → 2/10 → 1/10,
trending to chance — multi-formalism program repair does not take at this dose
from these parents) and protocol (tied the parent every time — the parent
already carries the skill, so the lesson is redundant dose). The corrected bar
worked as designed and the remaining deficit is CONTENT, not measurement.
Queued successor (calibrated): axis corpus v2 that keeps hygiene/explore,
replaces protocol with a lesson targeting capability the parent lacks, and
redesigns trace-repair from this line's own 432-completion-per-arm raw failure
outputs (own-experiment data, no benchmark exposure). Do not re-measure the
existing composites again; do not reuse sealed seeds.

## qwen35_4b_axis_corpus_v2_staged_repair (2026-07-15 — Kill rule fired; third-dose interference)

The forensics-driven v2 (staged repair lessons with demonstrated bounded
search; co-location-hardened hygiene; unchanged explore) trained cleanly and
met the frozen 154-task gate at seed 88,017 with normalized grading. Axis
holdout of 50: candidate 19, parent 19, replay_repeat3 25; per-kind
candidate/parent/replay: bugfind 3/0/3, bugmend 3/4/2, retrace 1/2/5, explore
5/7/9, hygiene 7/6/6. Retention 66/98/4 vs 71/98/3 vs 69/95/8. The kill rule
fired (`u_bugfind_win` and `u_bugmend_win` both false); seed 78,147 sealed.

Two program laws:

1. TRACE-REPAIR AXIS CLOSED. Two content designs (asserted search; demonstrated
   staged search built from quantified forensics) across four fresh-instrument
   events produced zero robust repair installs. The skill this axis needs is
   not installable in this model by ~30-55-row designed doses at rank-32,
   regardless of pedagogy. Any future attack requires a different mechanism
   argument (per the frozen rule, not a v3).
2. THIRD-DOSE INTERFERENCE. The third consecutive designed dose continued in
   place on one adapter lineage tied its parent on the axis total, lost the
   previously-installed explore edge, and dropped retention by five — while
   the third replay round won the entire axis holdout (25/50, explore 9/10,
   retrace 5/10). Combined with the stack trial's replay-drift reading, the
   adapter lineage is saturated as a vehicle for further designed doses:
   future doses need a fresh adapter from a clean parent, and replay
   continuation remains the strongest single broad-instrument move.

## qwen35_4b_hygiene_explore_destack_medium (2026-07-15 — Recovery confirmed; retention bands failed; the replay-interleaving law)

The de-stacking test trained the two replicated installs (hygiene 40 co-location-
hardened, explore 40) directly from the clean designed_fresh adapter against
matched replay. At the frozen 124-task gate (seed 88,018): axis holdout 15/20
vs replay 11 and parent 8, with BOTH preregistered recovery flags true (explore
7/4/6, hygiene 8/4/5) — the strongest axis result of the session. Retention:
58/93/11 vs 68/98/7 and 66/86/19 — the correct band failed against both
controls (−10/−8 vs −5), caps and parse against the parent. Not promoted; seed
78,148 sealed.

Three laws sharpen:

1. RECOVERY CONFIRMED. v2's stall was lineage interference, not content decay:
   the same lessons at the same dose on a clean lineage reinstall decisively at
   matched exposure. The escalation rule does not fire.
2. REPLAY INTERLEAVING PROTECTS RETENTION. The only retention-safe dose-two
   event (axis_on_replay: byte-equal retention) had a dedicated full replay
   round between doses; this direct dose paid ten retention points. Combined
   with every replay-refresh observation, the dose boundary is where replay
   belongs.
3. The gate architecture works as designed: it certified installs and refused
   a forgetting candidate in one event.

Queued successor (calibrated): the interleaved-replay dose — hygiene+explore
warm-started from this experiment's OWN replay_clean adapter (the replay round
already exists with receipts), same gate design at fresh seeds. That recipe
exactly reproduces the retention-safe precedent with the proven-install
content; honest gate probability is the session's highest yet.

## qwen35_4b_interleaved_replay_dose_medium (2026-07-15 — Interleaving refuted; escalation fired; dose-recipe search closed)

The direct test of the replay-interleaving retention law trained the verified
hygiene+explore corpus from the receipted interleaving replay round. At the
frozen 124-task gate (seed 88,019): axis candidate 11/20 vs parent 7 and
replay 6 — hygiene won its SIXTH consecutive event (7/2/3) — but explore lost
(4/5/3) and retention broke against both controls (59 vs 68/69; −9/−10 against
a −5 band), reproducing the direct dose's cost almost exactly DESPITE the
interleaved parent. Not promoted; seed 78,149 sealed.

Laws updated, one by refutation:

1. REPLAY-INTERLEAVING LAW REFUTED. Replay at the dose boundary does not
   protect retention. The single retention-safe dose event (axis_on_replay)
   owes its safety to something else — corpus composition (160 rows/4 kinds),
   lineage depth, or screen-seed fortune. Cross-receipt inference proposed the
   law; the preregistered direct test killed it. Record both.
2. HYGIENE IS UNCONDITIONAL. Six consecutive kind wins across every parent,
   dose size, and recipe — the single most robust installed lesson the program
   has produced.
3. ESCALATION FIRED. Per the frozen rule, the dose-recipe search is closed:
   the ~10-point retention cost of this two-lesson dose is not a scheduling
   artifact. The only funded successor in this line is a dose-vehicle
   mechanism study (adapter rank/capacity, loss weighting, dose size, or
   optimizer dynamics), with its own intake. The 160-row/4-kind composition
   difference from the retention-safe precedent is that study's first
   variable.

## qwen35_4b_dose_diversity_mechanism_cell (2026-07-15 — REFUTED_INTRINSIC: the retention trade is priced)

The escalation rule's funded mechanism cell trained the twice-verified 160-row
corpus directly from the clean parent and gated it at fresh seed 88,020 beside
three published composites. Retention correct of 104: clean_parent 70,
replay_clean 65 (−5), axis160_direct 61 (−9), hygiene_explore_direct 60 (−10 —
its known cost reproduced exactly). Preregistered verdict: REFUTED_INTRINSIC.
Axis holdout: axis160_direct best at 26/40 with hygiene 10/10 (SEVEN
consecutive hygiene wins, now perfect) and best caps (5).

The four-lifecycle retention arc closes with a priced law: at this vehicle
(rank-32 LoRA continued in place, 190 updates, LR 1e-5), designed doses cost
~5–10 retention points intrinsically. Diversity does not protect it
(this cell); interleaving does not protect it (prior refutation); a pure
replay round itself costs ~5 on a fresh screen; the sole byte-equal precedent
was screen fortune. Installs remain unambiguous throughout — the trade is
real on both sides. Successors must change the vehicle (rank, loss weighting,
update count — single variables against this same gate design) or preregister
gates that price the trade. The recipe search remains closed.

## qwen35_4b_rank_capacity_vehicle_cell (2026-07-15 — SCREEN_INSTABILITY: the guard fired; bands need calibration)

The vehicle study's first cell trained a fresh rank-64/alpha-128 adapter on
the clean-parent composite (trainer's one-argument --model-path delta;
encode_row byte-identity enforced by an AST test) and gated it at seed 88,021
beside the published rank-32 arm and the parent. Retention: parent 69, r32 64
(−5), r64 62 (−7). The r32 arm's known −9 failed to reproduce, tripping the
preregistered SCREEN_INSTABILITY guard: no capacity inference. Axis: r32 21,
r64 19 (install_preserved false), parent 17.

Consolidated across the four most recent gates, same-composite retention
deltas scatter ±3–4 points between fresh screens (r32: −9 then −5;
two-lesson: −10, −10; replay: −5; r64: −7). The 104-task retention screen's
seed noise is comparable to the ±5 band, so single-screen band adjudications
near the edge — including parts of the intrinsic-tax chain — carry real draw
noise. Standing summary: doses cost retention ~5–10 points with screen noise
~±3; no five-point band should be adjudicated by one screen.

Funded successor (preregistered branch): an eval-only retention-screen
calibration study — the published composites re-measured across several fresh
screens to size seed variance directly and set bands (or pooled-screen
protocols) that separate real effects from draws. Vehicle inference (rank,
weighting, updates) stays open until then.

## qwen35_4b_retention_screen_calibration (2026-07-15 — CALIBRATION_READ_COMPLETE: the band was ~1.2 SD wide; the tax law revises downward)

The instability guard's funded successor measured the measuring stick: the
five published composites re-run on four fresh 104-row retention screens
(seeds 88,022–88,025; 20 authenticated engine runs; zero training). The
adversarial design review corrected the estimand pre-freeze — bands govern
same-screen DELTAS versus the parent, so the calibration pools the
delta-vs-parent SD (common screen difficulty cancels; the draft's level SD
was wrong in both directions).

Readings: delta SD pooled 4.27 (per-arm 5.68/4.27/3.59/3.10; level SD 4.81
descriptive) → recommended band 9 and frozen protocol `pooled_k3`. The ±5
single-screen band every prior gate used was ~1.2 SD wide — but ±5 applied
to the MEAN of three pooled fresh screens is almost exactly 2 SD
(2 × 4.27/√3 = 4.9), so the historical band size survives as a pooled-k3
rule. All five historical single-screen tax readings (−9 axis160_direct,
−10/−10 hygiene_explore_direct, −7 axis160_r64, −5 replay_clean) fall
INSIDE their arms' pooled ± 2·SD intervals; the pooled deltas are −3.75,
−2.25, −0.75, −0.75. Same-composite single readings swing −10 to +4 across
screens; screen 88,025 ran commonly hard (parent 64 vs 67–69).

The standing law revises: designed doses cost ~1–4 retention points pooled
(not 5–10; the old figure was single-screen draws from a ±4.3-SD process).
Installs remain unambiguous; the trade is real but several times cheaper
than priced. Vehicle, descriptive only: rank-64 pooled −0.75 versus
rank-32's −3.75 (+3.0 favoring capacity, within noise) — the capacity
question stays open and is now cheaply adjudicable under pooled_k3 with
both arms already published.

## qwen35_4b_menders_sirens_tier_forensics (2026-07-15 — CONSTANTS_ARE_INSTRUMENT_ARTIFACTS: the goal gate's venue moves to medium)

The backlog's queued prerequisite ran as pure receipt analysis (2,278
committed gateway files, 356 cleaned family-score rows, zero GPU, zero
seeds, `benchmarks/` never read). The goal-gap pilot's standing claim —
menders = 0 and sirens = 0.500 for every arm at every seed at quick/tb1024
— has three committed counterexamples at the line's own instrument (base
sirens 0.375 at 78,131; candidate menders 0.021 at 78,131; replay_refresh
menders 0.125 at 78,133): the constants are item-draw artifacts of the
quick tier's 1/8-step granularity, not model walls.

The decisive tier read, from paired within-event strict-win adjudication:
the goal gate (all ten families strictly above base) passed 9 of 94
historical medium arm-events versus 1 of 84 at quick — the medium mode is
8/10 strict wins with 20 events at 9/10. Base never sits at a family
ceiling at medium (0/95 events; quick 2/82), sirens leaves its 0.500
sticking point (base exactly-0.5 in 14/95 medium events vs 49/82 quick,
spanning 0.2–0.6), and menders stays beatable (base zero in 54/95, max
0.3; treated arms reached 0.4). Near-miss blockers: menders/sirens/warren
at quick; menders/rites/warren at medium.

Honest limit carried on every reading: the nine medium passers were
gym-trained arms from the old line (trained ON menagerie-family data);
instrument feasibility is established, line transfer is not — the
contamination-free universal arms (best 0.5081 quick aggregate) have
never been measured at medium. Funded successor: that measurement — base
plus the line's best published composites, one fresh sealed medium seed,
tb1024, paired same-backend, the goal gate recorded from the same event.

## qwen35_4b_universal_medium_tier_measurement (2026-07-15 — MEASUREMENT_READ_COMPLETE: eight wins, zero losses, two ties from the goal)

The forensics' funded successor ran the universal line's first medium-tier
paired event: four published composites (trees deep-verified) on sealed
fresh seed 78,150 at tb 1,024, one-seed write-ahead ledger, base inside the
historical envelope on every family, all arms within budget. The
seed-consuming runner was hardened pre-freeze by adversarial review (the
review-verdict and code-pin checks now live at the boundary itself; a
one-byte drift of the readings evaluator trips them).

Readings: hygiene_explore 0.3379 > designed_fresh 0.3197 > replay_repeat
0.2981 > base 0.0567 — the quick ordering INVERTED (replay_repeat, 0.5081
best-ever at quick, ranks last of the treated at medium): the non-convex
tier-Pareto frontier (C54) replicates inside the universal line, and the
install carrier leads where it matters. All three treated arms took 8/10
strict family wins versus base — the historical mode — and
hygiene_explore/replay_repeat lost NOTHING: ties only at menders and rites
(both 0.0). designed_fresh's sole strict loss was warren (0.050 vs 0.067).
Sirens resolved to a strict win (base 0.4, every arm 0.6) exactly as the
forensics predicted.

Program position: the recorded goal gate is two tie-flips wide for
hygiene_explore. rites is elicitable in this lineage (designed_fresh 0.1 in
this same event; replay flipped it at quick). menders is the binding
constraint — 0 for every clean arm at both tiers on every seed except one
quick item (replay_refresh 0.125 at 78,133), while gym-trained arms
historically reached 0.3–0.4 there; the same-shape trace-repair dose is
closed by kill rule, so the successor must bring a genuinely new mechanism
argument for menders, carry rites alongside, start from the
hygiene_explore parent, and gate retention under pooled_k3.

## qwen35_4b_feedback_loop_state_chain_install (2026-07-15 — NOT_PROMOTED, split install: state-chains teach, feedback-repair fails a third time)

The two-tie install ran the full ladder cleanly (paired fresh rank-32
adapters from the hygiene_explore parent, exact zero-delta exposure,
control first, merges pinned, 12-run authenticated gate). The adversarial
review had corrected one MAJOR pre-freeze (unbounded op grammars versus
the finite uniqueness enumeration — 15 rows admitted out-of-grammar valid
fixes; every parameterized op now carries a documented legality clause and
an extended-grammar exclusion audit).

Verdict NOT_PROMOTED on three frozen bars, and the split IS the reading:
u_statechain INSTALLED — 11/20 on fresh instances, strict over parent (7)
and the strong replay control (10); narrated hidden-state tracking is
teachable at an 80-row dose (C14's state-chain law reaching the episode
protocol). u_feedloop FAILED COMPLETELY — 0/20 on fresh instances of the
four formalisms it trained 80 rows on, below both untrained controls
(1/20): repair-with-feedback is the THIRD failed pedagogy at the
menders-shaped skill (after asserted single-turn repair and demonstrated
bounded search, both closed by kill rule). Retention: candidate −3.0
pooled vs the parent (inside the revised 1–4-point tax law) but −5.67 vs
the replay control — 0.67 outside the calibrated ±5 pooled band — because
replay itself GAINED +2.67 over the parent, repeating the
replay-compounding law at the retention instrument. The pooled_k3
protocol's first live use measured event delta SD 4.08 versus the
calibration's 4.27: the instrument performs as designed. Axis total tied
replay 11–11 (ties fail). Sealed seed 78,151 was never opened and is
permanently sealed.

Program consequences: (1) extend the repair kill rule — no small designed
dose of ANY tested pedagogy (asserted, demonstrated-search,
episode-feedback) installs the repair-shaped skill; menders is open only
to genuinely different mechanism classes (scale, scaffolding, or
non-SFT levers). (2) The statechain lesson is a proven install and the
rites-relevant successor is a statechain-only dose (drop the dead
feedloop rows), which also relieves the retention pressure that came from
competing against replay's own gain.

## qwen35_4b_medium_budget_probe_measurement (2026-07-15 — BUDGET_GATE_STOP: the 8x thinking lever is infeasible at medium)

The budget probe asked whether serial-compute room alone moves the
menders/rites floors (the 9-versus-10 goal-ceiling question) and closed on
its preregistered stop at the minimum possible cost: the trusted gateway's
hard per-arm wall budget refused `base` at medium/tb8192 (safe diagnostic
`budget_gate_failed`, exit 2, no score emitted, nothing exposed) before
any treated arm ran — the frozen order ran base first precisely because
the line's quick-tier power statements flagged this risk. Seed 78,152 is
spent by the write-ahead ledger's opened record; no retry and no
lower-budget re-run are permitted inside the directory.

The delta review had corrected two MAJORs pre-freeze (movement booleans
scoped to arm/family pairs at zero in the pinned tb1024 event —
designed_fresh's rites was already 0.1 there, falsifying the original
premise; and fail-closed benchmark-implementation-signature equality
before any cross-budget contrast). Both amendments carry to any
successor.

Standing read: the budget lever survives only at intermediate settings
(tb2048–4096); base ran medium/tb1024 in 157 s but hygiene_explore was
the slowest arm (230 s), so either could bind at the gate. One further
preregistered intermediate-budget probe is the lever's last believable
test; a second stop closes the lever entirely and fixes the statechain
successor's 9/10 ceiling as the program's honest position.

## qwen35_4b_medium_intermediate_budget_probe (2026-07-15 — second BUDGET_GATE_STOP: the thinking-budget lever is closed)

The lever's preregistered last test refused identically to the first: the
gateway's per-arm wall budget rejected `base` at medium/tb4096
(`budget_gate_failed`, exit 2, nothing exposed) before any treated arm
ran. Base fits medium at tb1024 (157 s) but not at 4× or 8× thinking room
— the wall budget binds between 1× and 4× for the slowest common
denominator, and per the frozen consequence the thinking-budget lever is
CLOSED ENTIRELY for paired medium events; no further budget probes at any
setting without a new mechanism argument. Total cost of the complete
answer: two sealed seeds (78,152 / 78,153), two single-arm refusals, zero
exposed scores.

Program position after the lever's closure: menders has defeated three
small-dose SFT pedagogies AND the deployment-budget lever. The reachable
ceiling for currently-believable training paths is 9/10 families
(hygiene_explore + a rites flip). The mechanism classes that remain
believable for menders — dose SCALE (C43: partial installs were
data-limited; all three failed menders attempts were 80–160 rows) and
on-policy episode training — each need their own intake, calibration, and
kill rules. The funded branch is the statechain-only dose toward the 9/10
position; the queued divergent bet is the dose-scale cell.

## qwen35_4b_statechain_only_dose (2026-07-15 — PROMOTED locally; rites CONVERTED; the parent recorded the first 10/10)

Lifecycle 15's funded successor ran the full ladder clean and produced the
program's densest single event. Local gate (the first pooled_k3
promotion): axis 21/40 strictly over replay_ctl2 (19) and the parent
(17); pooled retention 64.67 vs 66.67/67.33 — inside the calibrated ±5
bands (−2.0/−2.67; the revised tax law priced it right); per-formalism
counts recorded (peatstove lost to both controls; muletrack floor-hard).

The medium event at sealed seed 78,154 returned three readings:

1. PILOT NOT PROMOTED: the candidate strictly beat base (0.3494 vs
   0.0800) and its exposure-matched replay control (0.3157) but lost to
   its parent by 0.017 (lockpick/siftstack/sirens gave back what rites
   gained). The parent remains the portfolio's best single model.
2. THE CONVERSION: candidate rites 0.300 versus 0.100 for BOTH matched
   controls on the same seed — the first demonstrated
   local-install→family transfer in program history. The axis→family
   under-conversion law has its first counterexample, with an end-to-end
   causal chain: designed state-chain episodes → local holdout install →
   benchmark family movement, all paired and exposure-matched.
3. THE RECORDED PASS: hygiene_explore_parent goal_gate_pass TRUE — 10/10
   strict family wins vs base (menders 0.017; warren 0.150 vs 0.100),
   zero ties, zero losses, aggregate 0.3663 vs 0.0800. The first
   all-families pass by a contamination-free arm. The frozen "9/10
   ceiling" was a draw-dependent floor-tie, exactly as the tier
   forensics predicted: menders was marginal capability plus item luck,
   never an absolute wall.

Honest scope, frozen into every document: single-item margins at menders
and warren on ONE seed. The confirmation law — independent fresh sealed
seeds plus a same-backend matched-compute sample-more baseline — governs
before any claim. The confirmation cell is the immediate funded
successor; nothing else outranks it.

## qwen35_4b_goal_gate_confirmation (2026-07-15 — AGGREGATE_ONLY: the sweep repeated once; the goal narrows to one family)

The mandatory replication of the recorded 10/10 ran clean: three
independent sealed medium seeds, both arms authenticated, every closed
ledger record carrying receipt pins, the readout refusing anything not
provenance-anchored (the review had caught and fixed exactly that gap
pre-freeze). It is also the program's first standalone-compliant cell
under the owner directive: the full six-stage lineage package (copied
datasets, fixed-seed manifest, three vendored trainer variants + merger,
the C53-era root adapter vendored with its provenance boundary stated,
rebuild_lineage.py verified in smoke).

Verdict AGGREGATE_ONLY under the frozen ordered partition. The aggregate
transfer is unconditional: 0.3287/0.3737/0.3837 versus base
0.0586/0.1122/0.0982 — strict wins on all three seeds, 4/4 all-time with
the discovery, never close. The all-families sweep replicated once: seed
78,157 passed 10/10 (two full sweeps across four independent sealed
seeds). The frozen 2/3 bar failed because 78,155 (9/10) and 78,156 (8/10)
were blocked entirely by TIES — menders at a 0.0 margin on both, warren
once (warren WON +0.267 on 78,155) — with zero strict losses anywhere in
the event.

Program position, stated exactly: the goal's primary condition is
DEMONSTRATED on two of four independent sealed seeds and NOT CONFIRMED at
the preregistered majority bar. The gate is localized to a single family:
menders, where both arms sit at zero on most draws and every tested
small-dose pedagogy plus the budget lever are closed. The funded
successor is the menders dose-scale intake (C43 precedent: partial
installs were data-limited; all failed menders attempts were 80–160
rows) with a precisely-known success criterion — any reliable nonzero
menders yield completes the gate; the zero-root lineage rebuild stays
queued as the provenance question.
