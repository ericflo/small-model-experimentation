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
