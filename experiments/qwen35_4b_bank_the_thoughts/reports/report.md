# Bank-the-thoughts (Phase 1): banking correct decomposition PLANS beats banking ANSWERS — content-causally, via the thinking channel

## Summary

Motivated by C27 (test-time thinking on a *no-think*-banked model adds no planning — but the model was never
trained to reason). The clean test: does training on the REASONING install more usable depth-3 than training on
the ANSWER alone? Phase 1 uses **synthetic forward-decomposition plans** (`input → op1 → state → op2 → state →
op3 → output`, then code) — genuine plans, not the model's own thoughts (Phase 2 does those). Three fresh QLoRA
adapters from base, on **matched** data (identical prompt+code; only the trace differs): **A** = `prompt→code`,
**T** = `prompt→⟨plan⟩→code`, **T_corrupt** = same code with a *mismatched* plan (content-causality control).

## Result: deployability on frozen held-out depth-3 (n=80)
| cell | coverage@16 | greedy@1 |
|---|---|---|
| base | 0.000 | 0.000 |
| A = answers (no-think) | 0.200 | 0.025 |
| **T = plans (think)** | **0.325** | **0.050** |
| T_corrupt (think) | 0.113 | 0.013 |
| T = plans (no-think) | 0.013 | 0.013 |

- **Banking the PLAN beats banking the ANSWER**, and it stacks with multi-sampling: T coverage@16 **0.325** vs A
  **0.200** (greedy@1 0.050 vs 0.025). Training the model to *reason to* the solution installs more usable
  depth-3 than training it to emit the solution.
- **It is the plan CONTENT, not the think-format or extra test-compute.** T_corrupt uses the *same* thinking
  channel and the *same* code targets but a *wrong-for-the-task* plan — and collapses to **0.113, below even A**.
  Teaching correct decomposition helps; teaching plausible-but-wrong reasoning actively **hurts** (worse than
  teaching just the answer).
- **The capability is a TEST-TIME CHANNEL, not a weight-only lift.** T deployed no-think ≈ 0 (0.013): the model
  must generate its plan to solve. Banking plans installs a *reason-then-solve* skill that requires test-time
  thinking to cash out.

## What this resolves

**C26/C27 reconciled.** Those showed test-time thinking on a *no-think*-trained model adds no planning. Here,
once the reasoning is **banked** (the model is trained to plan), thinking *does* help — depth-3 deploys far
better (0.325 vs 0.200). So the earlier null was "the model was never taught to reason about this task," exactly
the confound the user flagged — not "thinking is useless for planning."

## Step-1 planning (partial)
| model | step-1 no-think | step-1 think(2048) |
|---|---|---|
| base | 0.017 | 0.100 |
| A = answers | 0.100 | 0.117 |
| T = plans | 0.000 | *(eval too slow to complete — see limits)* |

A (answers) lifts step-1 next-op ranking no-think to 0.100 (replicating C25: banking improves lookahead-distance
ranking). T's step-1-*think* ranking eval did not complete: the trained-to-plan model generates long degenerate
thinking, making the per-node ranking eval impractically slow this session. So whether banking plans installs
step-1 *lookahead* specifically (vs coverage-via-reasoning) remains **open**.

## Honest limits

- **Synthetic plans, not the model's own thoughts.** Phase 2 (the user's literal ask) rejection-samples the
  banked model's own verified reasoning; here the plans are templated from the verified op-sequences. So this
  shows "banking correct explicit decomposition," not "banking the model's own reasoning."
- **Test-compute asymmetry:** T deploys with thinking (more inference tokens) than A (no-think). The coverage
  win is partly more test-compute — but the T-vs-T_corrupt content-causality (both think, T_corrupt collapses)
  shows the *advantage over A* is the correct-reasoning content, not compute alone.
- Token-matched-A control (A trained to T's token budget) deferred; single seed; n=80 deploy / n=60 step-1;
  step-1-T-think incomplete.

## Phase 2 (the user's literal ask): bank the model's OWN rejection-sampled thoughts

Harvested 101 tasks the banked model solves with thinking (its own verified reasoning; 20% yield; **model
thinking median 3071 chars — long rambly rationalizations**). On these *same 101 tasks* with *identical
canonical code*, three matched arms (only the trace differs): **A_self** = answers, **T_self** = the model's
own thoughts, **T_synth** = an explicit forward-decomposition plan.

| arm (same 101 tasks, matched code) | coverage@16 | greedy@1 |
|---|---|---|
| A_self = answers (no-think) | 0.087 | 0.013 |
| **T_self = model's OWN thoughts (think)** | **0.087** | 0.037 |
| **T_synth = explicit plan (think)** | **0.225** | 0.050 |

- **Banking the model's OWN reasoning does NOT beat banking answers** (T_self 0.087 = A_self 0.087 on
  coverage). **Banking an EXPLICIT correct plan does** (T_synth 0.225, ~2.6×).
- So it is the **plan QUALITY**, not "banking reasoning" per se, and not test-compute: T_self *spends more*
  test-time thinking (3071-char rambles) than T_synth's 576-char plan, yet is **worse**. The model's own
  rejection-sampled "thoughts" are post-hoc **rationalizations** of an answer it already knows — inert to bank.
- Direct answer to the literal ask ("rejection-sample successful thoughts and bank those, see if it stacks"):
  **it does not stack** — the model's own thoughts add nothing over the answer. Genuine forward *plans* do.
- (T_selfcorrupt — mismatched model thoughts — was still running at write time; the core 3-arm result stands.)

## Step-1 planning ranking (Phase-2 models, capped budget 768, n=40)
| model | step-1 no-think | step-1 think(768) |
|---|---|---|
| base | 0.025 | 0.025 |
| A_self = answers | 0.075 | 0.075 |
| T_self = model's thoughts | 0.025 | 0.050 |
| T_synth = explicit plan | 0.000 | 0.025 |

**None of the Phase-2 models install clean step-1 lookahead** — all ≈ chance (0.031), *including* T_synth which
deploys best (coverage 0.225). So T_synth's deployment advantage is **coverage-via-reasoning** (it learned a
reason-then-solve procedure that reaches solutions more often when sampled), **not** installed single-
forward-pass lookahead. Caveat: the ranking probe (rank one op given state→goal) is **off-distribution** for the
plan-trained model — it was trained to *generate* a full plan, not rank a single op — and it structurally
favors the answer-format models (A_self is highest at 0.075). So the step-1 probe does not cleanly measure the
plan-trained model's planning; the valid finding is the deploy coverage. Net: banking explicit plans raises the
deployable coverage ceiling, but does not install the step-1 lookahead the base forward pass lacks (C25/C26).

## Next
- Length-vs-content control (verbose *correct* plan padded to ~3000 chars): is it the rambliness or the
  post-hoc-ness that makes the model's own thoughts inert? Add token-matched A + a seed.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Adapters (~180MB each) moved out of repo.
