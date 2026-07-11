# Adversarial design review — entropy-routed think-pivot round 2

Self-review was used because the session's operating policy does not authorize
delegating to sub-agents. The review was completed before row scoring, training,
or scientific evaluation. Round-1 aggregate artifacts and the explicitly
reported exploratory 615-row logit census were available; no round-2 outcome
was available.

## Verdict

**Sound with fixes.** The initial "confident wrong turns + pull up" sketch was
too underspecified to distinguish a selector fix from an objective fix, and it
did not yet honor the sample-more or real tool-loop requirements. The frozen
design below incorporates the fixes.

## Blocking risks and dispositions

1. **Entropy storytelling.** Entropy alone cannot identify a correct branch;
   a high-entropy fork may just be noise. Varentropy is also scale- and
   temperature-dependent. **Fix:** verifier outcomes remain the direction
   label; entropy/varentropy only qualify the geometry. Both are computed from
   the frozen base at the actual harvest temperature and reported in nats.
2. **Selector/objective confound.** Training only a new positive objective on
   new rows would not say which change mattered. **Fix:** `demote` and `uplift`
   use the exact same real rows. `uplift_shuffled` isolates outcome content.
3. **Round-1 threshold reuse.** Its initial chosen-win was ~0.43, so a 0.35
   repetition threshold was meaningless. **Fix:** every selected real row
   starts with the chosen below the rejected by at least 0.5 logits; demotion
   hit rate starts at zero. Uplift is measured against reference-logit gain and
   also starts at zero.
4. **Positive-only can still perturb softmax globally.** Raising a chosen logit
   is not free. **Fix:** raw-logit reference tether remains; rejected is a
   tightly tethered non-target in `uplift`; gain deactivates at +0.5 logits.
5. **Low distinct-row dose.** The parent pool has only 615 rows, and uncapping
   its prefix tree yields 885 raw nodes—not published-scale data. **Fix:** hard
   floor 128 and matched maximum 256; below floor, stop. Any null is explicitly
   a low-dose pilot, never an impossibility claim. Sign/control/termination can
   license or kill scaling.
6. **Training-context reuse masquerading as capability.** Targeted logits will
   move by construction. **Fix:** they are mechanism diagnostics only. Claims
   require fresh procedural task seeds; repository repair is entirely held out
   and structurally unlike the training prompts.
7. **Toy single-turn evaluation.** Round 1 did not answer the user's coding-
   agent question. **Fix:** add a real iterative filesystem/test/patch/submit
   harness over fresh mini-repositories with six procedural fault families.
8. **Weak baseline.** Base greedy is not enough. **Fix:** compare the trained
   eight-turn agent with two independent four-turn base trajectories under the
   same maximum of eight model calls and 6,144 sampled tokens per task.
9. **Backend and adapter hazards.** **Fix:** every compared generation uses
   vLLM 0.24; trained arms are merged composite checkpoints and each passes a
   C49 on-vs-off behavioral gate.
10. **Benchmark leakage/spend.** **Fix:** the coding suite is generated locally;
    menagerie is run only through its CLI, only after whitebox/tool-loop gates,
    with fresh union-checked seeds. No benchmark content is read or trained on.

## Residual risks accepted

- The same parent harvest informed round 1 and round 2. That is intentional for
  a controlled mechanism follow-up; all scored downstream items are fresh.
- The positive-only objective is new, so its +0.5-logit cap is a mechanistic
  choice rather than a published optimum. The shuffled arm and fixed cap make
  the result interpretable without tuning it on outcomes.
- Six synthetic repository families are closer to coding-agent work than atoms
  but are not full software-engineering distributions. Menagerie remains the
  blackbox transfer arbiter if the mechanism earns access.
