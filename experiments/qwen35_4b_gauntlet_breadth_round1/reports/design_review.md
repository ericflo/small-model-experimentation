# Design review — qwen35_4b_gauntlet_breadth_round1

Three adversarial lenses (workflow), run at the scaffold stage after a tiny
GPU smoke harvest and BEFORE the full harvest/train/eval spend. Verdicts:
confounds=sound_with_fixes, stats=sound_with_fixes, mechanism=sound_with_fixes.
All must-fixes were applied before the full run; resolutions are recorded
inline. 2026-07-09.

## Lens: confounds — sound_with_fixes

### Must fix

1. **Lucky-guess/rationalization gate missing (C28).** build_sft kept any
   score==1.0 sample; kilnrite flag atoms are binary, loomfix bug-line atoms
   reach ~50% chance at L1, runeward's UNSAT arm is guessable at 25%,
   burrowmaze direction atoms draw from ~4-6 words. Correct-by-chance samples
   carry rationalized thinking into SFT.
   **Resolved:** families now emit per-item `answer_domain`; build_sft keeps
   samples from an item only if `answer_domain >= 5` OR >=3 of K samples
   verified correct (`scripts/build_sft.py` atom_examples). Pre-registered in
   gym_design.md.
2. **Turn-level filter trained on refused/malformed actions** inside
   successful episodes (horizon slack makes them common).
   **Resolved:** the episode contract now requires `Episode.last_action_ok`;
   all 7 episode families set it on every accepted/corrective branch
   (selftest-enforced: garbage action → False, oracle action → True); the
   harness records per-turn `action_ok`; build_sft drops not-ok turns.
3. **Family cap was a sort, not an interleave** — trimmed the hardest strata.
   **Resolved:** stratified (level, kind) round-robin, shortest-think-first
   within cell (`scripts/build_sft.py`).
4. **Pre-registration contradicted the config** (temp 1.0/think 1024 vs
   operative 0.8/4096, pass@6 vs K=4).
   **Resolved:** gym_design.md harvest section rewritten to the operative
   regime with the smoke-driven revision history; revisions predate the full
   harvest.

### Should fix (adopted)

- Forbidden-word screening now runs on every selftest episode observation
  (gym/base.py `_run_episode`).
- max_think_tokens=2000 vs quick's deployed 1024: justified in gym_design.md
  (medium/slow/deep deploy at 2048–4096; prefer-shortest biases low).
- Per-episode rollout cap added (max_rollouts_per_episode=2).
- Protocol-shape vs axis-gain inference rule pre-registered in gym_design.md.
- Small-latent-space caveat for L1 trained-family held-out items recorded in
  gym_design.md; eval strata never harvested (L4 atoms, L3 episodes) are
  labeled extrapolation rungs.
- bench.py per_family extraction hardened; adapter-arm LoRA-kernel numerics
  noted as the only non-weight arm difference.

## Lens: stats — sound_with_fixes

### Must fix

1. **Family cap bias** — same as confounds #3. Resolved (round-robin).
2. **No yield gate/fallback for the naturally-closed-only filter** (risk:
   <5% yield selects a weird subpopulation and silently shrinks breadth).
   **Resolved:** pre-registered yield gate — families under 100 SFT examples
   get a K=8 top-up harvest; still-starved families are flagged and the
   breadth claim scoped to represented families (gym_design.md).
3. **The 0.011 "noise floor" is a cross-backend single realization, not the
   H0 of this comparison.**
   **Resolved:** pre-registered base-vs-base null calibration on the same
   fresh seed (plus a second seed) before any adapter event; the calibration
   replaces the 0.011 figure in all inference.
4. **~25-30% power at the +0.03/two-seed bar poisons the negative reading.**
   **Resolved:** pre-registered three-way decision rule (positive ≥ +0.03
   two-seed mean with both positive AND medium ≥ +0.02; negative requires
   quick mean ≤ +0.01 AND medium ≤ +0.01 AND flat held-out-family transfer;
   otherwise inconclusive → iterate). Quick-only single-event effects below
   ~+0.06 are treated as undetectable by design.

## Lens: mechanism — sound_with_fixes

### Must fix

1. **Terminal special-token pollution corrupted scoring, episode driving, and
   SFT targets end-to-end** (runner stops on `<|endoftext|>` and generates
   through `<|im_end|>`; gym parsers kept the literal marker; verified on the
   smoke harvest: correct 40→94/240 after stripping; every episode action was
   judged malformed; 35/40 keepable SFT targets were polluted and train_think
   would have produced a double-`<|im_end|>` tail).
   **Resolved:** `gym/base.py` strips TERMINAL_MARKERS from the answer region
   in split_think (inherited by extract_answer/extract_action and by
   build_sft's split_target); marker-tolerance is now selftest-enforced
   (polluted oracle replies must score); episode families reroute raw action
   parsing through base.extract_action.
2. **Family cap** — same as above. Resolved.

### Verified clean (mechanism notes)

- Train/deploy channel identity: deploy prompt suffix ids match the trained
  rendering; target close sequence `\n</think>\n\n` tokenizes to the runner's
  exact close_ids; the -100 mask boundary property held on all probes.
- The pinned Qwen3.5 template injects NO empty think blocks into history
  assistant turns, so train-time episode contexts are byte-identical to
  generation-time contexts.
- build_sft correctly excludes force-closed atoms and force-closed/truncated
  episode turns; mid-episode turns terminated with tok.eos_token
  (`<|im_end|>`) match the model's natural per-turn channel.
- Seed namespaces disjoint (harvest 11001/21000+, gym-eval 90001/90501+,
  menagerie fresh-per-event with reuse mechanically blocked and 31337
  rejected).

### Should fix (adopted)

- train_think.py now pins MODEL_REVISION for tokenizer and model.
- Dead `sft.max_total_tokens` config key removed (train-side over-length skip
  is the operative guard).
- bench.py rewrites stored menagerie payloads to aggregate-only fields.
- README artifact list updated (this file now exists).

## Residual accepted risks

- train.* hyperparameters live as train_think.py CLI defaults rather than
  config reads (values match configs/default.yaml today; drift risk accepted
  for round 1).
- Episode SFT examples reconstruct context from stored turns; identity is
  guaranteed by construction and was verified by the mechanism lens, but any
  future harness change to message assembly must re-verify.
- K=4 smoke cells (n=8) make per-family yield estimates coarse; the full
  harvest yields are the operative numbers.
