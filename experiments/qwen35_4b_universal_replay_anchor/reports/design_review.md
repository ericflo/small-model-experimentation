# Adversarial Design Review

## Verdict

Proceed only as a result-separated integration experiment. The parent result proves
that local synthetic improvement is insufficient; retention must be a first-class gate.

## Threats and required repairs

1. **Adaptive benchmark fitting.** The design was motivated by aggregate/public-family
   deltas, so it lives in a new experiment and receives a fresh seed. Curriculum rows,
   labels, and task types remain frozen before that event.
2. **Replay can hide a null install.** A candidate that merely reproduces `blend` does
   not validate designed data. It must retain broad behavior and improve fresh designed
   tasks; replay-only is the mechanism control.
3. **Extra updates are a confound.** Match replay-only and union arms at 190 optimizer
   steps and effective batch 8, share 1,120 replay rows byte-for-byte, and replace the
   candidate's 400 designed rows with 400 additional replay rows in the control. The
   control has 17.3% more forward-token compute, so any candidate win is conservative.
4. **Warm-start displacement is already known.** Use replay in every update window and
   a fivefold lower learning rate than the failed sequential arm. Stop before benchmark
   if local retention regresses.
5. **Long replay rows can silently skip or OOM.** Freeze max length 4,096, batch 1,
   accumulation 8, exact tokenizer dose, zero skips, and fail-closed receipts. Preserve
   the parent's batch-2 `device not ready` failure as the reason for batch 1.
6. **Quick family cells are discrete.** One event is only a screen. The target is strict
   improvement of every score, but a claim requires independent seeds, paired
   uncertainty, and medium transfer.
7. **Backend or packaging drift can fake a delta.** Reserialize base and explicitly
   merge adapters through the same composite save path; require identical config,
   tokenizer, chat template, backend, seed, tier, and thinking cap.
8. **Matched-compute sampling remains the baseline.** Even a replicated SFT win is not
   complete until it beats the repository's matched-compute sampling control.

## Gates

- Corpus: inherited frozen hashes, exact nested-dose tests and executable truth tests
  green, zero tokenizer skips, no benchmark-shaped additions.
- Training: finite loss, complete nonzero adapter, exact pinned base, zero skips.
- Local candidate: on frozen seed 88,003, parse rate at least 0.90, at most 2/26 cap
  contacts, accuracy at least 0.65, and no repeated abstention on feasible route items.
  The replay-only arm is retained as a comparison, not subjected to a task family it
  never trained on.
- Pilot: candidate aggregate above base, every candidate-minus-base family delta
  strictly positive, and candidate aggregate no worse than immutable `blend`.
- Confirmation: fresh quick seeds plus medium@2,048 with strict positive mean family
  deltas and paired uncertainty; then matched-compute sampling.

## Stop rule

If the frozen candidate and replay-only control fail, record the integration result and
branch before changing dose, learning rate, curriculum mix, or substrate.
