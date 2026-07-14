# Confirmation Attempt-1 Recovery Review

Attempt 1 failed before score publication because ordinary runner outputs did
not physically carry a field that the scoring harness already projected as an
empty list. This operational recovery review is deliberately separate from the
immutable preregistered `design_review.md`.

## Verdict

Proceed only after the producer schema repair passes its direct regression and
full experiment tests, the failed transaction remains terminal, current
authorization/admission/raw paths are empty, and new no-clobber receipts bind
the corrected source inventory.

## Mandatory Constraints

1. Keep the strict journal projection unchanged; repair the producer schema so
   the validator is not weakened after observing a failure.
2. Emit only the pre-existing empty-list value. Do not change sampled tokens,
   decoded text, scoring semantics, task generation, seeds, or engine geometry.
3. Cover the exact naturally closed budget path that produced 5,487 malformed
   journal rows, not merely a synthetic validator fixture.
4. Preserve the complete failed transaction, authorization, and admission as a
   terminal archive. Do not finalize, copy, or resume its generated payload.
5. Treat the sealed blocks as unpeeked only because inspection was restricted
   to marker metadata, file hashes, and output key-presence counts. Any prompt,
   gold, text, item score, or aggregate inspection would instead require fresh
   block seeds.
6. Remove the archived authorization/admission from their live no-clobber
   paths, then issue fresh receipts bound to the corrected source inventory.
7. Start the campaign from an empty current raw/score tree and rerun both blocks
   in full. No archived generation receives compute credit or evidence status.

The implemented repair satisfies items 1–3 and all 212 experiment tests pass.
The terminal archive and no-peek boundary satisfy items 4–5. Fresh admission
and the empty-tree full rerun in items 6–7 remain runtime gates.

## Authorization Status

The fresh controls authorization now passes and hashes to
`2b9b86aa76bfb87169a2c70313f967f20c13a09e62fbab25069120e29f0ef9f1`.
It binds the restored frozen design review, unchanged 13-arm map, and corrected
runner in control inventory `690a5b5e…`. Item 6 still requires fresh global
admission; item 7 remains wholly unstarted.
