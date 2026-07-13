# Attempt-2 adversarial incident review

Verdict: **BLOCK** any continuation or replay in this experiment. Continue only
through a separately registered successor with fresh request identities and a
fresh seed domain.

The independent read-only review verified:

- the v2 live preflight is durably present at SHA-256
  `16961d80437835fdfdcad0fa78d482a847c00a243d25fa0a5d86062dc5fcea25`;
- the sole invocation receipt is `suffix_materialized.started.json` at
  `f6aa447b1936fac397a353fc13183f008e31884b5006ed7fc50ac78deed3387a`;
- the runner intentionally ignores tokenizer EOS `248046` and stops on model
  EOS `248044`;
- the authenticator incorrectly required both metadata fields to be `248044`;
- the confirmed exception occurred only after all 52 logical rows and metadata
  returned in memory and passed the earlier count/runtime/sampling/thinking
  metadata checks;
- authentication had not yet reached individual token-row validation;
- raw rows and metadata are written only after authentication, so zero sampled
  output bytes survived the process;
- no later invocation began; and
- the experiment's own transaction invariant classifies a partial `STARTED`
  state as terminal and prohibits deletion or resampling.

The review requires preserving the v2 incident bytes without fabricating
`raw`, `metadata`, or `COMPLETE` artifacts. A successor must authenticate the
exact `248044`/`248046` pair pre-outcome and use fresh task/record identities
and sampling seeds for every paired arm. Reusing the frozen IDs and seeds would
resample the terminal invocation.

The reviewer edited no files, made no model or GPU call, and accessed no
benchmark or prohibited data.
