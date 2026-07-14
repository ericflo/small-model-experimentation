# Calibration Implementation Review

## Status

`HOLD_LIVE_CALLS` pending a second independent adversarial review of the exact
pushed, green repair commit. No model load or generation is authorized.

## First adversarial review

The first review examined commit
`0533fb01fe53f562f9aee2ecbcb34dd469ea21e7` and returned
`HOLD_IMPLEMENTATION`. It identified eight release-blocking findings:

1. Coordinated output/metadata rewrites could forge seeds and token costs.
2. Thinking cells were pair-consistent but not directly bound to each exact
   persisted thought-source row.
3. Runner sidecars were mutually consistent but not absolutely bound to the
   live preflight engine and runtime.
4. A Markdown substring and a hard-coded verdict could mint a lock without a
   machine-readable review receipt for an exact commit.
5. The pre-import bootstrap trusted a lock-controlled superset allowlist and
   an arbitrary local implementation commit.
6. Frozen mechanics blobs were checked against their original commit rather
   than the current live `HEAD`.
7. The recorded live preflight had no exact schema and did not compare its
   runtime with the loaded runner runtime.
8. A failed lock contender could unlink the lock path and let a third process
   lock a new inode while the original holder was still running.

## Repair requirements

The repair must remain model-free, close all eight findings with mutation or
contention tests, pass the full repository gates, and be pushed to `main` with
both required workflows green. A second independent reviewer must then name
that exact commit and return `PASS_IMPLEMENTATION`.

Only a canonical
`reports/calibration_implementation_review.json` receipt committed after that
review may authorize lock publication. The receipt and report must name the
exact reviewed commit; reviewed runtime bytes must remain unchanged; the
receipt commit, release commit, and eventual lock commit must all be published
and green. Until those conditions are machine-authenticated, this report is a
hold, not an authorization.
