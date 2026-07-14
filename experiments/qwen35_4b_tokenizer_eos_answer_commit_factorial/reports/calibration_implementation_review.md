# Calibration Implementation Review

## Status

`HOLD_LIVE_CALLS` pending a fourth clean adversarial review of the exact
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

## Second adversarial review

The second review examined exact published-green commit
`462bd06922a338f841f7f20e365638f8709d64e4`. Its boundary/scoring audit passed,
but the combined verdict was `HOLD_IMPLEMENTATION` on four additional release
blockers:

1. Shadowable standard-library imports ran before the bootstrap under normal
   script-directory and `PYTHONPATH` semantics.
2. Lock, missing, and malformed stages could reach local imports without the
   review/release bootstrap.
3. Live-preflight runtime omitted exact Git dirty-state transition and tracked
   lock-to-live-to-current ancestry checks.
4. Absolute bundle attestation omitted adapter and RNG-isolation metadata.

The prospective repair requires isolated `-I` execution before shadowable
imports, a review-authenticated lock-publication bootstrap, exact clean-to-dirty
runtime transition plus lock ancestry, and adapter/RNG binding for all five
bundles. A third review must test these paths without reading construction,
mechanics, ciphertext, key, benchmark, or hidden artifacts.

## Third adversarial review (disqualified)

The third review examined exact published-green commit
`28855d21cfed8c96ecfb85106640a78d9efd4520` and returned
`HOLD_IMPLEMENTATION`. It found four reproducible blockers:

1. `-I` still inherited an attacker-controlled `PATH`, and a synthetic fake
   `git` executed before provenance authentication.
2. Adapter presence and RNG attestation types were not exact; missing adapter,
   Boolean/integer aliases, and floating numeric aliases authenticated.
3. A one-round PASS receipt could satisfy a release requiring three valid
   adversarial rounds.
4. The runner's Mamba-cache CLI re-exec did not preserve `-I` or the sanitized
   environment.

This review is permanently disqualified from PASS-receipt provenance: its
search exclusions were incorrectly scoped and traversed protected experiment
paths, visibly exposing mechanics-public/prepared content and potentially
scanning mechanics audit/ciphertext. It did not access `benchmarks/` or call the
model/GPU. Its synthetic failures remain useful repair evidence, but it does
not count toward `adversarial_review_rounds`.

The next review must be fresh and clean. The machine receipt requires at least
three valid independent rounds; with the disqualified attempt excluded, a clean
fourth attempt would be the third countable round.
