# Qwen3.5-4B Deep-Advantage MOPD Report

## Status

The preregistered CPU/scientific smoke, pinned-model preflight, installation
canary, fresh route qualification, five-update exact-logit locality pilot, and
all three four-round integrations pass. All matched trained controls and fixed
parameter soups also pass their artifact/training gates. The sealed capability
comparison has not run yet.

## Research Program Fit

This is the direct intervention follow-up to
`qwen35_4b_same_prefix_advantage_routing`. That experiment independently
qualified deep but stopped before MOPD because its separate quick prerequisite
failed. This new directory preserves that negative and tests only the surviving
deep mechanism.

## Method

Two fresh split-branch blocks must requalify the unchanged strict deep route on
the immutable 40/60 soup. If and only if that passes, a five-update exact-logit
pilot may authorize four rounds of deep-top-50 corrected reverse-KL MOPD with a
25% frozen-soup anchor. Matched controls move deep targets to non-advantage
states, replace deep with quick on the exact selected states, or imitate the
best deep continuation off-policy.

## Results

Preflight and installation checks pass. On the eight fixed canary prompts,
quick, deep, and soup each changed all 8 outputs relative to base; quick and
deep differed on 8/8, while soup differed from quick on 8/8 and deep on 7/8.
The source/soup gate therefore authorized fresh route qualification.

Qualification then passed. Across 384 new states and 9,216 same-prefix
continuations, the selector routed 54 states to deep (28/26 by block). Deep's
independent audit advantage over soup was `+0.16499`/`+0.12205`, pooled
`+0.14209` with one-sided 95% lower bound `+0.12297`. Against quick it was
`+0.20003`/`+0.14203`, pooled `+0.16910` with lower bound `+0.15337`. The
minimum support, both block signs, and both uncertainty gates passed.

The diagnostic quick route also passed on these fresh blocks: 29/18 routed
states, quick-over-soup `+0.08198`/`+0.17054` and pooled lower bound
`+0.10008`; quick-over-deep `+0.05378`/`+0.27883` and lower bound `+0.12129`.
This does not change the locked deep-only treatment. It strengthens the case
that a later two-teacher attempt should use the preregistered cross-fitted
direct-advantage predictor and a third untouched block rather than reusing
these outcomes.

The online locality round required three candidate batches. Across 576 failed
states, the selector found 90 deep routes; the assembler froze exactly 60 deep
capability units, 20 soup anchors, and 60 disjoint matched non-advantage
controls. Control matching was exact-cell for 57 units and family/kind for
three. The all-policy cache contains 140 samples, 35,147 active positions, and
hash-bound quick/deep/soup top-50 targets.

The locality pilot then consumed exactly 15 deep and five soup units over five
updates. Mean corrected top-50 loss was `0.05242`, below the `0.10` training
ceiling; held-probe loss improved `0.04773→0.02947` and top-50 overlap improved
`0.84840→0.85163`. On batch-of-one exact probes, centered non-target logit
drift was `0.02760`, relative entropy drop was `0.03112`, and target loss
improved `0.01293→0.01170`. All frozen checks passed and the machine-readable
authorization is `four_round_mopd`.

Seed 42 then completed all four full-dose rounds. Deep-route supply was
90/81/78/83, and every round selected 60 deep units plus 20 soup anchors,
completed 20 consume-once updates, stayed below the `0.10` mean-loss ceiling,
and non-decreased held-probe overlap. Mean corrected losses were
`0.05669`/`0.04901`/`0.04855`/`0.05404`; probe losses improved in every round.
The terminal merge receipt is
`88512a57ebb190f0392118a30258eee5fb3bc58d5d34ae04e384afc8842f9122`.
Entropy nevertheless contracted `10.28%`/`12.33%`/`8.90%`/`11.42%`. This is
not a registered full-round stop, but it is a material collapse-risk warning.

Seed 43 independently completed all four rounds. Deep-route supply was
90/60/82/60; rounds 1 and 3 reached quota after two candidate batches, while
round 2 used three. Mean corrected losses were
`0.05638`/`0.05172`/`0.05588`/`0.05130`; every round reduced held-probe mean
loss and non-decreased overlap. Round-3 probe loss improved
`0.07297→0.04417` and overlap `0.83025→0.83288`. Its terminal merge receipt is
`4af497550de22d9bbafdd9de97dd95eabeb6b16b6fa9a7516bf78c4c719d6ecf`.
Probe entropy contracted `12.49%`/`9.54%`/`10.66%`/`13.11%`, repeating the
same collapse-risk warning. The independent four-round provenance audit passes.
This establishes optimizer-seed/route-supply robustness only; it is not a
deployed capability result.

Seed 44 independently completed all four rounds. Deep-route supply was
`90`/`82`/`69`/`65`; rounds 0 and 1 used three candidate batches and rounds 2
and 3 used two. Mean corrected losses were
`0.05674`/`0.05561`/`0.04284`/`0.05147`; every round reduced held-probe mean
loss and increased top-50 overlap. Round-3 probe loss improved
`0.05601→0.02952` and overlap `0.82982→0.83459`. Its terminal merge receipt
is `33ae673db2abda3bfee69f311f5d5d5b8e1bda29fb2c1b286b3adbe514d4ba00`.
Probe entropy contracted `11.05%`/`9.53%`/`10.62%`/`7.55%`; rounds 0 and 2
therefore repeat the non-gating collapse-risk warning. The independent
four-round provenance audit passes with integration-receipt hash
`ff329ebccdc888689b6d6c985a558e66a2385aaa701babf8724d61444428bf1f`.

All controls then completed. Full-prefix non-advantage deep MOPD passed all four
round gates with mean corrected losses
`0.05393`/`0.05036`/`0.04990`/`0.04619`; probe loss improved and top-50 overlap
non-decreased in every round. The rematcher reproduced the original mapping
exactly in rounds 0, 2, and 3, and deterministically replaced the sole
zero-truncation-ineligible round-1 match. Wrong-teacher quick MOPD also passed
all four rounds, with losses `0.07040`/`0.06537`/`0.06047`/`0.06949` and the
same probe-loss/overlap directions. Off-policy best-deep-continuation SFT
completed all registered updates with mean CEs
`0.10926`/`0.11021`/`0.09851`/`0.09942` and reduced probe loss in every round.
Its frozen gate does not impose an MOPD overlap or CE threshold.

The 25%/50%/75% deep parameter soups each contain 128/128 nonzero merged LoRA
modules and an exhaustive inference inventory. The final trained-control merge
receipts are `99e4d3258f450173204466bd4a2b4f1dfadfc54d706008e6fc3944a5f7bd57f5`,
`90ba5ad70a6dede8e0181c1c05f80ffa9a0d9651b604a1cc27659a8da69df544`,
and `5f6b2c9c1d2a68001b7556c30324976c8312c3c4f170fe489496f0580853c435`.
The aggregate receipt hash is
`103ef4cc0b24d7c10666b6f0adfcd4dfae4720415c7fbbc76b681ab79162640b`.
Independent replay revalidated every canonical ledger, source binding,
adapter/merge chain, parameter mixture, and model byte. These are readiness and
optimizer-safety results only; superiority and causal routing remain wholly
reserved for sealed confirmation.

The subsequent no-clobber semantic authorization passed with canonical 13-arm
map hash `709694b7d770b5cbb09afe8b932bba3891ab4fea39c54c625fc84c5da973072d`
and receipt hash
`f4a5456844adeafd39e2e4f2a8036ed9fff2c78830b2eab9d4a7bfa1300d2278`.
The complete control-code inventory hash was unchanged on both sides of
publication. This authorizes creation of global confirmation admission; no
confirmation score exists yet.

Together, the three frozen optimizer seeds establish that the registered
four-round update can complete safely and that strict-deep route supply
persists on fresh seed-local states after the shared round-0 block. They do not
establish capability gain, causal advantage routing, superiority to any
source, router, control, parameter soup, or sample-more baseline, retention,
transfer, or composability. Those questions remain reserved for the sealed
confirmation.

The valid interpretation-only NF4/bf16 diagnostic further weakens any inference
from trainer-side improvement. Across 32 fixed consumed units and 7,970 target
positions, mean NF4 objective gain was `+0.02191`, while the explicit bf16
merges averaged `-0.000224`; gain-sign agreement was 15/32, gain correlation
was `-0.152`, and midpoint update cosine averaged `0.407`. Endpoint top-1
agreement was still 31/32. Thus endpoint similarity does not establish update
parity, and the diagnostic gives no authorization. Only sealed same-vLLM
procedural confirmation can determine whether the deployed checkpoint gained.

The exact locality measurement covers one midpoint active token for each of
the 20 consumed units, rather than every one of the 4,898 trained positions.
It therefore establishes the literal preregistered local-safety gate, not
global invariance. No procedural capability, control, sampling, routing, or
benchmark result exists yet.

Before starting controls, a fail-closed review found that the prior semantic
authorization did not fully reauthenticate the frozen quick/deep/soup source
bytes and could adopt a model mutation between authorization and global
confirmation admission. No control authorization, admission, or confirmation
output existed, so the boundary was hardened without changing a result or a
frozen scientific choice. The shared validator now requires the exact
seven-file checkpoint root and one of two frozen load profiles, authenticates
all model and receipt bytes, seals the canonical 13-arm map in controls
authorization, rechecks the full map on both sides of `ADMISSION`, and rechecks
each arm around evaluation. All provenance and transition regressions pass.

## Oracle Versus Deployable Boundary

Same-prefix verifier scores are training-only acquisition evidence. The
deployable primary must be one explicit merged Qwen3.5-4B checkpoint with no
teacher, verifier, router, or tool hidden at inference time.

## Interpretation Contract

Route, locality, training, control, procedural confirmation, and blackbox
benchmark results are distinct stages. An unreached stage is not negative and
will not be inferred from an earlier stop.

## Artifact Manifest

Large source, soup, adapter, and merged checkpoints remain external under the
paths in `artifact_manifest.yaml`. Small configs, receipts, analyses, tests,
and reports remain in git.
