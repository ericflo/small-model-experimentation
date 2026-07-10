# Qwen3.5-4B verified-macro capacity-fit vLLM rerun log

## 2026-07-10 — experiment split and protocol freeze

- Created a new experiment rather than changing the result-bearing direct parent.
- Copied the frozen tasks, demonstrations, libraries, macro DSL, model harness, and vLLM runner.
  The six byte identities are recorded in `data/source_provenance.json`; the runner remains
  byte-identical at SHA-256 `fd9972bdcb3a9e8b9841b45ed8e2849017a6e80b601e924817cdaaa5144b8782`.
- Imported no predecessor model-output artifact. Extracted only the ordered prompt ID, input hash,
  rendered-prompt hash, and prompt-token count from predecessor preflights into
  `data/prompt_manifest.json`.
- Corrected the research language: these are frozen, previously unscored v2 smoke tasks under an
  independent protocol, not newly model-unseen tasks.
- Froze the capacity-fit ladder at 49,152/19 and 61,440/15 with a 65,536-token model context and
  512-token answer allowance. The live engine, not the historical cache number, must prove fit.
- Separated termination-only K=4 probes from selectable K=12 base/designed bundles by namespace,
  role, receipt geometry, catalog ID, and selection validation.
- Added a fresh external root and a fixed nonblocking experiment lock. The predecessor root,
  parent/child aliases, symlinks, unknown files, and partial bundles fail closed.
- Added checkpointed catalog reconciliation. The only legal mutation is a registered
  preflight-only entry becoming receipt-complete; disappearing or silently appearing bundles fail.
- Added whole-history first-adequate selection and a terminal `pass:false, selected=null` record for
  61k exhaustion. The analyzer recomputes every recorded lower-rung termination metric before any
  decoded or hidden-content access.

## 2026-07-10 — model-free verification

- Frozen-protocol validation passed with the exact smoke-v2 record hashes:
  base `bd66aa64942f9e57e1fe55ae716c154ea1231480d6163f1811a07828ba364907` and
  designed `c5a6cd00d9600b7a63c8e2c132e202b25da30f30af299afb3735a8f5525d9e86`.
- All 37 CPU tests passed. Coverage includes cache-fit rejection, live-context enforcement,
  predecessor-root exclusion, old `max_num_seqs=64` rejection, same-rung runtime identity,
  K4 nonpromotion, phase transitions, receipt/catalog crash reconciliation, first-adequate success,
  terminal 61k failure, idempotence, lower-tier outcome-shopping detection, and hidden-content
  nonaccess before termination eligibility.
- No GPU engine or model call was launched from this experiment. Launch remains pending explicit
  coordination with the owner of the current GPU process.

## 2026-07-10 — 49k capacity-fit probe rejected; 61k probe started

- The fresh base-only K=4 think@49,152 preflight passed against the constructed engine's live
  cache: max-seqs 19, 997,888 KV-cache tokens, 528-token blocks, 963,072 block-rounded tokens of
  demand, and 34,816 tokens of margin.
- The last-written receipt and tracked catalog verify against all external file sizes and SHA-256
  digests. The 12 records produced 48 completions and 2,364,643 sampled tokens in 5,012.451 seconds
  (471.754 sampled tokens/s).
- Recomputed the frozen content-blind gate without exposing decoded or scored content. All 48
  samples ended by stage-one length, were force-closed, and contacted the reasoning boundary;
  token-ID periodicity classified 37 exact loops, 11 contacts remained unresolved, and 9 answers
  reached the 512-token limit. The exact periods and receipt hashes are preserved in
  `analysis/scientific_smoke_49k_termination_audit.json`.
- The 49k rung fails all three registered termination thresholds and is excluded before parsing,
  correctness, hidden labels, or any macro comparison. This is not a negative macro result.
- Strict capacity fitting did not maximize throughput: 471.754 tokens/s is 19.6% below the
  predecessor max-seqs-64 diagnostic's 586.471 tokens/s. Cache-safe scheduler geometry is necessary
  for validity but is not automatically throughput-optimal.
- Advanced exactly one rung. A fresh base-only K=4 think@61,440 probe at max-seqs 15 is now active;
  no 49k K=12 arm was launched.

## 2026-07-10 — 61k probe manually stopped pre-receipt after capture audit

- The 61k engine constructed successfully and wrote its checkpointed preflight before generation.
  The live audit passed with max-seqs 15, 997,888 KV-cache tokens, 528-token blocks, 63,360 rounded
  tokens per sequence, 950,400 tokens of demand, and 47,488 tokens of margin.
- The preserved 7,053-byte preflight has SHA-256
  `a2a3ef1f4ba9e68909374460030bc947712f10a488870a0db1bf081e368b8a5a`. The tracked catalog records
  `probe/think_61440/base` as `preflight_only`; there are no rows, runner metadata, or receipt.
- A content-blind audit of the installed vLLM 0.24 scheduler/dispatcher found a separate
  capture-configuration mismatch. The frozen runner requested `max_cudagraph_capture_size=15`, but
  vLLM's default capture
  ladder retained only `[1, 2, 4, 8]` and resolved the effective maximum to 8. Decode token batches
  wider than 8 therefore dispatched without CUDA graphs. The same issue affected the completed
  max-seqs-19 probe at widths 17--19 because its effective graph maximum was 16.
- The 61k process was manually stopped before it could produce a result. No decoded output, token
  termination classification, parse result, correctness value, hidden label, or score was read or
  inferred. The interruption is not a failed 61k termination probe and does not advance the frozen
  state machine.
- The prior 49k throughput comparison is consequently confounded: max-seqs 64 supplied a captured
  width-48 decode shape but overcommitted the KV cache, whereas max-seqs 19 fit the cache but lost
  CUDA graphs at its maximum widths. Its 1,977-token output-count difference was below 0.1%, versus
  a 24.2% wall-time difference. Moreover, sampled-token throughput omits model work spent
  recomputing preempted prefixes. Neither timing identifies the throughput optimum or changes the
  termination-only scientific verdict.
- This result-bearing experiment, its source, and its frozen protocol remain unchanged. A corrected
  capture list changes Ada batch geometry and must be tested with fresh rows under a separate
  follow-up protocol rather than by resuming this preflight.
