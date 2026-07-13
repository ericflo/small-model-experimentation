# Adversarial implementation review: mechanics boundary

Reviewed on 2026-07-13 before model construction or generation. Three
independent code audits attacked prompt placement, raw-result authentication,
resume semantics, context gates, full-program coverage, and live vLLM
geometry. The implementation may proceed to a separately committed lock only
after every resolution below passes tests and deterministic preparation.

## Decision

**PASS AFTER MANDATORY REPAIRS.** No Qwen model was loaded and no scientific
outcome was available during review. The repairs strengthen or correctly
implement the registered mechanics gate; none lowers a threshold.

## Findings and resolutions

1. **Raw rows could be fabricated or detached from prepared prompts.** The
   analyzer originally trusted raw IDs, metadata, and text. It now requires
   exact row count/order/uniqueness, byte-equivalent prepared metadata, exact
   prompt-token hashes and decoded prompt hashes, the custom thinking channel,
   deterministic stage seeds, one output, and complete runner metadata.

2. **Scored text could differ from completion token IDs.** Authentication now
   decodes the recorded completion IDs with the pinned tokenizer and requires
   exact text equality. It validates natural and forced-continuation schemas,
   retained thinking, injected close IDs, EOS trimming, every token count,
   prompt-length arithmetic, and aggregate runner counts. Extra sampled
   `</think>` tokens are preserved as genuine outputs and recorded
   descriptively rather than misclassified as artifact corruption.

3. **The implementation lock accepted partial maps and unpublished commits.**
   A code-defined allowlist must match exactly, with no missing or extra path.
   Symlinks and repository escapes fail. Current bytes and blobs at the
   implementation commit must have the registered hashes. The lock itself must
   be tracked, `origin/main` is fetched, and design, amendment,
   implementation, HEAD, and `origin/main` ancestry must agree.

4. **Control padding changed the continuation boundary.** Duplicate and
   placebo filler tokens are now inserted before a preserved final newline.
   Every matched row asserts equal injection length and equal terminal token ID
   across systematic, deranged, duplicate, and placebo arms.

5. **The result parser did not enforce its exact terminal-line ABI.** It now
   full-matches one `RESULT: [...]` answer after the final `</think>`, after
   stripping only approved runner terminal markers. Arbitrary extra answer
   text fails parsing and contributes to the registered interface gate.

6. **A globally passing adherence rate could hide one failed context.** Each of
   four contexts now independently gates systematic execution (`.75`),
   systematic candidate adherence (`.60`), deranged supplied-operation
   execution (`.60`), and systematic-minus-deranged registered execution
   (`+.35`). The summary preserves every numerator, denominator, threshold,
   check, and context decision.

7. **`candidate_adherence_min` was dead configuration.** It is now an explicit
   aggregate and per-context systematic gate, separate from the stronger
   execution threshold.

8. **The first full-program ceiling contained only four parameter-free cases.**
   The pre-model amendment freezes eight cases: four parameter-free and one
   each from `add_k`, `mul_k`, `take_k`, and `rotate_k`. Qualification requires
   `.50` visible pass both overall and within the parameterized stratum, plus
   `.90` parse and at most `.05` cap contact. With no candidate-blind program
   comparator, this remains explicitly non-causal reachability evidence.

9. **Program rows lacked the primary arms' integrity checks.** Program requests
   and public rows are now independently rebuilt, then authenticated under the
   same ID/meta/prompt/seed/model/runner/sampling/engine/count rules as primary
   mechanics. Exactly one output is required before parsing.

10. **No live KV-capacity receipt was preserved.** After engine construction
    and before generation, the code records live block size, KV-token capacity,
    Mamba blocks, cache concurrency, model dtype/length, scheduler geometry,
    DP/TP/world size, and resolved CUDA graphs. It block-rounds the largest
    prompt plus full reserve and deliberately requires the frozen maximum
    active width to fit without preemption. This conservative no-preemption
    rule is now explicit in the amendment.

11. **A partial run could force resampling or become unauditable.** Each of five
    invocations uses an immutable receipt-last state machine. A valid complete
    invocation is skipped. `STARTED` plus an authentic raw/metadata pair can be
    finalized without generation. `STARTED` alone is ambiguous and terminal;
    it can never trigger a second call. An OS file lock excludes concurrent
    runners.

12. **Runtime identity was incompletely checked.** Live execution now pins
    vLLM `0.24.0+cu129`, verifies every `==` package in the repository lock,
    checks the environment-lock hash, and requires exact engine arguments,
    bf16, DP=TP=world-size 1, synchronous scheduling, prefix caching off, Mamba
    mode `none`, and the registered full-decode CUDA-graph list.

13. **Prepared semantics were protected only by their own hashes.** Before
    generation and analysis, all 384 diagnostic prompts and eight program
    prompts are independently reconstructed from public inputs, the pinned
    tokenizer, operation schedule, control seeds, and deterministic task
    generator. Exact rows must match the committed prepared artifacts.

14. **Direct `--stage analyze` bypassed the lock.** Run and analyze now both
    require the tracked published implementation lock, prepared receipt, live
    preflight, all five complete receipts, and an aggregate authentication
    receipt.

## Required pre-model evidence

- deterministic prepare rerun with identical receipt hash;
- exact 96-row geometry for each diagnostic arm and eight program cases;
- full experiment-local tests, including malicious parser cases, contextwise
  failure, parameterized-ceiling failure, raw tampering, verification-only
  finalization, and ambiguous-start refusal;
- refreshed CPU smoke with `model_loaded=false` and `outcomes_loaded=false`;
- `make check`;
- a pushed design-amendment commit followed by a pushed implementation commit;
  and
- a separately pushed implementation-lock receipt whose critical hashes match
  both current files and the implementation commit.

Only after all of those pass may the vLLM engine be constructed.

## Locked prepare evidence

The final deterministic prepare was run twice with identical receipt bytes.
It records `model_loaded=false`, `outcomes_loaded=false`, 96 rows per diagnostic
arm, eight program rows, four distinct context derangements, and terminal
injection token ID `198` in every context.

| Prepared artifact | Rows | SHA-256 |
|---|---:|---|
| `systematic_requests.jsonl` | 96 | `40df4007f09493cf757bd913ed8d04eb859d2e28a7a7128897e1fe086f223e36` |
| `deranged_requests.jsonl` | 96 | `c49327b549ebc791b28075c981130eda67b86cff17b11e8c3828ace472247e1d` |
| `duplicate_requests.jsonl` | 96 | `6e212a20d210532719d197335adfdc80cd3ab70868796980e3926ccae6f08374` |
| `placebo_requests.jsonl` | 96 | `49ac003d1dfdf32be0f581cfd121fb843523dec4f0fa624a53e268a6975c2914` |
| `program_ceiling_requests.jsonl` | 8 | `16a3aad8ba0840a6b15095cb0eef0e5b78787a2424530aca8ae978ef8171b6e6` |
| `program_ceiling_public.jsonl` | 8 | `c4c49f64a03fa2a02650eb79bb65e93f9ef18b1adff5766d9e63b5d0c7a09806` |

The enclosing `preoutcome_receipt.json` SHA-256 is
`2d6b668a6d43e1bd657124c3645d85ea9996d9aaaea8f81225b97472a2f5b292`.
The full suite passes 39 tests plus 33 parameterized subtests, and refreshed
smoke remains `CPU_SMOKE_PASS`. No model was loaded and no outcome was observed.
