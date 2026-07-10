# Qwen3.5-4B verified-macro capacity-fit vLLM rerun report

## Summary

**Stopped infrastructure status:** the fresh capacity-fit K=4 probe at think@49,152 passed its live
KV-cache fit check but failed the registered content-blind termination gate. The subsequent
think@61,440 K=4 probe passed preflight, then was manually stopped before any result when a
CUDA-graph capture-geometry mismatch was identified. Its preflight-only artifact is preserved. No
decoded or scored content has been inspected, and no macro claim is eligible.

The key correction is to treat long-context inference as a joint context-and-concurrency problem.
The direct parent's larger thinking allowance retained `max_num_seqs=64`, which did not
conservatively fit 64 maximum-length contexts in the observed KV cache. This follow-up uses 19
sequences at 49,152 thinking tokens or 15 at 61,440 and requires the newly constructed vLLM engine
to prove live block-rounded cache fit before generation.

## Research program fit

This experiment belongs to `operator_and_skill_inventories`. It does not change the macro mechanism
or claim that designed macros improve capability. It repairs the inference-validity gate needed
before the frozen base/designed smoke can say anything about the interface.

## Method

- Only `Qwen/Qwen3.5-4B` at the pinned revision, through the byte-identical local vLLM runner.
- Frozen 12-task v2 smoke, base and designed libraries, demonstrations, prompts, sampling, and
  semantic rules.
- A new K4 base termination probe at each rung, permanently excluded from semantic selection.
- New K12 base and designed arms at the same first adequate rung.
- Live KV token capacity, block rounding, context, and worst rendered prompt checked before rows.
- Content-blind termination selection followed by full receipt/history re-verification before
  decoded text is parsed or inspected and before hidden task data is loaded.
- Fresh external namespace with receipt-last commits, checkpoint catalog, fixed lock, and strict
  crash reconciliation.

These v2 tasks are frozen and previously unscored under this independent protocol, but are not
model-unseen: predecessor diagnostics already called the same prompt identities. No predecessor
output is imported or scored.

## Results

### Model-free gates

- Frozen protocol and exact record identities: pass.
- Unit tests: 37/37 pass.
- vLLM runner byte parity with direct parent: pass.
- GPU generation: 49k K=4 complete; 61k K=4 stopped before result.
- Termination selection: 49k rejected; no 61k termination result exists.
- Semantic smoke: ineligible until a complete K12 matrix passes termination.

### Scientific result

The 49k probe's live capacity audit passed before generation:

- live KV capacity: 997,888 tokens;
- cache block size: 528 tokens;
- capacity-fit concurrency: max-seqs 19;
- block-rounded demand: 963,072 tokens; and
- remaining margin: 34,816 tokens.

Its complete receipt binds 12 records × K=4 = 48 samples. Every sample ended by stage-one length,
was force-closed, and contacted the reasoning boundary. The frozen token-ID periodicity rule found
37 exact periodic loops, with periods recorded in
`analysis/scientific_smoke_49k_termination_audit.json`; the other 11 contacts remained unresolved.
Nine answer stages reached the 512-token limit. Thus unresolved (22.92%), loop (77.08%), and answer
limit (18.75%) rates each fail its registered threshold. This rung is rejected before decoding or
scoring.

The probe sampled 2,364,643 tokens in 5,012.451 seconds (471.754 sampled tokens/s). It was 19.6%
slower than the predecessor's max-seqs-64 diagnostic at 586.471 tokens/s, but this is not a clean
test of cache-fit overhead. Max-seqs 64 initially admitted all 48 logical sequences and supplied a
captured width-48 decode graph. Its 995,328-token cache could hold only about 20,736 context tokens
per sequence at that width, so later cache pressure could preempt requests and recompute prefixes.
Max-seqs 19 bounded worst-case cache demand, but vLLM resolved its requested CUDA-graph maximum of
19 to `[1, 2, 4, 8, 16]`, leaving decode widths 17--19 uncaptured. The two probes differed by only
1,977 sampled tokens (less than 0.1%) while wall time differed by 24.2%, and sampled-token throughput
does not count recomputed prefix work. The comparison therefore distinguishes neither actual model
compute nor a throughput optimum.

This is a termination/provisioning result only. It is not evidence for or against verified macros,
and it does not authorize semantic analysis.

### Clean 61k interruption

The fresh max-seqs-15 61k invocation passed its live preflight before generation. Fifteen active
sequences required 950,400 block-rounded tokens from 997,888 live KV-cache tokens, leaving 47,488
tokens of margin with 528-token blocks. The preserved preflight is 7,053 bytes with SHA-256
`a2a3ef1f4ba9e68909374460030bc947712f10a488870a0db1bf081e368b8a5a`.

The installed vLLM 0.24 source then exposed a second geometry constraint: the runner requested a
maximum CUDA-graph capture size of 15, while the default capture ladder contains 1, 2, 4, and then
multiples of 8. The effective maximum was consequently 8, and decode widths 9--15 used no CUDA
graph. The process was manually stopped before rows, runner metadata, or a receipt existed. The
catalog correctly preserves only `probe/think_61440/base.preflight.json` and marks the bundle
`preflight_only`. No 61k token counts, timing, termination verdict, decoded output, or score exists;
the interruption cannot be interpreted as either a positive or negative result.

## Controls

The implementation rejects the predecessor root and old max-seqs-64 metadata, proves K4 cannot
resolve a selected K12 arm, enforces same-rung runtime/engine identity, disallows phase carryover,
recomputes the entire first-adequate history, and writes an explicit unselected terminal history if
61k fails. Tests also establish that changing decoded answers or correctness fields cannot change
termination selection.

## Oracle versus deployable evidence

If termination clears, sample selection is visible-only: earliest valid candidate among those with
maximal visible score. Hidden examples grade only after that index is frozen. Oracle hidden coverage
is a nondeployable smoke diagnostic and cannot support a capability claim.

## Interpretation

The current learned lesson is operational: increasing `max_model_len` or generation allowance is
not sufficient for long-context batched inference. `max_num_seqs` must be fitted to live
block-rounded KV capacity at the intended per-sequence reserve, and CUDA-graph capture sizes must
cover the intended decode concurrency. Both resolved values are part of the inference protocol.
Maximum cache-safe concurrency is not automatically a throughput optimum, but the present timings
cannot quantify that tradeoff because cache overcommit and graph coverage changed together.

The eventual branches are preregistered:

- live fit failure means infrastructure mismatch;
- terminal 61k termination failure remains setup-inconclusive;
- termination pass plus semantic failure indicates an interface/induction ceiling under the frozen
  protocol; and
- smoke pass licenses a separate matched-compute mechanism experiment, not a direct positive claim.

## Next action

Do not resume the preflight-only 61k bundle. A capture-aware rerun changes Ada batch geometry and
must begin with fresh rows under a separate follow-up protocol. This experiment remains stopped
with one rejected 49k termination probe and one preserved 61k preflight; no K12 matrix is authorized.
