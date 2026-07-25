# Depth Recurrence Probe Report

## Summary

Running four middle layers of the frozen Qwen3.5-4B a second time, inside a single forward pass with
**zero tokens emitted**, lifts forced-answer induction accuracy from 0.105 to **0.245** (n=400) on
held-out shift induction — matching what C59 measured for full chain-of-thought generation (0.235). On
out-of-family affine it goes 0.217 → **0.278**, which is notable because chain-of-thought *collapses*
there (C59: 0.020). Four adversarial controls survived on both substrates. No training; frozen weights.

This narrows C59's law. "Serial compute crosses the induction wall only via reasoning CONTENT" was
established against input-embedding hidden-state feedback and content-free filler tokens. It does not
extend to weight-shared mid-stack looping.

## Research Program Fit

`test_time_reasoning_budget` asks which ways of spending test-time compute actually buy capability in
the fixed 4B. C44 established the forward-pass induction wall is a serial-compute limit rather than a
knowledge limit; C59 then found that only generated reasoning content crosses it, which pushed every
remedy toward longer outputs and more sampling. If depth *inside* the pass substitutes, the same
capability may be reachable without paying for tokens — and a trained selective-looping variant becomes
worth building.

## Method

Forced-answer digit read, identical to C59's `forced` arm: chat template with
`enable_thinking=False`, `"Answer: "` appended, one forward pass, argmax over the ten single-token
digit ids, `use_cache=False`, greedy, bf16, eager attention, pinned revision `851bf6e8`.

Looping is implemented by temporarily duplicating entries of `model.model.layers` so the model's own
forward supplies all plumbing (rotary embeddings, per-layer-type masks). Qwen3.5-4B is a hybrid stack —
8 full-attention layers at indices 3, 7, 11, 15, 19, 23, 27, 31 interleaved with 24 gated-delta-net
layers — and masks are selected per layer *type* from `config.layer_types[i]`, positionally, so
`layer_types` is duplicated in lockstep and blocks are chosen on the 4-layer period.

Damped looping (`alpha` < 1) blends across each repeat iteration, `h <- h_in + alpha*(block(h_in) -
h_in)`, an Euler step on the layer's implied ODE — the form the May 2026 training-free looping result
uses. Only repeat iterations are damped, so k=1 is byte-identically the base model.

## Results

| substrate | arm | accuracy | balanced accuracy |
|---|---|---|---|
| held-out shift (n=400) | base single pass | 0.105 | 0.112 |
| held-out shift | **loop 12:16, k=2** | **0.245** | **0.251** |
| held-out shift | *C59 ref: real chain-of-thought* | *0.235* | — |
| held-out shift | *C59 ref: latent recurrence N=8 / filler N=32* | *0.060 / 0.095* | — |
| affine, out-of-family (n=400) | base single pass | 0.217 | 0.218 |
| affine | **loop 12:16, k=2** | **0.278** | **0.275** |
| affine | *C59 ref: real chain-of-thought* | *0.020* | — |

Damping dose-response on shift (n=200): 0.240 (α=1.0), 0.195 (0.5), 0.155 (0.25), 0.095 (0.1). The
effect scales with how much of the second pass is admitted.

Larger blocks hurt: 12:20 and 16:24 at k=2 or k=3 land between −0.070 and +0.015 of baseline.

## Controls

1. **Label-prior exploitation — ruled out.** Gold labels are skewed (majority class 0.18), so a
   degraded constant predictor would beat a chance-level baseline for free. Balanced accuracy (mean
   per-class recall, exactly 0.10 for any constant predictor) rises 0.112 → 0.251 (shift) and
   0.218 → 0.275 (affine); predictions span all ten digits with the same top-1 share as baseline.
2. **Positional coincidence — ruled out.** Every 4-layer block swept at k=2 on shift: 4:8 0.113,
   8:12 0.102, **12:16 0.245**, 16:20 0.175, 20:24 0.072, 24:28 0.117, 28:32 0.100 (baseline 0.105).
3. **Generic added depth — ruled out.** Inserting a copy of a *different* 4-layer block at the same
   stack position (identical added depth and parameter count) gives 0.000 (copy of 4:8) and 0.007
   (copy of 20:24) on shift, and baseline-level 0.085 on affine.
4. **Sample size — replicated on disjoint halves.** Shift 0.240 / 0.250 (baseline 0.085 / 0.125);
   affine 0.265 / 0.290 (baseline 0.205 / 0.230).

Coherence control: mean next-token logprob on fixed prose moves −0.021 nats at the winning arm, so the
looped stack remains a fluent LM rather than a degraded one. The α→0 continuity check verifies the
damping path (−2.592 → −2.602 against a −2.603 baseline).

### Two harness bugs that manufactured fake results first

- **Silent stack truncation.** The decoder iterates
  `for i, layer in enumerate(self.layers[: self.config.num_hidden_layers])`, so a lengthened
  `ModuleList` is cut back to 32 entries and the "deeper" model ran *fewer* real layers, dropping its
  tail. The first sweep reported a spurious +0.125 and a 6-nat coherence collapse from exactly this;
  byte-identical k=2/k=3 numbers were the tell. `num_hidden_layers` must be bumped alongside
  `layer_types`, and every arm now asserts its executed depth by counting layer invocations.
- **Cross-batch hook state.** Damping hooks counting call parity globally leaked a hidden state between
  batches (a 16-row tensor popped during the final 8-row batch). State is reset per forward pass and
  blends are shape-guarded.

## Oracle Versus Deployable Evidence

All numbers here are **diagnostic, not deployable**. The forced single-pass read is a probe chosen to
isolate forward-pass computation by removing the model's usual recourse to tokens; it is not how the
model is deployed. Nothing here is an oracle in the label-leaking sense — no arm sees the answer — but
neither is any arm a deployment protocol.

## Interpretation

The wall that C44/C59 mapped is not purely a token-content phenomenon. On these substrates the missing
computation can be supplied by *reusing the right weights at the right depth*, which is a statement
about where the model's induction machinery sits rather than about reasoning verbalisation. The
convergence with C19/C31 is the most suggestive detail: the winning block ends at layer 15, exactly
where op-type becomes maximally decodable from the residual stream, and looping past that point hurts.

What is *not* shown: that this helps generation (where CoT is already available), that it survives on
any deployable task, or that general capability is preserved beyond a prose-logprob check. A gain on a
chance-level forced read is the easiest regime in which to find an effect.

## Next Experiments

1. **Does looping STACK with generation, or only substitute for it?** Same arms with free generation
   and answer parsing. This decides whether the result is an inference-time lever or a curiosity.
2. **A real capability check** — MBPP or HumanEval through the standard harness, thinking-on — to test
   whether "coherence preserved" means "capability preserved".
3. **Trained selective looping** (Think-at-Hard-style, arXiv 2511.08577): loop only tokens a small
   decider marks hard, with per-iteration LoRA. Only worth funding if (1) or (2) is positive.
4. **Serving feasibility**, which gates any deployment claim: vLLM cannot currently serve a looped
   hybrid GDN stack, so a deployable version needs either an HF-eager serving path or engine work.

## Artifact Manifest

All outputs are small and committed under `reports/`: `recur_results.json` (block × k × α sweep with
executed-depth receipts and the α-continuity diagnostic), `verify_results.json` (shift controls),
`verify_affine.json` (affine controls). Substrate data is copied into `data/` per the
standalone-experiment directive. See `artifact_manifest.yaml`.
