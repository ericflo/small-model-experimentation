# Depth Recurrence Probe

Training-free mid-stack layer looping on Qwen3.5-4B: does added SERIAL DEPTH inside the forward pass
move the induction wall that C59 showed only reasoning-token CONTENT crosses?

**Status:** in-progress · since 2026-07-25 · headline effect measured, adversarially verified, and
replicated on two substrates; generation-mode and deployable transfer still open

## Research Program

- Program: `test_time_reasoning_budget`
- Program question: which ways of spending test-time compute actually buy capability in the fixed 4B?
- Prior anchors: **C59** (serial compute crosses the induction wall ONLY via reasoning CONTENT: on
  held-out shift induction at n=200, forced single-pass 0.090, latent recurrence N=8 0.060, filler
  tokens 0.070–0.095, real chain-of-thought 0.235; on out-of-family affine, forced 0.193 and
  real-CoT **0.020** — CoT fails there entirely); **C44** (the forward-pass induction wall is a
  serial-compute limit, not a knowledge limit); **C19/C31** (op-type is linearly decodable from the
  residual stream, peaking at layer 15).

## Question

C59 ruled out *one* mechanism for adding compute without tokens: feeding the last hidden state back in
as an INPUT EMBEDDING, which the model was never trained to consume and is maximally out of
distribution. Layer looping is a different mechanism — hidden states are fed to a layer that already
consumes hidden states of exactly that kind, at the same depth in the residual stream. Does it cross
the wall? This is C59's own pre-registered next test, and the 2024–2026 literature says the mechanism
is real when trained in (McLeish et al. 2511.07384: retrofitted TinyLlama GSM8K 46.2→52.0 at
recurrence 32; Saunshi et al. 2502.17416: a k-layer model looped L times approaches kL layers), with a
May 2026 result claiming *training-free* mid-stack looping is worth +2.64pp MMLU-Pro on Qwen3-4B.

## Result: looping layers 12:16 crosses the wall without emitting a single token

Forced-answer digit read (one forward pass, **zero tokens generated**), n=400, greedy,
`enable_thinking=False` — the exact protocol of C59's `forced` arm. Balanced accuracy is mean per-class
recall, which any constant predictor scores 0.10 on by construction.

| substrate | arm | accuracy | balanced acc |
|---|---|---|---|
| held-out shift | baseline single pass | 0.105 | 0.112 |
| held-out shift | **loop 12:16, k=2** | **0.245** | **0.251** |
| held-out shift | *C59 ref: real chain-of-thought* | *0.235* | — |
| held-out shift | *C59 ref: latent recurrence N=8 / filler N=32* | *0.060 / 0.095* | — |
| affine (out-of-family) | baseline single pass | 0.217 | 0.218 |
| affine (out-of-family) | **loop 12:16, k=2** | **0.278** | **0.275** |
| affine (out-of-family) | *C59 ref: real chain-of-thought* | *0.020* | — |

Two things stand out. On shift, the looped forward pass **matches what C59 measured for full
chain-of-thought generation** (0.245 vs 0.235) while emitting nothing. On affine — the substrate where
C59 found CoT *collapses* to 0.020 — looping still adds +0.061, so the mechanism is not simply
"reproducing what CoT does".

Damping gives a clean monotone dose-response on shift: 0.240 (α=1.0), 0.195 (0.5), 0.155 (0.25), 0.095
(0.1) at n=200. The effect scales with how much of the second pass is admitted, which is what a
mechanism rather than a fluke looks like.

### Four adversarial controls, all survived, on both substrates

1. **Label-prior exploitation — ruled out.** The forced read is an argmax over 10 digits and gold labels
   are skewed (majority class 0.18 on shift), so a degraded constant predictor would "beat" the
   baseline for free. Balanced accuracy goes 0.112 → **0.251** (shift) and 0.218 → **0.275** (affine),
   with predictions spanning all 10 digits and the same top-1 share as baseline (0.20 vs 0.19).
2. **Positional coincidence — ruled out.** Sweeping every 4-layer block, only 12:16 and 16:20 help.
   Shift: 12:16 0.245, 16:20 0.175, vs 4:8 0.113, 8:12 0.102, 20:24 0.072, 24:28 0.117, 28:32 0.100
   (baseline 0.105). Affine: 12:16 0.278 is the maximum, and 20:24/24:28/28:32 fall *below* baseline.
   The effect is localised to one band, not "more compute is better".
3. **Generic added depth — ruled out, sharply.** Inserting a copy of a *different* 4-layer block at the
   same stack position (identical added depth and parameter count) is catastrophic on shift — 0.000 for
   a copy of 4:8, 0.007 for 20:24 — and merely baseline-level on affine (0.085 both). Re-running *these*
   layers is load-bearing; depth per se is not.
4. **Sample size — replicated on disjoint halves.** Shift: 0.240 / 0.250 (baseline 0.085 / 0.125).
   Affine: 0.265 / 0.290 (baseline 0.205 / 0.230).

Coherence (mean next-token logprob on fixed prose) is preserved at the winning arm (−0.021 nats), so
this is not a broken model scoring by accident. Note the convergence with **C19/C31**: layer 15 — the
last layer of the winning block — is exactly where op-type becomes maximally decodable from the
residual stream (0.99 at depth 1). Looping the block that *ends* at the most-decodable layer is what
helps, and looping past it (20:24 onward) hurts.

### Two harness bugs found and fixed first — both produced fake results

- **Silent stack truncation.** The Qwen3.5 decoder iterates
  `for i, layer in enumerate(self.layers[: self.config.num_hidden_layers])`, so a lengthened
  `ModuleList` is cut back to 32 entries: the "looped" model actually ran *fewer* real layers, dropping
  its tail. The first sweep reported a spurious +0.125 and a 6-nat coherence collapse from exactly this,
  and gave byte-identical k=2/k=3 numbers — the tell. `config.num_hidden_layers` must be bumped in
  lockstep with `layer_types`. Every arm now **asserts its own executed depth** by counting layer
  invocations and refuses to report numbers otherwise.
- **Cross-batch hook state.** Damping hooks that counted call parity globally leaked a hidden state
  between batches (a 16-row tensor popped during the final 8-row batch). State is now reset per forward
  pass and blends are shape-guarded.

The α→0 continuity check is the standing guard that damping does what its name says: coherence must
return to baseline as α→0 (measured −2.592 → −2.602 against a −2.603 baseline).

## What this does and does not establish

**Does:** on two induction substrates, weight-shared looping of one specific mid-stack block converts a
near-chance forward pass into one at or above the chain-of-thought number — untrained, on a frozen
checkpoint, with coherence intact and four adversarial controls passed. That **narrows C59's law**:
"compute-depth does not help" holds for input-embedding feedback and filler tokens, and does *not*
generalise to mid-stack layer looping.

**Does not:** (a) the gain is on a *forced single-pass read*, chosen to isolate forward-pass
computation — it says nothing yet about generation, where the model already has CoT available;
(b) no deployable task has been measured (MBPP, pi-coding-agent); (c) prose logprob is a weak
general-capability check, so "coherence preserved" is not "capability preserved"; (d) both substrates
are digit-mapping induction from the same generator family. Anything claim-bearing needs (a)–(c), and
the obvious next step is whether looping *stacks with* generation rather than substituting for it.

## Run

```bash
scripts/recur.py --n 200 --ks 2,3 --blocks 12:20,16:24,12:16 --alphas 1.0,0.5,0.25,0.1
scripts/verify.py --n 400                                                   # four adversarial controls
scripts/verify.py --data ../data/test_affine.jsonl --n 400 --out ../reports/verify_affine.json
```

Run under the repo `.venv`, inside
`experiments/qwen35_4b_realrepo_agentic_instrument/scripts/guard.sh` (cgroup memory ceiling — see
`docs/wsl_stability.md`).

## Artifacts

- `scripts/recur.py` — looping context manager (depth-verified), block/k/α sweep, α-continuity diagnostic
- `scripts/verify.py` — balanced accuracy, positional sweep, insert-a-different-block control, split-half
- `data/heldout_shift.jsonl`, `data/test_affine.jsonl` — copied in (standalone-experiment directive)
- `reports/recur_results.json`, `reports/verify_results.json`, `reports/verify_affine.json`
