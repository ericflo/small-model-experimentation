#!/usr/bin/env python3
"""Training-free mid-stack layer looping on Qwen3.5-4B, measured against the C59 arms.

THE QUESTION. C59 established that test-time compute crosses the induction wall only through the
CONTENT of generated reasoning tokens: on held-out shift induction (n=200) forced single-pass scored
0.090, LATENT recurrence (N=8 hidden-state feedback) 0.060, content-free FILLER tokens 0.070-0.095 --
all flat -- while real chain-of-thought generation reached 0.235. That killed one specific mechanism:
feeding the last hidden state back in as an INPUT EMBEDDING, which the model was never trained to
interpret and is therefore maximally out-of-distribution.

Layer looping is a DIFFERENT mechanism: hidden states are fed back to a layer that already consumes
hidden states of exactly that kind, at the same depth in the residual stream. The 2024-2026 literature
separates the two cleanly -- retrofitted depth recurrence buys real reasoning (McLeish et al. 2511.07384
took TinyLlama GSM8K 46.2 -> 52.0 at test-recurrence 32; Saunshi et al. 2502.17416 show a k-layer model
looped L times approaches kL layers), and a May 2026 result reports TRAINING-FREE mid-stack looping
worth +2.64pp MMLU-Pro on Qwen3-4B-Instruct. This is C59's own pre-registered next test, and it is
cheap: no training, one GPU-hour.

PRE-REGISTERED KILL RULE. If looped accuracy stays within noise of forced single-pass at every k and
every block, untrained depth is dead on this substrate and only the trained branch (Think-at-Hard-style
selective loop LoRA, 2511.08577) survives. A DROP is not a null: it is the "latent overthinking"
signature (uniform looping revises correct first-pass tokens into errors), which is why the coherence
control below is mandatory -- without it, "looping did not help" and "looping broke the model" are
indistinguishable, and only the second one licenses the trained-selective-looping follow-up.

IMPLEMENTATION. The block is looped by temporarily DUPLICATING entries of `model.model.layers`, so the
model's own forward loop supplies every piece of plumbing (rotary embeddings, per-layer-type masks,
cache handling) instead of this script re-deriving it. Qwen3.5-4B is a HYBRID stack -- 8 full-attention
layers at indices 3,7,11,15,19,23,27,31 interleaved with 24 gated-delta-net (linear-attention) layers --
and the forward builds masks per layer TYPE from `config.layer_types[i]`, indexed positionally. So
`layer_types` must be duplicated in lockstep or the loop either raises IndexError or, worse, silently
applies an attention mask of the wrong kind. Blocks are also chosen on the 4-layer period so a looped
block contains whole full/linear groups.

Run under the repo `.venv` (HF/torch). Single forward passes only, `use_cache=False`: the recurrent
conv/state of the GDN layers is then computed fresh within each pass over the whole sequence, which is
well defined; reusing a KV/state cache across duplicated layers would NOT be.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DIGIT_IDS = [15 + d for d in range(10)]          # single-token digits, as used by C44/C59

# C59 comparators on held-out shift induction at n=200 (claim ledger C59). Printed alongside our
# numbers so the contrast is legible without going back to the ledger.
C59_REFERENCE = {"forced_single_pass": 0.090, "latent_recurrence_N8": 0.060,
                 "filler_N8": 0.070, "filler_N32": 0.095, "real_cot": 0.235}

COHERENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog. Software engineering is the discipline of building "
    "reliable systems from unreliable parts, and the most valuable skill is knowing which failure "
    "modes matter. When a test suite passes but the program is wrong, the tests were measuring the "
    "wrong thing. A good measurement tells you what to do next."
)


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


class LoopedLayers:
    """Context manager: duplicate layers[a:b] k times in place, restoring the model on exit.

    `alpha` < 1 enables DAMPED looping, which is the actual published training-free method rather
    than the straw man. Re-applying a block outright adds a second full residual update, so the
    residual stream leaves the scale the stack was trained on -- the smoke test measured a 6.5-nat
    coherence collapse at k=2. Damping instead takes a fractional step per extra iteration,
    h <- h_in + alpha * (block(h_in) - h_in), i.e. an Euler step on the layer's implied ODE, which is
    what the May 2026 training-free looping result uses (+2.64pp MMLU-Pro on Qwen3-4B-Instruct).

    Only the REPEAT iterations are damped; the first pass through the block stays native, so k=1 is
    byte-identically the base model and every delta is attributable to the added depth.

    Restoration restores BOTH the ModuleList and config.layer_types and removes all hooks: leaving a
    duplicated stack or a live hook would silently contaminate every later arm in this process.
    """

    def __init__(self, model, a, b, k, alpha=1.0):
        self.model, self.a, self.b, self.k, self.alpha = model, a, b, k, alpha
        self.inner = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self.orig_layers = self.inner.layers
        self.cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.orig_types = list(getattr(self.cfg, "layer_types", []) or [])
        self.orig_nlayers = getattr(self.cfg, "num_hidden_layers", None)
        self.handles = []
        self.state = {"a_calls": 0, "stack": [], "diag": None}

    def enable_diag(self):
        """Collect block-boundary norm/cosine stats on repeat iterations."""
        self.state["diag"] = []
        return self

    def _install_damping(self, layers):
        """Blend across each repeat iteration using boundary hooks.

        Hooks (not a wrapper module) because the model's forward selects a per-layer-type attention
        mask for EACH layer it iterates -- Qwen3.5 is a hybrid full/linear stack -- so collapsing the
        block into one module would hand its mixed-type inner layers a single wrong mask.
        """
        first, last = layers[self.a], layers[self.b - 1]
        st, k, alpha = self.state, self.k, self.alpha

        def reset(_mod, _args, _kwargs):
            """Zero the loop state at the start of every top-level forward.

            Counting call parity globally across batches is fragile: any pass that ends with an
            unpopped entry silently poisons the NEXT batch, which surfaced as a shape mismatch
            (a 16-row hidden state popped during the final 8-row batch). Per-forward reset makes
            each pass self-contained.
            """
            st["a_calls"] = 0
            st["stack"].clear()
            return None

        def pre(_mod, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if st["a_calls"] % k != 0:          # a repeat iteration, not the native first pass
                st["stack"].append(h)
            st["a_calls"] += 1
            return None

        def post(_mod, _args, _kwargs, out):
            if not st["stack"]:
                return None                      # native pass -> leave untouched
            h_in = st["stack"].pop()
            h_out = out[0] if isinstance(out, tuple) else out
            if h_in.shape != h_out.shape:        # never blend mismatched states; count it instead
                st["skipped"] = st.get("skipped", 0) + 1
                return None
            if st.get("diag") is not None:
                # Is the perturbation small when alpha is small? If ||h_out|| >> ||h_in|| the block's
                # second application leaves the trained regime outright, and blending at the boundary
                # cannot rescue it -- that distinguishes "damping is broken" from "damping is
                # insufficient", which are different conclusions about the mechanism.
                with torch.no_grad():
                    ni = h_in.float().norm(dim=-1).mean().item()
                    no = h_out.float().norm(dim=-1).mean().item()
                    cos = torch.nn.functional.cosine_similarity(
                        h_in.float().flatten(0, 1), h_out.float().flatten(0, 1), dim=-1).mean().item()
                st["diag"].append({"norm_in": round(ni, 2), "norm_out": round(no, 2),
                                   "ratio": round(no / max(ni, 1e-6), 3), "cos": round(cos, 4)})
            blended = h_in + alpha * (h_out - h_in)
            return (blended,) + tuple(out[1:]) if isinstance(out, tuple) else blended

        self.handles.append(self.inner.register_forward_pre_hook(reset, with_kwargs=True))
        self.handles.append(first.register_forward_pre_hook(pre, with_kwargs=True))
        self.handles.append(last.register_forward_hook(post, with_kwargs=True))

    def __enter__(self):
        layers = list(self.orig_layers)
        a, b, k = self.a, self.b, self.k
        if k > 1:
            new_layers = layers[:b] + layers[a:b] * (k - 1) + layers[b:]
            self.inner.layers = nn.ModuleList(new_layers)
            if self.orig_types:
                t = self.orig_types
                self.cfg.layer_types = t[:b] + t[a:b] * (k - 1) + t[b:]
            # MUST bump num_hidden_layers: the decoder forward iterates
            #     for i, layer in enumerate(self.layers[: self.config.num_hidden_layers])
            # so a longer ModuleList is SILENTLY TRUNCATED back to 32 entries. Without this line the
            # "looped" model actually runs FEWER real layers (the tail is cut off), which reads as a
            # coherence collapse plus a spurious accuracy change -- the first version of this sweep
            # produced exactly that artifact, including a fake +0.125 that was a degraded model
            # exploiting a skewed label prior (most-common-class 0.18 vs baseline 0.085).
            self.cfg.num_hidden_layers = len(new_layers)
            if self.alpha < 1.0:
                self.state["a_calls"] = 0
                self.state["stack"].clear()
                self._install_damping(layers)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.inner.layers = self.orig_layers
        if self.orig_types:
            self.cfg.layer_types = self.orig_types
        if self.orig_nlayers is not None:
            self.cfg.num_hidden_layers = self.orig_nlayers
        return False


@torch.no_grad()
def forced_accuracy(model, tok, episodes, a, b, k, alpha=1.0, bs=16):
    """Forced-answer digit read, identical protocol to C59's `forced` arm (no tokens generated)."""
    ok = 0
    with LoopedLayers(model, a, b, k, alpha):
        for s in range(0, len(episodes), bs):
            sub = episodes[s:s + bs]
            prompts = [chat(tok, e["prompt"]) + "Answer: " for e in sub]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            logits = model(**enc, use_cache=False).logits[:, -1, DIGIT_IDS]
            for e, d in zip(sub, logits.argmax(-1).tolist()):
                ok += int(str(d) == e["answer"])
    return ok / max(1, len(episodes))


@torch.no_grad()
def executed_depth(model, tok, a, b, k):
    """Count layer executions in one forward. The harness must prove its own depth.

    A silent truncation (config.num_hidden_layers) already invalidated one whole sweep here, so every
    arm asserts its effective depth instead of assuming the ModuleList length is what ran. One hook
    per UNIQUE module: a module appearing twice in the list fires its single hook twice, so the tally
    is the number of executed POSITIONS.
    """
    ids = tok("depth check", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    n = {"calls": 0}
    with LoopedLayers(model, a, b, k) as ctx:
        uniq = {id(m): m for m in ctx.inner.layers}
        handles = [m.register_forward_hook(lambda *_: n.__setitem__("calls", n["calls"] + 1))
                   for m in uniq.values()]
        try:
            model(ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
    return n["calls"]


@torch.no_grad()
def boundary_diag(model, tok, a, b, k, alpha):
    """One forward with instrumentation: what does the repeated block do to the residual stream?"""
    ids = tok(COHERENCE_TEXT, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    ctx = LoopedLayers(model, a, b, k, alpha)
    with ctx:
        ctx.enable_diag()
        model(ids, use_cache=False)
        d = list(ctx.state["diag"] or [])
    return d


@torch.no_grad()
def coherence(model, tok, a, b, k, alpha=1.0):
    """Mean next-token logprob on fixed prose: is the looped stack still a functioning LM?

    THE control that separates "depth does not help" from "depth broke the model". A large drop here
    means the looped stack is out of distribution, which reframes a flat accuracy result entirely.
    """
    ids = tok(COHERENCE_TEXT, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with LoopedLayers(model, a, b, k, alpha):
        out = model(ids, use_cache=False)
    lp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    return lp.gather(-1, tgt[..., None]).squeeze(-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "heldout_shift.jsonl"))
    ap.add_argument("--n", type=int, default=200, help="C59 replicated at n=200; keep it comparable")
    ap.add_argument("--ks", default="1,2,3,4")
    ap.add_argument("--alphas", default="1.0,0.5,0.25",
                    help="damping per repeat iteration; 1.0 is naive re-application")
    ap.add_argument("--blocks", default="12:20,16:24,8:24",
                    help="a:b half-open layer ranges, chosen on the 4-layer hybrid period")
    ap.add_argument("--out", default=str(EXP / "reports" / "recur_results.json"))
    args = ap.parse_args()

    episodes = [json.loads(l) for l in open(args.data)][: args.n]
    print(f"loading {MODEL_ID} @ {REV[:8]} (bf16, eager)", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="eager")  # footgun-ok: the PUBLISHED forced-read numbers (0.085/0.245/0.278) were measured under eager; sdpa shifts them ~0.005-0.010 via bf16 argmax flips, so switching would break comparability with committed results. New work should use sdpa.
    model.eval()
    inner = model.model.language_model if hasattr(model.model, "language_model") else model.model
    types = list(getattr(model.config.get_text_config() if hasattr(model.config, "get_text_config")
                         else model.config, "layer_types", []) or [])
    print(f"{len(inner.layers)} layers | layer_types present: {bool(types)}", flush=True)
    if types:
        full = [i for i, t in enumerate(types) if "full" in t]
        print(f"full-attention layers at {full}", flush=True)

    ks = [int(x) for x in args.ks.split(",")]
    blocks = [tuple(int(v) for v in blk.split(":")) for blk in args.blocks.split(",")]
    results = {"model": MODEL_ID, "revision": REV, "n": len(episodes),
               "substrate": Path(args.data).name, "c59_reference": C59_REFERENCE, "arms": []}

    base_acc = forced_accuracy(model, tok, episodes, 0, 0, 1)
    base_coh = coherence(model, tok, 0, 0, 1)
    print(f"\nBASELINE forced single-pass: acc {base_acc:.3f} | coherence logprob {base_coh:.3f}"
          f"   (C59 forced {C59_REFERENCE['forced_single_pass']:.3f})", flush=True)
    results["baseline"] = {"acc": base_acc, "coherence_logprob": base_coh}

    alphas = [float(x) for x in args.alphas.split(",")]
    for (a, b) in blocks:
        for k in ks:
            if k == 1:
                continue
            want_depth = len(inner.layers) + (b - a) * (k - 1)
            got_depth = executed_depth(model, tok, a, b, k)
            if got_depth != want_depth:
                raise SystemExit(f"DEPTH CHECK FAILED for block {a}:{b} k={k}: "
                                 f"executed {got_depth} layers, expected {want_depth}. "
                                 f"The loop is not actually deepening the stack -- refusing to "
                                 f"report numbers from a mis-specified model.")
            for alpha in alphas:
                acc = forced_accuracy(model, tok, episodes, a, b, k, alpha)
                coh = coherence(model, tok, a, b, k, alpha)
                depth = len(inner.layers) + (b - a) * (k - 1)
                arm = {"block": f"{a}:{b}", "k": k, "alpha": alpha, "executed_depth": got_depth,
                       "acc": acc,
                       "delta_vs_base": round(acc - base_acc, 4),
                       "coherence_logprob": coh, "coherence_delta": round(coh - base_coh, 4),
                       "effective_depth": depth}
                results["arms"].append(arm)
                print(f"  block {a:2d}:{b:2d} k={k} a={alpha:<4} depth {depth:3d}  acc {acc:.3f} "
                      f"({acc - base_acc:+.3f})  coherence {coh:+.3f} ({coh - base_coh:+.3f})",
                      flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=1))

    # Alpha-continuity diagnostic: coherence must approach the baseline as alpha -> 0, or the
    # damping path is not doing what its name says. Cheap (coherence is n-independent).
    a0, b0 = blocks[0]
    results["alpha_continuity"] = []
    for alpha in (1.0, 0.5, 0.25, 0.1, 0.01, 0.001):
        coh = coherence(model, tok, a0, b0, 2, alpha)
        d = boundary_diag(model, tok, a0, b0, 2, alpha)
        row = {"alpha": alpha, "coherence_logprob": round(coh, 4),
               "coherence_delta": round(coh - base_coh, 4),
               "boundary": d[0] if d else None}
        results["alpha_continuity"].append(row)
        print(f"  alpha-continuity block {a0}:{b0} k=2 a={alpha:<6} coherence {coh:+.3f} "
              f"({coh - base_coh:+.3f})  boundary {row['boundary']}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))

    best = max(results["arms"], key=lambda r: r["acc"], default=None)
    print("\n=== VERDICT ===", flush=True)
    if best is None:
        print("no looped arms ran", flush=True)
    else:
        # A 200-episode binomial at p~0.1 has SE ~0.021, so ~2 SE is the smallest credible move.
        se = (max(base_acc, 1e-6) * (1 - base_acc) / max(1, len(episodes))) ** 0.5
        print(f"best looped arm: block {best['block']} k={best['k']} acc {best['acc']:.3f} "
              f"({best['delta_vs_base']:+.3f}); 1 SE ~ {se:.3f}", flush=True)
        if best["delta_vs_base"] <= 2 * se:
            print("KILL RULE MET: untrained depth is flat on this substrate (C59's content-only law "
                  "extends from latent feedback to layer looping).", flush=True)
        if any(r["coherence_delta"] < -0.5 for r in results["arms"]):
            print("NOTE: coherence degraded -- the looped stack is out of distribution, so this is a "
                  "'broken', not a 'no-op'; the trained selective-loop branch remains open.", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
