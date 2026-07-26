#!/usr/bin/env python3
"""Prove the cache-safe loop is correct BEFORE any number is measured with it.

Three gates, each targeting a way this could be silently wrong. The looping line already produced one
fake result from a silent truncation, so the rule here is that the implementation must demonstrate its
own correctness rather than look plausible.

GATE 1 (identity): with k=1 the stack must be byte-identical to the untouched model. If cloning or the
  config edits perturb anything, every reported delta is measured against the wrong baseline.
GATE 2 (depth): the number of executed layer positions must equal 32 + (b-a)*(k-1). This is the gate
  that catches `config.num_hidden_layers` truncation, which previously made a "deeper" model run FEWER
  layers and fabricated a +0.125.
GATE 3 (cache equivalence): the whole point of this module. Greedy generation WITH cache must produce
  the same tokens as greedy generation WITHOUT cache under the same looped stack. If clones shared a
  cache index, the second pass would overwrite the first pass's entries and cached generation would
  drift from uncached — a corruption that produces fluent, plausible, wrong output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import CacheSafeLoop  # noqa: E402

@torch.no_grad()
def _cache_drift(model, ids, steps=12):
    """Greedy-decode `steps` tokens twice -- with and without cache -- under whatever stack is active.

    Returns (max |logit difference| at any step, fraction of steps whose argmax agreed).

    Exact token equality is the WRONG gate: the cached path attends one query token against stored keys
    while the uncached path recomputes the whole sequence, and in bf16 those are numerically different
    kernels, so greedy decoding can legitimately diverge at a near-tie even on the UNMODIFIED model.
    So this is measured against the base model's own drift as a noise floor; only drift far above that
    floor indicates the clones are actually sharing cache slots.

    BOTH STREAMS ARE TEACHER-FORCED ON THE SAME TOKENS. Letting each follow its own argmax makes the
    measurement meaningless after the first disagreement: the streams then hold different prefixes, so
    the logit gap reflects divergent context rather than cache correctness. Measured that way the
    UNMODIFIED model showed max|dlogit| 18.9 and only 0.67 agreement -- a "noise floor" that was really
    just prefix drift. On a shared prefix the true numeric gap is ~0.2.
    """
    seq = ids.clone()
    max_dev, agreed = 0.0, 0
    past = None
    for step in range(steps):
        out_c = model(seq[:, -1:] if past is not None else seq, past_key_values=past, use_cache=True)
        past = out_c.past_key_values
        lg_c = out_c.logits[:, -1].float()
        lg_u = model(seq, use_cache=False).logits[:, -1].float()   # full recompute, same prefix
        max_dev = max(max_dev, (lg_c - lg_u).abs().max().item())
        agreed += int(lg_c.argmax(-1).item() == lg_u.argmax(-1).item())
        seq = torch.cat([seq, lg_c.argmax(-1, keepdim=True)], dim=-1)   # one shared continuation
    return max_dev, agreed / steps


MODEL_ID = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROMPT = "Write one sentence about why measurement discipline matters in research."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="eager")  # footgun-ok: the PUBLISHED forced-read numbers (0.085/0.245/0.278) were measured under eager; sdpa shifts them ~0.005-0.010 via bf16 argmax flips, so switching would break comparability with committed results. New work should use sdpa.
    model.eval()
    inner = model.model.language_model if hasattr(model.model, "language_model") else model.model
    n_base = len(inner.layers)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False),
              return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    ok = True

    # ---- GATE 1: identity at k=1 -----------------------------------------------------------------
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits[:, -1].float().clone()
    with CacheSafeLoop(model, 12, 16, 1):
        with torch.no_grad():
            got = model(ids, use_cache=False).logits[:, -1].float()
    max_abs = (ref - got).abs().max().item()
    print(f"GATE 1 identity@k=1: max|dlogit| = {max_abs:.3e}  {'PASS' if max_abs == 0 else 'FAIL'}")
    ok &= max_abs == 0

    # Noise floor: the UNMODIFIED model's own cached-vs-uncached drift. Every looped arm is judged
    # against this, so bf16 kernel differences cannot be mistaken for cache corruption.
    base_dev, base_agree = _cache_drift(model, ids, steps=12)
    print(f"noise floor (base, no looping): max|dlogit| {base_dev:.4f}, token agreement {base_agree:.2f}")

    for (a, b, k) in [(12, 16, 2), (12, 16, 3), (12, 20, 2)]:
        print(f"\n--- block {a}:{b}, k={k} ---")
        # ---- GATE 2: executed depth --------------------------------------------------------------
        calls = {"n": 0}
        with CacheSafeLoop(model, a, b, k) as ctx:
            uniq = {id(m): m for m in ctx.inner.layers}
            handles = [m.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
                       for m in uniq.values()]
            try:
                with torch.no_grad():
                    model(ids, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            want = n_base + (b - a) * (k - 1)
            print(f"GATE 2 depth: executed {calls['n']} positions, expected {want} "
                  f"(distinct modules {len(uniq)})  {'PASS' if calls['n'] == want else 'FAIL'}")
            ok &= calls["n"] == want

            # ---- GATE 3: cached vs uncached decoding, against the model's OWN noise floor --------
            dev, agree = _cache_drift(model, ids, steps=12)
        print(f"GATE 3 cache drift: max|dlogit| {dev:.4f}, token agreement {agree:.2f} "
              f"(noise floor: base {base_dev:.4f} / {base_agree:.2f})  "
              f"{'PASS' if dev <= max(3 * base_dev, 0.05) else 'FAIL'}")
        ok &= dev <= max(3 * base_dev, 0.05)

    # weight sharing: clones must not allocate new parameter storage
    with CacheSafeLoop(model, 12, 16, 3) as ctx:
        ptrs = {p.data_ptr() for lay in ctx.inner.layers for p in lay.parameters()}
        base_ptrs = {p.data_ptr() for lay in ctx.orig_layers for p in lay.parameters()}
    print(f"\nweight sharing: {len(ptrs)} distinct tensors under k=3 vs {len(base_ptrs)} in base  "
          f"{'PASS (shared)' if ptrs == base_ptrs else 'FAIL (new storage allocated)'}")
    ok &= ptrs == base_ptrs

    print(f"\n{'ALL GATES PASS' if ok else 'GATES FAILED - do not measure with this'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
