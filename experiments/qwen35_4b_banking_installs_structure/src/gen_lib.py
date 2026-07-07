"""Qwen3.5-4B generation (capturing full token sequences) + answer-token activation extraction.

For the separability probe we need the model's hidden state at the answer token of its OWN
generated solution, under different thinking conditions. Generation reuses the s1-style
budget-forcing + shuffled-thinking logic from the sibling thinking_budget_scaling experiment,
but returns the FULL clean token sequence (prompt + thinking + </think> + answer) so a clean
forward pass can read the last-token hidden state per layer.

Activation extraction uses RIGHT padding (so the Mamba-style linear-attention recurrence sees
real tokens first and the last real token's state is uncorrupted by pad).
"""
from __future__ import annotations

import time
import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
THINK_CLOSE = 248069
SYSTEM = ("You are a Python coding assistant. Provide a single correct Python function. "
          "Put the final function in one ```python ... ``` code block.")
THINK_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
NOTHINK_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
GREEDY = dict(do_sample=False)


class Probe:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa").eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set([eos] if isinstance(eos, int) else list(eos))
        self.close_ids = self.tok("</think>\n\n", add_special_tokens=False).input_ids
        self.n_layers = self.model.config.num_hidden_layers
        self.load_secs = time.time() - t0

    # -- prompts / helpers --------------------------------------------------
    def prompt(self, user: str, enable_thinking: bool) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _ids(self, text):
        return self.tok(text, return_tensors=None).input_ids

    def _trim_eos(self, ids):
        for i, t in enumerate(ids):
            if t in self.eos_ids:
                return ids[:i]
        return ids

    def _has_eos(self, ids):
        return any(t in self.eos_ids for t in ids)

    @torch.no_grad()
    def _gen(self, prefix_ids, max_new, sampling):
        maxlen = max(len(p) for p in prefix_ids)
        pad = self.tok.pad_token_id
        inp = torch.tensor([[pad] * (maxlen - len(p)) + p for p in prefix_ids], device=self.device)
        attn = (inp != pad).long()
        out = self.model.generate(input_ids=inp, attention_mask=attn, max_new_tokens=max_new,
                                  pad_token_id=pad, **sampling)
        return out[:, maxlen:].tolist()

    # -- generation returning FULL clean token sequences --------------------
    def gen_sequences(self, prompts, *, think, budget, shuffle=False, greedy=False,
                      answer_max=512, think_cap=4096, batch_size=48):
        """Return list of dicts: {seq_ids (prompt+gen, EOS-trimmed), n_think, forced}."""
        results = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

        def process(idxs, bs):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                try:
                    outs = self._seq_batch([prompts[i] for i in sub], think, budget, shuffle,
                                           greedy, answer_max, think_cap)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2)); continue
                for r, i in zip(outs, sub):
                    results[i] = r
        process(order, batch_size)
        return results

    def _seq_batch(self, prompts, think, budget, shuffle, greedy, answer_max, think_cap):
        sampling = GREEDY if greedy else (THINK_SAMPLING if think else NOTHINK_SAMPLING)
        prefix = [self._ids(p) for p in prompts]
        if not think:
            gens = self._gen(prefix, answer_max, sampling)
            return [{"seq_ids": prefix[j] + self._trim_eos(g), "n_think": 0, "forced": False}
                    for j, g in enumerate(gens)]
        if budget is None:
            gens = self._gen(prefix, think_cap, sampling)
            return [{"seq_ids": prefix[j] + self._trim_eos(g),
                     "n_think": (g.index(THINK_CLOSE) if THINK_CLOSE in self._trim_eos(g) else 0),
                     "forced": False} for j, g in enumerate(gens)]
        # budget-forced thinking
        gen1 = self._gen(prefix, budget, sampling)
        results = [None] * len(prompts)
        cont, meta = [], []
        for j, g in enumerate(gen1):
            has_close = THINK_CLOSE in g
            if has_close and self._has_eos(g[g.index(THINK_CLOSE):]):
                results[j] = {"seq_ids": prefix[j] + self._trim_eos(g),
                              "n_think": g.index(THINK_CLOSE), "forced": False}
                continue
            if has_close:
                tr = g[:g.index(THINK_CLOSE)]; forced = False
            else:
                tr = self._trim_eos(g); forced = True
            n_think = len(tr)
            if shuffle and tr:
                perm = torch.randperm(len(tr)).tolist(); tr = [tr[k] for k in perm]
            cont.append(prefix[j] + tr + self.close_ids); meta.append((j, n_think, forced))
        if cont:
            gen2 = self._gen(cont, answer_max, sampling)
            for (j, n_think, forced), ans, pre in zip(meta, gen2, cont):
                results[j] = {"seq_ids": pre + self._trim_eos(ans), "n_think": n_think, "forced": forced}
        return results

    # -- ladder: capture thinking tokens (gen_real) + answer-only regen (gen_answer) --
    def gen_real(self, prompts, *, budget, batch_size=48):
        """Generate real thinking at `budget`; return per item {think_tokens, seq_ids, n_think, forced}."""
        results = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

        def process(idxs, bs):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                try:
                    outs = self._real_batch([prompts[i] for i in sub], budget)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2)); continue
                for r, i in zip(outs, sub):
                    results[i] = r
        process(order, batch_size)
        return results

    def _real_batch(self, prompts, budget):
        prefix = [self._ids(p) for p in prompts]
        gen1 = self._gen(prefix, budget, THINK_SAMPLING)
        results = [None] * len(prompts)
        cont, meta = [], []
        for j, g in enumerate(gen1):
            has_close = THINK_CLOSE in g
            if has_close and self._has_eos(g[g.index(THINK_CLOSE):]):
                tr = g[:g.index(THINK_CLOSE)]
                results[j] = {"think_tokens": tr, "seq_ids": prefix[j] + self._trim_eos(g),
                              "n_think": len(tr), "forced": False}
                continue
            tr = g[:g.index(THINK_CLOSE)] if has_close else self._trim_eos(g)
            cont.append(prefix[j] + tr + self.close_ids); meta.append((j, tr, not has_close))
        if cont:
            gen2 = self._gen(cont, 512, THINK_SAMPLING)
            for (j, tr, forced), ans, pre in zip(meta, gen2, cont):
                results[j] = {"think_tokens": tr, "seq_ids": pre + self._trim_eos(ans),
                              "n_think": len(tr), "forced": forced}
        return results

    def gen_answer(self, prefixes, *, batch_size=48, answer_max=512):
        """Given prefixes (prompt + thinking-variant + </think>), generate the answer.
        Returns per item {seq_ids = prefix + answer}."""
        results = [None] * len(prefixes)
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))

        def process(idxs, bs):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                pre = [prefixes[i] for i in sub]
                try:
                    ans = self._gen(pre, answer_max, THINK_SAMPLING)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2)); continue
                for a, i, pr in zip(ans, sub, pre):
                    results[i] = {"seq_ids": pr + self._trim_eos(a)}
        process(order, batch_size)
        return results

    # -- activation extraction: per-layer last-token hidden state -----------
    @torch.no_grad()
    def activations(self, seqs, batch_size=16):
        """seqs: list of token-id lists. Returns float16 array [N, n_layers+1, H] (last real token)."""
        self.tok.padding_side = "right"  # right-pad: keep linear-attn recurrence clean
        pad = self.tok.pad_token_id
        out = []
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        buf = [None] * len(seqs)

        def process(idxs, bs):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                seqlist = [seqs[i] for i in sub]
                maxlen = max(len(x) for x in seqlist)
                try:
                    ids = torch.tensor([x + [pad] * (maxlen - len(x)) for x in seqlist], device=self.device)
                    attn = (ids != pad).long()
                    o = self.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
                    bidx = torch.arange(len(sub), device=self.device)
                    last = torch.tensor([len(x) - 1 for x in seqlist], device=self.device)
                    # slice each layer's last-real-token vector BEFORE stacking (avoid the
                    # huge [B, L+1, T, H] tensor that crashed the driver on long sequences)
                    per_layer = [h[bidx, last, :] for h in o.hidden_states]  # list of [B, H]
                    vecs = torch.stack(per_layer, dim=1).to(torch.float16).cpu().numpy()  # [B, L+1, H]
                    del o, per_layer
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2)); continue
                for k, i in enumerate(sub):
                    buf[i] = vecs[k]
        process(order, batch_size)
        self.tok.padding_side = "left"
        return np.stack(buf)

    # -- verification judge: P(candidate correct) from the A/B logit ----------
    A_ID = 32   # "A" = correct
    B_ID = 33   # "B" = incorrect

    @torch.no_grad()
    def _judge_logit(self, prefixes, batch_size=16):
        """Left-pad each prefix, forward, read P(A) vs P(B) at the last token. Returns list of P(A)."""
        out = [None] * len(prefixes)
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
        pad = self.tok.pad_token_id

        def process(idxs, bs):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                seqs = [prefixes[i] for i in sub]
                maxlen = max(len(x) for x in seqs)
                try:
                    ids = torch.tensor([[pad] * (maxlen - len(x)) + x for x in seqs], device=self.device)
                    attn = (ids != pad).long()
                    o = self.model(input_ids=ids, attention_mask=attn, logits_to_keep=1)
                    lg = o.logits[:, -1, :].float()  # [B, V] (last token, left-padded)
                    pab = torch.softmax(torch.stack([lg[:, self.A_ID], lg[:, self.B_ID]], dim=1), dim=1)
                    pa = pab[:, 0].cpu().tolist()
                    del o, lg
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2)); continue
                for i, v in zip(sub, pa):
                    out[i] = v
        process(order, batch_size)
        return out

    def judge_prompt(self, task_text, code, enable_thinking):
        sys = "You are a strict Python code reviewer. Decide if a candidate solution solves the task."
        user = (f"Task:\n{task_text}\n\nCandidate solution:\n```python\n{code}\n```\n\n"
                f"Does this solution correctly solve the task (pass all valid tests)? "
                f"Answer with a single letter: A = correct, B = incorrect.")
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def judge_nothink(self, prompts, batch_size=16):
        """prompts: rendered no-think judge prompts. Append 'Answer: ' and read P(A)."""
        ans = self.tok("Answer: ", add_special_tokens=False).input_ids
        prefixes = [self._ids(p) + ans for p in prompts]
        return self._judge_logit(prefixes, batch_size)

    def judge_think(self, prompts, *, budget=1024, gen_batch=32, logit_batch=16):
        """prompts: rendered thinking judge prompts. Generate thinking (chunked), force close, read P(A)."""
        base = [self._ids(p) for p in prompts]
        ansfix = self.tok("</think>\n\nAnswer: ", add_special_tokens=False).input_ids
        prefixes = [None] * len(prompts)
        forced = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(base[i]))

        def process(idxs, bs):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                try:
                    gen1 = self._gen([base[i] for i in sub], budget, THINK_SAMPLING)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2)); continue
                for g, i in zip(gen1, sub):
                    if THINK_CLOSE in g:
                        tr = g[:g.index(THINK_CLOSE)]; forced[i] = False
                    else:
                        tr = self._trim_eos(g); forced[i] = True
                    prefixes[i] = base[i] + tr + ansfix
        process(order, gen_batch)
        pa = self._judge_logit(prefixes, logit_batch)
        return pa, forced
