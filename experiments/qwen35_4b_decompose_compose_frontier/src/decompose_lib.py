"""Decompose-and-compose search: build a depth-D solution as a pipeline of primitives, executing each
step via the interpreter to materialize intermediate state. Model-guided (the 4B ranks the next
primitive via a letter-logit read of current-state->target) vs brute-force (all primitives), sharing an
interpreter-call budget. The interpreter is a calculator; the intelligence stays in the 4B's ranking.
"""
from __future__ import annotations

import torch
import gen_tasks as G

PARAM_OPTS = {
    "add_k": [-3, -2, -1, 1, 2, 3, 5], "mul_k": [-2, -1, 2, 3], "mod_k": [2, 3, 4, 5],
    "filter_gt_k": list(range(-4, 5)), "filter_lt_k": list(range(-4, 5)),
    "take_k": [1, 2, 3, 4], "drop_k": [1, 2, 3, 4], "chunk_sum_k": [1, 2, 3, 4], "rotate_k": [1, 2, 3, 4],
}
DESC = {
    "reverse": "reverse the list", "sort_asc": "sort ascending", "sort_desc": "sort descending",
    "unique_stable": "drop later duplicates, keep order", "dedup_adjacent": "collapse runs of equal neighbors",
    "abs_all": "absolute value of each", "square": "square each", "negate": "negate each",
    "filter_even": "keep even", "filter_odd": "keep odd", "keep_positive": "keep x>0",
    "running_sum": "prefix sums", "running_max": "prefix maxima", "adjacent_diff": "consecutive differences",
    "add_k": "add k to each", "mul_k": "multiply each by k", "mod_k": "each mod k",
    "filter_gt_k": "keep x>k", "filter_lt_k": "keep x<k", "take_k": "first k", "drop_k": "drop first k",
    "rotate_k": "rotate left by k", "chunk_sum_k": "sum each consecutive chunk of k",
}
NAMES = [p["name"] for p in G.PRIMS]
ARITY = {p["name"]: p["arity"] for p in G.PRIMS}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(NAMES)]
_MENU = "\n".join(f"{LETTERS[i]}) {n}({'k' if ARITY[n] else ''}): {DESC[n]}" for i, n in enumerate(NAMES))


def expand(name):
    """A primitive name -> its (name, param) candidates (all params for parameterized ops)."""
    return [(name, k) for k in PARAM_OPTS[name]] if ARITY[name] else [(name, None)]


def apply_prim(op, k, states):
    p = G.PRIM_BY_NAME[op]
    out = []
    for s in states:
        try:
            r = p["fn"](list(s), k) if p["arity"] else p["fn"](list(s))
        except Exception:
            return None
        if len(r) > 64 or any(abs(v) > 10 ** 7 for v in r):
            return None
        out.append(tuple(r))
    return tuple(out)


def propose_prompt(current, target):
    ex = "\n".join(f"  {list(c)}  ->  {list(t)}" for c, t in list(zip(current, target))[:6])
    return (
        "You are building a pipeline of list operations one step at a time. Options:\n"
        f"{_MENU}\n\nThe lists are currently on the left and must become the target on the right:\n{ex}\n\n"
        "Which SINGLE option, applied to the current lists, best progresses toward the target? "
        "Answer with just the option letter.")


class Decomposer:
    def __init__(self, probe):
        self.p = probe
        if probe is not None:
            self.letter_ids = [probe.tok(L, add_special_tokens=False).input_ids[-1] for L in LETTERS]
            self.ans_ids = probe.tok("Answer: ", add_special_tokens=False).input_ids

    @torch.no_grad()
    def _rank(self, states_list, target, batch_size=16):
        """Per state, return primitive names ranked by the model's letter-logit over the 23 options."""
        prefixes = [self.p._ids(self.p.prompt(propose_prompt(s, target), enable_thinking=False)) + self.ans_ids
                    for s in states_list]
        out = [None] * len(prefixes)
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
        pad = self.p.tok.pad_token_id
        for s in range(0, len(order), batch_size):
            sub = order[s:s + batch_size]
            seqs = [prefixes[i] for i in sub]
            m = max(len(x) for x in seqs)
            ids = torch.tensor([[pad] * (m - len(x)) + x for x in seqs], device=self.p.device)
            attn = (ids != pad).long()
            logits = self.p.model(input_ids=ids, attention_mask=attn, logits_to_keep=1).logits[:, -1, :].float()
            ll = logits[:, self.letter_ids]  # [B, 23]
            rankidx = torch.argsort(ll, dim=1, descending=True).cpu().tolist()
            for k, i in enumerate(sub):
                out[i] = [NAMES[j] for j in rankidx[k]]
        return out

    def search(self, task, mode, *, beam_width, top_p, max_depth, call_budget):
        """Beam search over primitive pipelines. mode in {'guided','brute'}."""
        inp = tuple(tuple(ex["input"]) for ex in task["visible"])
        target = tuple(tuple(ex["output"]) for ex in task["visible"])
        level = [(inp, [])]
        seen = {inp}
        calls = 0
        for depth in range(max_depth):
            if mode == "guided":
                ranked = self._rank([s for s, _ in level], target)
                namelists = [names[:top_p] for names in ranked]
            else:
                namelists = [NAMES for _ in level]
            cand = []
            for (state, prims), names in zip(level, namelists):
                ops = [c for nm in names for c in expand(nm)]
                for rank, (op, k) in enumerate(ops):
                    if calls >= call_budget:
                        break
                    new = apply_prim(op, k, state)
                    calls += 1
                    if new is None:
                        continue
                    if new == target:
                        return {"solved": True, "prims": prims + [(op, k)], "calls": calls, "depth": depth + 1}
                    if new in seen:
                        continue
                    seen.add(new)
                    cand.append((new, prims + [(op, k)], rank))
                if calls >= call_budget:
                    break
            cand.sort(key=lambda c: c[2])
            level = [(c[0], c[1]) for c in cand[:beam_width]]
            if not level or calls >= call_budget:
                break
        return {"solved": False, "prims": None, "calls": calls}


def verify_hidden(task, prims):
    states = [tuple(ex["input"]) for ex in task["hidden"]]
    for op, k in prims:
        states = apply_prim(op, k, states)
        if states is None:
            return False
    return list(states) == [tuple(ex["output"]) for ex in task["hidden"]]


def prims_to_code(prims):
    body = ["    xs = list(xs)"]
    for op, k in prims:
        body.append("    " + (G.PYSRC[op].format(k=k) if k is not None else G.PYSRC[op]))
    body.append("    return xs")
    return "def transform(xs):\n" + "\n".join(body)
