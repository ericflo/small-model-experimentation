#!/usr/bin/env python3
"""Can the model's JUDGE role evaluate what its own ACTOR role wrote? (Probe A)

THE GAP THIS TESTS. Three facts sit in tension in this corpus:
  * prompted AS A JUDGE on real code, a single-token readout scores AUROC 0.837 (C46);
  * explicit self-verification WHILE ACTING is at chance, and verbalized confidence is a constant (C40);
  * agentic episodes that fail do 2.8x the tool calls at the SAME error rate and write comparable code --
    they simply never converge (behavior_scan.py), and on induction the model is right 0.972 of the time
    when it commits versus 0.685 overall.
So a competent judge circuit exists in these weights and never fires in the actor loop. If it can grade
the actor's OWN output, then "commit when ready" is a wiring problem -- elicitation, which this corpus
has repeatedly shown works -- rather than a missing capability, which has failed ~20 times.

WHAT IT MEASURES. For every logged episode we can recover both halves of a well-posed judging task from
the trajectory itself: the `edit` tool call's `oldText` carries the target function's signature AND
docstring (the spec), and `newText` carries the implementation the agent actually wrote. Ground truth is
the episode's execution-verified `solved` flag. So: AUROC of the judge's P(True) against real outcomes,
on the model's own work, with no new rollouts.

CONTROLS, because a bare AUROC proves little here:
  * MAJORITY baseline -- what "always say solved" scores, given the class balance.
  * SURFACE baseline -- implementation length. C40 established that a surface feature reaches 0.61 while
    the internal signal reaches 0.95; a judge that fails to beat length is reading nothing but size.
  * SHUFFLED control -- judge scores paired against permuted labels, which must collapse to ~0.5.
Readout follows C46/C57 exactly (strict-reviewer system prompt, single-token A/B after "Answer: ",
thinking off) because that is the configuration the 0.837 was measured in; token ids are derived at
runtime and asserted single-token rather than hardcoded, since the vLLM path needs space-prefixed ids
(357/417) while HF needs bare ones (32/33) and mixing them silently yields a degenerate 0.5.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUTD = HERE.parents[2] / "large_artifacts" / "qwen35_4b_realrepo_agentic_instrument"
MODEL_ID = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STUB = "raise NotImplementedError  # TODO: implement"

SYS = ("You are a strict Python code reviewer. You judge whether an implementation correctly and "
       "completely satisfies its specification.")


def extract(traj: Path):
    """Recover (spec, FINAL implementation) for the target function from an episode's edit calls.

    Edits are CHAINED, not sampled. An agent that edits the same function three times would otherwise
    have an intermediate version judged against the episode's FINAL execution outcome -- label noise
    injected by the harness rather than by the model. So the stub-containing edit seeds the text and each
    later edit whose oldText still appears in it is applied in order, yielding the version the tests
    actually ran against.
    """
    spec = impl = None
    try:
        with open(traj, errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") != "tool_execution_start":
                    continue
                if (ev.get("toolName") or "") not in ("edit", "write", "str_replace"):
                    continue
                for ed in ((ev.get("args") or {}).get("edits") or []):
                    if not isinstance(ed, dict):
                        continue
                    old, new = ed.get("oldText") or "", ed.get("newText") or ""
                    if not new.strip():
                        continue
                    if STUB in old:
                        spec = old.replace(STUB, "").rstrip()   # signature + docstring
                        impl = new
                    elif impl and old and old in impl:          # a refinement of what we already have
                        impl = impl.replace(old, new)
    except OSError:
        return None, None
    return spec, impl


def build_prompt(tok, spec, impl):
    user = (f"Specification (signature and docstring of the function to implement):\n"
            f"```python\n{spec}\n```\n\n"
            f"Candidate implementation:\n```python\n{impl}\n```\n\n"
            f"Does the candidate implementation correctly and completely satisfy the specification, "
            f"such that the project's existing unit tests for it would pass?\n"
            f"Answer with a single letter: A = correct, B = incorrect.")
    text = tok.apply_chat_template([{"role": "system", "content": SYS},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)  # footgun-ok: C46's judge readout is measured no-think; thinking changes the substrate
    return text + "Answer: "


def auroc(scores, labels):
    """Rank-based AUROC with tie handling; returns 0.5 when a class is missing."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="base,warmstart")
    ap.add_argument("--out", default=str(HERE.parent / "reports" / "judge_probe.json"))
    a = ap.parse_args()

    rows = []
    for label in a.labels.split(","):
        label = label.strip()
        rep = OUTD / "reports" / f"pi_{label}_test.json"
        if not rep.exists():
            continue
        for e in json.loads(rep.read_text()).get("episodes", []):
            if not e.get("traj"):
                continue
            spec, impl = extract(OUTD / e["traj"])
            if spec and impl:
                rows.append({"arm": label, "task": e["task_id"], "solved": bool(e.get("solved")),
                             "spec": spec, "impl": impl})
    print(f"recovered {len(rows)} (spec, implementation) pairs with ground truth", flush=True)
    if not rows:
        print("nothing to judge"); return
    pos = sum(r["solved"] for r in rows)
    print(f"class balance: {pos} solved / {len(rows)-pos} not  -> MAJORITY baseline "
          f"{max(pos, len(rows)-pos)/len(rows):.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV)
    ids_a = tok("A", add_special_tokens=False).input_ids
    ids_b = tok("B", add_special_tokens=False).input_ids
    assert len(ids_a) == 1 and len(ids_b) == 1, f"A/B not single-token: {ids_a} {ids_b}"
    ID_A, ID_B = ids_a[0], ids_b[0]
    print(f"readout token ids: A={ID_A} B={ID_B} (derived, not hardcoded)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model.eval()

    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            prompt = build_prompt(tok, r["spec"], r["impl"][:6000])
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            lg = model(**enc, use_cache=False).logits[0, -1]
            pa, pb = lg[ID_A].float(), lg[ID_B].float()
            r["p_true"] = torch.softmax(torch.stack([pa, pb]), 0)[0].item()
            if i % 50 == 0:
                print(f"  judged {i}/{len(rows)}", flush=True)

    scores = [r["p_true"] for r in rows]
    labels = [r["solved"] for r in rows]
    lengths = [len(r["impl"]) for r in rows]
    rnd = random.Random(0)
    shuffled = labels[:]
    rnd.shuffle(shuffled)

    res = {"n": len(rows), "n_solved": pos,
           "majority_baseline": round(max(pos, len(rows) - pos) / len(rows), 4),
           "judge_auroc": round(auroc(scores, labels), 4),
           "surface_length_auroc": round(auroc(lengths, labels), 4),
           "shuffled_control_auroc": round(auroc(scores, shuffled), 4),
           "mean_p_true_solved": round(sum(s for s, l in zip(scores, labels) if l) / max(1, pos), 4),
           "mean_p_true_failed": round(sum(s for s, l in zip(scores, labels) if not l) / max(1, len(rows) - pos), 4),
           "per_arm": {}}
    for arm in {r["arm"] for r in rows}:
        sub = [(r["p_true"], r["solved"]) for r in rows if r["arm"] == arm]
        res["per_arm"][arm] = {"n": len(sub),
                               "auroc": round(auroc([s for s, _ in sub], [l for _, l in sub]), 4)}

    print("\n=== JUDGE-ON-OWN-WORK ===", flush=True)
    print(f"  judge AUROC            {res['judge_auroc']:.3f}   (C46 reference on MBPP: 0.837)", flush=True)
    print(f"  surface (length) AUROC {res['surface_length_auroc']:.3f}   <- must be beaten to mean anything", flush=True)
    print(f"  shuffled control       {res['shuffled_control_auroc']:.3f}   <- must be ~0.5", flush=True)
    print(f"  mean P(True): solved {res['mean_p_true_solved']:.3f} vs failed {res['mean_p_true_failed']:.3f}", flush=True)
    print(f"  per arm: {res['per_arm']}", flush=True)
    gap = res["judge_auroc"] - max(res["surface_length_auroc"], 1 - res["surface_length_auroc"])
    print(f"\n  VERDICT: {'ALIVE -- the judge grades its own work above surface' if gap > 0.05 else 'DEAD in this form -- judge does not beat a length feature'}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({**res, "rows": [{k: v for k, v in r.items()
                                                        if k not in ("spec", "impl")} for r in rows]}, indent=1))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
