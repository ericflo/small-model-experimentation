#!/usr/bin/env python3
"""Does the confidence toolkit (C40/C41) survive on REAL code? MBPP: per problem, greedy + k temperature samples.
Per sample: ground truth = hidden-assert execution (full_pass); IMPLICIT confidence = mean token-logprob of the
generated completion (teacher-forced, same model); EXPLICIT = P(True) via the repo C10-style code-reviewer judge
(no-think; thinking subsample in a follow-up pass). Also records visible_all_pass (public-test execution-select
reference) and behavior_signature (execution-clustering self-consistency) and code length (surface baseline)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from src.coverage_utils import sampling_prompt, candidate_from_completion, mbpp_record, load_humaneval_records  # noqa: E402
sys.path.insert(0, str(EXP / "src"))
import gen_lib as GL  # noqa: E402


def load_sanitized_test(count, offset, visible_tests=1, timeout_s=5.0):
    """Offline cache has the SANITIZED config: 'prompt' (not 'text'), test_list as a JSON string. Adapt to the
    mbpp_record shape."""
    import ast as _ast
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp")["test"]
    records = []
    for raw in list(ds)[offset: offset + count * 2]:
        tl = raw.get("test_list")
        if isinstance(tl, str):
            try: tl = _ast.literal_eval(tl)
            except Exception: continue
        imports = raw.get("test_imports")
        if isinstance(imports, str):
            try: imports = _ast.literal_eval(imports)
            except Exception: imports = []
        adapted = {"task_id": raw["task_id"], "text": raw["prompt"], "test_list": tl,
                   "test_setup_code": "\n".join(imports or []), "code": raw.get("code", "")}
        rec = mbpp_record(adapted, "heldout", visible_tests, timeout_s)
        if rec is not None:
            records.append(rec)
        if len(records) >= count: break
    return records


@torch.no_grad()
def mean_logprobs(p, prompt_texts, gens, batch_size=4, chunk=128):
    """Teacher-forced mean logprob of each generated continuation. Memory-safe: keep logits bf16, log-softmax in
    float32 over sequence CHUNKS (full-vocab float32 logits OOM at batch 8 x 800 tokens)."""
    out = [None] * len(gens); pad = p.tok.pad_token_id
    order = sorted(range(len(gens)), key=lambda i: len(gens[i][1]))
    for s in range(0, len(order), batch_size):
        sub = order[s:s + batch_size]
        seqs = [gens[i][1] for i in sub]; ml = max(len(x) for x in seqs)
        ids = torch.tensor([[pad] * (ml - len(x)) + x for x in seqs], device=p.device)
        attn = (ids != pad).long()
        logits = p.model(input_ids=ids, attention_mask=attn).logits       # bf16 [B, L, V]
        tgt = ids[:, 1:]
        lps = []
        for c0 in range(0, logits.shape[1] - 1, chunk):
            c1 = min(c0 + chunk, logits.shape[1] - 1)
            lsm = torch.log_softmax(logits[:, c0:c1].float(), dim=-1)
            lps.append(lsm.gather(-1, tgt[:, c0:c1].unsqueeze(-1)).squeeze(-1))
            del lsm
        tok_lp = torch.cat(lps, dim=1)                                     # [B, L-1]
        for row, i in enumerate(sub):
            plen, seq = gens[i]; L = len(seq); off = ml - L
            span = tok_lp[row, off + plen - 1: off + L - 1]
            out[i] = float(span.mean()) if span.numel() else -99.0
        del logits, tok_lp, lps
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mbpp", "humaneval"], default="mbpp")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--visible-tests", type=int, default=1)
    ap.add_argument("--answer-max", type=int, default=420)
    ap.add_argument("--judge-batch-size", type=int, default=16)
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    if a.dataset == "humaneval":
        records = load_humaneval_records(a.n, a.offset, a.visible_tests, timeout_s=5.0)
    else:
        records = load_sanitized_test(a.n, a.offset, visible_tests=a.visible_tests)
    out_name = a.out_name or ("code_conf" if a.dataset == "mbpp" else f"{a.dataset}_code_conf")
    print(f"[cc] {len(records)} {a.dataset} test records", flush=True)
    p = GL.Probe()
    prompts = [sampling_prompt(r, p.tok) for r in records]

    def run(greedy, tag):
        gg = p.gen_sequences(prompts if greedy else [pr for pr in prompts for _ in range(a.k)],
                             think=False, budget=None, greedy=greedy, answer_max=a.answer_max, batch_size=32)
        return gg

    greedy_g = run(True, "greedy")
    samp_g = run(False, "samples")

    out_records = []
    all_gens, gen_index = [], []   # for batched logprob
    for ri, (rec, pr) in enumerate(zip(records, prompts)):
        plen = len(p._ids(pr))
        entries = []
        g = greedy_g[ri]
        entries.append(("greedy", g))
        for j in range(a.k):
            entries.append((f"s{j}", samp_g[ri * a.k + j]))
        row = {
            "task_id": rec["task_id"],
            "dataset": rec.get("dataset", a.dataset),
            "public_case_count": len(rec.get("public_cases", [])),
            "cands": [],
        }
        for tag, g in entries:
            text = p.tok.decode(g["seq_ids"][plen:], skip_special_tokens=True)
            cand = candidate_from_completion(text, rec, source=tag, order=len(row["cands"]))
            row["cands"].append({"tag": tag, "full_pass": bool(cand["full_pass"]),
                                 "visible_all_pass": bool(cand.get("visible_all_pass", False)),
                                 "behavior_signature": cand.get("behavior_signature", ""),
                                 "deployable_behavior_signature": cand.get("deployable_behavior_signature", ""),
                                 "public_signature": cand.get("public_signature", ""),
                                 "parse_ok": cand["parse_status"] == "parsed",
                                 "code": cand["code"], "code_len": len(cand["code"])})
            gen_index.append((ri, len(row["cands"]) - 1))
            all_gens.append((plen, list(g["seq_ids"])))
        out_records.append(row)
        if (ri + 1) % 25 == 0: print(f"[cc] executed {ri+1}/{len(records)}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump(out_records, open(EXP / "runs" / f"{out_name}_checkpoint.json", "w"))  # persist before logprob phase
    print("[cc] checkpoint saved; computing mean logprobs...", flush=True)
    lps = mean_logprobs(p, prompts, all_gens)
    for (ri, ci), lp in zip(gen_index, lps):
        out_records[ri]["cands"][ci]["mean_logprob"] = round(lp, 4)
    json.dump(out_records, open(EXP / "runs" / f"{out_name}_logprob.json", "w"))
    # explicit P(True), no-think, for every candidate with parsed code
    print("[cc] computing P(True) judge...", flush=True)
    jp, jidx = [], []
    for ri, row in enumerate(out_records):
        for ci, c in enumerate(row["cands"]):
            if c["parse_ok"] and c["code"]:
                jp.append(p.judge_prompt(records[ri]["task_text"], c["code"], enable_thinking=False))
                jidx.append((ri, ci))
    ptrue = p.judge_nothink(jp, batch_size=a.judge_batch_size)
    for (ri, ci), v in zip(jidx, ptrue):
        out_records[ri]["cands"][ci]["p_true"] = round(v, 4)
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump(out_records, open(EXP / "runs" / f"{out_name}.json", "w"))
    n_pass = sum(c["full_pass"] for r in out_records for c in r["cands"])
    n_all = sum(len(r["cands"]) for r in out_records)
    print(f"[cc] wrote runs/{out_name}.json | {len(out_records)} problems x {1+a.k} cands | overall pass-rate {n_pass/n_all:.2f}", flush=True)


if __name__ == "__main__":
    main()
