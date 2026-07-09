#!/usr/bin/env python3
"""GPU gates (run as separate processes so each frees the GPU when done).

--gate c45     Regen-sanity gate: the regenerated C45 adapter must reproduce C45's headline --
               held-out a7 induction-via-generation on the committed seed-pinned episode set
               (meta_induction data/gen_heldfam_a7.jsonl; no-think channel, greedy, parse
               `Answer: <digit>`) -- >= c45_headline_min before its DSL eval is spent.

--gate install Skill-installed gate for dsl_sft: on held-out depth-1 tasks in the DEPLOY channel
               (think mode, greedy), the adapter must reproduce the taught trace format on
               >= install_format_min AND its depth-1 full-solve must not collapse more than
               install_d1_collapse_max vs base (C29/C43 forgetting check).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hv_common as C  # noqa: E402
from gen_traces import trace_format_ok  # noqa: E402

EXP = C.EXP
sys.path.insert(0, str(EXP / "src"))
ANS = re.compile(r"[Aa]nswer:\s*\**\s*(\d)")


def gate_c45(args, cfg, sfx):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data_path = (EXP / cfg["c45_regen"]["train"]).resolve().parent / "gen_heldfam_a7.jsonl"
    assert data_path.exists(), f"missing committed a7 episode set: {data_path}"
    n = 20 if args.smoke else cfg["c45_regen"]["gate_n_episodes"]
    rows = C.load_jsonl(data_path)[:n]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True,
                                                 device_map="cuda", dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    print(f"[gate:c45] adapter {args.adapter} | {len(rows)} a7 episodes", flush=True)

    correct = 0
    bs = 32
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            sub = rows[s:s + bs]
            prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for r in sub]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, seq in zip(sub, out):
                txt = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                m = ANS.findall(txt)
                correct += int((m[-1] if m else "") == r["answer"])
            print(f"[gate:c45] {min(s + bs, len(rows))}/{len(rows)} acc so far "
                  f"{correct / min(s + bs, len(rows)):.3f}", flush=True)
    acc = correct / len(rows)
    res = {"gate": "c45", "acc": round(acc, 4), "n": len(rows),
           "threshold": cfg["gates"]["c45_headline_min"],
           "passed": bool(acc >= cfg["gates"]["c45_headline_min"])}
    print(f"[gate:c45] a7 induction-via-generation acc={acc:.3f} (need >= "
          f"{cfg['gates']['c45_headline_min']}) -> {'PASS' if res['passed'] else 'FAIL'}", flush=True)
    return res, EXP / "runs" / f"gate_c45{sfx}.json"


def gate_install(args, cfg, sfx):
    import gen_lib as GL
    GL.THINK_SAMPLING = dict(do_sample=True, temperature=cfg["eval"]["temperature"],
                             top_p=cfg["eval"]["top_p"], top_k=20)
    budget = 192 if args.smoke else cfg["eval"]["budget"]
    answer_max = 192 if args.smoke else cfg["eval"]["answer_max"]
    tasks = C.load_jsonl(EXP / "data" / f"gate_d1{sfx}.jsonl")
    p = GL.Probe()
    prompts = [p.prompt(C.ident_prompt(C.fam_of(t["family"]), t), enable_thinking=True) for t in tasks]

    def run_pass(tag):
        gg = p.gen_sequences(prompts, think=True, budget=budget, greedy=True,
                             answer_max=answer_max, batch_size=24)
        solved, fmt_ok = 0, 0
        for t, pr, g in zip(tasks, prompts, gg):
            gen_ids = g["seq_ids"][len(p._ids(pr)):]
            text = p.tok.decode(gen_ids)
            think_region = text.split("</think>")[0]
            fmt_ok += int(trace_format_ok(think_region))
            code = C.extract_code(text, think_mode=True)
            solved += int(C.py_solves(code, t))
        print(f"[gate:install] {tag}: d1 solve {solved}/{len(tasks)} | trace-format "
              f"{fmt_ok}/{len(tasks)}", flush=True)
        return solved / len(tasks), fmt_ok / len(tasks)

    base_solve, _ = run_pass("base")
    from peft import PeftModel
    p.model = PeftModel.from_pretrained(p.model, str(args.adapter)).eval()
    dsl_solve, dsl_fmt = run_pass("dsl_sft")

    g = cfg["gates"]
    passed = dsl_fmt >= g["install_format_min"] and dsl_solve >= base_solve - g["install_d1_collapse_max"]
    res = {"gate": "install", "n": len(tasks), "format_rate": round(dsl_fmt, 4),
           "d1_base": round(base_solve, 4), "d1_dsl": round(dsl_solve, 4),
           "format_min": g["install_format_min"], "collapse_max": g["install_d1_collapse_max"],
           "passed": bool(passed)}
    print(f"[gate:install] format {dsl_fmt:.2f} (need >= {g['install_format_min']}), d1 "
          f"{dsl_solve:.2f} vs base {base_solve:.2f} (max drop {g['install_d1_collapse_max']}) "
          f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    return res, EXP / "runs" / f"gate_install{sfx}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["c45", "install"])
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = C.load_cfg()
    sfx = C.sfx(args.smoke)
    res, out_path = (gate_c45 if args.gate == "c45" else gate_install)(args, cfg, sfx)
    out_path.parent.mkdir(exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"[gate:{args.gate}] wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
