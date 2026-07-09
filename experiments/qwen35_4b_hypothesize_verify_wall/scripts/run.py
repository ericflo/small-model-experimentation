#!/usr/bin/env python3
"""Orchestrator: make_tasks -> TRAP GATE -> gen_traces -> train dsl_sft -> c45 regen + REGEN GATE
-> INSTALL GATE -> eval arms (base no-think anchor, base, scaffold, c45_zero, dsl_sft).

Idempotent: every stage is skipped when its output artifact exists, so the pipeline is safe to
re-run after interruption. Gates hard-stop the pipeline on failure (pre-registered) unless
--force; in --smoke mode gate FAILURES warn-and-continue (a 1-step smoke adapter cannot pass
real gates -- smoke validates plumbing, and smoke artifacts are strictly _smoke-suffixed so they
can never poison a full run).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))
import hv_common as C  # noqa: E402


def sh(script, *extra):
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print(f"[run] $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(EXP), check=True)


def gate_stop(name, passed, smoke, force):
    if passed:
        print(f"[run] GATE {name}: PASS", flush=True)
        return
    if smoke:
        print(f"[run] GATE {name}: FAIL -- smoke mode, continuing (plumbing check only)", flush=True)
    elif force:
        print(f"[run] GATE {name}: FAIL -- --force given, continuing AT YOUR OWN RISK", flush=True)
    else:
        print(f"[run] GATE {name}: FAIL -- stopping (re-run with --force to override)", flush=True)
        sys.exit(1)


def trap_gate(cfg, sfx, smoke, force):
    """Oracle-skelfill >= threshold per family x depth (CPU) + rand-skelfill@K anchor (C32)."""
    out_path = EXP / "runs" / f"trap_gate{sfx}.json"
    if out_path.exists():
        res = json.load(open(out_path))
        print(f"[run] trap gate cached: {out_path.name}", flush=True)
    else:
        tasks = C.load_jsonl(EXP / "data" / f"eval_tasks{sfx}.jsonl")
        cells, srng = {}, random.Random(999)
        for t in tasks:
            fam = C.fam_of(t["family"])
            key = f"{t['family']}_d{t['depth']}"
            cell = cells.setdefault(key, {"n": 0, "oracle": 0, "rand": 0})
            cell["n"] += 1
            cell["oracle"] += int(C.skeletonfill_hidden(fam, C.true_types(t), t))
            hit = False
            for _ in range(cfg["eval"]["K"]):  # rand-skelfill at matched R=K attempts
                sk = [srng.choice(C.types_of(fam)) for _ in range(t["depth"])]
                if C.skeletonfill_hidden(fam, sk, t):
                    hit = True
                    break
            cell["rand"] += int(hit)
        thr = cfg["gates"]["oracle_skelfill_min"]
        for key, cell in cells.items():
            cell["oracle_rate"] = round(cell["oracle"] / cell["n"], 4)
            cell["rand_rate"] = round(cell["rand"] / cell["n"], 4)
            print(f"[run] trap {key}: oracle-skelfill {cell['oracle_rate']:.3f} "
                  f"(need >= {thr}) | rand-skelfill@{cfg['eval']['K']} {cell['rand_rate']:.3f}", flush=True)
        res = {"cells": cells, "threshold": thr,
               "passed": bool(all(c["oracle_rate"] >= thr for c in cells.values()))}
        out_path.parent.mkdir(exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=1)
    gate_stop("trap (oracle-skelfill)", res["passed"], smoke, force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="continue past a failed gate (full runs)")
    args = ap.parse_args()
    cfg = C.load_cfg()
    sfx = C.sfx(args.smoke)
    smoke_flag = ["--smoke"] if args.smoke else []
    t0 = time.time()
    (EXP / "runs").mkdir(exist_ok=True)

    # 1. frozen tasks -------------------------------------------------------------------------
    data = EXP / "data"
    need = [data / f"eval_tasks{sfx}.jsonl", data / f"sft_tasks{sfx}.jsonl", data / f"gate_d1{sfx}.jsonl"]
    if all(p.exists() for p in need):
        print("[run] stage make_tasks: outputs exist, skip", flush=True)
    else:
        print("[run] stage make_tasks", flush=True)
        sh("make_tasks.py", *smoke_flag)

    # 2. trap gate (pre-registered hard stop) -------------------------------------------------
    print("[run] stage trap_gate", flush=True)
    trap_gate(cfg, sfx, args.smoke, args.force)

    # 3. truth-blind traces -------------------------------------------------------------------
    if (data / f"traces{sfx}.jsonl").exists():
        print("[run] stage gen_traces: output exists, skip", flush=True)
    else:
        print("[run] stage gen_traces", flush=True)
        sh("gen_traces.py", *smoke_flag)

    # 4. dsl_sft adapter (think-channel reasoning-SFT) -----------------------------------------
    lora_dsl = EXP / "runs" / f"lora_dsl{sfx}"
    if (lora_dsl / "adapter_model.safetensors").exists():
        print("[run] stage train_dsl: adapter exists, skip", flush=True)
    else:
        print("[run] stage train_dsl", flush=True)
        sh("train_think.py", "--train", str(data / f"traces{sfx}.jsonl"), "--out", str(lora_dsl),
           "--epochs", str(cfg["sft"]["epochs"]), *smoke_flag)

    # 5. c45 adapter regen (committed train_general.jsonl, C45 recipe, seed-pinned) + gate -----
    lora_c45 = EXP / "runs" / f"lora_c45{sfx}"
    if (lora_c45 / "adapter_model.safetensors").exists():
        print("[run] stage train_c45: adapter exists, skip", flush=True)
    else:
        print("[run] stage train_c45", flush=True)
        c45_train = (EXP / cfg["c45_regen"]["train"]).resolve()
        assert c45_train.exists(), f"missing committed C45 train set: {c45_train}"
        sh("train_lora.py", "--train", str(c45_train), "--out", str(lora_c45),
           "--epochs", str(cfg["sft"]["epochs"]), "--seed", str(cfg["c45_regen"]["seed"]), *smoke_flag)
    gate_c45_path = EXP / "runs" / f"gate_c45{sfx}.json"
    if not gate_c45_path.exists():
        print("[run] stage gate_c45 (regen-sanity)", flush=True)
        sh("gate_gpu.py", "--gate", "c45", "--adapter", str(lora_c45), *smoke_flag)
    gate_stop("c45 regen-sanity", json.load(open(gate_c45_path))["passed"], args.smoke, args.force)

    # 6. install gate for dsl_sft (format + no depth-1 collapse, deploy channel) ---------------
    gate_inst_path = EXP / "runs" / f"gate_install{sfx}.json"
    if not gate_inst_path.exists():
        print("[run] stage gate_install", flush=True)
        sh("gate_gpu.py", "--gate", "install", "--adapter", str(lora_dsl), *smoke_flag)
    gate_stop("dsl_sft install", json.load(open(gate_inst_path))["passed"], args.smoke, args.force)

    # 7. eval arms (shared frozen eval; one arm per process) -----------------------------------
    arms = [("base_nothink", ["--arm", "base", "--nothink-anchor"]),
            ("base", ["--arm", "base"]),
            ("scaffold", ["--arm", "scaffold"]),
            ("c45_zero", ["--arm", "c45_zero", "--adapter", str(lora_c45)]),
            ("dsl_sft", ["--arm", "dsl_sft", "--adapter", str(lora_dsl)])]
    for tag, extra in arms:
        out = EXP / "runs" / f"eval_{tag}{sfx}.json"
        if out.exists():
            print(f"[run] stage eval_{tag}: output exists, skip", flush=True)
            continue
        print(f"[run] stage eval_{tag}", flush=True)
        sh("eval_arms.py", *extra, *smoke_flag)

    print(f"[run] PIPELINE COMPLETE ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
