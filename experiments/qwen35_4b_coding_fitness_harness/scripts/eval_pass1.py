#!/usr/bin/env python3
"""Lean, standalone, adapter-evaluable HumanEval + MBPP greedy pass@1 harness.

This is the shared *coding-fitness* signal for the curriculum-installation program: a
deterministic, execution-graded pass@1 on HumanEval (164) and MBPP (first-N test), that
runs the BASE ``Qwen/Qwen3.5-4B`` or any merged composite via ``--model-override``.

Design (reuses only proven machinery, all copied into ``src/`` so this dir is standalone):
  * Records + prompt + execution grading come from ``src.coverage_utils`` / ``src.code_env``
    (the exact loop behind the C46 code-confidence experiment):
      - ``load_humaneval_records`` / ``mbpp_record`` build the problem records,
      - ``sampling_prompt`` renders the code-chat prompt (no-think channel),
      - ``candidate_from_completion`` extracts the code and EXECUTES it against the
        problem's hidden tests, returning ``full_pass`` (the pass@1 unit).
  * Generation goes through the pinned single-file ``src.vllm_runner`` (the repo's
    high-throughput offline runner). Base uses ``Qwen/Qwen3.5-4B`` at the pinned revision;
    ``--model-override`` points at a merged Qwen3.5 composite dir (fingerprint-checked by
    the runner). Greedy, ``n=1``, thinking off, seed 0 -> deterministic.

We deliberately do NOT mix backends: base and every adapter/composite arm are all graded
through the same vLLM path, so their pass@1 numbers are directly comparable.

Output JSON:
  {dataset, n, model, model_id, model_revision, pass_at_1, passed, total,
   per_problem:[{task_id, passed}], ...diagnostics...}

Run under the vLLM venv, e.g.:
  .venv-vllm/bin/python scripts/eval_pass1.py --dataset humaneval --n 164 --out runs/base_humaneval.json
  .venv-vllm/bin/python scripts/eval_pass1.py --dataset mbpp --n 200 --out runs/base_mbpp.json
  .venv-vllm/bin/python scripts/eval_pass1.py --dataset humaneval --smoke --out runs/smoke.json
  .venv-vllm/bin/python scripts/eval_pass1.py --dataset humaneval --n 164 \
      --model-override /abs/path/to/large_artifacts/<exp>/merged/<composite> --out runs/comp_humaneval.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))

from src.coverage_utils import (  # noqa: E402
    candidate_from_completion,
    load_humaneval_records,
    mbpp_record,
    sampling_prompt,
)

RUNNER = EXP / "src" / "vllm_runner.py"
VLLM_PYTHON = ROOT / ".venv-vllm" / "bin" / "python"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


# --------------------------------------------------------------------------- records
def build_humaneval_records(n: int, visible_tests: int, timeout_s: float) -> list[dict]:
    """First ``n`` HumanEval problems (offset 0). Grade ALL of them: if a problem has no
    extractable doctest example, keep it with 0 visible examples (its docstring still
    carries any inline examples) rather than dropping it, so n=164 means all 164."""
    from datasets import load_dataset
    from src.coverage_utils import humaneval_record

    dataset = list(load_dataset("openai/openai_humaneval", split="test"))[:n]
    records: list[dict] = []
    for raw in dataset:
        rec = humaneval_record(raw, "heldout", visible_tests, timeout_s)
        if rec is None:  # no doctest examples at visible_tests>0 -> keep with 0 shown
            rec = humaneval_record(raw, "heldout", 0, timeout_s)
        if rec is not None:
            records.append(rec)
    return records


def build_mbpp_records(n: int, offset: int, config: str, visible_tests: int, timeout_s: float) -> list[dict]:
    """First ``n`` MBPP test problems via the copied ``mbpp_record`` grader. Config is
    explicit (both 'full' and 'sanitized' are cached, so an unspecified load is ambiguous
    offline). 'sanitized' stores 'prompt' + a stringified test_list; adapt to mbpp_record."""
    import ast as _ast

    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", config)["test"]
    records: list[dict] = []
    for raw in list(ds)[offset : offset + n * 2]:
        if config == "sanitized":
            tl = raw.get("test_list")
            if isinstance(tl, str):
                try:
                    tl = _ast.literal_eval(tl)
                except Exception:
                    continue
            imports = raw.get("test_imports")
            if isinstance(imports, str):
                try:
                    imports = _ast.literal_eval(imports)
                except Exception:
                    imports = []
            raw = {
                "task_id": raw["task_id"],
                "text": raw["prompt"],
                "test_list": tl,
                "test_setup_code": "\n".join(imports or []),
                "code": raw.get("code", ""),
                "challenge_test_list": raw.get("challenge_test_list", []),
            }
        rec = mbpp_record(raw, "heldout", visible_tests, timeout_s)
        if rec is not None:
            records.append(rec)
        if len(records) >= n:
            break
    return records


# --------------------------------------------------------------------------- tokenizer
def load_tokenizer(model_override: str | None):
    """Load the SAME tokenizer the runner uses, for prompt rendering + completion decode."""
    from transformers import AutoTokenizer

    if model_override:
        tok = AutoTokenizer.from_pretrained(
            str(Path(model_override).expanduser().resolve()),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------------------------------------------------------- generation
def run_generation(
    prompt_rows: list[dict],
    *,
    model_override: str | None,
    max_new_tokens: int,
    max_model_len: int,
    gpu_mem: float,
    max_num_seqs: int,
    workdir: Path,
) -> tuple[dict[str, dict], dict]:
    """Invoke the pinned vLLM runner CLI (greedy, thinking off, seed 0). Returns
    (outputs-by-id, runner-metadata). The CLI auto-handles the Qwen3.5 Mamba-cache
    concurrency cap via a single deterministic process re-exec."""
    in_path = workdir / "prompts.jsonl"
    out_path = workdir / "gen.jsonl"
    meta_path = workdir / "gen.meta.json"
    with in_path.open("w", encoding="utf-8") as fh:
        for row in prompt_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    cmd = [
        str(VLLM_PYTHON), "-B", str(RUNNER),
        "--input", str(in_path),
        "--output", str(out_path),
        "--metadata", str(meta_path),
        "--thinking", "off",
        "--greedy",
        "--n", "1",
        "--max-tokens", str(max_new_tokens),
        "--seed", "0",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-num-seqs", str(max_num_seqs),
    ]
    if model_override:
        cmd += ["--model-override", str(Path(model_override).expanduser().resolve())]

    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    print(f"[eval_pass1] launching vLLM runner: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.write(proc.stdout or "")
    sys.stdout.flush()
    if proc.returncode != 0:
        raise SystemExit(f"[eval_pass1] vLLM runner failed (exit {proc.returncode})")

    outputs: dict[str, dict] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            outputs[row["id"]] = row
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return outputs, meta


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--n", type=int, default=None, help="number of problems (default: all/164 for HE, 200 for MBPP)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model-override", default=None, help="merged Qwen3.5 composite dir; default = base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny n=4 validation run")
    ap.add_argument("--visible-tests", type=int, default=1, help="public examples shown in-prompt (grading uses hidden tests)")
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--mbpp-config", choices=["full", "sanitized"], default="full")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--keep-raw", action="store_true", help="keep the intermediate vLLM JSONL + metadata next to --out")
    a = ap.parse_args()

    if a.smoke:
        n = 4
    elif a.n is not None:
        n = a.n
    else:
        n = 164 if a.dataset == "humaneval" else 200

    t0 = time.perf_counter()
    if a.dataset == "humaneval":
        records = build_humaneval_records(a.offset + n, a.visible_tests, a.timeout_s)[a.offset : a.offset + n]
    else:
        records = build_mbpp_records(n, a.offset, a.mbpp_config, a.visible_tests, a.timeout_s)
    print(f"[eval_pass1] {len(records)} {a.dataset} records (requested n={n}, offset={a.offset})", flush=True)

    tok = load_tokenizer(a.model_override)
    prompt_rows = []
    rec_by_id: dict[str, dict] = {}
    for rec in records:
        rid = str(rec["task_id"])
        rec_by_id[rid] = rec
        prompt_rows.append({"id": rid, "prompt": sampling_prompt(rec, tok), "meta": {"task_id": rec["task_id"]}})

    raw_dir = a.out.parent if a.keep_raw else Path(tempfile.mkdtemp(prefix="eval_pass1_"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs, meta = run_generation(
        prompt_rows,
        model_override=a.model_override,
        max_new_tokens=a.max_new_tokens,
        max_model_len=a.max_model_len,
        gpu_mem=a.gpu_memory_utilization,
        max_num_seqs=a.max_num_seqs,
        workdir=raw_dir,
    )

    per_problem = []
    passed = n_parse_failed = n_truncated = 0
    for rid, rec in rec_by_id.items():
        row = outputs.get(rid)
        if row is None:
            raise SystemExit(f"[eval_pass1] missing generation output for {rid!r}")
        out = row["outputs"][0]
        # Match the reference decode exactly: strip special tokens from the completion.
        text = tok.decode(out["token_ids"], skip_special_tokens=True)
        cand = candidate_from_completion(text, rec, source="greedy", order=0)
        ok = bool(cand.get("full_pass"))
        passed += int(ok)
        n_parse_failed += int(cand.get("parse_status") != "parsed")
        n_truncated += int(bool(out.get("truncated")))
        per_problem.append({"task_id": rec["task_id"], "passed": ok,
                            "parse_ok": cand.get("parse_status") == "parsed",
                            "n_answer_tokens": out.get("n_answer_tokens"),
                            "truncated": bool(out.get("truncated"))})

    total = len(per_problem)
    result = {
        "dataset": a.dataset,
        "n": total,
        "model": "base" if not a.model_override else str(Path(a.model_override).expanduser().resolve()),
        "model_id": meta.get("model"),
        "model_revision": meta.get("model_revision"),
        "pass_at_1": (passed / total) if total else 0.0,
        "passed": passed,
        "total": total,
        "per_problem": per_problem,
        # provenance / diagnostics
        "greedy": True,
        "seed": 0,
        "thinking": "off",
        "max_new_tokens": a.max_new_tokens,
        "visible_tests": a.visible_tests,
        "timeout_s": a.timeout_s,
        "mbpp_config": a.mbpp_config if a.dataset == "mbpp" else None,
        "offset": a.offset,
        "n_parse_failed": n_parse_failed,
        "n_truncated": n_truncated,
        "runner_sha256": meta.get("runner_sha256"),
        "resolved_sampling": meta.get("resolved_sampling"),
        "generation_seconds": meta.get("timing", {}).get("generation_seconds"),
        "sampled_tokens_per_second": meta.get("timing", {}).get("sampled_tokens_per_second"),
        "model_load_seconds": meta.get("timing", {}).get("model_load_seconds"),
        "wall_seconds": time.perf_counter() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[eval_pass1] {a.dataset} pass@1 = {result['pass_at_1']:.4f} "
        f"({passed}/{total}) | parse_fail={n_parse_failed} truncated={n_truncated} | "
        f"wall={result['wall_seconds']:.1f}s -> {a.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
