#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.modeling import (  # noqa: E402
    load_model_for_generation,
    load_tokenizer,
    render_generation_prompt,
)
from repair_experiment.patching import extract_unified_diff, patch_stats  # noqa: E402


FLASK_RUNTIME_PINS = [
    "Werkzeug==2.2.3",
    "Jinja2==3.1.2",
    "itsdangerous==2.1.2",
    "click==8.1.3",
]

REQUESTS_RUNTIME_PINS = [
    "pytest<8",
    "pytest-httpbin==1.0.0",
    "pytest-mock==2.0.0",
    "pytest-cov",
    "PySocks",
    "trustme",
    "Flask==1.1.4",
    "Jinja2==2.11.3",
    "Werkzeug==1.0.1",
    "itsdangerous==1.1.0",
    "click==7.1.2",
    "httpbin==0.7.0",
    "MarkupSafe==2.0.1",
]


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )
    result = {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "output": proc.stdout[-12000:],
    }
    if check and proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def load_instance(dataset_name: str, split: str, instance_id: str) -> dict[str, Any]:
    dataset = load_dataset(dataset_name, split=split)
    for row in dataset:
        if row["instance_id"] == instance_id:
            return dict(row)
    raise SystemExit(f"instance not found: {instance_id}")


def ensure_repo(repo: str, repo_cache: Path) -> Path:
    owner, name = repo.split("/")
    path = repo_cache / name
    if not path.exists():
        run(["git", "clone", f"https://github.com/{owner}/{name}.git", str(path)], timeout=300, check=True)
    return path


def ensure_flask_venv(venv: Path, checkout: Path) -> Path:
    python = venv / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(venv)], timeout=120, check=True)
    run([str(python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], timeout=180, check=True)
    run([str(python), "-m", "pip", "install", "-e", str(checkout), "pytest<8"], timeout=240, check=True)
    tests_requirements = checkout / "requirements" / "tests.txt"
    if tests_requirements.exists():
        run([str(python), "-m", "pip", "install", "-r", str(tests_requirements)], timeout=240, check=True)
    run([str(python), "-m", "pip", "install", *FLASK_RUNTIME_PINS], timeout=180, check=True)
    return python


def ensure_requests_venv(venv: Path, checkout: Path) -> Path:
    python = venv / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(venv)], timeout=120, check=True)
    run([str(python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], timeout=180, check=True)
    run([str(python), "-m", "pip", "install", "-e", str(checkout), *REQUESTS_RUNTIME_PINS], timeout=300, check=True)
    return python


def repo_profile(repo: str) -> dict[str, Any]:
    profiles = {
        "pallets/flask": {
            "ensure_venv": ensure_flask_venv,
            "pythonpath_entries": ["src"],
            "caveat": "The runner uses a manually validated local Flask dependency profile on Python 3.12.",
        },
        "psf/requests": {
            "ensure_venv": ensure_requests_venv,
            "pythonpath_entries": ["."],
            "caveat": "The runner uses a manually validated local Requests dependency profile on Python 3.12.",
        },
    }
    if repo not in profiles:
        raise SystemExit(f"direct runner has no validated environment profile for {repo}")
    return profiles[repo]


def parse_json_list(text: str) -> list[str]:
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list, got {type(value).__name__}")
    return [str(item) for item in value]


def changed_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            path = parts[3]
            if path.startswith("b/"):
                path = path[2:]
            files.append(path)
    return files


def production_context_files(instance: dict[str, Any]) -> list[str]:
    files = []
    for path in changed_files(instance["patch"]):
        if path.startswith("tests/") or "/tests/" in path:
            continue
        files.append(path)
    return sorted(set(files))


def read_files(root: Path, paths: list[str]) -> dict[str, str]:
    out = {}
    for path in paths:
        full = root / path
        if full.exists():
            out[path] = full.read_text(encoding="utf-8", errors="replace")
    return out


def remove_worktree(repo: Path, path: Path) -> None:
    if path.exists():
        run(["git", "-C", str(repo), "worktree", "remove", "--force", str(path)], timeout=120)
        if path.exists():
            shutil.rmtree(path)


def make_worktree(repo: Path, base_commit: str, path: Path, test_patch: str) -> Path:
    remove_worktree(repo, path)
    run(["git", "-C", str(repo), "worktree", "add", "--detach", str(path), base_commit], timeout=180, check=True)
    apply_result = apply_patch(path, test_patch)
    if not apply_result["applied"]:
        raise RuntimeError(f"official test patch did not apply: {apply_result['output']}")
    return path


def apply_patch(root: Path, patch: str) -> dict[str, Any]:
    if not patch.strip():
        return {"applied": False, "output": "empty patch"}
    proc = run(
        ["git", "apply", "--recount", "--whitespace=nowarn", "-"],
        cwd=root,
        input_text=patch,
        timeout=120,
    )
    return {"applied": proc["returncode"] == 0, "output": proc["output"], "returncode": proc["returncode"]}


def run_tests(root: Path, python: Path, tests: list[str], pythonpath_entries: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    paths = [str(root if entry in ("", ".") else root / entry) for entry in pythonpath_entries]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return run(
        [str(python), "-m", "pytest", "-q", "-W", "ignore::DeprecationWarning", *tests],
        cwd=root,
        timeout=180,
        env=env,
    )


def build_record(
    instance: dict[str, Any],
    *,
    files: dict[str, str],
    wrong_patch: str = "",
    test_output: str = "",
) -> dict[str, Any]:
    return {
        "task_id": instance["instance_id"],
        "episode_id": instance["instance_id"],
        "split": "swebench_verified_direct",
        "issue": instance["problem_statement"],
        "buggy_files": files,
        "current_files": files,
        "wrong_patch": wrong_patch,
        "test_output_after_wrong_patch": test_output,
        "target_next_diff": instance["patch"],
        "base_buggy_diff": instance["patch"],
        "metadata": {
            "bug_family": "real_swebench_verified",
            "failure_class": "real",
        },
    }


def generate_patch(
    *,
    model_id: str,
    revision: str,
    adapter: str | None,
    tokenizer,
    record: dict[str, Any],
    prompt_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    model = load_model_for_generation(model_id, revision, adapter, load_in_4bit=True)
    prompt = render_generation_prompt(tokenizer, record, prompt_mode)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    elapsed = time.time() - start
    completion_ids = output[encoded["input_ids"].shape[1] :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    patch = extract_unified_diff(completion)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "completion": completion,
        "extracted_patch": patch,
        "generation_seconds": elapsed,
        **patch_stats(patch),
    }


def evaluate_patch(
    *,
    repo: Path,
    base_commit: str,
    worktree: Path,
    test_patch: str,
    candidate_patch: str,
    python: Path,
    tests: list[str],
    pythonpath_entries: list[str],
) -> dict[str, Any]:
    make_worktree(repo, base_commit, worktree, test_patch)
    applied = apply_patch(worktree, candidate_patch)
    if not applied["applied"]:
        return {
            "patch_applied": False,
            "patch_apply_output": applied["output"],
            "tests_passed": False,
            "test_output": applied["output"],
        }
    test_result = run_tests(worktree, python, tests, pythonpath_entries)
    return {
        "patch_applied": True,
        "patch_apply_output": applied["output"],
        "tests_passed": test_result["returncode"] == 0,
        "test_output": test_result["output"],
        "test_returncode": test_result["returncode"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-id", default="pallets__flask-5014")
    parser.add_argument("--repo-cache", type=Path, default=Path("runs/swebench_direct/repos"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/swebench_direct/eval_work"))
    parser.add_argument("--venv-dir", type=Path, default=Path("runs/swebench_direct/venvs"))
    parser.add_argument("--output", type=Path, default=Path("reports/swebench_direct_slice_results.json"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.repo_cache = args.repo_cache.resolve()
    args.work_dir = args.work_dir.resolve()
    args.venv_dir = args.venv_dir.resolve()
    args.output = args.output.resolve()

    instance = load_instance(args.dataset, args.split, args.instance_id)
    profile = repo_profile(instance["repo"])

    args.repo_cache.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.venv_dir.mkdir(parents=True, exist_ok=True)

    repo = ensure_repo(instance["repo"], args.repo_cache)
    base_commit = instance["base_commit"]
    tests = parse_json_list(instance["FAIL_TO_PASS"])
    context_paths = production_context_files(instance)
    task_work_dir = args.work_dir / args.instance_id
    task_work_dir.mkdir(parents=True, exist_ok=True)

    preflight_tree = make_worktree(repo, base_commit, task_work_dir / "preflight_base", instance["test_patch"])
    python = profile["ensure_venv"](args.venv_dir / args.instance_id, preflight_tree)
    pythonpath_entries = profile["pythonpath_entries"]
    base_test = run_tests(preflight_tree, python, tests, pythonpath_entries)
    gold_eval = evaluate_patch(
        repo=repo,
        base_commit=base_commit,
        worktree=task_work_dir / "preflight_gold",
        test_patch=instance["test_patch"],
        candidate_patch=instance["patch"],
        python=python,
        tests=tests,
        pythonpath_entries=pythonpath_entries,
    )
    base_files = read_files(preflight_tree, context_paths)
    if args.preflight_only:
        summary = {
            "status": "preflight_completed",
            "harness": "direct_pytest_not_official_docker",
            "dataset": args.dataset,
            "instance_id": args.instance_id,
            "repo": instance["repo"],
            "base_commit": base_commit,
            "tests": tests,
            "context_paths": context_paths,
            "pythonpath_entries": pythonpath_entries,
            "base_failed": base_test["returncode"] != 0,
            "gold_passed": bool(gold_eval["tests_passed"]),
        }
        payload = {
            "summary": summary,
            "preflight": {
                "base_test": base_test,
                "gold_eval": gold_eval,
            },
            "caveats": [
                "This is not the official Docker SWE-bench harness.",
                profile["caveat"],
                "The prompt context includes production files touched by the gold patch, matching the synthetic opened-file setup but not a full autonomous retrieval setting.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    tokenizer = load_tokenizer(args.model_id, args.revision)

    initial_record = build_record(instance, files=base_files)
    initial_gen = generate_patch(
        model_id=args.model_id,
        revision=args.revision,
        adapter=None,
        tokenizer=tokenizer,
        record=initial_record,
        prompt_mode="final_patch",
        max_new_tokens=args.max_new_tokens,
    )
    initial_eval = evaluate_patch(
        repo=repo,
        base_commit=base_commit,
        worktree=task_work_dir / "initial_frozen",
        test_patch=instance["test_patch"],
        candidate_patch=initial_gen["extracted_patch"],
        python=python,
        tests=tests,
        pythonpath_entries=pythonpath_entries,
    )

    wrong_tree = make_worktree(repo, base_commit, task_work_dir / "wrong_patch_tree", instance["test_patch"])
    wrong_apply = apply_patch(wrong_tree, initial_gen["extracted_patch"])
    if wrong_apply["applied"]:
        wrong_test = run_tests(wrong_tree, python, tests, pythonpath_entries)
        wrong_trace = wrong_test["output"]
        current_files = read_files(wrong_tree, context_paths)
    else:
        wrong_trace = wrong_apply["output"]
        current_files = base_files
    repair_record = build_record(
        instance,
        files=current_files,
        wrong_patch=initial_gen["extracted_patch"],
        test_output=wrong_trace,
    )

    repair_specs = [
        ("B_frozen_second_attempt", None, "trace"),
        ("C_final_patch_sft", "models/v2_final_patch_sft_lora", "trace"),
        ("D_no_trace_repair_sft", "models/v2_failure_conditioned_no_trace_lora", "no_trace"),
        ("E_trace_repair_sft", "models/v2_failure_conditioned_trace_lora", "trace"),
        ("F_shuffled_trace_repair_sft", "models/v2_failure_conditioned_shuffled_trace_lora", "trace"),
    ]
    repair_results = []
    for condition, adapter, prompt_mode in repair_specs:
        gen = generate_patch(
            model_id=args.model_id,
            revision=args.revision,
            adapter=adapter,
            tokenizer=tokenizer,
            record=repair_record,
            prompt_mode=prompt_mode,
            max_new_tokens=args.max_new_tokens,
        )
        if wrong_apply["applied"]:
            # Recreate the true two-step state for evaluation on top of the wrong patch.
            two_step_tree = make_worktree(repo, base_commit, task_work_dir / f"{condition}_two_step", instance["test_patch"])
            apply_patch(two_step_tree, initial_gen["extracted_patch"])
            repair_applied = apply_patch(two_step_tree, gen["extracted_patch"])
            if repair_applied["applied"]:
                test_result = run_tests(two_step_tree, python, tests, pythonpath_entries)
                repair_eval = {
                    "patch_applied": True,
                    "patch_apply_output": repair_applied["output"],
                    "tests_passed": test_result["returncode"] == 0,
                    "test_output": test_result["output"],
                    "test_returncode": test_result["returncode"],
                }
            else:
                repair_eval = {
                    "patch_applied": False,
                    "patch_apply_output": repair_applied["output"],
                    "tests_passed": False,
                    "test_output": repair_applied["output"],
                }
        else:
            # If the first patch did not apply, the repository is still at the original
            # tree. Evaluate the second attempt as a replacement patch conditioned on
            # the failed diff and apply-error trace.
            repair_eval = evaluate_patch(
                repo=repo,
                base_commit=base_commit,
                worktree=task_work_dir / f"{condition}_eval",
                test_patch=instance["test_patch"],
                candidate_patch=gen["extracted_patch"],
                python=python,
                tests=tests,
                pythonpath_entries=pythonpath_entries,
            )
        repair_results.append(
            {
                "condition": condition,
                "adapter": adapter,
                "prompt_mode": prompt_mode,
                **gen,
                **repair_eval,
            }
        )

    valid_initial_failure = not initial_eval["tests_passed"]
    summary = {
        "status": "completed",
        "harness": "direct_pytest_not_official_docker",
        "dataset": args.dataset,
        "instance_id": args.instance_id,
        "repo": instance["repo"],
        "base_commit": base_commit,
        "tests": tests,
        "context_paths": context_paths,
        "pythonpath_entries": pythonpath_entries,
        "base_failed": base_test["returncode"] != 0,
        "gold_passed": bool(gold_eval["tests_passed"]),
        "initial_resolved@1": 1.0 if initial_eval["tests_passed"] else 0.0,
        "valid_initial_failures": 1 if valid_initial_failure else 0,
        "repair_after_first_failure@1": {
            row["condition"]: (1.0 if valid_initial_failure and row["tests_passed"] else 0.0)
            for row in repair_results
        },
        "end_to_end_resolved@2": {
            row["condition"]: (1.0 if initial_eval["tests_passed"] or row["tests_passed"] else 0.0)
            for row in repair_results
        },
    }
    payload = {
        "summary": summary,
        "preflight": {
            "base_test": base_test,
            "gold_eval": gold_eval,
        },
        "initial": {
            **initial_gen,
            **initial_eval,
        },
        "repair_record": {
            "wrong_patch_applied": wrong_apply["applied"],
            "wrong_patch_apply_output": wrong_apply["output"],
            "test_output_after_wrong_patch": wrong_trace[-4000:],
        },
        "repairs": repair_results,
        "caveats": [
            "This is not the official Docker SWE-bench harness.",
            profile["caveat"],
            "The prompt context includes production files touched by the gold patch, matching the synthetic opened-file setup but not a full autonomous retrieval setting.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
