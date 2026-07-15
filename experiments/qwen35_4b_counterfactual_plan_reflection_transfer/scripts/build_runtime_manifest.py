#!/usr/bin/env python3
"""Build the deterministic pre-Python file/lease manifest for static launchers."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


EXPERIMENT = "experiments/qwen35_4b_counterfactual_plan_reflection_transfer"
SNAPSHOT = Path("/workspace/sme-reflection-runtime")
EXECUTION_ROOT = Path("/workspace/sme-reflection-exec")

STAGES = {
    "training": (
        "adapter_behavior_gate",
        "analyze",
        "authorize_stage",
        "build_eval_inputs",
        "build_literal_action_inputs",
        "build_literal_reflection_inputs",
        "calibration_gate",
        "matched_compute_gate",
        "merge_adapter",
        "retention_gate",
        "runtime_audit",
        "score",
        "score_literal",
        "tokenizer_receipt",
        "train",
    ),
    "vllm": ("run_frozen_reservoir", "runtime_audit", "vllm_runner"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"pre-Python manifest path is not a file: {path}")
    return resolved


def mapped_native_files(source_root: Path) -> set[Path]:
    experiment = source_root / EXPERIMENT
    probe = (
        "import pathlib,sys;"
        f"sys.path.insert(0,{str(experiment / 'src')!r});"
        "import runtime_contract;"
        "print('\\n'.join(sorted({f[-1] for l in "
        "pathlib.Path('/proc/self/maps').read_text().splitlines() "
        "if (f:=l.split()) and f[-1].startswith('/')})))"
    )
    environment = {
        "HOME": "/root",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "Etc/UTC",
        "LOCPATH": str(SNAPSHOT / "lib/locale"),
        "GCONV_PATH": str(SNAPSHOT / "runtime-libs/gconv"),
        "LD_LIBRARY_PATH": (
            f"{SNAPSHOT}/runtime-libs:/usr/local/cuda/lib64"
        ),
    }
    command = [
        str(SNAPSHOT / "runtime-libs/ld-linux-x86-64.so.2"),
        "--library-path",
        environment["LD_LIBRARY_PATH"],
        "--argv0",
        str(SNAPSHOT / "bin/python3.12"),
        str(SNAPSHOT / "bin/python3.12"),
        "-I",
        "-B",
        "-S",
        "-c",
        probe,
    ]
    result = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return {regular(Path(line)) for line in result.stdout.splitlines() if line}


def binary_dependencies(path: Path) -> set[Path]:
    result = subprocess.run(
        [
            str(SNAPSHOT / "runtime-libs/ld-linux-x86-64.so.2"),
            "--library-path",
            f"{SNAPSHOT}/runtime-libs:/usr/local/cuda/lib64",
            "--list",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dependencies: set[Path] = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split()
        candidates = [field for field in fields if field.startswith("/")]
        if candidates:
            dependency = regular(Path(candidates[0]))
            if dependency.name.startswith("ld-linux-"):
                continue
            if SNAPSHOT not in dependency.parents and Path("/usr/local/cuda") not in dependency.parents:
                raise ValueError(f"binary dependency escaped the runtime snapshot: {dependency}")
            dependencies.add(dependency)
    return dependencies


def add_file(rows: dict[Path, str], path: Path, role: str = "preauth") -> None:
    resolved = regular(path)
    prior = rows.get(resolved)
    if prior is not None and prior != role:
        if prior == "preauth":
            rows[resolved] = role
            return
        if role == "preauth":
            return
        raise ValueError(f"manifest file has conflicting roles: {resolved}")
    rows[resolved] = role


def build(source_root: Path) -> bytes:
    source_root = source_root.resolve()
    experiment = source_root / EXPERIMENT
    execution_experiment = EXECUTION_ROOT / EXPERIMENT
    files: dict[Path, str] = {}
    add_file(files, SNAPSHOT / "bin/python3.12", "interpreter")
    add_file(
        files,
        SNAPSHOT / "runtime-libs/ld-linux-x86-64.so.2",
        "loader",
    )
    add_file(files, SNAPSHOT / "pyvenv.cfg")
    add_file(files, SNAPSHOT / "tools/git", "git")
    for path in (SNAPSHOT / "lib/python3.12").rglob("*"):
        if path.is_file():
            add_file(files, path)
    for path in mapped_native_files(source_root):
        add_file(files, path)
    for path in binary_dependencies(SNAPSHOT / "tools/git"):
        add_file(files, path)
    for root in (SNAPSHOT / "lib/locale", SNAPSHOT / "runtime-libs/gconv", SNAPSHOT / "lib/git-core"):
        for path in root.rglob("*"):
            if path.is_file():
                add_file(files, path)
    add_file(
        files,
        experiment / "src/runtime_contract.py",
        "runtime_contract",
    )
    add_file(
        files,
        experiment / "src/load_window_guard.py",
        "load_window_guard",
    )
    add_file(files, source_root / "requirements-training.lock.txt")
    add_file(files, source_root / "requirements-vllm.lock.txt")

    output = ["schema\t1"]
    for path, role in sorted(files.items(), key=lambda item: str(item[0])):
        rendered = str(path)
        if source_root == path or source_root in path.parents:
            rendered = str(EXECUTION_ROOT / path.relative_to(source_root))
        output.append(f"file\t{role}\t{rendered}\t{sha256(path)}")
    for backend, stages in sorted(STAGES.items()):
        for stage in stages:
            relative = (
                Path("src/vllm_runner.py")
                if stage == "vllm_runner"
                else Path(f"scripts/{stage}.py")
            )
            source = regular(experiment / relative)
            target = execution_experiment / relative
            output.append(
                f"stage\t{backend}\t{stage}\t{target}\t{sha256(source)}"
            )
    return ("\n".join(output) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.source_root)
    args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
