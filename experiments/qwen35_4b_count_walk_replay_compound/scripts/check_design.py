#!/usr/bin/env python3
"""Frozen-design checker: normalized-hash pins on every fill-slot file.

A raw hash pin on a file with orchestrator-filled TODO-pin value slots
would break when the slots fill post-merge; a plain substring contract pins
declarations but not call sites. The NORMALIZED-HASH pin (lifecycle 22's
mechanism, carried through the count-don't-walk cell's six-slot machinery)
freezes every byte of each pinned file outside exactly its pin value
slots: the deterministic slot patterns below canonicalize the value group
(``None`` pre-fill, the filled form post-fill) to a fixed placeholder, and
the sha256 of the canonicalized bytes is frozen per file in
``NORMALIZED_PIN_SHA256``. Three files are pinned symmetrically:

- ``run_benchmark.py`` — the THREE candidate slots (replay_compound tree,
  weights, committed merge receipt sha);
- ``train_trial.py`` — the ONE ``PUBLISHED_ARM_HASHES`` value slot (the
  single-line sorted-key four-sha dict post-fill);
- ``eval_local_vllm.py`` — the ONE ``EXPECTED_TRAINED_TREE_SHA256`` value
  slot.

Every byte outside the slots — every guard call site included — is
byte-frozen pre- and post-fill; ``--check`` fails closed on one byte of
drift and re-runs at the seed-consuming boundary inside
``run_benchmark.py`` itself.

``--check`` additionally audits: the trusted gateway sha, the copied
frozen corpora (the training pool and the stage-7 production inputs), the
lifecycle substring contracts of the trainer/merger wrappers, and that no
script in this cell ever references the benchmark suite directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
RUN_BENCHMARK = SCRIPTS / "run_benchmark.py"
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
PIN_PLACEHOLDER = "__REPLAY_COMPOUND_TODO_PIN__"
# {pinned file: ((slot name, multiline regex, required match count), ...)}.
# Group 2 is the ONLY mutable region of each file; group 1 (and group 3
# where present) are kept verbatim so indentation, key, and trailing comma
# stay frozen. The train_trial slot enforces the SINGLE-LINE sorted-key
# four-sha dict as the only legal fill form.
PIN_SLOT_PATTERNS = {
    "run_benchmark.py": (
        (
            "replay_compound_tree_and_weights_dict_entries",
            r'^(    "replay_compound": )(None|"[0-9a-f]{64}")(,)$',
            2,
        ),
        (
            "replay_compound_merge_receipt_constant",
            r'^(REPLAY_COMPOUND_MERGE_RECEIPT_SHA256 = )(None|"[0-9a-f]{64}")$',
            1,
        ),
    ),
    "train_trial.py": (
        (
            "published_arm_hashes_value_slot",
            r'^(    "replay_compound": )(None|\{"adapter_config": "[0-9a-f]{64}", '
            r'"adapter_weights": "[0-9a-f]{64}", "log": "[0-9a-f]{64}", '
            r'"receipt": "[0-9a-f]{64}"\})(,)$',
            1,
        ),
    ),
    "eval_local_vllm.py": (
        (
            "expected_trained_tree_value_slot",
            r'^(    "replay_compound": )(None|"[0-9a-f]{64}")(,)$',
            1,
        ),
    ),
}
# Frozen normalized hashes per pinned file (recompute and re-freeze only
# under a new review); --check fails closed on one byte of drift anywhere
# outside the canonicalized pin value slots of each file.
NORMALIZED_PIN_SHA256 = {
    "run_benchmark.py": (
        "11a6cc140da470c57f85f2da0215f851f4722acf3e1ee057e44e3cd1605066f5"
    ),
    "train_trial.py": (
        "97c062979945fb1a672c66598d0617e66719524e5356f39132755a9e5a2756e7"
    ),
    "eval_local_vllm.py": (
        "1b2947921f09d94224eb2bed446ab8c3bef3adb676e5e0a53695947e98a3ff6a"
    ),
}
RUN_BENCHMARK_NORMALIZED_SHA256 = NORMALIZED_PIN_SHA256["run_benchmark.py"]
# Belt-and-braces call-site contracts (the normalized hash is the
# load-bearing control; these give readable diagnostics for the guard call
# sites a drifted runner would most plausibly lose).
RUN_BENCHMARK_CALL_SITE_CONTRACTS = (
    "        require_todo_pins_filled()",
    '        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")',
    "        promotion = authenticate_local_promotion(args.candidate)",
    "        require_clean_pushed_main(",
    "        require_unconsumed_ledger(LEDGER, opened_record, args.resume)",
    "        require_count_walk_parent_provenance(model)",
)
# The copied frozen corpora this standalone cell carries (fail-closed pins).
FROZEN_CORPORA = (
    ("data/sft_blend.jsonl",
     "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2", 2240),
    ("data/sft_count_walk.jsonl",
     "21e6f5cb705f447f7a4dfc9bff24673f798f48df312b99a6cf686505855ee096", 160),
    ("data/count_walk.jsonl",
     "71291542c3c901caccf9586543efb02da319b371244728ecfd1a0fc7cb92ed26", 1520),
    ("data/replay_ctl7.jsonl",
     "94e8259ec03800d0a4dcbf8075252c5180a668e2da74569fcf62497cf0f9de5a", 1520),
)
# Substring contracts that must survive later TODO-PIN fills (hashes would
# not); the load-bearing controls are the normalized runner pin and the
# design receipt's raw code pins.
REQUIRED_TRIAL_CONTRACTS = (
    "EXPECTED_ROWS = 2240",
    "TRAINING_SEED = 86",
    "OPTIMIZER_STEPS = 280",
    "LORA_RANK = 32",
    "LORA_ALPHA = 64",
    'MODEL_PATH_WEIGHTS_SHA256 = (\n    "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"\n)',
    'MODEL_PATH_TREE_SHA256 = (\n    "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"\n)',
    '"fresh_adapter": True,',
    "--model-path",
    "PASS_CONTROL_TRAINING",
    "PUBLISHED_ARM_HASHES",
    "check_parent_provenance()",
)
REQUIRED_MERGE_CONTRACTS = (
    'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
    'payload.get("applied_lora_modules") != 128',
    "--base-model",
    "PASS_CONTROL_MERGE",
    'BASE_COMPOSITE_RECEIPT_SHA256 = (\n    "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7"\n)',
    'BASE_COMPOSITE_WEIGHTS_SHA256 = (\n    "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"\n)',
    "if sha256_file(base_weights) != BASE_COMPOSITE_WEIGHTS_SHA256:",
)
BENCHMARK_READ_FORBIDDEN_SCRIPTS = (
    "run.py",
    "run_benchmark.py",
    "check_design.py",
    "check_local.py",
    "eval_local_vllm.py",
    "gen_local_gate.py",
    "rebuild_lineage.py",
    "train_trial.py",
    "merge_trained_arm.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_pinned_source(name: str, text: str) -> str:
    """Canonicalize exactly the file's TODO-pin value slots to the placeholder.

    Deterministic: each slot pattern must match its required count (None
    pre-fill or the frozen filled form post-fill) or normalization fails
    closed — a file whose pin slots drifted cannot even be hashed. Only
    the value group is replaced; every other byte passes through verbatim,
    so the normalized bytes are identical pre- and post-fill.
    """
    for slot_name, pattern, count in PIN_SLOT_PATTERNS[name]:
        compiled = re.compile(pattern, re.MULTILINE)
        matches = compiled.findall(text)
        if len(matches) != count:
            raise ValueError(
                f"{name} pin slot '{slot_name}' matched "
                f"{len(matches)} times, expected {count}; the "
                "normalized-hash pin cannot be computed on a drifted file"
            )
        text = compiled.sub(
            lambda m: m.group(1)
            + PIN_PLACEHOLDER
            + (m.group(3) if m.re.groups >= 3 else ""),
            text,
        )
    return text


def normalized_pinned_sha256(name: str, text: str) -> str:
    return sha256_bytes(normalize_pinned_source(name, text).encode("utf-8"))


def normalize_run_benchmark_source(text: str) -> str:
    return normalize_pinned_source("run_benchmark.py", text)


def normalized_runner_sha256(text: str) -> str:
    return normalized_pinned_sha256("run_benchmark.py", text)


def verify_normalized_pins() -> None:
    """The load-bearing fill-slot controls: the NORMALIZED-HASH pins.

    Everything in each pinned file except its canonicalized pin value
    slots is byte-frozen against NORMALIZED_PIN_SHA256 — for
    run_benchmark.py, deleting or reordering ANY guard call site
    (require_todo_pins_filled, require_verdict,
    authenticate_local_promotion, require_clean_pushed_main,
    require_unconsumed_ledger, require_count_walk_parent_provenance, ...)
    changes the normalized hash and fails --check, including the re-run at
    the seed-consuming boundary; for train_trial.py and
    eval_local_vllm.py the same mechanism freezes every guard around the
    PUBLISHED_ARM_HASHES and EXPECTED_TRAINED_TREE_SHA256 value slots. The
    call-site substring contracts are belt-and-braces diagnostics only.
    """
    for name, expected in sorted(NORMALIZED_PIN_SHA256.items()):
        if expected is None:
            raise ValueError(f"{name} normalized-hash pin is unfilled (TODO-PIN)")
        path = SCRIPTS / name
        if not path.is_file():
            raise ValueError(f"normalized-hash-pinned file is absent: {name}")
        text = path.read_text(encoding="utf-8")
        digest = normalized_pinned_sha256(name, text)
        if digest != expected:
            raise ValueError(
                f"{name} drifted outside its TODO-pin value slots: "
                f"normalized sha256 {digest} != frozen {expected}"
            )
        if PIN_PLACEHOLDER in text:
            raise ValueError(
                f"the pin placeholder must never appear in a live pinned "
                f"file: {name}"
            )
    runner_text = RUN_BENCHMARK.read_text(encoding="utf-8")
    missing = [
        value
        for value in RUN_BENCHMARK_CALL_SITE_CONTRACTS
        if value not in runner_text
    ]
    if missing:
        raise ValueError(
            f"frozen run_benchmark.py guard call sites missing: {missing}"
        )


def verify_gateway() -> None:
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise ValueError("trusted gateway is absent or changed")


def verify_frozen_corpora() -> int:
    checked = 0
    for relative, expected_sha, expected_rows in FROZEN_CORPORA:
        path = EXP / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"frozen corpus artifact is absent or changed: {path}")
        rows = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if rows != expected_rows:
            raise ValueError(
                f"frozen corpus row count changed: {path} has {rows}, "
                f"expected {expected_rows}"
            )
        checked += 1
    return checked


def verify_lifecycle_contracts() -> None:
    trial = (SCRIPTS / "train_trial.py").read_text(encoding="utf-8")
    merge = (SCRIPTS / "merge_trained_arm.py").read_text(encoding="utf-8")
    missing = [value for value in REQUIRED_TRIAL_CONTRACTS if value not in trial]
    if missing:
        raise ValueError(f"train_trial.py frozen contracts missing: {missing}")
    missing = [value for value in REQUIRED_MERGE_CONTRACTS if value not in merge]
    if missing:
        raise ValueError(f"merge_trained_arm.py frozen contracts missing: {missing}")


def verify_no_benchmark_reads() -> None:
    forbidden = "benchmarks" + "/"
    for name in BENCHMARK_READ_FORBIDDEN_SCRIPTS:
        if forbidden in (SCRIPTS / name).read_text(encoding="utf-8"):
            raise ValueError(f"script references the benchmark suite directory: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    assert args.check
    try:
        verify_normalized_pins()
        verify_gateway()
        corpora = verify_frozen_corpora()
        verify_lifecycle_contracts()
        verify_no_benchmark_reads()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "normalized_pin_sha256": NORMALIZED_PIN_SHA256,
                "pin_slots": {
                    name: [slot for slot, _, _ in slots]
                    for name, slots in sorted(PIN_SLOT_PATTERNS.items())
                },
                "pin_slot_count": {
                    name: sum(count for _, _, count in slots)
                    for name, slots in sorted(PIN_SLOT_PATTERNS.items())
                },
                "gateway_sha256": GATEWAY_SHA256,
                "frozen_corpora_checked": corpora,
                "ok": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
