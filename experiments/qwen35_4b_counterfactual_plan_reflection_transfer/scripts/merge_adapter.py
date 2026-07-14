#!/usr/bin/env python3
"""Merge a trained LoRA into the full composite Qwen3.5-4B checkpoint for vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import shutil
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402
from checkpoint_lineage import (  # noqa: E402
    adapter_tensor_inventory,
    merged_checkpoint_inventory,
)
from merge_replay import authenticate_base_snapshot, verify_merge_equation  # noqa: E402
from stages import read_and_validate_stage_receipt  # noqa: E402
from tensor_merge import write_tensor_level_merge  # noqa: E402

install_benchmark_firewall(EXP.parents[1])

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["authorization"]["evaluation"] is not True:
        raise SystemExit("adapter merge/evaluation is not authorized by the committed config")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    source_receipt_path = args.adapter / "training_receipt.json"
    source_receipt = json.loads(source_receipt_path.read_text())
    required_training_receipt = {
        "schema_version", "experiment_id", "arm", "seed", "model_id",
        "model_revision", "optimizer_steps", "train_loss", "config_sha256",
        "tokenizer_receipt_sha256", "stage_receipt_sha256",
        "copied_tokenizer_receipt_sha256", "copied_stage_receipt_sha256",
        "trainer_git_commit", "trainer_sha256", "recipe_sha256",
        "record_receipt_sha256", "parity_sha256",
        "adapter_tree_excluding_training_receipt_sha256",
    }
    if set(source_receipt) != required_training_receipt or source_receipt["schema_version"] != 2:
        raise ValueError("training receipt schema changed")
    observed_adapter_tree = _sha256_tree(
        args.adapter, excluded={"training_receipt.json"}
    )
    if (
        source_receipt.get("adapter_tree_excluding_training_receipt_sha256")
        != observed_adapter_tree
    ):
        raise ValueError("adapter tree differs from its training receipt")
    if subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout:
        raise ValueError("adapter merge requires a clean worktree")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    recipe = config["training"]["recipe"]
    allowed_arms = set(config["training"]["arms"])
    allowed_seeds = set(config["training"]["staged_seeds"].values())
    if (
        source_receipt["experiment_id"] != config["experiment_id"]
        or source_receipt["config_sha256"] != _sha256_file(config_path)
        or source_receipt["arm"] not in allowed_arms
        or source_receipt["seed"] not in allowed_seeds
        or int(source_receipt["optimizer_steps"])
        != int(config["training"]["schedule"]["optimizer_steps_total"])
        or source_receipt["trainer_git_commit"] != current_commit
        or source_receipt["trainer_sha256"]
        != _sha256_file(EXP / "scripts" / "train.py")
        or source_receipt["recipe_sha256"]
        != hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("training receipt differs from frozen experiment lineage")
    positive_arm = config["training"]["positive_control"]["arm"]
    if (
        source_receipt["arm"] == positive_arm
        and source_receipt["seed"] != config["training"]["staged_seeds"]["screen"]
    ):
        raise ValueError("positive-control replication adapter is unauthorized")
    copied_tokenizer = args.adapter / "source_tokenizer_receipt.json"
    copied_stage = args.adapter / "source_stage_receipt.json"
    if (
        _sha256_file(copied_tokenizer) != source_receipt["tokenizer_receipt_sha256"]
        or _sha256_file(copied_tokenizer)
        != source_receipt["copied_tokenizer_receipt_sha256"]
        or _sha256_file(copied_stage) != source_receipt["stage_receipt_sha256"]
        or _sha256_file(copied_stage) != source_receipt["copied_stage_receipt_sha256"]
    ):
        raise ValueError("copied prerequisite receipts differ from training lineage")
    tokenizer_receipt = json.loads(copied_tokenizer.read_text())
    if (
        tokenizer_receipt.get("experiment_id") != config["experiment_id"]
        or tokenizer_receipt.get("model_id") != config["model"]["id"]
        or tokenizer_receipt.get("model_revision") != config["model"]["revision"]
        or tokenizer_receipt.get("tokenizer_eos_token_id") != 248046
        or hashlib.sha256(
            json.dumps(
                tokenizer_receipt.get("record_receipt"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        != source_receipt["record_receipt_sha256"]
        or hashlib.sha256(
            json.dumps(
                tokenizer_receipt.get("parity"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        != source_receipt["parity_sha256"]
    ):
        raise ValueError("copied tokenizer receipt identity changed")
    expected_stage = (
        "screen_training"
        if source_receipt["seed"] == config["training"]["staged_seeds"]["screen"]
        else "replication_training"
    )
    read_and_validate_stage_receipt(
        copied_stage,
        config=config,
        config_path=config_path,
        expected_stage=expected_stage,
    )
    model_id = config["model"]["id"]
    revision = config["model"]["revision"]
    if (
        source_receipt["model_id"] != model_id
        or source_receipt["model_revision"] != revision
    ):
        raise ValueError("adapter training receipt has the wrong base identity")
    peft_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if int(peft_config["r"]) != int(recipe["lora_rank"]):
        raise ValueError("adapter rank differs from preregistration")
    if set(peft_config["target_modules"]) != set(
        recipe["target_modules"]
    ):
        raise ValueError("adapter target modules differ from preregistration")
    if (
        float(peft_config["lora_alpha"]) != float(recipe["lora_alpha"])
        or float(peft_config["lora_dropout"]) != float(recipe["lora_dropout"])
        or str(peft_config["bias"]) != str(recipe["lora_bias"])
        or str(peft_config.get("peft_type", "")).upper() != "LORA"
        or str(peft_config.get("task_type", "")).upper() != "CAUSAL_LM"
        or peft_config.get("base_model_name_or_path") != model_id
        or peft_config.get("init_lora_weights") is not True
        or peft_config.get("fan_in_fan_out") not in {False, None}
        or peft_config.get("use_dora") not in {False, None}
        or peft_config.get("use_rslora") not in {False, None}
        or peft_config.get("modules_to_save") not in (None, [])
        or peft_config.get("rank_pattern") not in (None, {})
        or peft_config.get("alpha_pattern") not in (None, {})
    ):
        raise ValueError("adapter PEFT recipe differs from preregistration")
    adapter_path = args.adapter / "adapter_model.safetensors"
    source_adapter_inventory = adapter_tensor_inventory(adapter_path)
    merge_contract = config["merge"]
    base_root, base_index, _structure = authenticate_base_snapshot()
    tensor_merge = write_tensor_level_merge(
        base_root=base_root,
        base_index=base_index,
        adapter_path=adapter_path,
        output=args.output,
        recipe=recipe,
        contract=merge_contract,
    )
    applied = int(tensor_merge["adapted_module_count"])
    lineage = args.output / "source_lineage"
    lineage.mkdir(parents=False, exist_ok=False)
    retained_adapter = lineage / "adapter_tree"
    shutil.copytree(args.adapter, retained_adapter)
    retained_training = retained_adapter / "training_receipt.json"
    retained_stage = retained_adapter / "source_stage_receipt.json"
    retained_tokenizer = retained_adapter / "source_tokenizer_receipt.json"
    retained_adapter_config = retained_adapter / "adapter_config.json"
    retained_adapter_weights = retained_adapter / "adapter_model.safetensors"
    if _sha256_tree(retained_adapter, excluded={"training_receipt.json"}) != observed_adapter_tree:
        raise ValueError("retained source adapter tree differs after copy")
    retained_inventory = adapter_tensor_inventory(retained_adapter_weights)
    if retained_inventory != source_adapter_inventory:
        raise ValueError("retained source adapter tensors differ after copy")
    checkpoint_inventory = merged_checkpoint_inventory(args.output)
    merge_replay = verify_merge_equation(
        merged_path=args.output,
        adapter_tree=retained_adapter,
        recipe=recipe,
    )
    receipt = {
        "schema_version": 5,
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256_file(config_path),
        "model_id": model_id,
        "model_revision": revision,
        "source_training_receipt_sha256": _sha256_file(retained_training),
        "source_stage_receipt_sha256": _sha256_file(retained_stage),
        "source_tokenizer_receipt_sha256": _sha256_file(retained_tokenizer),
        "source_trainer_sha256": source_receipt["trainer_sha256"],
        "source_trainer_git_commit": source_receipt["trainer_git_commit"],
        "source_recipe_sha256": source_receipt["recipe_sha256"],
        "source_adapter_tree_sha256": observed_adapter_tree,
        "source_adapter_sha256": _sha256_file(retained_adapter_weights),
        "source_adapter_config_sha256": _sha256_file(retained_adapter_config),
        "source_adapter_inventory": retained_inventory,
        "source_arm": source_receipt["arm"],
        "source_seed": source_receipt["seed"],
        "applied_lora_modules": applied,
        "merge_contract_sha256": hashlib.sha256(
            json.dumps(merge_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "tensor_merge": tensor_merge,
        "merge_script_sha256": _sha256_file(Path(__file__).resolve()),
        "merge_git_commit": current_commit,
        "merged_checkpoint_inventory": checkpoint_inventory,
        "merge_replay": merge_replay,
        "merged_tree_sha256": _sha256_tree(args.output),
    }
    (args.output / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
