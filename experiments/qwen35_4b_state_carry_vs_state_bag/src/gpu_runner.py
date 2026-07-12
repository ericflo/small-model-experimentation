"""GPU-only model loading, mechanics smoke, training, and evaluation.

All result-bearing arms stay on Hugging Face Transformers because the experiment
requires arbitrary internal layer calls.  Importing this module on a CPU setup
without the pinned training environment is expected to fail clearly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import platform
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

from .config import (
    MODEL_ID,
    MODEL_REVISION,
    config_sha256,
    require_confirmatory_config,
    source_contract_sha256,
)
from .data_pipeline import data_contract_sha256, read_jsonl
from .mechanics import recurrent_compute_receipt
from .state_loop_model import StateLoopModel
from .substrate import LETTERS, trajectory_targets


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
ANSWER_STRINGS = tuple(f" {letter}" for letter in LETTERS)
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _requirements_lock_sha256() -> str:
    if not REQUIREMENTS_LOCK.is_file():
        raise RuntimeError(f"pinned requirements lock is missing: {REQUIREMENTS_LOCK}")
    return _sha256(REQUIREMENTS_LOCK)


def _environment_receipt(*, include_device: bool) -> dict[str, Any]:
    import peft

    receipt: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if include_device:
        receipt["device"] = _require_cuda()
    return receipt


def _identity_receipt(config: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _requirements_lock_sha256(),
        "phase": phase,
    }


def _checkpoint_identity(metadata: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in metadata.items() if key != "checkpoint_identity_sha256"}
    )


def _with_receipt_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    return receipt


def _require_new_output(path: Path, *, kind: str, directory: bool) -> None:
    """Fail closed on every prior or partial artifact; result stages are non-resumable."""
    if path.exists():
        raise RuntimeError(f"refusing to resume or overwrite existing {kind}: {path}")
    if directory:
        path.mkdir(parents=True, exist_ok=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def _read_gate_receipt(
    path: Path,
    config: Mapping[str, Any],
    *,
    allowed_statuses: set[str],
    gate_name: str,
    expected_phase: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{gate_name} receipt is missing: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{gate_name} receipt is unreadable: {path}") from exc
    status = str(receipt.get("status", receipt.get("verdict", "")))
    if status not in allowed_statuses:
        raise RuntimeError(
            f"{gate_name} did not pass: expected one of {sorted(allowed_statuses)}, got {status!r}"
        )
    if receipt.get("phase") != expected_phase:
        raise RuntimeError(
            f"{gate_name} phase mismatch: {receipt.get('phase')!r} != {expected_phase!r}"
        )
    receipt_identity = receipt.get("receipt_identity_sha256")
    identity_payload = {
        key: value for key, value in receipt.items() if key != "receipt_identity_sha256"
    }
    if receipt_identity != _canonical_sha256(identity_payload):
        raise RuntimeError(f"{gate_name} receipt identity digest mismatch")
    expected = _identity_receipt(config, phase=str(receipt.get("phase", "gate")))
    for key in (
        "experiment_id",
        "config_sha256",
        "source_contract_sha256",
        "requirements_training_lock_sha256",
    ):
        if receipt.get(key) != expected[key]:
            raise RuntimeError(
                f"{gate_name} identity mismatch for {key}: {receipt.get(key)!r} != {expected[key]!r}"
            )
    return receipt


def _require_cuda(minimum_gib: float = 44.0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for every model-bearing stage")
    properties = torch.cuda.get_device_properties(0)
    gib = properties.total_memory / (1024**3)
    if gib < minimum_gib:
        raise RuntimeError(
            f"this experiment requires at least {minimum_gib:.0f} GiB VRAM; "
            f"device {properties.name} exposes {gib:.1f} GiB"
        )
    if not is_flash_linear_attention_available() or not is_causal_conv1d_available():
        raise RuntimeError(
            "Qwen3.5 fast path is incomplete. Rebuild flash-linear-attention and causal-conv1d "
            "exactly as docs/compute_environment.md specifies."
        )
    return {
        "name": properties.name,
        "total_memory_gib": gib,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }


def _render(tokenizer: Any, row: Mapping[str, Any], *, thinking: bool = False) -> str:
    messages = [{"role": "user", "content": row["prompt"]}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Qwen3.5's template accepts enable_thinking; fail if a future template drops it.
    kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def _without_workspace(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the same task without latent placeholders for the text baseline."""
    state_token = str(config["architecture"]["state_token"])
    lines = [
        line
        for line in str(row["prompt"]).splitlines()
        if state_token not in line
        and line.strip() != "Internal state workspace (the question deliberately comes later):"
    ]
    clean = dict(row)
    clean["prompt"] = "\n".join(lines).replace(
        "Answer with only the choice letter.",
        "Reason through each transition, then give the choice letter.",
    )
    return clean


def _validate_tokenizer(tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    state_token = str(config["architecture"]["state_token"])
    state_ids = tokenizer.encode(state_token, add_special_tokens=False)
    if len(state_ids) != 1 or state_ids[0] == tokenizer.unk_token_id:
        raise RuntimeError(f"state token must be one known token, got {state_ids}")
    answer_ids = []
    for answer in ANSWER_STRINGS:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"answer {answer!r} must be a single token, got {token_ids}")
        answer_ids.append(token_ids[0])
    if len(set(answer_ids)) != len(answer_ids):
        raise RuntimeError("answer-letter token IDs must be distinct")
    return {"state_token_id": state_ids[0], "answer_token_ids": answer_ids}


def _discover_loop_linear_targets(model: nn.Module, start: int, end: int) -> list[str]:
    targets = []
    pattern = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.")
    for name, module in model.named_modules():
        match = pattern.search(name)
        if not match or not isinstance(module, nn.Linear):
            continue
        layer = int(match.group(1))
        if start <= layer < end:
            targets.append(name)
    if not targets:
        raise RuntimeError("no Qwen loop-block linear modules were discovered for LoRA")
    return sorted(targets)


def _trainable_parameter_receipt(wrapper: StateLoopModel) -> dict[str, Any]:
    named = sorted(
        (
            (name, parameter)
            for name, parameter in wrapper.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    lora = sum(parameter.numel() for name, parameter in named if "lora_" in name)
    loop_modules = sum(parameter.numel() for name, parameter in named if "lora_" not in name)
    value_digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        value_digest.update(name.encode("utf-8"))
        value_digest.update(str(tensor.dtype).encode("ascii"))
        value_digest.update(str(tuple(tensor.shape)).encode("ascii"))
        value_digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return {
        "total": lora + loop_modules,
        "lora": lora,
        "state_modules": loop_modules,
        "tensor_count": len(named),
        "names_sha256": hashlib.sha256(
            "\n".join(name for name, _ in named).encode("utf-8")
        ).hexdigest(),
        "values_sha256": value_digest.hexdigest(),
    }


def _load_base(config: Mapping[str, Any]) -> tuple[Any, nn.Module, dict[str, Any]]:
    model_config = config["model"]
    expected_transformers = str(model_config["transformers_version"])
    if transformers.__version__ != expected_transformers:
        raise RuntimeError(
            f"Transformers drift: expected {expected_transformers}, got {transformers.__version__}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=bool(model_config["trust_remote_code"]),
    )
    token_receipt = _validate_tokenizer(tokenizer, config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=bool(model_config["trust_remote_code"]),
        dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash is not None and commit_hash != MODEL_REVISION:
        raise RuntimeError(
            f"loaded model commit {commit_hash} differs from pinned revision {MODEL_REVISION}"
        )
    model.requires_grad_(False)
    return tokenizer, model, token_receipt


def _attach_new_adapter(model: nn.Module, config: Mapping[str, Any]) -> tuple[nn.Module, list[str]]:
    architecture = config["architecture"]
    targets = _discover_loop_linear_targets(
        model, int(architecture["loop_start"]), int(architecture["loop_end"])
    )
    lora = architecture["lora"]
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            target_modules=targets,
        ),
    )
    bad = []
    layer_pattern = re.compile(r"\.layers\.(\d+)\.")
    for name, parameter in peft_model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = layer_pattern.search(name)
        if not match or not (
            int(architecture["loop_start"]) <= int(match.group(1)) < int(architecture["loop_end"])
        ):
            bad.append(name)
    if bad:
        raise RuntimeError(f"LoRA escaped the recurrent block: {bad[:8]}")
    return peft_model, targets


def _build_new(config: Mapping[str, Any]) -> tuple[Any, StateLoopModel, dict[str, Any]]:
    tokenizer, base, token_receipt = _load_base(config)
    peft_model, targets = _attach_new_adapter(base, config)
    wrapper = StateLoopModel(peft_model, config).cuda()
    receipt = {
        "tokenizer": token_receipt,
        "lora_targets": targets,
        "lora_targets_sha256": hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest(),
        "trainable_parameters": _trainable_parameter_receipt(wrapper),
        "environment": _environment_receipt(include_device=True),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _requirements_lock_sha256(),
    }
    return tokenizer, wrapper, receipt


def _validate_checkpoint_files(
    config: Mapping[str, Any],
    checkpoint: Path,
    metadata: Mapping[str, Any],
    *,
    require_loop_state: bool,
    expected_phase: str | None = None,
    expected_step: int | None = None,
    expected_seed: int | None = None,
) -> None:
    expected = {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _requirements_lock_sha256(),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint identity mismatch: {mismatches}")
    if expected_phase is not None and metadata.get("phase") != expected_phase:
        raise RuntimeError(
            f"checkpoint phase mismatch: {metadata.get('phase')!r} != {expected_phase!r}"
        )
    if expected_step is not None and int(metadata.get("step", -1)) != expected_step:
        raise RuntimeError(
            f"checkpoint step mismatch: {metadata.get('step')!r} != {expected_step}"
        )
    if expected_seed is not None and int(metadata.get("train_seed", -1)) != expected_seed:
        raise RuntimeError(
            f"checkpoint seed mismatch: {metadata.get('train_seed')!r} != {expected_seed}"
        )
    expected_identity = _checkpoint_identity(metadata)
    if metadata.get("checkpoint_identity_sha256") != expected_identity:
        raise RuntimeError("checkpoint identity digest mismatch")
    stored_environment = metadata.get("environment")
    current_environment = _environment_receipt(include_device=False)
    if not isinstance(stored_environment, Mapping):
        raise RuntimeError("checkpoint has no environment receipt")
    for key in ("python", "torch", "transformers", "peft", "cuda_runtime"):
        if stored_environment.get(key) != current_environment.get(key):
            raise RuntimeError(
                f"checkpoint runtime mismatch for {key}: "
                f"{stored_environment.get(key)!r} != {current_environment.get(key)!r}"
            )
    adapter_files = metadata.get("adapter_files")
    if not isinstance(adapter_files, Mapping) or not adapter_files:
        raise RuntimeError("checkpoint has no adapter-file hash receipt")
    for filename, expected_hash in adapter_files.items():
        path = checkpoint / "adapter" / str(filename)
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"checkpoint adapter hash mismatch: {filename}")
    if require_loop_state:
        loop_path = checkpoint / "loop_state.pt"
        if (
            not loop_path.is_file()
            or _sha256(loop_path) != metadata.get("loop_state_sha256")
        ):
            raise RuntimeError("checkpoint loop-state hash mismatch")


def _load_checkpoint(
    config: Mapping[str, Any],
    checkpoint: Path,
    *,
    trainable: bool = False,
    expected_phase: str | None = None,
    expected_step: int | None = None,
    expected_seed: int | None = None,
) -> tuple[Any, StateLoopModel, dict[str, Any]]:
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    _validate_checkpoint_files(
        config,
        checkpoint,
        metadata,
        require_loop_state=True,
        expected_phase=expected_phase,
        expected_step=expected_step,
        expected_seed=expected_seed,
    )
    tokenizer, base, token_receipt = _load_base(config)
    peft_model = PeftModel.from_pretrained(
        base, str(checkpoint / "adapter"), is_trainable=trainable
    )
    wrapper = StateLoopModel(peft_model, config).cuda()
    payload = torch.load(checkpoint / "loop_state.pt", map_location="cpu", weights_only=True)
    wrapper.load_extra_state_dict(payload)
    receipt = {
        "tokenizer": token_receipt,
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "trainable_parameters": _trainable_parameter_receipt(wrapper),
    }
    return tokenizer, wrapper, receipt


def _encode_row(
    tokenizer: Any,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    k: int,
    device: torch.device,
) -> dict[str, Any]:
    rendered = _render(tokenizer, row, thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    token_receipt = _validate_tokenizer(tokenizer, config)
    base_ids = input_ids[0].tolist()
    for answer, answer_id in zip(ANSWER_STRINGS, token_receipt["answer_token_ids"]):
        contextual = tokenizer.encode(rendered + answer, add_special_tokens=False)
        if contextual[:-1] != base_ids or contextual[-1] != answer_id:
            raise RuntimeError(
                f"answer tokenization is not prefix-stable in context for {answer!r}"
            )
    state_mask = input_ids.eq(int(token_receipt["state_token_id"]))
    count = int(state_mask.sum().item())
    expected = int(config["architecture"]["state_slots"])
    if count != expected:
        raise RuntimeError(f"tokenized prompt has {count} state slots, expected {expected}")
    query_character = rendered.index("Query:")
    query_token = len(tokenizer.encode(rendered[:query_character], add_special_tokens=False))
    if int(torch.where(state_mask)[1].max().item()) >= query_token:
        raise RuntimeError("state slot is not causally before the query")
    targets = trajectory_targets(dict(row), k)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "state_mask": state_mask,
        "answer_positions": torch.tensor([input_ids.shape[1] - 1], device=device),
        "answer_targets": torch.tensor(
            [token_receipt["answer_token_ids"][int(row["correct_choice"])]], device=device
        ),
        "state_targets": {
            name: torch.tensor([values], device=device, dtype=torch.long)
            for name, values in targets.items()
        },
        "depths": torch.tensor([int(row["depth"])], device=device, dtype=torch.long),
        "answer_token_ids": token_receipt["answer_token_ids"],
        "rendered_prompt": rendered,
        "prompt_tokens": input_ids.shape[1],
    }


def _forward(
    wrapper: StateLoopModel,
    batch: Mapping[str, Any],
    *,
    k: int,
    mode: str,
    state_override: Mapping[int, torch.Tensor] | None = None,
) -> Any:
    return wrapper(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state_mask=batch["state_mask"],
        answer_positions=batch["answer_positions"],
        answer_targets=batch["answer_targets"],
        state_targets=batch["state_targets"],
        depths=batch["depths"],
        k=k,
        mode=mode,
        state_override=state_override,
    )


def _choice_prediction(output: Any, answer_token_ids: Sequence[int]) -> tuple[int, bool, float]:
    logits = output.answer_logits[0]
    choice_logits = logits[torch.tensor(answer_token_ids, device=logits.device)]
    predicted = int(choice_logits.argmax().item())
    full_top = int(logits.argmax().item())
    probabilities = torch.softmax(logits.float(), dim=-1)
    answer_mass = float(probabilities[list(answer_token_ids)].sum().item())
    return predicted, full_top in set(answer_token_ids), answer_mass


def _gradient_receipt(wrapper: StateLoopModel) -> dict[str, Any]:
    groups = {"lora": 0.0, "state": 0.0, "sufficiency": 0.0, "step": 0.0}
    counts = {key: 0 for key in groups}
    for name, parameter in wrapper.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm().cpu())
        if "lora_" in name:
            group = "lora"
        elif "sufficiency" in name:
            group = "sufficiency"
        elif "step_encoder" in name:
            group = "step"
        else:
            group = "state"
        groups[group] += norm
        counts[group] += 1
    return {key: {"summed_norm": groups[key], "tensors": counts[key]} for key in groups}


def _k1_parity_error(wrapper: StateLoopModel, batch: Mapping[str, Any]) -> float:
    wrapper.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        k1 = _forward(wrapper, batch, k=1, mode="carry")
        with wrapper.peft_model.disable_adapter():
            direct = wrapper.core(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :]
    return float((k1.answer_logits - direct).abs().max().cpu())


def _smoke_row(config: Mapping[str, Any], *, worst_case: bool = False) -> dict[str, Any]:
    from .substrate import generate_example

    substrate = config["substrate"]
    architecture = config["architecture"]
    return generate_example(
        seed=99103 if worst_case else 99101,
        split="model_smoke",
        family=(substrate["heldout_family"] if worst_case else substrate["train_families"][0]),
        # Prose has the longest registered per-node rendering; paired with the
        # longest braided rule this is the conservative K=max geometry smoke.
        template="prose" if worst_case else substrate["train_templates"][0],
        depth=(int(architecture["max_recurrence"]) if worst_case else 4),
        node_count=int(substrate["node_count"]),
        checksum_modulus=int(substrate["checksum_modulus"]),
        num_choices=int(substrate["num_choices"]),
        state_token=architecture["state_token"],
        state_slots=int(architecture["state_slots"]),
        max_attempts=int(substrate["max_generation_attempts"]),
    )


def model_smoke(config: Mapping[str, Any], output: Path) -> None:
    require_confirmatory_config(config)
    _require_new_output(output, kind="model-smoke receipt", directory=False)
    started = time.time()
    device_receipt = _require_cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(99102)
    tokenizer, wrapper, setup = _build_new(config)
    row = _smoke_row(config)
    worst_row = _smoke_row(config, worst_case=True)
    batch = _encode_row(tokenizer, row, config, k=4, device=torch.device("cuda"))
    k1_batch = _encode_row(tokenizer, row, config, k=1, device=torch.device("cuda"))
    worst_k = int(config["architecture"]["max_recurrence"])
    worst_batch = _encode_row(
        tokenizer, worst_row, config, k=worst_k, device=torch.device("cuda")
    )
    wrapper.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        carry = _forward(wrapper, batch, k=4, mode="carry")
        bag = _forward(wrapper, batch, k=4, mode="bag")
        carry_k1 = _forward(wrapper, k1_batch, k=1, mode="carry")
        bag_k1 = _forward(wrapper, k1_batch, k=1, mode="bag")
    torch.cuda.synchronize()
    worst_started = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        carry_worst = _forward(wrapper, worst_batch, k=worst_k, mode="carry")
        bag_worst = _forward(wrapper, worst_batch, k=worst_k, mode="bag")
    torch.cuda.synchronize()
    worst_forward_seconds = time.time() - worst_started
    max_error = _k1_parity_error(wrapper, k1_batch)
    allowed = float(config["gates"]["k1_max_logit_abs_error"])
    if max_error > allowed:
        raise RuntimeError(f"K=1 parity failed: max logit error {max_error} > {allowed}")
    if carry.answer_logits.shape != bag.answer_logits.shape:
        raise RuntimeError("carry and bag answer shapes differ")
    k1_mode_error = float((carry_k1.answer_logits - bag_k1.answer_logits).abs().max().cpu())
    if k1_mode_error != 0.0:
        raise RuntimeError(f"Carry/Bag K=1 equality failed: {k1_mode_error}")
    worst_finite = {
        "carry": bool(torch.isfinite(carry_worst.answer_logits).all().item()),
        "bag": bool(torch.isfinite(bag_worst.answer_logits).all().item()),
    }
    if not all(worst_finite.values()):
        raise RuntimeError(f"worst-K forward produced nonfinite logits: {worst_finite}")

    arm_parameter_receipts = {
        arm: dict(setup["trainable_parameters"]) for arm in ("carry", "bag")
    }
    parameter_receipts_equal = (
        _canonical_sha256(arm_parameter_receipts["carry"])
        == _canonical_sha256(arm_parameter_receipts["bag"])
    )
    if not parameter_receipts_equal:
        raise RuntimeError("Carry/Bag shared-wrapper parameter receipts differ")
    gradients_by_arm = {}
    backward_seconds = {}
    wrapper.train()
    for arm in ("carry", "bag"):
        wrapper.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        backward_started = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backward = _forward(wrapper, batch, k=4, mode=arm)
        assert backward.loss is not None
        backward.loss.backward()
        torch.cuda.synchronize()
        backward_seconds[arm] = time.time() - backward_started
        gradients = _gradient_receipt(wrapper)
        for required in ("lora", "state", "sufficiency", "step"):
            norm = float(gradients[required]["summed_norm"])
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"model smoke found no {required} gradient for {arm}")
        gradients_by_arm[arm] = gradients
    if not math.isfinite(max_error):
        raise RuntimeError("K=1 parity produced a nonfinite error")

    receipt = _with_receipt_identity({
        "status": "MODEL_SMOKE_PASS",
        **_identity_receipt(config, phase="g0"),
        "scientific_evidence": False,
        "device": device_receipt,
        "environment": setup["environment"],
        "setup": setup,
        "prompt_tokens": batch["prompt_tokens"],
        "state_slots": int(batch["state_mask"].sum().item()),
        "k1_max_logit_abs_error": max_error,
        "k1_carry_bag_max_logit_abs_error": k1_mode_error,
        "k1_allowed_error": allowed,
        "carry_bag_shared_wrapper": True,
        "carry_bag_parameter_receipts": arm_parameter_receipts,
        "carry_bag_parameter_receipts_equal": parameter_receipts_equal,
        "carry_shape": list(carry.answer_logits.shape),
        "bag_shape": list(bag.answer_logits.shape),
        "gradient_receipts": gradients_by_arm,
        "k4_forward_backward_seconds": backward_seconds,
        "worst_k": worst_k,
        "worst_k_prompt_tokens": worst_batch["prompt_tokens"],
        "worst_k_forward_seconds": worst_forward_seconds,
        "worst_k_finite": worst_finite,
        "worst_k_shapes": {
            "carry": list(carry_worst.answer_logits.shape),
            "bag": list(bag_worst.answer_logits.shape),
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "elapsed_seconds": time.time() - started,
    })
    _write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def _data_dir(config: Mapping[str, Any]) -> Path:
    path = (ROOT / config["paths"]["data_dir"]).resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"prepared data missing at {path}; run scripts/run.py --stage prepare-data first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != config["experiment_id"]:
        raise RuntimeError("prepared-data experiment identity mismatch")
    if manifest.get("data_contract_sha256") != data_contract_sha256(config):
        raise RuntimeError("prepared data does not match this substrate contract")
    if (
        manifest.get("benchmark_files_read") != 0
        or manifest.get("cross_split_structural_duplicates") != 0
    ):
        raise RuntimeError("prepared-data contamination receipt is not clean")
    files = manifest.get("files", {})
    required_splits = {
        "train",
        "validation",
        "depth_extrapolation",
        "family_holdout",
        "template_holdout",
        "joint_holdout",
        "counterfactual",
        "pilot_depth",
        "pilot_joint",
        "pilot_counterfactual",
        "pilot_validation",
    }
    if set(files) != required_splits:
        raise RuntimeError("prepared-data manifest has missing or unexpected splits")
    for split, receipt in files.items():
        data_path = path / f"{split}.jsonl.gz"
        if not data_path.exists() or _sha256(data_path) != receipt.get("sha256"):
            raise RuntimeError(f"prepared-data hash mismatch for {split}")
    return path


def _schedule(step: int, total: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _save_checkpoint(
    wrapper: StateLoopModel,
    config: Mapping[str, Any],
    output: Path,
    *,
    arm: str,
    seed: int,
    step: int,
    phase: str,
    pilot: bool,
    setup: Mapping[str, Any],
    data_manifest_sha256: str,
    training_prompt_tokens: int,
    training_layer_token_applications: int,
    training_order_sha256: str,
) -> None:
    _require_new_output(output, kind="checkpoint", directory=True)
    wrapper.peft_model.save_pretrained(output / "adapter", safe_serialization=True)
    loop_state_path = output / "loop_state.pt"
    torch.save(wrapper.extra_state_dict(), loop_state_path)
    metadata = {
        "schema_version": 1,
        **_identity_receipt(config, phase=phase),
        "train_arm": arm,
        "train_seed": seed,
        "step": step,
        "pilot": pilot,
        "trainable_parameters": setup["trainable_parameters"],
        "environment": setup["environment"],
        "data_manifest_sha256": data_manifest_sha256,
        "training_prompt_tokens": training_prompt_tokens,
        "training_layer_token_applications": training_layer_token_applications,
        "training_order_sha256": training_order_sha256,
        "adapter_files": {},
        "loop_state_sha256": _sha256(loop_state_path),
    }
    for path in (output / "adapter").glob("*"):
        if path.is_file():
            metadata["adapter_files"][path.name] = _sha256(path)
    metadata["checkpoint_identity_sha256"] = _checkpoint_identity(metadata)
    _write_json(output / "checkpoint.json", metadata)


def _validation_accuracy(
    wrapper: StateLoopModel,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    arm: str,
    k: int,
    limit: int = 128,
) -> float:
    wrapper.eval()
    correct = 0
    with torch.no_grad():
        for row in rows[:limit]:
            batch = _encode_row(tokenizer, row, config, k=k, device=torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _forward(wrapper, batch, k=k, mode=arm)
            predicted, _, _ = _choice_prediction(output, batch["answer_token_ids"])
            correct += predicted == int(row["correct_choice"])
    wrapper.train()
    return correct / min(len(rows), limit)


def train(
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    pilot: bool = False,
    model_smoke_receipt: Path,
    promotion_receipt: Path | None = None,
) -> None:
    require_confirmatory_config(config)
    _require_cuda()
    if arm not in {"carry", "bag"}:
        raise ValueError(arm)
    training = config["training"]
    expected_seeds = (
        {int(training["pilot_seed"])}
        if pilot
        else set(map(int, training["train_seeds"]))
    )
    if seed not in expected_seeds:
        raise RuntimeError(
            f"seed {seed} is not registered for {'pilot' if pilot else 'full'} training: "
            f"{sorted(expected_seeds)}"
        )
    _read_gate_receipt(
        model_smoke_receipt,
        config,
        allowed_statuses={"MODEL_SMOKE_PASS"},
        gate_name="G0 model-smoke",
        expected_phase="g0",
    )
    if not pilot:
        if promotion_receipt is None:
            raise RuntimeError("full training requires the G1 promotion receipt")
        _read_gate_receipt(
            promotion_receipt,
            config,
            allowed_statuses={"PILOT_PROMOTION_READY"},
            gate_name="G1 pilot promotion",
            expected_phase="pilot",
        )
    _require_new_output(
        output_dir,
        kind=f"{'pilot' if pilot else 'full'} {arm} training run",
        directory=True,
    )
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer, wrapper, setup = _build_new(config)
    data_dir = _data_dir(config)
    data_manifest_hash = _sha256(data_dir / "manifest.json")
    train_rows = read_jsonl(data_dir / "train.jsonl.gz")
    validation_split = "pilot_validation" if pilot else "validation"
    validation = read_jsonl(data_dir / f"{validation_split}.jsonl.gz")
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    total_steps = int(training["pilot_steps"] if pilot else training["train_steps"])
    accumulation = int(training["gradient_accumulation"])
    train_k = int(training["train_k"])
    parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    metrics_path = output_dir / "train_metrics.jsonl"
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    example_index = 0
    training_prompt_tokens = 0
    training_layer_token_applications = 0
    training_order_digest = hashlib.sha256()
    loop_layers = int(config["architecture"]["loop_end"]) - int(
        config["architecture"]["loop_start"]
    )
    started = time.time()
    for step in range(1, total_steps + 1):
        totals = {"loss": 0.0, "answer": 0.0, "state": 0.0, "fixed": 0.0}
        for _ in range(accumulation):
            row = train_rows[example_index % len(train_rows)]
            example_index += 1
            batch = _encode_row(
                tokenizer, row, config, k=train_k, device=torch.device("cuda")
            )
            example_layer_token_applications = recurrent_compute_receipt(
                sequence_tokens=int(batch["prompt_tokens"]),
                total_layers=int(config["architecture"]["expected_num_layers"]),
                loop_layers=loop_layers,
                k=train_k,
            ).total_layer_token_applications
            training_order_digest.update(
                json.dumps(
                    {
                        "index": example_index,
                        "id": row["id"],
                        "k": train_k,
                        "prompt_tokens": int(batch["prompt_tokens"]),
                        "layer_token_applications": example_layer_token_applications,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            training_order_digest.update(b"\n")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _forward(wrapper, batch, k=train_k, mode=arm)
                if output.loss is None:
                    raise RuntimeError("training forward returned no loss")
                if not bool(torch.isfinite(output.loss).item()):
                    raise RuntimeError(f"nonfinite training loss at optimizer step {step}")
                scaled = output.loss / accumulation
            scaled.backward()
            training_prompt_tokens += int(batch["prompt_tokens"])
            training_layer_token_applications += example_layer_token_applications
            totals["loss"] += float(output.loss.detach().cpu()) / accumulation
            totals["answer"] += float(output.answer_loss.detach().cpu()) / accumulation
            totals["state"] += float(output.state_loss.detach().cpu()) / accumulation
            totals["fixed"] += float(output.fixed_point_loss.detach().cpu()) / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["gradient_clip"])
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"nonfinite gradient norm at optimizer step {step}")
        multiplier = _schedule(step - 1, total_steps, float(training["warmup_fraction"]))
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate"]) * multiplier
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            _append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "arm": arm,
                    "seed": seed,
                    **totals,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
                    "elapsed_seconds": time.time() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                },
            )
        if step % int(training["eval_every_steps"]) == 0 or step == total_steps:
            accuracy = _validation_accuracy(
                wrapper, tokenizer, validation, config, arm=arm, k=train_k
            )
            _append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "arm": arm,
                    "seed": seed,
                    "validation_accuracy": accuracy,
                    "event": "validation",
                },
            )
        if step % int(training["save_every_steps"]) == 0 or step == total_steps:
            checkpoint = output_dir / f"checkpoint_{step:06d}"
            _save_checkpoint(
                wrapper,
                config,
                checkpoint,
                arm=arm,
                seed=seed,
                step=step,
                phase="pilot" if pilot else "full",
                pilot=pilot,
                setup=setup,
                data_manifest_sha256=data_manifest_hash,
                training_prompt_tokens=training_prompt_tokens,
                training_layer_token_applications=training_layer_token_applications,
                training_order_sha256=training_order_digest.hexdigest(),
            )
    _write_json(
        output_dir / "run.json",
        _with_receipt_identity({
            "status": "TRAINING_COMPLETE",
            **_identity_receipt(config, phase="pilot" if pilot else "full"),
            "arm": arm,
            "seed": seed,
            "steps": total_steps,
            "pilot": pilot,
            "data_manifest_sha256": data_manifest_hash,
            "training_prompt_tokens": training_prompt_tokens,
            "training_layer_token_applications": training_layer_token_applications,
            "training_order_sha256": training_order_digest.hexdigest(),
            "elapsed_seconds": time.time() - started,
            "environment": setup["environment"],
            "setup": setup,
        }),
    )


def _evaluate_cell(
    wrapper: StateLoopModel,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    train_arm: str,
    eval_mode: str,
    split: str,
    k: int,
    output_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    correct = node_correct = phase_correct = checksum_correct = 0
    node_step_total = phase_step_total = checksum_step_total = joint_step_total = 0.0
    full_top_answer = 0
    answer_mass = 0.0
    count = min(len(rows), limit) if limit else len(rows)
    loop_layers = int(config["architecture"]["loop_end"]) - int(config["architecture"]["loop_start"])
    wrapper.eval()
    with torch.no_grad():
        for row in rows[:count]:
            batch = _encode_row(tokenizer, row, config, k=k, device=torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _forward(wrapper, batch, k=k, mode=eval_mode)
            predicted, top_is_answer, mass = _choice_prediction(
                output, batch["answer_token_ids"]
            )
            is_correct = predicted == int(row["correct_choice"])
            correct += is_correct
            full_top_answer += top_is_answer
            answer_mass += mass
            node_predictions = output.node_logits.argmax(dim=-1)
            phase_predictions = output.phase_logits.argmax(dim=-1)
            checksum_predictions = output.checksum_logits.argmax(dim=-1)
            node_matches = node_predictions == batch["state_targets"]["node"]
            phase_matches = phase_predictions == batch["state_targets"]["phase"]
            checksum_matches = checksum_predictions == batch["state_targets"]["checksum"]
            joint_matches = node_matches & phase_matches & checksum_matches
            state_delta_rms_by_step = [
                float(
                    (output.states[index] - output.states[index - 1])
                    .float()
                    .pow(2)
                    .mean()
                    .sqrt()
                    .cpu()
                )
                for index in range(1, len(output.states))
            ]
            node_correct += int(node_matches.all().item())
            phase_correct += int(phase_matches.all().item())
            checksum_correct += int(
                checksum_matches.all().item()
            )
            node_step_accuracy = float(node_matches.float().mean().item())
            phase_step_accuracy = float(phase_matches.float().mean().item())
            checksum_step_accuracy = float(checksum_matches.float().mean().item())
            joint_step_accuracy = float(joint_matches.float().mean().item())
            node_step_total += node_step_accuracy
            phase_step_total += phase_step_accuracy
            checksum_step_total += checksum_step_accuracy
            joint_step_total += joint_step_accuracy
            compute = recurrent_compute_receipt(
                sequence_tokens=batch["prompt_tokens"],
                total_layers=int(config["architecture"]["expected_num_layers"]),
                loop_layers=loop_layers,
                k=k,
            )
            _append_jsonl(
                output_path,
                {
                    "id": row["id"],
                    "split": split,
                    "family": row["family"],
                    "template": row["template"],
                    "query_kind": row["query_kind"],
                    "depth": row["depth"],
                    "k": k,
                    "train_arm": train_arm,
                    "eval_mode": eval_mode,
                    "correct_choice": row["correct_choice"],
                    "predicted_choice": predicted,
                    "correct": is_correct,
                    "full_top_is_answer": top_is_answer,
                    "answer_token_mass": mass,
                    "node_trajectory_exact": bool(node_matches.all().item()),
                    "phase_trajectory_exact": bool(phase_matches.all().item()),
                    "checksum_trajectory_exact": bool(checksum_matches.all().item()),
                    "node_step_accuracy": node_step_accuracy,
                    "phase_step_accuracy": phase_step_accuracy,
                    "checksum_step_accuracy": checksum_step_accuracy,
                    "joint_step_accuracy": joint_step_accuracy,
                    "state_delta_rms_by_step": state_delta_rms_by_step,
                    "mean_state_delta_rms": (
                        sum(state_delta_rms_by_step) / len(state_delta_rms_by_step)
                        if state_delta_rms_by_step
                        else 0.0
                    ),
                    "loop_diagnostics": output.diagnostics,
                    "prompt_tokens": batch["prompt_tokens"],
                    "layer_token_applications": compute.total_layer_token_applications,
                },
            )
    return {
        "split": split,
        "k": k,
        "train_arm": train_arm,
        "eval_mode": eval_mode,
        "n": count,
        "accuracy": correct / count,
        "node_trajectory_exact": node_correct / count,
        "phase_trajectory_exact": phase_correct / count,
        "checksum_trajectory_exact": checksum_correct / count,
        "node_step_accuracy": node_step_total / count,
        "phase_step_accuracy": phase_step_total / count,
        "checksum_step_accuracy": checksum_step_total / count,
        "joint_step_accuracy": joint_step_total / count,
        "full_top_is_answer": full_top_answer / count,
        "mean_answer_token_mass": answer_mass / count,
    }


def _counterfactual_swaps(
    wrapper: StateLoopModel,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    output_path: Path,
    *,
    limit_pairs: int | None,
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    selected = sorted(pairs.items())[:limit_pairs] if limit_pairs else sorted(pairs.items())
    donor_follow = recipient_preserve = baseline_correct = baseline_donor_follow = 0
    evaluated_pairs = evaluated_directions = 0
    wrapper.eval()
    with torch.no_grad():
        for pair_id, pair in selected:
            if len(pair) != 2:
                raise RuntimeError(f"counterfactual pair {pair_id} does not have exactly two rows")
            first, second = pair
            if int(first["depth"]) != int(second["depth"]):
                raise RuntimeError(f"counterfactual pair {pair_id} has unequal depths")
            k = int(first["depth"])
            swap_step = max(1, k // 2)
            first_batch = _encode_row(tokenizer, first, config, k=k, device=torch.device("cuda"))
            second_batch = _encode_row(tokenizer, second, config, k=k, device=torch.device("cuda"))
            geometry_equal = (
                first_batch["input_ids"].shape == second_batch["input_ids"].shape
                and torch.equal(first_batch["state_mask"], second_batch["state_mask"])
                and torch.equal(
                    first_batch["answer_positions"], second_batch["answer_positions"]
                )
                and int(first_batch["prompt_tokens"]) == int(second_batch["prompt_tokens"])
            )
            if not geometry_equal:
                raise RuntimeError(
                    f"counterfactual pair {pair_id} has incompatible token/state geometry"
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                first_output = _forward(wrapper, first_batch, k=k, mode="carry")
                second_output = _forward(wrapper, second_batch, k=k, mode="carry")
            for direction, donor, recipient, donor_batch, recipient_batch, donor_output, recipient_output in (
                (
                    "a_to_b",
                    first,
                    second,
                    first_batch,
                    second_batch,
                    first_output,
                    second_output,
                ),
                (
                    "b_to_a",
                    second,
                    first,
                    second_batch,
                    first_batch,
                    second_output,
                    first_output,
                ),
            ):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    swapped_output = _forward(
                        wrapper,
                        recipient_batch,
                        k=k,
                        mode="carry",
                        state_override={swap_step: donor_output.states[swap_step - 1]},
                    )
                donor_value = donor["choices"][donor["correct_choice"]]
                donor_choice_in_recipient = recipient["choices"].index(donor_value)
                baseline_prediction, _, _ = _choice_prediction(
                    recipient_output, recipient_batch["answer_token_ids"]
                )
                donor_baseline_prediction, _, _ = _choice_prediction(
                    donor_output, donor_batch["answer_token_ids"]
                )
                swapped_prediction, _, _ = _choice_prediction(
                    swapped_output, recipient_batch["answer_token_ids"]
                )
                baseline_hit = baseline_prediction == int(recipient["correct_choice"])
                baseline_follows_donor = baseline_prediction == donor_choice_in_recipient
                follows = swapped_prediction == donor_choice_in_recipient
                preserves = swapped_prediction == int(recipient["correct_choice"])
                baseline_correct += baseline_hit
                baseline_donor_follow += baseline_follows_donor
                donor_follow += follows
                recipient_preserve += preserves
                evaluated_directions += 1
                _append_jsonl(
                    output_path,
                    {
                        "pair_id": pair_id,
                        "direction": direction,
                        "depth": k,
                        "swap_step": swap_step,
                        "geometry_equal": True,
                        "prompt_tokens": int(recipient_batch["prompt_tokens"]),
                        "baseline_prediction": baseline_prediction,
                        "donor_baseline_prediction": donor_baseline_prediction,
                        "swapped_prediction": swapped_prediction,
                        "recipient_choice": recipient["correct_choice"],
                        "donor_choice_in_recipient": donor_choice_in_recipient,
                        "baseline_correct": baseline_hit,
                        "baseline_recipient_correct": baseline_hit,
                        "baseline_donor_follow": baseline_follows_donor,
                        "donor_baseline_correct": (
                            donor_baseline_prediction == int(donor["correct_choice"])
                        ),
                        "donor_follow": follows,
                        "recipient_preserve": preserves,
                    },
                )
            evaluated_pairs += 1
    return {
        "pairs": evaluated_pairs,
        "directions": evaluated_directions,
        "baseline_accuracy": baseline_correct / evaluated_directions,
        "baseline_donor_follow_rate": baseline_donor_follow / evaluated_directions,
        "donor_follow_rate": donor_follow / evaluated_directions,
        "recipient_preserve_rate": recipient_preserve / evaluated_directions,
    }


def evaluate(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    arm: str,
    expected_seed: int,
    output_dir: Path,
    pilot: bool = False,
) -> None:
    require_confirmatory_config(config)
    _require_cuda()
    _require_new_output(output_dir, kind="evaluation run", directory=True)
    expected_phase = "pilot" if pilot else "full"
    expected_step = int(
        config["training"]["pilot_steps"] if pilot else config["training"]["train_steps"]
    )
    tokenizer, wrapper, setup = _load_checkpoint(
        config,
        checkpoint,
        expected_phase=expected_phase,
        expected_step=expected_step,
        expected_seed=expected_seed,
    )
    train_arm = setup["checkpoint_metadata"]["train_arm"]
    if bool(setup["checkpoint_metadata"].get("pilot")) != pilot:
        raise RuntimeError("checkpoint pilot flag disagrees with evaluation phase")
    if arm not in {"carry", "bag"}:
        raise ValueError(arm)
    data_dir = _data_dir(config)
    data_manifest_hash = _sha256(data_dir / "manifest.json")
    if setup["checkpoint_metadata"].get("data_manifest_sha256") != data_manifest_hash:
        raise RuntimeError("checkpoint and evaluation data manifests differ")
    checkpoint_k1_error = None
    if train_arm in {"carry", "bag"}:
        parity_split = "pilot_validation" if pilot else "validation"
        parity_row = read_jsonl(data_dir / f"{parity_split}.jsonl.gz")[0]
        parity_batch = _encode_row(
            tokenizer, parity_row, config, k=1, device=torch.device("cuda")
        )
        checkpoint_k1_error = _k1_parity_error(wrapper, parity_batch)
        allowed = float(config["gates"]["k1_max_logit_abs_error"])
        if not math.isfinite(checkpoint_k1_error) or checkpoint_k1_error > allowed:
            raise RuntimeError(
                f"checkpoint K=1 parity failed: {checkpoint_k1_error} > {allowed}"
            )
    rows_path = output_dir / "rows.jsonl"
    summaries = []
    started = time.time()
    evaluation_config = config["evaluation"]

    def run_cell(split: str, rows: Sequence[dict[str, Any]], k: int, limit: int | None) -> None:
        summaries.append(
            _evaluate_cell(
                wrapper,
                tokenizer,
                rows,
                config,
                train_arm=train_arm,
                eval_mode=arm,
                split=split,
                k=k,
                output_path=rows_path,
                limit=limit,
            )
        )

    edge_cut = train_arm == "carry" and arm == "bag"
    if edge_cut:
        if pilot:
            raise RuntimeError("edge-cut evaluation is a full-phase causal endpoint")
        depth_rows = read_jsonl(data_dir / "depth_extrapolation.jsonl.gz")
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in depth_rows:
            grouped.setdefault(int(row["depth"]), []).append(row)
        for depth, group in sorted(grouped.items()):
            run_cell("depth_extrapolation", group, depth, None)
    elif pilot:
        depth_rows = read_jsonl(data_dir / "pilot_depth.jsonl.gz")
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for row in depth_rows:
            by_depth.setdefault(int(row["depth"]), []).append(row)
        for depth, group in sorted(by_depth.items()):
            for k in sorted({int(config["training"]["train_k"]), depth}):
                run_cell("pilot_depth", group, k, None)
        joint_rows = read_jsonl(data_dir / "pilot_joint.jsonl.gz")
        joint_grouped: dict[int, list[dict[str, Any]]] = {}
        for row in joint_rows:
            joint_grouped.setdefault(int(row["depth"]), []).append(row)
        for depth, group in sorted(joint_grouped.items()):
            run_cell("pilot_joint", group, depth, None)
    else:
        validation = read_jsonl(data_dir / "validation.jsonl.gz")
        for k in (1, 4, 8, 12):
            run_cell(
                "validation",
                validation,
                k,
                None,
            )

        depth_rows = read_jsonl(data_dir / "depth_extrapolation.jsonl.gz")
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for row in depth_rows:
            by_depth.setdefault(int(row["depth"]), []).append(row)
        for depth, group in sorted(by_depth.items()):
            full_ks = {int(config["training"]["train_k"]), depth}
            for k in evaluation_config["k_values"]:
                run_cell(
                    "depth_extrapolation",
                    group,
                    int(k),
                    None
                    if int(k) in full_ks
                    else int(evaluation_config["curve_items_per_depth"]),
                )

        holdout_splits = (
            "family_holdout",
            "template_holdout",
            "joint_holdout",
        )
        for split in holdout_splits:
            rows = read_jsonl(data_dir / f"{split}.jsonl.gz")
            grouped: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(int(row["depth"]), []).append(row)
            for depth, group in sorted(grouped.items()):
                run_cell(
                    split,
                    group,
                    depth,
                    int(evaluation_config["holdout_items_per_depth"]),
                )
    swap_summary = None
    if arm == "carry":
        swap_split = "pilot_counterfactual" if pilot else "counterfactual"
        counterfactual = read_jsonl(data_dir / f"{swap_split}.jsonl.gz")
        swap_path = output_dir / "counterfactual_swaps.jsonl"
        swap_summary = _counterfactual_swaps(
            wrapper,
            tokenizer,
            counterfactual,
            config,
            swap_path,
            limit_pairs=None,
        )
        swap_summary["counterfactual_swap_row_file"] = swap_path.name
        swap_summary["counterfactual_swap_row_file_sha256"] = _sha256(swap_path)
    receipt = _with_receipt_identity({
        "status": "EVALUATION_COMPLETE",
        **_identity_receipt(config, phase=expected_phase),
        "pilot": pilot,
        "edge_cut_primary_only": edge_cut,
        "train_arm": train_arm,
        "eval_mode": arm,
        "expected_seed": expected_seed,
        "checkpoint": str(checkpoint),
        "checkpoint_identity_sha256": setup["checkpoint_metadata"][
            "checkpoint_identity_sha256"
        ],
        "data_manifest_sha256": data_manifest_hash,
        "checkpoint_k1_max_logit_abs_error": checkpoint_k1_error,
        "backend": "transformers",
        "summaries": summaries,
        "counterfactual_swaps": swap_summary,
        "row_file": rows_path.name,
        "row_file_sha256": _sha256(rows_path),
        "elapsed_seconds": time.time() - started,
        "environment": _environment_receipt(include_device=False),
        "setup": setup,
    })
    _write_json(output_dir / "summary.json", receipt)


def train_text_baseline(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: Path,
    mechanism_receipt: Path,
) -> None:
    """Train the explicit-state-trace baseline on the identical procedural rows.

    This path intentionally uses the same loop-layer LoRA parameterization but
    standard Qwen autoregression.  It is phase-gated and should run only after
    the carry-vs-bag mechanism gate opens.
    """
    require_confirmatory_config(config)
    _require_cuda()
    expected_seeds = set(map(int, config["training"]["train_seeds"]))
    if seed not in expected_seeds:
        raise RuntimeError(f"text seed {seed} is not registered: {sorted(expected_seeds)}")
    _read_gate_receipt(
        mechanism_receipt,
        config,
        allowed_statuses={"MECHANISTIC_DEPTH_POSITIVE"},
        gate_name="G2/G3 mechanism",
        expected_phase="full",
    )
    _require_new_output(output_dir, kind="text-baseline training run", directory=True)
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer, base, token_receipt = _load_base(config)
    peft_model, targets = _attach_new_adapter(base, config)
    trainable_receipt = _trainable_parameter_receipt(peft_model)
    data_dir = _data_dir(config)
    rows = read_jsonl(data_dir / "train.jsonl.gz")
    random.Random(seed).shuffle(rows)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    metrics = output_dir / "train_metrics.jsonl"
    peft_model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(training["gradient_accumulation"])
    total_steps = int(training["train_steps"])
    index = 0
    training_order_digest = hashlib.sha256()
    training_prompt_tokens = 0
    training_layer_token_applications = 0
    for step in range(1, total_steps + 1):
        loss_sum = 0.0
        for _ in range(accumulation):
            row = rows[index % len(rows)]
            index += 1
            prompt = _render(tokenizer, _without_workspace(row, config), thinking=True)
            trace_parts = []
            for transition_index, state in enumerate(row["trajectory"][1:], start=1):
                label = row["world"]["labels"][state["node"]]
                trace_parts.append(
                    f"Step {transition_index}: node={label}, phase={state['phase']}, "
                    f"checksum={state['checksum']}."
                )
            target = " ".join(trace_parts) + f"\n</think>\n\n{row['answer_letter']}<|im_end|>"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(prompt + target, add_special_tokens=False)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError("text-baseline target retokenized the frozen prompt prefix")
            target_ids = full_ids[len(prompt_ids) :]
            example_layer_token_applications = len(full_ids) * int(
                config["architecture"]["expected_num_layers"]
            )
            training_order_digest.update(
                json.dumps(
                    {
                        "index": index,
                        "id": row["id"],
                        "k": None,
                        "prompt_tokens": len(prompt_ids),
                        "layer_token_applications": example_layer_token_applications,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            training_order_digest.update(b"\n")
            training_prompt_tokens += len(prompt_ids)
            training_layer_token_applications += example_layer_token_applications
            ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
            labels = torch.tensor(
                [[-100] * len(prompt_ids) + target_ids], device="cuda", dtype=torch.long
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = peft_model(input_ids=ids, labels=labels, use_cache=False).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"nonfinite text-baseline loss at optimizer step {step}")
            (loss / accumulation).backward()
            loss_sum += float(loss.detach().cpu()) / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in peft_model.parameters() if p.requires_grad],
            float(training["gradient_clip"]),
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"nonfinite text-baseline gradient at optimizer step {step}")
        multiplier = _schedule(
            step - 1, total_steps, float(training["warmup_fraction"])
        )
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate"]) * multiplier
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            _append_jsonl(
                metrics,
                {
                    "step": step,
                    "loss": loss_sum,
                    "seed": seed,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
                },
            )
    peft_model.save_pretrained(output_dir / "adapter", safe_serialization=True)
    adapter_files = {
        path.name: _sha256(path)
        for path in (output_dir / "adapter").glob("*")
        if path.is_file()
    }
    checkpoint_metadata = {
            "schema_version": 1,
            **_identity_receipt(config, phase="text"),
            "train_arm": "text_baseline",
            "train_seed": seed,
            "step": total_steps,
            "pilot": False,
            "data_manifest_sha256": _sha256(data_dir / "manifest.json"),
            "lora_targets": targets,
            "trainable_parameters": trainable_receipt,
            "environment": _environment_receipt(include_device=True),
            "tokenizer": token_receipt,
            "adapter_files": adapter_files,
            "training_prompt_tokens": training_prompt_tokens,
            "training_layer_token_applications": training_layer_token_applications,
            "training_order_sha256": training_order_digest.hexdigest(),
        }
    checkpoint_metadata["checkpoint_identity_sha256"] = _checkpoint_identity(
        checkpoint_metadata
    )
    _write_json(output_dir / "checkpoint.json", checkpoint_metadata)
    _write_json(
        output_dir / "run.json",
        _with_receipt_identity(
            {
                "status": "TEXT_TRAINING_COMPLETE",
                **_identity_receipt(config, phase="text"),
                "seed": seed,
                "step": total_steps,
                "checkpoint_identity_sha256": checkpoint_metadata[
                    "checkpoint_identity_sha256"
                ],
                "training_prompt_tokens": training_prompt_tokens,
                "training_layer_token_applications": training_layer_token_applications,
                "training_order_sha256": training_order_digest.hexdigest(),
                "environment": checkpoint_metadata["environment"],
            }
        ),
    )


def _parse_generated_choice(text: str) -> int | None:
    if "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[1]
    matches = re.findall(
        r"(?:Answer\s*:\s*)?\b([ABCD])\b", visible, flags=re.IGNORECASE
    )
    if not matches:
        return None
    return LETTERS.index(matches[-1].upper())


def evaluate_sample_more(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    expected_seed: int,
    output_dir: Path,
    mechanism_receipt: Path,
) -> None:
    """Explicit-CoT sample-more comparator under frozen layer-token allocation."""
    require_confirmatory_config(config)
    _require_cuda()
    _read_gate_receipt(
        mechanism_receipt,
        config,
        allowed_statuses={"MECHANISTIC_DEPTH_POSITIVE"},
        gate_name="G2/G3 mechanism",
        expected_phase="full",
    )
    _require_new_output(output_dir, kind="sample-more evaluation", directory=True)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata["train_arm"] != "text_baseline":
        raise RuntimeError("sample-more requires an explicit text-baseline checkpoint")
    _validate_checkpoint_files(
        config,
        checkpoint,
        metadata,
        require_loop_state=False,
        expected_phase="text",
        expected_step=int(config["training"]["train_steps"]),
        expected_seed=expected_seed,
    )
    tokenizer, base, token_receipt = _load_base(config)
    model = PeftModel.from_pretrained(
        base, str(checkpoint / "adapter"), is_trainable=False
    )
    model.eval()
    data_dir = _data_dir(config)
    data_manifest_hash = _sha256(data_dir / "manifest.json")
    if metadata.get("data_manifest_sha256") != data_manifest_hash:
        raise RuntimeError("text checkpoint and sample-more data manifests differ")
    rows = read_jsonl(data_dir / "depth_extrapolation.jsonl.gz")
    rows_path = output_dir / "rows.jsonl"
    layer_count = int(config["architecture"]["expected_num_layers"])
    loop_layers = int(config["architecture"]["loop_end"]) - int(
        config["architecture"]["loop_start"]
    )
    sampling = config["evaluation"]["sample_more"]
    total_generation_seconds = 0.0
    total_sampled_tokens = 0
    total_allocated_new_tokens = 0
    total_natural_closes = 0
    total_cap_contacts = 0
    for row in rows:
        recurrent_rendered = _render(tokenizer, row, thinking=False)
        recurrent_prompt_tokens = len(
            tokenizer.encode(recurrent_rendered, add_special_tokens=False)
        )
        rendered = _render(tokenizer, _without_workspace(row, config), thinking=True)
        prompt_ids = tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt"
        ).input_ids.cuda()
        k = int(row["depth"])
        recurrent_budget = recurrent_compute_receipt(
            sequence_tokens=recurrent_prompt_tokens,
            total_layers=layer_count,
            loop_layers=loop_layers,
            k=k,
        ).total_layer_token_applications
        desired_new = min(
            int(sampling["max_new_tokens"]),
            int(sampling["fixed_overhead_tokens"])
            + int(sampling["desired_tokens_per_transition"]) * k,
        )
        feasible = [
            candidate_n
            for candidate_n in range(1, int(sampling["max_samples"]) + 1)
            if recurrent_budget // (layer_count * candidate_n) - prompt_ids.shape[1]
            >= desired_new
        ]
        if not feasible:
            raise RuntimeError(
                "sample-more budget cannot fund the frozen per-sample reasoning allowance"
            )
        n = feasible[-1]
        max_new = min(
            int(sampling["max_new_tokens"]),
            int(recurrent_budget // (layer_count * n) - prompt_ids.shape[1]),
        )
        sample_seed = int.from_bytes(
            hashlib.blake2b(row["id"].encode("utf-8"), digest_size=8).digest(), "big"
        ) % (2**31)
        torch.manual_seed(sample_seed)
        generation_kwargs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "do_sample": bool(sampling["do_sample"]),
            "max_new_tokens": max_new,
            "num_return_sequences": n,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if bool(sampling["do_sample"]):
            generation_kwargs.update(
                {
                    "temperature": float(sampling["temperature"]),
                    "top_p": float(sampling["top_p"]),
                    "top_k": int(sampling["top_k"]),
                }
            )
        torch.cuda.synchronize()
        generation_started = time.time()
        with torch.no_grad():
            generated = model.generate(**generation_kwargs)
        torch.cuda.synchronize()
        generation_seconds = time.time() - generation_started
        continuations = generated[:, prompt_ids.shape[1] :]
        generated_token_counts = []
        cap_contacts = []
        continuation_token_ids = []
        for continuation in continuations:
            token_list = continuation.tolist()
            try:
                generated_count = token_list.index(tokenizer.eos_token_id) + 1
            except ValueError:
                generated_count = len(token_list)
            generated_token_counts.append(generated_count)
            cap_contacts.append(
                tokenizer.eos_token_id not in token_list and generated_count >= max_new
            )
            continuation_token_ids.append(token_list[:generated_count])
        samples = tokenizer.batch_decode(continuations, skip_special_tokens=False)
        natural_closes = ["</think>" in sample for sample in samples]
        choices = [_parse_generated_choice(sample) for sample in samples]
        valid = [choice for choice in choices if choice is not None]
        counts = {choice: valid.count(choice) for choice in range(4)}
        majority = max(counts, key=lambda choice: (counts[choice], -choice)) if valid else None
        correct = int(row["correct_choice"])
        sample_budget = n * layer_count * (prompt_ids.shape[1] + max_new)
        if sample_budget > recurrent_budget:
            raise RuntimeError(
                f"sample-more budget exceeded recurrence: {sample_budget} > {recurrent_budget}"
            )
        total_generation_seconds += generation_seconds
        total_sampled_tokens += sum(generated_token_counts)
        total_allocated_new_tokens += n * max_new
        total_natural_closes += sum(natural_closes)
        total_cap_contacts += sum(cap_contacts)
        _append_jsonl(
            rows_path,
            {
                "id": row["id"],
                "depth": row["depth"],
                "query_kind": row["query_kind"],
                "correct_choice": correct,
                "n": n,
                "max_new_tokens": max_new,
                "recurrent_prompt_tokens": recurrent_prompt_tokens,
                "sample_prompt_tokens": int(prompt_ids.shape[1]),
                "model_layers": layer_count,
                "loop_layers": loop_layers,
                "recurrent_layer_token_budget": recurrent_budget,
                "sample_layer_token_budget": sample_budget,
                "sample_seed": sample_seed,
                "text_train_seed": metadata["train_seed"],
                "checkpoint_identity_sha256": metadata[
                    "checkpoint_identity_sha256"
                ],
                "sampling_config": dict(sampling),
                "generated_token_counts": generated_token_counts,
                "continuation_token_ids": continuation_token_ids,
                "decoded_samples": samples,
                "natural_close": natural_closes,
                "cap_contact": cap_contacts,
                "sampled_tokens": sum(generated_token_counts),
                "allocated_new_tokens": n * max_new,
                "generation_seconds": generation_seconds,
                "choices": choices,
                "majority_correct": majority == correct,
                "pass_at_n": correct in valid,
                "parse_rate": len(valid) / n,
            },
        )
    _write_json(
        output_dir / "summary.json",
        _with_receipt_identity(
            {
                "status": "SAMPLE_MORE_COMPLETE",
                **_identity_receipt(config, phase="sample_more"),
                "checkpoint": str(checkpoint),
                "checkpoint_identity_sha256": metadata[
                    "checkpoint_identity_sha256"
                ],
                "data_manifest_sha256": data_manifest_hash,
                "expected_seed": expected_seed,
                "text_train_seed": metadata["train_seed"],
                "rows": rows_path.name,
                "rows_sha256": _sha256(rows_path),
                "sampled_tokens": total_sampled_tokens,
                "allocated_new_tokens": total_allocated_new_tokens,
                "natural_closes": total_natural_closes,
                "cap_contacts": total_cap_contacts,
                "generation_seconds": total_generation_seconds,
                "sampling_config": dict(sampling),
                "environment": _environment_receipt(include_device=False),
                "tokenizer": token_receipt,
            }
        ),
    )
