#!/usr/bin/env python3
"""Exact-token, weighted long-horizon QLoRA SFT for one frozen arm."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import time
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    create_recurrent_attention_mask,
)

EXP = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP / "configs" / "default.yaml"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FULL_LOGIT_MAX_LENGTH = 8192
LONG_LOSS_CHUNK = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_encode(rec: dict[str, Any], *, max_length: int, w_think: float) -> dict[str, Any]:
    prompt = [int(value) for value in rec["prompt_token_ids"]]
    trace = [int(value) for value in rec["trace_token_ids"]]
    boundary = [int(value) for value in rec["answer_boundary_token_ids"]]
    answer = [int(value) for value in rec["answer_token_ids"]]
    eos = [int(rec["eos_token_id"])]
    ids = [*prompt, *trace, *boundary, *answer, *eos]
    if len(ids) != int(rec["total_tokens"]):
        raise ValueError(f"stored token count mismatch for {rec['record_id']}")
    if len(ids) > max_length:
        raise ValueError(
            f"selected target exceeds max_length and may not be truncated: "
            f"{rec['record_id']} {len(ids)} > {max_length}"
        )
    weights = (
        [0.0] * len(prompt)
        + [w_think] * len(trace)
        + [1.0] * (len(boundary) + len(answer) + 1)
    )
    labels = [-100 if weight == 0 else token for token, weight in zip(ids, weights)]
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "loss_weights": weights,
    }


class ThinkSftData(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maximum = max(len(row["input_ids"]) for row in features)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] + [self.pad_token_id] * (maximum - len(row["input_ids"])) for row in features]
            ),
            "attention_mask": torch.tensor(
                [row["attention_mask"] + [0] * (maximum - len(row["input_ids"])) for row in features]
            ),
            "labels": torch.tensor(
                [row["labels"] + [-100] * (maximum - len(row["input_ids"])) for row in features]
            ),
            "loss_weights": torch.tensor(
                [row["loss_weights"] + [0.0] * (maximum - len(row["input_ids"])) for row in features],
                dtype=torch.float32,
            ),
        }


def text_checkpoint_key_mapping() -> dict[str, str]:
    """Map the composite checkpoint's language tower into Qwen3_5ForCausalLM."""
    index_path = Path(
        hf_hub_download(
            MODEL_ID,
            "model.safetensors.index.json",
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    keys = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model."
    mapping = {
        key: f"model.{key[len(prefix):]}"
        for key in keys
        if key.startswith(prefix)
    }
    if len(mapping) < 400:
        raise RuntimeError(f"unexpectedly small Qwen3.5 text key map: {len(mapping)}")
    return mapping


def xformers_causal_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Memory-efficient causal attention for Qwen's 256-d full heads."""
    del kwargs
    if attention_mask is not None:
        raise ValueError("xFormers training path requires unpadded batch-one rows")
    from xformers.ops import LowerTriangularMask, memory_efficient_attention

    if module.num_key_value_groups > 1:
        key = key.repeat_interleave(module.num_key_value_groups, dim=1)
        value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    output = memory_efficient_attention(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        attn_bias=LowerTriangularMask(),
        p=dropout,
        scale=scaling,
    )
    return output, None


def memory_bounded_text_forward(
    self: Any,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Any,
) -> BaseModelOutputWithPast:
    """Qwen3.5 text forward with the missing long-row layer checkpoints.

    Transformers 5.13 sets ``gradient_checkpointing=True`` on this model but
    its generated Qwen3.5 text loop never calls ``_gradient_checkpointing_func``.
    For >8k training rows we checkpoint each decoder layer explicitly.  Short
    rows retain the stock direct loop and its measured 4.2 s smoke speed.
    """
    del kwargs
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("specify exactly one of input_ids or inputs_embeds")
    if use_cache:
        raise ValueError("training forward does not permit cache state")
    if past_key_values is not None:
        raise ValueError("training forward does not accept past_key_values")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = position_ids.view(1, 1, -1).expand(
            4, inputs_embeds.shape[0], -1
        )
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(
            4, position_ids.shape[0], -1
        )
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
    else:
        text_position_ids = None
        rope_position_ids = position_ids
    if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
        raise ValueError("batch-one exact-token training may not contain padding")
    masks = {
        "full_attention": None,
        "linear_attention": create_recurrent_attention_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        ),
    }
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
    checkpoint_layers = hidden_states.shape[1] > FULL_LOGIT_MAX_LENGTH
    for index, decoder_layer in enumerate(
        self.layers[: self.config.num_hidden_layers]
    ):
        layer_mask = masks[self.config.layer_types[index]]
        def layer_forward(
            values: torch.Tensor,
            layer: Any = decoder_layer,
            selected_mask: Any = layer_mask,
        ) -> torch.Tensor:
            return layer(
                values,
                position_embeddings=position_embeddings,
                attention_mask=selected_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )

        hidden_states = (
            checkpoint(layer_forward, hidden_states, use_reentrant=True)
            if checkpoint_layers
            else layer_forward(hidden_states)
        )
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)


def load_text_model(rank: int, alpha: int, dropout: float) -> Any:
    outer = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=outer.text_config,
        key_mapping=text_checkpoint_key_mapping(),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=bnb,
        attn_implementation="sdpa",
    )
    ALL_ATTENTION_FUNCTIONS.register("xformers_memory_efficient", xformers_causal_attention)
    model.config._attn_implementation = "xformers_memory_efficient"
    model.model.config._attn_implementation = "xformers_memory_efficient"
    model.model.forward = types.MethodType(
        memory_bounded_text_forward, model.model
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.arm not in config["sft"]["arms"]:
        raise ValueError(f"unknown frozen arm: {args.arm}")
    dataset = args.dataset or (
        Path(config["artifacts"]["external_root"]) / "sft" / f"{args.arm}.jsonl.gz"
    )
    records = read_jsonl_gz(dataset)
    random.Random(args.seed).shuffle(records)
    epochs = float(config["sft"]["epochs"])
    if args.smoke:
        records = records[:2]
        epochs = 1.0

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [
        exact_encode(
            row,
            max_length=int(config["sft"]["max_length"]),
            w_think=float(config["sft"]["weight_think"]),
        )
        for row in records
    ]
    # With batch size one, deterministic global length ordering minimizes
    # allocator churn while preserving exactly the same optimizer examples.
    encoded.sort(key=lambda row: len(row["input_ids"]))
    print(
        f"[train] arm={args.arm} rows={len(encoded)} "
        f"tokens={sum(len(row['input_ids']) for row in encoded)} max={max(len(row['input_ids']) for row in encoded)}",
        flush=True,
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = load_text_model(
        int(config["sft"]["rank"]),
        int(config["sft"]["alpha"]),
        float(config["sft"]["dropout"]),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=epochs,
        per_device_train_batch_size=int(config["sft"]["batch_size"]),
        gradient_accumulation_steps=int(config["sft"]["gradient_accumulation"]),
        learning_rate=float(config["sft"]["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    class ExactWeightedTrainer(Trainer):
        def _get_train_sampler(self, *unused_args: Any, **unused_kwargs: Any) -> Any:
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

        def compute_loss(
            self,
            peft_model: Any,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            **unused_kwargs: Any,
        ) -> Any:
            weights = inputs.pop("loss_weights")[:, 1:].contiguous()
            labels = inputs.pop("labels")[:, 1:].contiguous()
            mask = (labels != -100).float() * weights
            if inputs["input_ids"].shape[1] <= FULL_LOGIT_MAX_LENGTH:
                outputs = peft_model(**inputs, use_cache=False, return_dict=True)
                logits = outputs.logits[:, :-1, :].contiguous()
                # Keep logits bf16.  This path is ~23x faster on 3--4k rows
                # and remains comfortably inside 48 GB through 8k.
                losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1).clamp(min=0),
                    reduction="none",
                ).view_as(labels)
                loss = (losses * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                causal_model = peft_model.get_base_model()
                outputs = causal_model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state[:, :-1, :]

                def chunk_numerator(
                    chunk_hidden: torch.Tensor,
                    chunk_labels: torch.Tensor,
                    chunk_mask: torch.Tensor,
                ) -> torch.Tensor:
                    chunk_logits = causal_model.lm_head(chunk_hidden)
                    chunk_losses = torch.nn.functional.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.size(-1)),
                        chunk_labels.reshape(-1).clamp(min=0),
                        reduction="none",
                    ).view_as(chunk_labels)
                    return (chunk_losses * chunk_mask).sum()

                numerator = hidden.new_zeros((), dtype=torch.float32)
                for start in range(0, hidden.shape[1], LONG_LOSS_CHUNK):
                    stop = min(hidden.shape[1], start + LONG_LOSS_CHUNK)
                    numerator = numerator + checkpoint(
                        chunk_numerator,
                        hidden[:, start:stop, :],
                        labels[:, start:stop],
                        mask[:, start:stop],
                        use_reentrant=True,
                    )
                loss = numerator / mask.sum().clamp(min=1.0)
            return (loss, outputs) if return_outputs else loss

    trainer = ExactWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ThinkSftData(encoded),
        data_collator=Collator(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.out.iterdir())
        if path.is_file()
    }
    receipt = {
        "schema_version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "smoke": args.smoke,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "bitsandbytes",
                "accelerate",
                "xformers",
                "flash-linear-attention",
                "causal-conv1d",
            )
        },
        "full_attention_backend": "xformers_memory_efficient",
        "long_row_threshold": FULL_LOGIT_MAX_LENGTH,
        "long_loss_chunk": LONG_LOSS_CHUNK,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(encoded),
        "forward_tokens": sum(len(row["input_ids"]) for row in encoded),
        "supervised_weighted_tokens": sum(sum(row["loss_weights"]) for row in encoded),
        "skipped_rows": 0,
        "full_logit_rows": sum(
            len(row["input_ids"]) <= FULL_LOGIT_MAX_LENGTH for row in encoded
        ),
        "chunked_loss_rows": sum(
            len(row["input_ids"]) > FULL_LOGIT_MAX_LENGTH for row in encoded
        ),
        "epochs": epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "train_metrics": train_result.metrics,
        "artifacts": artifacts,
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[train] saved {args.arm} to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
