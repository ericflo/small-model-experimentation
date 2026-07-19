"""SFT warm-start: install loop discipline (explore->edit->test->iterate) from harvested
completed+passing multi-turn tool-calling trajectories.

TRL SFTTrainer + the Qwen3.5 THINK *training* chat template, which wraps each assistant turn in
{% generation %} tags so `assistant_only_loss=True` trains ONLY the model's own spans
(`<think>{reasoning}</think>` + content + tool_calls) and MASKS system/user/tool-result tokens.
Reasoning is the base's own harvested trace (C60: harvest, never author). LoRA; merge separately.
"""
import argparse, json, os, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import trl
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
TRAIN_TEMPLATE = Path(trl.__file__).parent / "chat_templates" / "qwen3_5_think_training.jinja"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default=str(ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "adapters" / "warmstart"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=8192)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    print(f"loaded {len(rows)} SFT rows; template={TRAIN_TEMPLATE.name}", flush=True)
    ds = Dataset.from_list(rows)

    peft_config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=a.out + "_out",
        chat_template_path=str(TRAIN_TEMPLATE),
        assistant_only_loss=True,        # train only on the model's own spans (incl. reasoning + tool_calls)
        max_length=a.max_length,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16"},
        gradient_checkpointing=True,
    )
    # Pass a TOKENIZER (not AutoProcessor): Qwen3.5-4B is multimodal-capable, so TRL's default
    # AutoProcessor sets _is_vlm=True and REFUSES assistant_only_loss. Our data is pure text.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    trainer = SFTTrainer(model=a.model, args=args, train_dataset=ds, peft_config=peft_config,
                         processing_class=tokenizer)
    print("=== training SFT warm-start ===", flush=True)
    trainer.train()
    trainer.save_model(a.out)
    print(f"=== saved warm-start adapter -> {a.out} ===", flush=True)


if __name__ == "__main__":
    main()
