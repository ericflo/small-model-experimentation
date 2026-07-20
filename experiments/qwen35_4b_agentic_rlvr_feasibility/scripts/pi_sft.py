"""Fit pi-coding-agent's execution-verified successful trajectories (the RFT step).

Closes the loop: pi generates -> tests decide -> keep only passing episodes -> fit -> merge -> eval.
This is policy improvement from execution reward using rollouts from the REAL deployment scaffold,
which is the form "RLVR through pi rollouts" can take given pi ships no logprobs (TRL's GRPO needs
`per_token_logps`, so policy-gradient RL over pi rollouts is not possible without patching pi).

Trains ON TOP OF the current policy (the merged warm-start that generated the rollouts), which is
what makes it an iteration rather than a restart.

Prompt fidelity matters here more than usual: rows carry pi's OWN system prompt and its OWN tool
schemas (read/edit/write/bash), captured from the wire by pi_proxy. TRL's SFTTrainer reads the
`tools` column per example (sft_trainer.py:1474) and renders it into the prompt, so the model is
trained in the exact context it will be deployed in. Our previous warm-start trained on
read_file/write_file/list_dir/run_bash -- names that do not exist in pi -- which is precisely the
mismatch this step exists to correct.
"""
import argparse, json, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path

import trl
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"
TRAIN_TEMPLATE = Path(trl.__file__).parent / "chat_templates" / "qwen3_5_think_training.jinja"


def _arrow_safe(row):
    """Make tool-argument values uniformly strings so Arrow can infer one schema.

    pi's tools take heterogeneous argument types and datasets/pyarrow refuses the mix
    ("cannot mix list and non-list, non-null values"): across a real harvest `edits` arrived as a
    list 78x and a str 4x, and limit/offset/timeout arrived as both int and str. JSON-encoding the
    non-strings is also the FAITHFUL choice, not merely the convenient one: the Qwen template
    renders `<parameter=k>\n<value>` as text, and what the model originally emitted for a list
    argument was JSON text -- so json.dumps reproduces the original surface, whereas Python's repr
    (single quotes, True/False) would not.
    """
    out = dict(row)
    msgs = []
    for m in row["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments") or {}
                fn["arguments"] = {k: (v if isinstance(v, str) else json.dumps(v))
                                   for k, v in args.items()}
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        # pi sends USER content as a list of {type:text} blocks while every other role uses a
        # plain string -- that list/str mix is what Arrow rejects. Flatten to text; our data is
        # text-only, and this is exactly what the template would render anyway.
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(b.get("text", "") for b in c if isinstance(b, dict))
        elif c is None:
            m["content"] = ""
        # one harvested tool_call arrived without an `id`; keep the key set uniform for Arrow
        if m.get("tool_calls"):
            for i, tc in enumerate(m["tool_calls"]):
                tc.setdefault("id", f"call_{i}")
                tc.setdefault("type", "function")
        msgs.append(m)
    out["messages"] = msgs
    # Serialize the tool schemas too. pi's JSON-Schema fragments are structurally heterogeneous
    # (array params carry `items`, scalars do not), which trips the same Arrow inference. TRL
    # accepts a JSON string here and json.loads it back before rendering (sft_trainer.py:1475).
    out["tools"] = json.dumps(row.get("tools") or [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(OUTD / "pi_sft_rows.jsonl"))
    ap.add_argument("--model", default=str(OUTD / "merged" / "warmstart"),
                    help="the policy that GENERATED the rollouts -- RFT iterates on it, not on base")
    ap.add_argument("--out", default=str(OUTD / "adapters" / "pi_rft"))
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="1 epoch by standing directive: dial up UNIQUE data, not repeats")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=8192)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    keep = [_arrow_safe(r) for r in rows if r.get("messages")]
    print(f"pi RFT rows: {len(keep)} | tools rendered per row: "
          f"{len(json.loads(keep[0]['tools']))} schemas", flush=True)
    turns = [sum(1 for m in r["messages"] if m.get("role") == "assistant") for r in keep]
    print(f"assistant turns per trajectory: min {min(turns)} / mean {sum(turns)/len(turns):.1f} / max {max(turns)}",
          flush=True)
    ds = Dataset.from_list(keep)

    peft_config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=a.out + "_out",
        chat_template_path=str(TRAIN_TEMPLATE),
        assistant_only_loss=True,     # train only the model's own spans; pi's tool results are masked
        max_length=a.max_length,      # pi trajectories are long: 2.5k-char system prompt + many turns
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        # host RAM (15GB) is binding: plain from_pretrained leaves a ~9GB resident copy and the
        # OOM-killer takes the run down with no traceback
        model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Qwen3.5's vocab is 248320, so the LM head dominates activation memory (measured: 78% of it
        # at seq 4096). Liger's chunked loss never materializes the [seq, 248320] logits, which is
        # what makes an 8192-token agentic trajectory trainable on a 24GB card.
        use_liger_kernel=True,
    )
    from transformers import AutoTokenizer
    # tokenizer, NOT AutoProcessor: Qwen3.5-4B is multimodal-capable and TRL's VLM guard refuses
    # assistant_only_loss for processors. Our data is pure text.
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    trainer = SFTTrainer(model=a.model, args=args, train_dataset=ds, peft_config=peft_config,
                         processing_class=tokenizer)
    print("=== fitting pi-verified trajectories ===", flush=True)
    trainer.train()
    trainer.save_model(a.out)
    print(f"=== saved -> {a.out} ===", flush=True)


if __name__ == "__main__":
    main()
