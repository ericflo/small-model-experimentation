"""DPO on pi pass-vs-timeout pairs — the negative gradient RFT structurally lacks (task #62).

The RFT iterations proved execution-filtered SFT is SAFE but not net-positive: fitting successes
reinforces what already works and has NO signal to suppress the timeouts that are the real failure.
DPO supplies exactly that — it raises log p(solve-and-stop trajectory) while LOWERING
log p(loop-forever trajectory), so the loop-to-the-wall behaviour is actively penalised, not merely
un-reinforced. No logprobs from pi are needed (DPO computes them from our own policy).

Trains a fresh LoRA ON TOP OF the merged warm-start (the 0.606 deployment baseline that generated the
chosen trajectories). With `peft_config` and no explicit ref_model, DPO uses the SAME base with the
new adapter DISABLED as the KL reference -- i.e. it anchors to the warm-start, so DPO polishes rather
than restarts, and there is no second model resident (critical on 24GB).

Memory: Qwen3.5-4B's 248320-vocab LM head dominates activation memory; DPO scores FOUR sequences per
step (chosen/rejected x policy/reference), so `use_liger_kernel=True` (LigerFusedLinearDPOLoss, which
never materialises the [seq, 248320] logits) is what makes this fit at all.
"""
import argparse, json, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Safe here (unlike RLVR colocate): no vLLM shares this device, so expandable_segments cannot collide
# with vLLM's sleep-mode CuMemAllocator. It reclaims the fragmentation the OOM message flagged.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import trl
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"
TRAIN_TEMPLATE = Path(trl.__file__).parent / "chat_templates" / "qwen3_5_think_training.jinja"


def _norm_msgs(msgs):
    """Same Arrow-safety pi_sft._arrow_safe applies, but to a prompt/chosen/rejected message list.

    pi sends user content as a list of {type:text} blocks while other roles use plain strings, and
    tool_call arguments carry heterogeneous value types -- both make pyarrow raise "cannot mix list
    and non-list". Flatten content to text and stringify non-string argument values (which is also
    the faithful surface: the Qwen template renders a list argument as its JSON text anyway).
    """
    out = []
    for m in msgs:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(b.get("text", "") for b in c if isinstance(b, dict))
        elif c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for i, tc in enumerate(m["tool_calls"]):
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments") or {}
                if isinstance(args, dict):
                    fn["arguments"] = {k: (v if isinstance(v, str) else json.dumps(v))
                                       for k, v in args.items()}
                tc["function"] = fn
                tc.setdefault("id", f"call_{i}")
                tc.setdefault("type", "function")
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(OUTD / "pi_dpo_pairs.jsonl"))
    ap.add_argument("--model", default=str(OUTD / "merged" / "warmstart"),
                    help="the 0.606 deployment baseline; DPO polishes it and KL-anchors to it")
    ap.add_argument("--out", default=str(OUTD / "adapters" / "pi_dpo"))
    ap.add_argument("--epochs", type=float, default=2.0,
                    help="preference pairs are few; a couple epochs over UNIQUE pairs, not repeats")
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO polishing LR. RFT taught that 1e-4 SFT on a working policy is too hot; "
                         "DPO is gentler still since the loss is a logratio, not token CE")
    ap.add_argument("--beta", type=float, default=0.1, help="KL strength vs the warm-start reference")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="DPO forwards chosen+rejected under policy AND reference concurrently (Liger "
                         "fused DPO loss can't precompute the ref); 6144 OOM'd needing +11.4GiB, so seq "
                         "is the memory knob. 3072 + keep_end preserves the tests-passed decision point")
    ap.add_argument("--grad-accum", type=int, default=8)
    a = ap.parse_args()

    pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
    # keep only the columns DPO consumes; Arrow-normalize the message lists and JSON-encode the tool
    # schemas (DPOTrainer json.loads a string tools column before rendering -- dpo_trainer.py:996 --
    # and a string sidesteps Arrow's inference over pi's structurally-heterogeneous JSON-Schema).
    keep = [{"prompt": _norm_msgs(p["prompt"]), "chosen": _norm_msgs(p["chosen"]),
             "rejected": _norm_msgs(p["rejected"]), "tools": json.dumps(p.get("tools") or [])}
            for p in pairs]
    ct = sum(1 for p in pairs if p.get("rejected_exit") == 124)
    print(f"DPO pairs: {len(keep)} ({ct} rejected are real timeouts) | "
          f"mean chosen turns {sum(p.get('chosen_turns',0) for p in pairs)/max(1,len(pairs)):.1f} | "
          f"mean rejected turns {sum(p.get('rejected_turns',0) for p in pairs)/max(1,len(pairs)):.1f}",
          flush=True)
    ds = Dataset.from_list(keep)

    peft_config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
    args = DPOConfig(
        output_dir=a.out + "_out",
        max_length=a.max_length,
        # The decision point (tests-just-passed -> stop vs continue) is at the END of the prompt.
        # keep_start (the default) would truncate the END off the long pairs and destroy the signal;
        # keep_end left-truncates, preserving the ALL-PASS observation the stop decision hinges on.
        truncation_mode="keep_end",
        beta=a.beta,
        loss_type="sigmoid",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 248320-vocab LM head is 78% of activation memory; DPO scores 4 sequences/step -> without
        # the fused chunked loss the [seq, 248320] logits OOM. This is the load-bearing flag.
        use_liger_kernel=True,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    # DPO renders via the TOKENIZER's chat template (no chat_template_path field like SFT). Use the
    # same Qwen3.5 think template the warm-start was trained under so the context is identical.
    tokenizer.chat_template = TRAIN_TEMPLATE.read_text()

    trainer = DPOTrainer(model=a.model, args=args, train_dataset=ds, peft_config=peft_config,
                         processing_class=tokenizer)
    print("=== DPO: raise log p(solve+stop) - log p(loop-forever) ===", flush=True)
    trainer.train()
    trainer.save_model(a.out)
    print(f"=== saved DPO adapter -> {a.out} ===", flush=True)


if __name__ == "__main__":
    main()
