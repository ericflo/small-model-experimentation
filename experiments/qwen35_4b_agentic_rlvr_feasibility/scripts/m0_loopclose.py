"""M0 GATE: does the single-GPU GRPO colocate loop physically CLOSE for Qwen3.5-4B?

C49 says vLLM 0.24 runtime-LoRA is a silent no-op for Qwen3.5-4B PEFT adapters. If TRL's colocate
per-step LoRA->vLLM sync is a no-op, GRPO trains against a FROZEN policy: reward never moves.
This smoke uses a TRIVIAL, easily-movable reward and checks whether the vLLM-served policy actually
shifts toward it. enforce_eager is forced via monkeypatch (Qwen3.5 hybrid arch hangs on CUDA graphs).
"""
import os, sys, re
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- force enforce_eager into TRL's colocate vLLM (patch the class method) ---
import vllm
_orig_llm_init = vllm.LLM.__init__
def _patched_llm_init(self, *a, **kw):
    kw["enforce_eager"] = True
    print(f"[patch] vllm.LLM(enforce_eager=True) gpu_mem={kw.get('gpu_memory_utilization')} "
          f"sleep={kw.get('enable_sleep_mode')} max_len={kw.get('max_model_len')}", flush=True)
    return _orig_llm_init(self, *a, **kw)
vllm.LLM.__init__ = _patched_llm_init

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen3.5-4B"

# --- loop-closure reward: normalized completion LENGTH (push longer). Sampled completions ALWAYS
# vary in length -> nonzero within-group advantage from step 1 -> a clean test of weight-sync:
# if the policy updates through vLLM, mean length climbs; if C49 freezes it, length stays flat.
def reward_length(completions, **kwargs):
    out = []
    for c in completions:
        text = c if isinstance(c, str) else (c[-1]["content"] if isinstance(c, list) else str(c))
        out.append(1.0 - min(len(text), 400) / 400.0)  # push SHORTER: base verbose -> low -> climbs if loop closes
    return out

prompts = [
    "Write a short line of text:",
    "Say something:",
    "Continue:",
    "Your response:",
    "Reply here:",
    "Output:",
    "Answer:",
    "Text:",
]
ds = Dataset.from_dict({"prompt": prompts * 8})  # 64 rows

peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")

args = GRPOConfig(
    output_dir="/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/877cfc7b-ff96-4334-b8b5-d31dd4c686fa/scratchpad/rlvr/m0_out",
    per_device_train_batch_size=8,
    num_generations=8,
    max_completion_length=128,
    learning_rate=1e-4,
    logging_steps=1,
    max_steps=20,
    save_strategy="no",
    report_to=[],
    bf16=True,
    model_init_kwargs={"dtype": "bfloat16"},  # load base in bf16 (8.5GB) not fp32 (16GB) — frees room for colocate vLLM
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_enable_sleep_mode=True,
    vllm_gpu_memory_utilization=0.55,
    vllm_max_model_length=1024,
    temperature=1.0,
    loss_type="dr_grpo",
    importance_sampling_level="sequence",
    beta=0.0,  # no KL, let it move freely for the smoke
    chat_template_kwargs={"enable_thinking": False},  # smoke: fast, no thinking needed to test weight-sync
)

print("=== building GRPOTrainer (colocate) ===", flush=True)
trainer = GRPOTrainer(model=MODEL, reward_funcs=reward_length, args=args,
                      train_dataset=ds, peft_config=peft_config)

# capture per-step mean reward via a callback
from transformers import TrainerCallback
class RewardTrend(TrainerCallback):
    def __init__(self): self.rewards = []
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "reward" in logs:
            self.rewards.append(logs["reward"])
            print(f"[step {len(self.rewards)}] reward={logs['reward']:.4f}", flush=True)
cb = RewardTrend()
trainer.add_callback(cb)

print("=== training 20 steps ===", flush=True)
trainer.train()

r = cb.rewards
print("\n=== M0 VERDICT ===", flush=True)
print("reward trend:", [round(x, 4) for x in r], flush=True)
if len(r) >= 4:
    early = sum(r[:2]) / 2
    late = sum(r[-2:]) / 2
    moved = late - early
    print(f"early(mean first2)={early:.4f}  late(mean last2)={late:.4f}  delta={moved:+.4f}", flush=True)
    if moved > 0.05:
        print("LOOP CLOSES: served policy moved toward the reward (runtime-LoRA sync WORKS).", flush=True)
    else:
        print("LOOP DID NOT CLOSE: reward flat -> C49 runtime-LoRA no-op suspected -> need merge-and-reload.", flush=True)
else:
    print("insufficient steps logged", flush=True)
