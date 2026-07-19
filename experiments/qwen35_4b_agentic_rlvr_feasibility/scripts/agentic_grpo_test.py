"""Agentic GRPO integration test: TRL drives Qwen3.5-4B (thinking-on) through CodingEnv's tools
(multi-turn tool-calling) on real toolz tasks, grades with get_reward (test pass), does GRPO steps.

Goal: confirm the WHOLE RLVR loop runs end-to-end without error and produces reward variance —
NOT to learn yet. If this runs, the pi-mirroring env + TRL colocate + thinking + tool-loop + reward
pipeline is proven, and we scale to real training.
"""
import os, sys, json
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import vllm
_orig = vllm.LLM.__init__
def _patched(self, *a, **kw):
    kw["enforce_eager"] = True
    return _orig(self, *a, **kw)
vllm.LLM.__init__ = _patched

sys.path.insert(0, "/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/877cfc7b-ff96-4334-b8b5-d31dd4c686fa/scratchpad/rlvr")
from coding_env import CodingEnv
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen3.5-4B"
STORE = "/home/ericflo/Development/small-model-experimentation/large_artifacts/_taskrepos"
TOOLZ_PY = f"{STORE}/toolz/.venv-test/bin/python"

# build dataset from a few toolz tasks (moderate difficulty)
tasks = json.load(open("/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/877cfc7b-ff96-4334-b8b5-d31dd4c686fa/scratchpad/agentic/toolz_tasks.json"))
picked = [t for t in tasks if 4 <= t["body_lines"] <= 14][:8]
rows = []
for t in picked:
    rows.append({
        "prompt": [{"role": "system", "content": 'You are an expert Python coding agent. Think step by step, then use the tools to inspect files, write code, and run the tests until they pass. You MUST edit the target file with write_file and MUST run the tests with run_bash to verify before finishing. Do not give up after only reading files — always make your implementation and test it. Keep iterating until the tests pass.'}, {"role": "user", "content": "Complete the coding task in this repository using the available tools."}],
        "repo_dir": f"{STORE}/{t['repo']}", "rel_file": t["rel_file"], "func_name": t["func_name"],
        "python": TOOLZ_PY, "test_cmd": "python -m pytest -q",
    })
ds = Dataset.from_list(rows * 4)  # repeat for a few steps
print(f"dataset: {len(ds)} rows from {len(picked)} distinct tasks", flush=True)

peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")

args = GRPOConfig(
    output_dir="/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/877cfc7b-ff96-4334-b8b5-d31dd4c686fa/scratchpad/rlvr/agentic_out",
    per_device_train_batch_size=8,
    num_generations=8,
    max_completion_length=1280,   # multi-turn budget (thinking + tool loop)
    max_tool_calling_iterations=6,
    learning_rate=1e-5,
    logging_steps=1,
    max_steps=4,
    save_strategy="no",
    report_to=[],
    bf16=True,
    model_init_kwargs={"dtype": "bfloat16"},
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_enable_sleep_mode=True,
    vllm_gpu_memory_utilization=0.55,
    vllm_max_model_length=4096,
    temperature=1.0,
    loss_type="dr_grpo",
    importance_sampling_level="sequence",
    beta=0.0,
    chat_template_kwargs={"enable_thinking": True},  # thinking-on (the program's non-negotiable)
)

print("=== building GRPOTrainer (agentic, environment_factory=CodingEnv) ===", flush=True)
trainer = GRPOTrainer(model=MODEL, args=args, train_dataset=ds,
                      peft_config=peft_config, environment_factory=lambda: CodingEnv())

from transformers import TrainerCallback
class Log(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            keys = {k: round(v, 4) for k, v in logs.items() if isinstance(v, (int, float)) and (k in ('reward','reward_std','loss','grad_norm','frac_reward_zero_std','entropy') or 'mean_length' in k or 'call_frequency' in k)}
            if keys:
                print(f"[log] {keys}", flush=True)
trainer.add_callback(Log())

print("=== training 4 agentic GRPO steps ===", flush=True)
trainer.train()
print("=== AGENTIC GRPO LOOP RAN END-TO-END ===", flush=True)
