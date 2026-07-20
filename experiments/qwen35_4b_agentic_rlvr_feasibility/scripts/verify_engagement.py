"""Verify the SFT warm-start unblocked RLVR signal.

C61 established the blocker: the RAW base yields reward_std = 0 in the agentic GRPO env (it explores
1-2 reads then quits WITHOUT writing), so GRPO gets zero advantage -> zero gradient. This runs a few
GRPO steps against CodingEnv with a given model and reports the decisive metrics:

  reward_std > 0  and  frac_reward_zero_std < 1   -> variance unblocked, RLVR can learn
  completions/mean_length up, write_file appearing -> the loop discipline actually installed

Usage: verify_engagement.py --model <hf-id-or-merged-composite> [--steps 4]
"""
import argparse, json, os, sys
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import vllm
_orig = vllm.LLM.__init__
def _patched(self, *a, **kw):
    kw["enforce_eager"] = True          # Qwen3.5 hybrid arch hangs on CUDAGraph capture
    return _orig(self, *a, **kw)
vllm.LLM.__init__ = _patched

# TRL sleeps the colocate engine at level=2, which OFFLOADS the 8.5GB of weights to HOST RAM and
# reloads them on every wake. This box has 15GB host RAM -> the reload gets OOM-killed. Level 1
# frees only the KV cache and keeps weights resident on the GPU: same VRAM time-sharing, no host hit.
_orig_sleep = vllm.LLM.sleep
def _sleep_level1(self, level=2):
    return _orig_sleep(self, level=1)
vllm.LLM.sleep = _sleep_level1

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
from coding_env import CodingEnv
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

STORE = ROOT / "large_artifacts" / "_taskrepos"
TASKS = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "toolz_tasks.json"

SYSTEM = ("You are an expert Python coding agent. Think step by step, then use the tools to inspect "
          "files, write code, and run the tests until they pass. You MUST edit the target file with "
          "write_file and MUST run the tests with run_bash to verify before finishing. Do not give up "
          "after only reading files. Keep iterating until the tests pass.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=1280)
    ap.add_argument("--vllm-util", type=float, default=0.50)
    ap.add_argument("--no-sleep", action="store_true", help="disable vLLM sleep entirely")
    a = ap.parse_args()

    tasks = [t for t in json.load(open(TASKS)) if 4 <= t["body_lines"] <= 14][:8]
    rows = [{"prompt": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "Complete the coding task in this repository using the available tools."}],
             "repo_dir": str(STORE / t["repo"]), "rel_file": t["rel_file"], "func_name": t["func_name"],
             "python": str(STORE / "toolz" / ".venv-test" / "bin" / "python"),
             "test_cmd": "python -m pytest -q"} for t in tasks]
    ds = Dataset.from_list(rows * 4)
    print(f"model={a.model}  tasks={len(tasks)}  steps={a.steps}  num_gen={a.num_generations}", flush=True)

    args = GRPOConfig(
        output_dir="/tmp/verify_engagement_out",
        per_device_train_batch_size=a.num_generations, num_generations=a.num_generations,
        max_completion_length=a.max_completion_length, max_tool_calling_iterations=8,
        learning_rate=1e-6, logging_steps=1, max_steps=a.steps, save_strategy="no", report_to=[],
        bf16=True, model_init_kwargs={"dtype": "bfloat16"}, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", vllm_enable_sleep_mode=not a.no_sleep,
        vllm_gpu_memory_utilization=a.vllm_util, vllm_max_model_length=4096,
        temperature=1.0, loss_type="dr_grpo", importance_sampling_level="sequence", beta=0.0,
        chat_template_kwargs={"enable_thinking": True},
    )
    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM")

    stats = []
    class Log(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if logs and "reward" in logs:
                keep = {k: round(v, 4) for k, v in logs.items() if isinstance(v, (int, float)) and (
                    k in ("reward", "reward_std", "frac_reward_zero_std") or "mean_length" in k or "call_frequency" in k)}
                stats.append(keep)
                print(f"[step {len(stats)}] {keep}", flush=True)

    trainer = GRPOTrainer(model=a.model, args=args, train_dataset=ds, peft_config=peft_config,
                          environment_factory=lambda: CodingEnv())
    trainer.add_callback(Log())
    trainer.train()

    print("\n=== ENGAGEMENT VERDICT ===", flush=True)
    if not stats:
        print("no steps logged"); return
    mean_std = sum(s.get("reward_std", 0) for s in stats) / len(stats)
    mean_rew = sum(s.get("reward", 0) for s in stats) / len(stats)
    zero_frac = sum(s.get("frac_reward_zero_std", 1) for s in stats) / len(stats)
    mean_len = sum(s.get("completions/mean_length", 0) for s in stats) / len(stats)
    mean_calls = sum(s.get("tools/call_frequency", 0) for s in stats) / len(stats)
    print(f"mean reward={mean_rew:.4f}  mean reward_std={mean_std:.4f}  "
          f"frac_reward_zero_std={zero_frac:.2f}  mean completion len={mean_len:.0f}  "
          f"mean tool calls={mean_calls:.2f}", flush=True)
    if mean_std > 0.01 and zero_frac < 1.0:
        print("VARIANCE UNBLOCKED: reward_std > 0 -> GRPO has advantage signal -> RLVR can learn.", flush=True)
    else:
        print("STILL BLOCKED: reward_std ~ 0 -> no advantage -> RLVR cannot learn yet.", flush=True)


if __name__ == "__main__":
    main()
