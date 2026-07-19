"""RLVR: GRPO from execution reward on real-repo agentic coding tasks.

Policy = Qwen3.5-4B (merged SFT warm-start composite, per C49: deploy merged, runtime LoRA is a
vLLM no-op). Rollouts = TRL-driven multi-turn tool-calling through CodingEnv (tools mirror
pi-coding-agent's interface so training transfers to pi deployment); TRL owns the token logprobs
GRPO needs. Reward = CodingEnv.get_reward (engagement-gated + partial test credit, full pass = 1.0).

Single-4090 recipe (C61, all required): enforce_eager monkeypatch (hybrid arch CUDAGraph hang),
bf16 model_init load (not fp32), vllm_gpu_memory_utilization 0.55 + sleep_mode. Feasible group size
on 24GB is ~4 (8 OOMs). Stability: M0's reward oscillated at lr 1e-4/beta 0 -> default to a lower lr
with a small KL anchor.

TRAIN/TEST FIREWALL: --tasks-json must be the TRAIN split only; never train on a scenario used for
evaluation.
"""
import argparse, json, os, sys
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import vllm
_orig = vllm.LLM.__init__
def _patched(self, *a, **kw):
    kw["enforce_eager"] = True
    return _orig(self, *a, **kw)
vllm.LLM.__init__ = _patched

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
from coding_env import CodingEnv
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

STORE = ROOT / "large_artifacts" / "_taskrepos"
SYSTEM = ("You are an expert Python coding agent. Think step by step, then use the tools to inspect "
          "files, write code, and run the tests until they pass. You MUST edit the target file with "
          "write_file and MUST run the tests with run_bash to verify before finishing. Do not give up "
          "after only reading files. Keep iterating until the tests pass.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged SFT warm-start composite (or base)")
    ap.add_argument("--tasks-json", default=str(ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "toolz_tasks.json"))
    ap.add_argument("--out", default=str(ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "adapters" / "rlvr"))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.02, help="KL anchor (M0 oscillated at beta=0)")
    ap.add_argument("--min-body", type=int, default=4)
    ap.add_argument("--max-body", type=int, default=30)
    a = ap.parse_args()

    tasks = [t for t in json.load(open(a.tasks_json)) if a.min_body <= t["body_lines"] <= a.max_body]
    rows = [{"prompt": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "Complete the coding task in this repository using the available tools."}],
             "repo_dir": str(STORE / t["repo"]), "rel_file": t["rel_file"], "func_name": t["func_name"],
             "python": str(STORE / "toolz" / ".venv-test" / "bin" / "python"),
             "test_cmd": "python -m pytest -q"} for t in tasks]
    ds = Dataset.from_list(rows * 8)
    print(f"RLVR: model={a.model}  train tasks={len(tasks)}  rows={len(ds)}  steps={a.steps}", flush=True)

    args = GRPOConfig(
        output_dir=a.out + "_out",
        per_device_train_batch_size=a.num_generations, num_generations=a.num_generations,
        max_completion_length=a.max_completion_length, max_tool_calling_iterations=10,
        learning_rate=a.lr, logging_steps=1, max_steps=a.steps, save_strategy="no", report_to=[],
        bf16=True, model_init_kwargs={"dtype": "bfloat16"}, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=0.55, vllm_max_model_length=4096,
        temperature=1.0, loss_type="dr_grpo", importance_sampling_level="sequence", beta=a.beta,
        chat_template_kwargs={"enable_thinking": True},
    )
    peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")

    hist = []
    class Log(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if logs and "reward" in logs:
                keep = {k: round(v, 4) for k, v in logs.items() if isinstance(v, (int, float)) and (
                    k in ("reward", "reward_std", "frac_reward_zero_std", "loss") or "mean_length" in k)}
                hist.append(keep)
                print(f"[step {len(hist)}] {keep}", flush=True)

    trainer = GRPOTrainer(model=a.model, args=args, train_dataset=ds, peft_config=peft_config,
                          environment_factory=lambda: CodingEnv())
    trainer.add_callback(Log())
    trainer.train()
    trainer.save_model(a.out)
    Path(a.out + "_history.json").write_text(json.dumps(hist, indent=1))
    if len(hist) >= 6:
        early = sum(h.get("reward", 0) for h in hist[:3]) / 3
        late = sum(h.get("reward", 0) for h in hist[-3:]) / 3
        print(f"\n=== RLVR reward: early {early:.4f} -> late {late:.4f} (delta {late-early:+.4f}) ===", flush=True)
    print(f"=== saved RLVR adapter -> {a.out} ===", flush=True)


if __name__ == "__main__":
    main()
