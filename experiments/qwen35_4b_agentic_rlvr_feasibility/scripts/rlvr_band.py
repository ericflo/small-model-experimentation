"""RLVR (GRPO, execution reward) on the calibrated difficulty BAND, from the SFT warm-start.

The chain that got here: M0 proved the single-GPU GRPO loop closes (C61) -> the raw base gave
reward_std=0 (quits without writing) -> an SFT warm-start on harvested completed trajectories
installed ENGAGEMENT (4x episode length) -> per-task calibration found the band where the
warm-started policy passes SOMETIMES (0.25-0.75), which is where GRPO actually has advantage signal.
This trains on exactly that band.

Reward = execution (tests pass = 1.0). Policy = merged warm-start composite (C49: deploy merged).
Single-4090 recipe: enforce_eager patch, sleep level-1 (level-2 offloads weights to host RAM and
OOMs a 15GB box), bf16 load, util 0.50.
"""
import argparse, json, os, sys
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import vllm
_orig_init = vllm.LLM.__init__
def _patched_init(self, *a, **kw):
    kw["enforce_eager"] = True
    return _orig_init(self, *a, **kw)
vllm.LLM.__init__ = _patched_init
_orig_sleep = vllm.LLM.sleep
def _sleep_level1(self, level=2):
    return _orig_sleep(self, level=1)
vllm.LLM.sleep = _sleep_level1

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
from coding_env import SynthEnv
import synth_scenarios
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"
SYSTEM = ("You are an expert Python coding agent. Think step by step, then use the tools to inspect "
          "files, write code, and run the tests until they pass. You MUST edit the target file with "
          "write_file and MUST run the tests with run_bash to verify before finishing. Do not give up "
          "after only reading files. Keep iterating until the tests pass.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--band", default=str(OUTD / "difficulty.json"))
    ap.add_argument("--out", default=str(OUTD / "adapters" / "rlvr_band"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--vllm-util", type=float, default=0.50)
    ap.add_argument("--tool-iters", type=int, default=12, help="max multi-turn tool iterations per episode")
    ap.add_argument("--model-len", type=int, default=16384, help="vLLM context for LONG agentic rollouts")
    ap.add_argument("--grad-accum", type=int, default=12,
                    help="KEY: micro-batch x grad-accum = generation batch, which must be divisible by\n"
                         "num_generations. Accumulation lets the GROUP stay large (signal) while the\n"
                         "logprob-forward micro-batch stays small (memory).")
    ap.add_argument("--micro-batch", type=int, default=1,
                    help="per_device_train_batch_size: sequences in ONE logprob forward (memory knob)")
    a = ap.parse_args()

    band = set(t.split(":", 1)[1] for t in json.load(open(a.band))["band"] if t.startswith("synth:"))
    scen = {s["id"]: s for s in synth_scenarios.SCENARIOS}
    picked = [scen[i] for i in sorted(band) if i in scen]
    if not picked:
        print("no band tasks found — run calibrate_difficulty.py first"); return
    print(f"RLVR band tasks ({len(picked)}): {[s['id'] for s in picked]}", flush=True)

    rows = [{"prompt": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "Complete the coding task using the available tools."}],
             "scenario_id": s["id"], "files": json.dumps(s["files"]), "check": s["check"],
             # SynthEnv.reset appends this to the user turn
             "prompt_text": s["prompt"]} for s in picked]
    # reset(**row) receives every column; name the task text `prompt_text` -> map to what reset expects
    for r in rows:
        r["prompt_text"] = r["prompt_text"]
    ds = Dataset.from_list(rows * 12)

    args = GRPOConfig(
        output_dir=a.out + "_out",
        per_device_train_batch_size=a.micro_batch, num_generations=a.num_generations,
        max_completion_length=a.max_completion_length, max_tool_calling_iterations=a.tool_iters,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, logging_steps=1, max_steps=a.steps, save_strategy="no", report_to=[],
        bf16=True, model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True},
        gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=a.vllm_util, vllm_max_model_length=a.model_len,
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
                Path(a.out + "_history.json").parent.mkdir(parents=True, exist_ok=True)
                Path(a.out + "_history.json").write_text(json.dumps(hist, indent=1))

    trainer = GRPOTrainer(model=a.model, args=args, train_dataset=ds, peft_config=peft_config,
                          environment_factory=lambda: SynthEnv())
    trainer.add_callback(Log())
    trainer.train()
    trainer.save_model(a.out)
    if len(hist) >= 6:
        n = max(3, len(hist) // 4)
        early = sum(h.get("reward", 0) for h in hist[:n]) / n
        late = sum(h.get("reward", 0) for h in hist[-n:]) / n
        print(f"\n=== RLVR reward: early {early:.4f} -> late {late:.4f} (delta {late-early:+.4f}) ===", flush=True)
        print("RLVR LEARNED" if late - early > 0.05 else "no clear movement yet", flush=True)
    print(f"=== saved -> {a.out} ===", flush=True)


if __name__ == "__main__":
    main()
