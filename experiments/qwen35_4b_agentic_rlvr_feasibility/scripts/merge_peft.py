"""Merge a PEFT LoRA adapter into a full Qwen3.5-4B composite that vLLM can load.

Why not the vendored merge_adapter.py: that merger hand-parses adapter key names for adapters
trained against the text-only `base_reserialized` composite. Adapters trained against the
multimodal `Qwen/Qwen3.5-4B` carry a `language_model.` path segment
(`base_model.model.model.language_model.layers.N...`) which it rejects. PEFT knows its own key
layout, so `merge_and_unload()` handles either.

Loads the MULTIMODAL class (AutoModelForImageTextToText) so the saved checkpoint keeps the full
Qwen3.5 structure vLLM expects, and copies the preprocessor configs the merger otherwise drops.
Merges on GPU (device_map) because host RAM (15GB) is the binding constraint on this box.
"""
import argparse, shutil, sys
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

EXTRA_CONFIGS = ["preprocessor_config.json", "video_preprocessor_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0", help="merge device (host RAM is tight)")
    a = ap.parse_args()

    print(f"[merge] loading base {a.base_model} on {a.device}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        a.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=a.device)
    print(f"[merge] applying adapter {a.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, a.adapter)
    model = model.merge_and_unload()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving composite -> {out}", flush=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base_model).save_pretrained(str(out))

    # vLLM needs the multimodal preprocessor configs present alongside the weights
    try:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(a.base_model, allow_patterns=EXTRA_CONFIGS))
        for name in EXTRA_CONFIGS:
            src = snap / name
            if src.exists() and not (out / name).exists():
                shutil.copy(src, out / name)
                print(f"[merge] copied {name}", flush=True)
    except Exception as e:
        print(f"[merge] preprocessor config copy skipped: {type(e).__name__}: {e}", flush=True)
    print("[merge] done", flush=True)


if __name__ == "__main__":
    main()
