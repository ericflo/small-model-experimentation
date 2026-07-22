"""Install the token-cap ladder into pi's model config (the C63-follow-on termination experiment).

MECHANISM UNDER TEST (from the design synthesis, grounded in measured data): pi sends its model
entry's `maxTokens` as `max_completion_tokens` on EVERY call. At the measured ~14.5 tok/s per-stream
decode (eager mode), 8192 tokens = ~565s -- approximately the entire 600s episode wall. So one
permitted-length <think>-dominated call can consume the whole episode before pi's agent loop gets a
second turn; the observed "loop to the wall" is often ONE runaway generation, not many turns (the
pi_rft trace showed exactly this: 92 stream deltas, one message_end). Capping per-call output at 2560
(~177s worst case) mechanically restores the invariant that an episode is a LOOP (>=3 maxed calls,
~20 typical calls within the wall). pi's own agent loop handles truncation: truncated-with-tool-calls
gets a "re-issue with complete arguments" correction and continues; truncated-pure-think ends the
turn cleanly and the workspace is still scored from disk.

This is the first termination lever that is MECHANICAL (removes permission to continue) rather than
persuasive (asks the policy to stop) -- persuasion failed four times in training and is soft in
prompting.

Adds (idempotent, additive; original backed up once):
- provider `kiln-proxy` -> baseUrl http://127.0.0.1:8421/v1 (the instrumented pi_proxy), carrying
  the FULL ladder: qwen35-4b-pi8k (8192, re-anchor arm), qwen35-4b-pi2560 (primary), qwen35-4b-pi4096
  (contingent). All contextWindow 40960 -- do NOT copy the stale 262144 entries, they trip vLLM's
  prompt+max_completion_tokens length check.
- the pi2560/pi4096 entries into `kiln-local` too, for proxy-less runs.

Serve all arms from ONE weight load:
  vllm serve <merged/warmstart> --served-model-name qwen35-4b-pi8k qwen35-4b-pi2560 qwen35-4b-pi4096 \
    --enforce-eager --max-model-len 40960 --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --port 8420
"""
import json
import shutil
from pathlib import Path

CFG = Path.home() / ".pi/agent/models.json"
CTX = 40960
LADDER = [("qwen35-4b-pi8k", 8192), ("qwen35-4b-pi2560", 2560), ("qwen35-4b-pi4096", 4096)]


def entry(mid, cap):
    return {"contextWindow": CTX, "id": mid, "input": ["text"], "maxTokens": cap,
            "name": f"Qwen 3.5 4B ({cap} cap)"}


def main():
    d = json.loads(CFG.read_text())
    bak = CFG.with_suffix(".json.bak-capladder")
    if not bak.exists():
        shutil.copy(CFG, bak)

    prov = d["providers"].setdefault("kiln-proxy", {
        "api": "openai-completions", "apiKey": "dummy",
        "baseUrl": "http://127.0.0.1:8421/v1",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
        "models": [],
    })
    for mid, cap in LADDER:
        if not any(m["id"] == mid for m in prov["models"]):
            prov["models"].append(entry(mid, cap))
    local = d["providers"]["kiln-local"]["models"]
    for mid, cap in LADDER[1:]:
        if not any(m["id"] == mid for m in local):
            local.append(entry(mid, cap))

    CFG.write_text(json.dumps(d, indent=2))
    print("applied. kiln-proxy models:",
          [(m["id"], m["maxTokens"]) for m in d["providers"]["kiln-proxy"]["models"]])
    print("kiln-local ladder additions:",
          [(m["id"], m["maxTokens"]) for m in local if m["id"] in {x[0] for x in LADDER}])


if __name__ == "__main__":
    main()
