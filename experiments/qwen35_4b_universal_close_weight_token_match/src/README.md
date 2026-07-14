# Source

`vllm_runner.py` is the experiment-local pinned Qwen3.5-4B aggregate inference
runner copied from the authenticated parent line. It is used only after explicit
adapter merges and only through the trusted aggregate benchmark gateway.

Training, stream construction, local evaluation, merge, and gateway wrappers live
under `scripts/` so this result-bearing follow-up remains self-contained.
