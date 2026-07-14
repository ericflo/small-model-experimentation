# Source

`vllm_runner.py` is the repository's pinned Qwen3.5-4B bulk runner. This copy keeps
the template's sampling/provenance contract and adds the established explicit merged
composite override because runtime LoRA is a verified silent no-op for this model in
the pinned vLLM stack.

The override validates the exact Qwen3.5-4B config fingerprint, is mutually exclusive
with `--adapter`, loads tokenizer/config locally, and records a null hub revision.
Tests cover the accepted and rejected fingerprints.
