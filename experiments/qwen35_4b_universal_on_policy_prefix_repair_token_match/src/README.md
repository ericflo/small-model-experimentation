# Source

`vllm_runner.py` is the pinned experiment-local bulk-generation template. It is
present for design review but is not yet authorized or invoked: the reviewed design
must first decide whether on-policy prefix collection requires the same Transformers
backend as local deployment or an explicitly merged same-policy vLLM path. Backend
mixing between paired arms is forbidden.
