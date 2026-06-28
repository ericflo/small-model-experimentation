# Analysis Summary

- Main QLoRA trace run reached 68.0% direct executable bytecode accuracy on fresh paired prompts and 85.9% with answer-verified local search.
- Hard-composition direct bytecode accuracy for the main run was 55.5%.
- The frozen-Qwen trace-head control reached 66.4% fresh paired direct bytecode accuracy, only +1.6 pp behind the live QLoRA trace run.
- The answer-only QLoRA control reached 14.8% fresh paired answer accuracy, far below executable trace supervision.
- Expert iteration improved fresh paired direct bytecode from 21.9% after seed training to 48.4% after two rounds, but did not beat dense 512-trace supervision.
- Expert target found rate rose from 58.3% to 68.3%; changed-target rate fell from 68.8% to 49.1%.
