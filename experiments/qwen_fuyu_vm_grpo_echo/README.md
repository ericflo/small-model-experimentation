# Qwen Fuyu VM GRPO-ECHO

This standalone experiment trains `Qwen/Qwen3-4B` as a Fuyu-style recurrent
controller for a typed bytecode VM.

The model does not emit natural-language action tokens. Each VM step is one
full Qwen forward pass over prompt embeddings plus dense VM-state embeddings.
Structured heads choose one VM edit action or `STOP`; the VM executes that
action; the resulting observation is projected into dense state tokens for the
next full-model pass.

The key intervention is trajectory-level GRPO-style training with structured
ECHO observation prediction. Rewards are assigned to complete VM rollouts, while
the auxiliary ECHO loss trains the same hidden state to predict the VM
observation caused by each sampled action.

Large checkpoints are stored outside this directory:

```text
large_artifacts/qwen_fuyu_vm_grpo_echo/checkpoints/
```

Read `experiment_log.md` for the iteration record. Reports are written under
`reports/`.
