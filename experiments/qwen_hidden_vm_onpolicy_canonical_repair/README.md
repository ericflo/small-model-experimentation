# Qwen Hidden VM On-Policy Canonical Repair

Standalone experiment for training `Qwen/Qwen3-4B` to compile mixed natural-language tasks into a hidden typed VM, then improve the compiler with canonical on-policy repair targets.

Small files live here:

```text
experiments/qwen_hidden_vm_onpolicy_canonical_repair/
```

Large checkpoints live separately here:

```text
large_artifacts/qwen_hidden_vm_onpolicy_canonical_repair/checkpoints/
```

The main question is whether verified state-equivalent local repairs can be folded back into a Qwen-attached hidden VM compiler without damaging the trace-trained policy.
