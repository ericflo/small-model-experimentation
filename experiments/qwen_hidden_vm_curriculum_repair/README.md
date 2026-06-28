# Qwen Hidden VM Curriculum Repair

Standalone experiment for training `Qwen/Qwen3-4B` to compile natural-language tasks into a fixed hidden VM with a length curriculum and verifier-guided program repair.

Small files live here:

```text
experiments/qwen_hidden_vm_curriculum_repair/
```

Large checkpoints live separately here:

```text
large_artifacts/qwen_hidden_vm_curriculum_repair/checkpoints/
```

The main question is whether a Qwen-attached hidden VM compiler can become more length-robust when trained with staged program lengths and then distilled from locally verified program repairs.
