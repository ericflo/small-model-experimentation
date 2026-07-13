# Qwen Span-Free Compiler

**Status:** finished

This experiment tests whether a small trainable latent compiler can read a
frozen Qwen hidden sequence and configure an executable modular program without
being given token-span features for the numeric values or operation words.

Lightweight files live in this directory. Saved checkpoints live under:

```text
large_artifacts/qwen_span_free_compiler/checkpoints/
```

Run outputs are written under `runs/`, analysis outputs under `analysis/`, and
the chronological experiment log plus final report under `reports/`.
