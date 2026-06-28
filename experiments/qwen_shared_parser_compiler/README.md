# Qwen Shared Parser Compiler

This experiment tests whether a shared token parser can read a frozen Qwen
hidden sequence, recover ordered program symbols, and configure an executable
latent modular program without token-span inputs at inference time.

Lightweight code, logs, run JSON/CSV, analysis, and reports live in this
directory. Saved checkpoints live under:

```text
large_artifacts/qwen_shared_parser_compiler/checkpoints/
```

The compiler uses shared token-level role and symbol heads, a monotonic
operation slot reader, and an after-operation argument reader, so parser weights
are reused across operation steps without span inputs at inference time.
