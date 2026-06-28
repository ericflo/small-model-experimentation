# Experiment Log

## 2026-06-27

- Created standalone direct-vs-ABI comparison package.
- Scope: exact held-out `TestAnswer` table on Foofah cases.
- Planned arms: frozen ABI held-out coverage, first-visible ABI selection, direct Qwen generation.
- Imported the frozen Foofah ABI gate outputs into `data/abi_*` for a standalone apples-to-apples comparison.
- Smoke-tested direct Qwen with no-thinking chat template: 3/3 parse, 3/3 exact on the first three cases.
- Launched full direct Qwen greedy JSON generation with `max_new_tokens=768`.
- Full run complete: direct Qwen 138/250 exact (55.2%), parse 236/250 (94.4%).
- Frozen ABI oracle coverage 45/250 (18.0%), first-visible 43/250 (17.2%).
- Direct-or-ABI-first-visible fallback 147/250 (58.8%), a +9 case lift over direct alone.
- Interpretation: the Foofah structural-transform ABI is complementary but not the main route; direct generation dominates the frozen ABI gate on this external benchmark.
