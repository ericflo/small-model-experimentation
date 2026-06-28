# qwen35_4b_foofah_direct_vs_abi

Standalone external benchmark comparison:

- Frozen ABI oracle coverage on Foofah table transformations.
- Direct Qwen3.5-4B generation of the held-out `TestAnswer` table from the example pair and `TestingTable`.

The experiment asks whether the ABI/compiler route adds value over simply asking the base model to perform the transformation directly on external structural table tasks.

Source benchmark clone:

`/workspace/large_artifacts/external_sources/foofah_benchmarks`

## Reproduce

```bash
python scripts/build_cases.py
python scripts/eval_direct_qwen.py --limit 3 --max-new-tokens 220
python scripts/eval_direct_qwen.py --max-new-tokens 768 --progress-every 10
python scripts/make_report.py
```

The full direct-generation arm uses greedy decoding with `enable_thinking=False`
and a 768-token output cap. Outputs are scored by exact equality to Foofah's
held-out `TestAnswer` table after string normalization.
