# qwen35_4b_foofah_external_transform_gate

Standalone external transformation gate using the Foofah benchmark format.

The experiment freezes a compact table-transformation ABI, loads benchmark cases from an external source, and measures:

- raw oracle coverage on `InputTable` -> `OutputTable`
- held-out coverage on `TestingTable` -> `TestAnswer`
- coverage by benchmark family and sample count
- constrained selection accuracy for simple baselines

No model training is performed unless oracle coverage is high enough to make compiler selection meaningful.

Source benchmark clone:

`/workspace/large_artifacts/external_sources/foofah_benchmarks`

## Reproduce

```bash
python scripts/run_gate.py
python scripts/make_report.py
```
