# qwen35_4b_foofah_ephemeral_program_induction

**Status:** finished

Standalone experiment on Foofah table transformations:

- Direct Qwen3.5-4B table generation.
- Ephemeral Python program induction: Qwen writes a `transform(table)` function from examples.
- Visible-example verification of generated programs.
- Held-out execution on Foofah `TestingTable`.
- Agreement analysis between direct output and executable program output.

The experiment asks whether "write a bespoke executable transformer, verify it, then execute it" adds value over direct output generation on an external table-transformation benchmark.

Benchmark source:

`/workspace/large_artifacts/external_sources/foofah_benchmarks`

## Reproduce

```bash
python scripts/build_cases.py
python scripts/eval_qwen.py --limit 8 --program-prompt induce --max-direct-tokens 384 --max-code-tokens 512
python scripts/eval_qwen.py --program-prompt context --max-direct-tokens 768 --max-code-tokens 768 --progress-every 10
python scripts/make_report.py --records reports/eval_records_context.jsonl
```

All scoring is exact table equality against held-out `TestAnswer` after converting cells to strings.
