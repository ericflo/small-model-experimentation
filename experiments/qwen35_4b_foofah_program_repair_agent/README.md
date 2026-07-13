# qwen35_4b_foofah_program_repair_agent

**Status:** finished

Standalone experiment on Foofah table transformations.

The experiment asks whether Qwen3.5-4B becomes more useful when it writes an
executable table-transform program, observes visible-example failures, and
repairs the program over several rounds before executing it on the held-out
input table.

Arms and measurements:

- Direct JSON output for the held-out table.
- Initial generated `transform(table)` program.
- Repair loop with visible feedback from the example input/output.
- Final visible-verified program, scored on held-out `TestAnswer`.
- Direct/program oracle union and deployable fallback metrics.
- False-pass rate: visible-example pass but held-out failure.

Benchmark source:

`/workspace/large_artifacts/external_sources/foofah_benchmarks`

## Reproduce

```bash
python scripts/build_cases.py
python scripts/eval_repair_agent.py --limit 6 --max-repairs 2 --max-direct-tokens 384 --max-code-tokens 512
python scripts/eval_repair_agent.py --max-repairs 3 --max-direct-tokens 768 --max-code-tokens 768 --progress-every 10
python scripts/make_report.py
```

All scoring uses exact equality to Foofah's held-out `TestAnswer` table after
converting all cells to strings.
