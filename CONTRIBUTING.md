# Contributing

## Add An Experiment

1. Pick a stable snake_case id.
2. Copy `templates/experiment/` to `experiments/<id>/`.
3. Fill in `README.md` before running the full experiment.
4. Add a cheap smoke run and record how to run it.
5. Put code, data, runs, analysis, and reports inside the experiment folder.
6. Run `make catalog` and `make validate`.

## Report A Result

Every result-bearing experiment should answer:

- What question did this test?
- What was the strongest baseline or control?
- What evidence would have falsified the idea?
- What changed after the result?
- What should the next experiment do differently?

## Artifact Policy

Keep small reproducible artifacts in git. Use Git LFS for model/checkpoint files. Put externally stored artifacts in a manifest that names the path, checksum when available, and how to regenerate or retrieve them.

