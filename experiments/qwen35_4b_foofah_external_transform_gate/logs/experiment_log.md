# Experiment Log

## 2026-06-27

- Created standalone external transformation gate package.
- Selected Foofah benchmark format because each case provides input/output examples and separate held-out test tables.
- Frozen ABI scope: projection, header dropping, unpivot/melt, key:value folding, and regex extraction.
- Planned gate: run oracle coverage first; only run model selection if held-out oracle coverage is large enough to interpret.
- Smoke run on 20 files completed: 5/20 held-out coverage.
- Full run on 250 files completed: raw coverage 57/250 (22.8%), held-out coverage 45/250 (18.0%).
- First-visible selection solved 43/250 (17.2%), nearly all held-out-covered cases, so model-side selection was not run.
- Generated final report and charts under `reports/`.
