#!/usr/bin/env python3
"""Retired unsafe search entry point retained to fail closed for old commands."""

raise SystemExit(
    "retired: the historical search mixed HF/vLLM backends and read benchmark result "
    "details. Use train_trial.py, eval_curriculum.py, merge_trial.py, and "
    "run_benchmark.py under the frozen preregistration."
)
