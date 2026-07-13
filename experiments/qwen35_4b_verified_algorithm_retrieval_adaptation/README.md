# qwen35_4b_verified_algorithm_retrieval_adaptation

**Status:** finished

Standalone experiment package for testing verified algorithm retrieval plus Qwen3.5-4B adaptation on MBPP-style code tasks.

The experiment builds a verified algorithm library from training tasks, retrieves nearest algorithms for held-out tasks, asks Qwen to adapt retrieved code to the target function and public tests, and evaluates whether this recovers tasks missed by direct sampling.

Large artifacts, if any are added later, should be stored outside this directory under:

`/workspace/large_artifacts/qwen35_4b_verified_algorithm_retrieval_adaptation`

Final report: `reports/final_report.md`
