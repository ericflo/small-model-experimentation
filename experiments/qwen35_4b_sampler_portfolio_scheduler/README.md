# qwen35_4b_sampler_portfolio_scheduler

**Status:** finished

Standalone experiment package for testing whether multiple Qwen3.5-4B generation policies can be combined more efficiently than simply sampling more from one policy.

The experiment treats each generation policy as an arm that can contribute candidate programs. It evaluates static portfolios, oracle portfolio headroom, and a small learned scheduler that observes only a short base-hot prefix before choosing which policy block receives the remaining budget.

Large artifacts, if any are added later, should be stored outside this directory under:

`/workspace/large_artifacts/qwen35_4b_sampler_portfolio_scheduler`

Final report: `reports/final_report.md`
