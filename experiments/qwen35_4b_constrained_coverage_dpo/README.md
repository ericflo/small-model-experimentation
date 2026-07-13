# qwen35_4b_constrained_coverage_dpo

**Status:** finished

Standalone experiment package for a constrained coverage preference objective on Qwen3.5-4B code generation.

The experiment asks whether a small local adapter can improve coverage/sample efficiency without paying for it through parse collapse or first-sample degradation. The objective combines hard-negative DPO with reference-logprob anchoring and positive NLL anchoring. Success is defined on the coverage/pass@1/token Pareto frontier against inference-only sample-more baselines, not merely against a smaller K baseline.

Large model artifacts are stored outside this directory under:

`/workspace/large_artifacts/qwen35_4b_constrained_coverage_dpo`

Final report: `reports/final_report.md`

Pilot readout: constrained DPO preserved pass@1/parseability and beat the shuffled control, but did not reach the K=8 sample-more coverage reference. The useful next lead is policy-portfolio scheduling, not scaling this single constrained sampler unchanged.
