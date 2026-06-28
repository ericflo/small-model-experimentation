# Analysis Summary

- Main verifier held-out prefix AUC reached 0.937.
- Fresh paired greedy accuracy was 64.1%; best verifier beam was 64.8% (`beam_verifier_w2`); local answer repair was 82.0%.
- Fresh paired logprob beam oracle was 84.4%, so correct programs were often in the beam even when top-1 selection did not improve much.
- Hard-composition greedy accuracy was 41.4%; best verifier beam was 44.5% (`beam_verifier_w2`); local answer repair was 70.3%.
- Hard-composition logprob beam oracle was 70.3%, exposing a large remaining reranking gap.
