# Analysis Summary

- Main compiler quick validation bytecode accuracy reached 64.1% at the final logged checkpoint.
- Main train prefix labels: exact positives 7.7%, raw semantic positives 56.6%, filtered semantic positives 35.2%.
- Exact-prefix value AUC reached 0.941; semantic value AUC reached 0.853.
- Fresh paired: greedy/logprob were 68.8%/68.8%; best exact value was 71.1% (`beam_exact_w0.5`); best semantic value was 68.0% (`beam_semantic_w0.25`); answer repair was 82.8%.
- Hard composition: greedy/logprob were 51.6%/51.6%; best exact value was 51.6% (`beam_exact_w0.25`); best semantic value was 51.6% (`beam_semantic_w2`); answer repair was 78.9%.
- Conclusion: bounded semantic reachability creates a broader, learnable target, but in this setup it does not beat exact-prefix supervision for top-1 no-answer beam selection. The remaining gap is not candidate containment; it is ranking/calibration of reachable candidates.
