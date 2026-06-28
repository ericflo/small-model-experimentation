# Analysis Summary

- Main compiler quick validation bytecode accuracy reached 71.9%.
- Main train prefix labels: exact positives 4.3%, found positives 72.1%, mean Q 0.259, mean advantage 0.415.
- Value-model best held-out AUCs: exact 0.950, found 0.814, qvalue 0.628, advantage 0.666.
- Fresh paired: greedy/logprob were 67.2%/67.2%; best exact was 68.8% (`beam_exact_w1`); best found was 68.8% (`beam_found_w0.25`); best advantage was 69.5% (`beam_advantage_w0.25`); answer repair was 82.0%.
- Hard composition: greedy/logprob were 56.2%/56.2%; best exact was 57.0% (`beam_exact_w0.25`); best advantage was 57.0% (`beam_advantage_w1`); answer repair was 78.9%.
- Conclusion: budgeted action-value labels expose a large recoverability set and produce small no-answer beam gains, but the decisive signal in this setup is still supervised candidate execution against the target answer. The next experiment should move supervision closer to the answer-verified oracle instead of only learning a lightweight partial-program ranker.
