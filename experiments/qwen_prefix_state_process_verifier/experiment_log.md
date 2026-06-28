# Experiment Log

## 2026-06-23

- Created standalone experiment directory with `src/`, `runs/`, `analysis/`, `reports/`, and a separate large-artifact checkpoint root.
- Design choice: use frozen Qwen prompt features plus a trained bytecode compiler head, then test whether a prefix-state process verifier can improve typed beam search over greedy decoding and simple local answer-verified search.
- Planned conditions:
  - compiler-only greedy constrained decoding;
  - fixed local answer-verified repair over complete programs;
  - typed beam search scored only by compiler log probability;
  - typed beam search scored by compiler log probability plus the learned prefix-state verifier.
- Implemented the self-contained experiment harness in `src/qwen_prefix_state_process_verifier_experiment.py`.
- Verified VM execution and prefix-sample generation locally.
- Ran `smoke_prefix_state_verifier`, an end-to-end Qwen-backed smoke test with tiny data. It loaded `Qwen/Qwen3-4B`, extracted frozen features, trained a tiny compiler, collected prefix samples, trained the verifier, wrote metrics, and saved the checkpoint under the large-artifact root.
- Ran `pilot_prefix_state_verifier_s128`. The verifier reached held-out prefix AUC 0.921, but top-1 beam gains were modest with the weak 128-example compiler. Fresh paired greedy was 9.4%, local answer repair was 46.9%, compiler-logprob beam was 9.4%, and the best verifier beam was 12.5%. This suggests the main run needs a stronger compiler and a wider verifier-weight sweep.
- Started `main_prefix_state_verifier_s512`; interrupted the first attempt during final evaluation because the naive beam implementation scored every prefix expansion with a separate verifier forward pass.
- Patched typed beam search to score each beam layer in batches and reran `main_prefix_state_verifier_s512` cleanly.
- Main result: verifier AUC reached 0.936 on held-out prefix states. Fresh paired greedy was 64.1%, local answer repair was 82.0%, compiler-logprob beam was 64.1%, and the best verifier beam was 64.8%. Hard-composition greedy was 41.4%, local answer repair was 70.3%, compiler-logprob beam was 41.4%, and the best verifier beam was 44.5%.
