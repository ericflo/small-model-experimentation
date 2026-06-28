# Qwen Tail Repair Stability Critic

Standalone experiment for testing whether a learned tail-repair critic can turn
mostly-correct length-24 modular programs into stable exact executable programs.

The experiment uses frozen source compilers as generators, enumerates local
tail edits around each generated program, labels candidates with exact execution
trace checks, and trains a small critic to select repair candidates without
using target answers or target states at inference.

Large checkpoints and cached candidate groups live under:

`/workspace/large_artifacts/qwen_tail_repair_stability_critic/`

## Reports

- Markdown: `reports/qwen_tail_repair_stability_critic_report.md`
- HTML: `reports/qwen_tail_repair_stability_critic_report.html`
- Figures: `reports/figures/`

## Result

The repair candidate set had high oracle coverage, but the learned feature
critic did not improve the deployable selector. On standard length-24 examples,
no repair, the iteration-1 critic, and the iteration-2 focused critic all ended
at 44.8% mean exact accuracy with 44.6% source-seed standard deviation. The
answer-oracle candidate selector reached 91.1%, so candidate coverage was not
the limiting factor. The stability gate failed because mean accuracy and
source-seed variance were unchanged.
