# Qwen3.5-4B Generator-Verifier Gap Report

## Summary

Is the C2 selection wall (coverage ≫ deployable selection) a **capability** limit or a **plumbing**
limit? We measure a frozen Qwen3.5-4B's intrinsic **verification** skill (black-box: judge a candidate
correct/incorrect from the A/B logit, no execution, no hidden tests) against its **generation** skill
(pass@1) on the same MBPP items, at no-think and thinking-on. Result: **checking is easier than doing —
but only with thinking.** No-think self-verification is weak and heavily yes-biased (balanced-acc 0.627,
AUROC 0.773, says "correct" 91% of the time — essentially the generator agreeing with itself). Thinking
turns it into a real critic (balanced-acc **0.827**, AUROC **0.926**). And the model's own thinking-verifier
— **zero training, fully deployable** — selects best-of-8 at **0.860**, closing **75%** of the pass@1→oracle
gap (0.771 → 0.890), vs 24% for no-think. So **C2 is plumbing, not capability**: the selection program has
real headroom, and the lever is thinking-augmented self-verification. This also inverts C9: thinking helps
*verification* (+0.20 balanced-acc) at least as much as it helped *generation* — its deepest value may be
helping the model *know* which answer is right, not just *produce* right ones.

## Research Program Fit

Bridges `evidence_conditioned_selection` (C2) and `test_time_reasoning_budget` (C9). C2 has been treated
as an empirical wall across ~160 experiments, all of which build trained selectors and report selected
accuracy; none isolated the upstream black-box primitive of self-verification, or asked whether the wall
is verification capability vs evidence plumbing. This is that one measurement.

## Method

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8 no-think candidates,
  execution-labeled (pass rate 0.771).
- **Black-box verification judge:** present the task spec + one example assert + the candidate code, read
  P(correct) from the A(=correct, tok 32) vs B(=incorrect, tok 33) logit at a forced answer position — no
  execution, no hidden tests. No-think = one forward; thinking = generate up to 1024 thinking tokens, force
  `</think>`, then read the A/B logit at "Answer: ".
- **Metrics:** generation pass@1 / oracle pass@k; verification balanced-accuracy + AUROC (discriminate the
  model's own correct vs incorrect candidates); verifier-selected best-of-k vs pass@1 vs oracle (gap closed);
  the thinking asymmetry. **Controls:** foreign-solution judge (a different task's candidate → should be
  rejected); say-A rate + balanced accuracy control the yes-bias.

## Results

| quantity | no-think | thinking |
| --- | ---: | ---: |
| verification balanced accuracy | 0.627 | **0.827** |
| verification AUROC | 0.773 | **0.926** |
| say-A (correct) rate | 0.91 | 0.83 |
| verifier-selected best-of-8 | 0.800 | **0.860** |
| pass@1→oracle gap closed | +24% | **+75%** |
| foreign reject rate | 1.00 | 1.00 |

Generation: pass@1 **0.771**, oracle pass@8 **0.890**. Figure: `analysis/gen_verify.png`.

### Finding 1 — checking is easier than doing, but only with thinking
No-think self-verification is weak (balanced-acc 0.627, AUROC 0.773) with a strong yes-bias (say-A 0.91 >
base pass 0.771) — the no-think "verifier" mostly re-agrees with the generator. Thinking makes it a
genuine critic: balanced-acc 0.827, AUROC 0.926, and a lower yes-bias (0.83). So the model *can* tell its
own good solutions from its bad ones — when it thinks.

### Finding 2 — C2 is plumbing, not a capability wall
Using the model's own (black-box, training-free, deployable) verifier to pick best-of-8 lifts pass@1 0.771
→ 0.860 with thinking (0.800 no-think), against an oracle of 0.890 — closing **75%** of the achievable gap
(24% no-think). The selection program the corpus never stopped building has real headroom; the missing
piece was thinking-augmented self-verification, not a better trained selector.

### Finding 3 — thinking helps verification (the C9 inversion)
Thinking raised verification balanced-acc +0.20 (AUROC +0.15), comparable to or exceeding its effect on
generation (C9: greedy +15pp). Thinking's value is not only in *producing* correct answers but in
*recognizing* them — which is exactly the primitive selection needs.

## Controls

Foreign-solution judgments (a different task's candidate spliced in) are rejected at rate 1.00 in both
modes → the verifier genuinely reads the task, not a length/format heuristic. Balanced accuracy and the
say-A rate control for the yes-bias (a constant "A" predictor scores 0.5 balanced-acc; no-think's 0.627
shows weak-but-real discrimination, thinking's 0.827 shows strong discrimination).

## Oracle Versus Deployable Evidence

The verification judge uses **only** the task spec + candidate (no execution, no hidden tests), so
verifier-selected accuracy (0.860 thinking) is **deployable**. pass@8 = 0.890 is the non-deployable oracle
ceiling. So a deployable, zero-training thinking-verifier recovers 75% of the oracle headroom — notably,
without the visible-test signal the earlier thinking controller relied on (which was bounded by visible-test
false-passes at ~0.91 deployable / 0.93 oracle on a different pool).

## Interpretation

For a small model on this benchmark, the selection bottleneck is not that the model *can't tell* right from
wrong — it can, quite well, once it thinks (AUROC 0.93). The corpus's C2 wall is an evidence-plumbing
problem, and the highest-leverage plumbing is the model's own thinking-verifier. This unifies the corpus's
two strongest recent threads: thinking (C9) and selection (C2) meet in *thinking-augmented verification*,
which is cheaper and stronger than the trained selectors the selection program has favored, and a natural
next controller signal.

### Limitations
- MBPP (basic, likely partly contaminated), n=100, single seed; candidates are no-think generations (a
  think-generated pool may be harder to verify). The verifier sees one example assert (as the generator did).
  The "checking > doing" comparison is across different scales (discrimination AUROC vs generation pass@1);
  the deployable selection result (verifier-selected vs pass@1 vs oracle) is the apples-to-apples headline.

## Next Experiments

- Wire the thinking-verifier as the controller signal (vs / combined with the visible test) and measure the
  deployable accuracy-vs-token Pareto — does self-verification beat the visible-test C2 wall at matched cost?
- Verify a *think-generated* candidate pool (harder negatives) and on a contamination-controlled substrate.
- Iterated generate→self-verify→revise loop: does the strong thinking-verifier drive self-correction?

## Artifact Manifest

See `artifact_manifest.yaml`. Small records/labels + summary + figure in-repo; no external artifacts.
