# Literature as a Failure Map

The goal was not to find a paper to replicate. It was to identify the easiest ways a recurrence experiment can look profound without answering the deeper-representation counterfactual.

| Work | What it establishes | What remains or can fool us | Design response here |
|---|---|---|---|
| [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431) | Residual paths can behave like collections of shorter computations. | More layers or residual calls do not prove one representation was serially transformed. | Compare one carried state to an equal-call reset-state bag. |
| [Universal Transformers](https://arxiv.org/abs/1807.03819) | Weight-tied recurrent depth can express iterative algorithms. | From-scratch recurrence does not show a pretrained LM can be reorganized, and extra depth can be the only comparator. | Retrofit the fixed repository model and preserve an exact K=1 path. |
| [Coconut](https://arxiv.org/abs/2412.06769) | Hidden states can be fed back as continuous thoughts. | A continuous feedback channel may compress or mimic CoT without demonstrating a progressively sufficient world state. | Query-after-state supervision and causal swaps; explicit CoT remains a comparator. |
| [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171) | A recurrent block can scale latent test-time computation to substantial model size. | “More recurrent compute helps” does not isolate serial state from equal-compute shallow representations. | State-Carry versus separately optimized State-Bag is the headline endpoint. |
| [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://openreview.net/forum?id=din0lGfZFd) | Looped effective depth can approximate deeper untied computation on controlled tasks. | Expressivity or synthetic success is not evidence that a particular pretrained 4B learns causally useful state. | Full Qwen retrofit, held-out depth/family, state sufficiency, and donor interventions. |
| [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384) | Recurrence can be retrofitted into pretrained models using a recurrence curriculum. | A recurrence retrofit is now a baseline rather than a breakthrough; performance can still reflect compute or training recipe. | Exact reset-state twin, same initialization/data/parameters, and a fixed final checkpoint. |
| [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822) | Properly trained recurrent-depth models can extrapolate by increasing inference recurrence. | Excess recurrence produces overthinking; trained depth extrapolation can be fragile. | K=4 training, K=5–12 evaluation, stationary step encoding, fixed-point loss, and overthinking curves. |
| [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) | Naive block reapplication commonly degrades; damped substeps can stabilize it. | Training-free gains may be small and do not identify semantic state. | Damped recurrence is an engineering prior; it cannot pass without Carry>Bag and state causality. |
| [DiscoLoop](https://arxiv.org/abs/2607.00341) | A bridge entity may be decoded yet misaligned with its token embedding; mixed discrete/continuous recurrence repairs the interface. | A mixed channel alone is no longer a novel deeper-representation contribution. | Semantic echo is a triggered ablation. A positive is labeled interface-only unless continuous serial state and causal gates also pass. |

## Repository Evidence That Changes the Bet

### Serial computation is real

C44 found reasoning-SFT at 1.00 through generated reasoning and 0.01 when forced into one answer pass. C45 showed the installed hypothesize/verify procedure transfers within a held-out affine family. This makes “the 4B cannot perform serial algorithms” untenable; the open question is whether those algorithms can live in an internal reusable state instead of emitted tokens.

### Static weights are depth-local

C11–C24 show self-training and banking install what their data covers but do not automatically climb the next depth. C54 moves a deep tier through compressed successful traces yet finds a quick/deep Pareto frontier in one static adapter. Dynamic recurrence is attractive because it can be absent at K=1 and spend temporary state/compute only where depth is needed.

### Structure, not arithmetic, is the wall

C13/C16/C32/C36 show Qwen executes a supplied procedure well but struggles to propose deeper structure. Random pointer worlds therefore avoid making arithmetic the bottleneck and ask recurrence to maintain the missing joint structure directly.

### Readable is not usable

C19 decodes shallow composition information; C20's ActAdd is inert. C30 gains only when a complete decoded operation is externalized into an interface the model consumes. The later Jacobian transport replication demonstrates that a properly chosen context-local state can be causally consumed. This motivates both donor-following swaps and the semantic-echo branch.

### Thin recurrence already failed

`qwen_fastweight_hook` inserted a 256-dimensional recurrent fast-weight module late in Qwen and trained mostly from answer-letter NLL. Favorable 100-example K bumps disappeared on 250 examples, and a K=0-only control produced similar bumps. This experiment responds with full-width repeated Qwen motifs, state supervision, exact paired comparisons, larger samples, three seeds, and a fixed final checkpoint.

## Ideas Pruned as Primaries

- **Longer native CoT:** already establishes serial token computation, not silent deeper representation.
- **Naive layer looping:** now a literature replication and vulnerable to instability.
- **Continuous-state feedback alone:** close to Coconut and insufficiently causal.
- **Mixed discrete/continuous recurrence alone:** close to DiscoLoop and primarily an interface result.
- **Another small fast-weight adapter:** directly contradicted by the nearest repository experiment at its tested bandwidth/supervision.
- **Untied added layers:** changes parameter count and confounds optimization with scaling.
- **External DSL/VM repair:** gives structure to an external executor rather than demonstrating deeper internal representation.
- **Activation probes:** decodability cannot establish use.

The surviving experiment is the one whose positive cannot be explained by any of those easier stories.
