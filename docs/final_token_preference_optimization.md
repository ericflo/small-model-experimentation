# Final-Token Preference Optimization (FTPO)

A reference for a family of techniques that detect and correct **localized generation
pathologies** in language models — runaway repetition loops and systematically
over-represented lexical patterns — by training a preference at the *single token position*
where the pathological continuation begins, using the model's own next-token alternatives as
the positive signal. No external teacher, no gold answers, no larger model anywhere in the
loop: the correction signal is extracted entirely from the model's own probability
distribution plus a rule-based detector. That makes the method compatible with this
repository's standing constraint that capability may never be imported from a larger or
external model (see [Relevance to this corpus](#relevance-to-this-corpus)).

**Lineage.** The technique was developed by Sam Paech and collaborators as a three-part
framework — a backtracking suppression sampler, an automated model-profiling pipeline, and the
FTPO training objective — published at ICLR 2026 (arXiv:2510.15061, Paech, Roush, Goldfeder &
Shwartz-Ziv) with the lexical-pattern problem as its target. Liquid AI then adapted the same
single-token preference machinery to runaway repetition in reasoning models (blog post + open
pipeline, July 2026, with Paech as primary author), swapping the pattern detector for a
repetition-loop detector. Both codebases are open; full links in [References](#references).
Throughout this document, "the lexical pipeline" refers to Paech's automated implementation
and "the repetition pipeline" to Liquid AI's.

## The failure modes

### 1. Periodic repetition loops

A model mid-generation emits a span — often a discourse connective like `Wait,` or
`Alternatively,` followed by a restatement — and then repeats that span verbatim until the
context window or token budget is exhausted. On hard math and coding tasks this affects a
double-digit share of completions for small reasoning models: Liquid AI measured **10.2%** of
completions for an early LFM2.5-2.6B checkpoint, and **22.9%** for Qwen3.5-4B under greedy
sampling.

Three mechanisms have to line up:

- **Over-trained initiator tokens.** Heavy (often synthetic) reasoning-data training makes a
  handful of high-frequency function and connective tokens unusually attractive continuations.
  When the model is uncertain, these tokens dominate the next-token distribution without
  moving the reasoning forward. In the LFM2.5-2.6B measurements the most common
  loop-initiating token was the article `' the'` (11.39% of loops), followed by `' So'`
  (4.51%), `'Alternatively'` (3.22%), `'Wait'` (2.56%), and `' But'` (2.46%).
- **Context self-reinforcement.** Once a span has appeared, the prior context makes the same
  span more likely to appear again; with each repetition the probability of every token in the
  looping span climbs toward 1.0. Follow-up analysis work links this to a characteristic
  V-shaped attention pattern and finds that *semantic* repetition (being stuck on an idea)
  precedes *textual* repetition (Duan et al. 2026; see also Pipis et al. 2025).
- **Low-temperature decoding.** Reasoning models are typically run near temperature 0 for
  stable, reproducible traces. At the argmax, a locally reinforced loop has no escape route —
  and once self-reinforcement has pushed the loop token's probability near 1, even moderate
  temperatures rarely escape (looping is still significant at T=0.67).

This corpus has met this failure mode directly: in the
[verified-macro long-context rerun](../experiments/qwen35_4b_verified_macro_long_context_rerun/reports/report.md),
exact periodic loops dominated the unfinished reasoning tails at high thinking budgets —
81/144 samples at think@32,768 (a 32,768-token reasoning budget), with loops still dominant in
follow-up probes at 49k — and the [model playbook](model_playbook.md) now requires gating
exact periodic loops separately before reading correctness.

### 2. Over-represented lexical patterns

The stylistic sibling of the same defect: specific words, phrases, n-grams, and sentence
constructions appear in model output at frequencies far above human-writing baselines — the
stereotyped register of assistant prose. The magnitudes are extreme and measurable: the ICLR
paper reports individual words appearing **hundreds to tens of thousands of times** more
frequently than in human text (e.g. the invented name "elara" at 85,513× in one model's
creative writing; "unsettlingly" at 3,833×), and structural templates like *"It's not X, it's
Y"* at 6.3× human frequency. These over-use profiles are stable enough to cluster models into
families by their pattern lists alone, and they resist prompting-based fixes (the
"pink elephant" problem: instructing a model not to use a phrase makes the phrase salient).

### The shared structure

Both pathologies are **localized distributional defects**. At an identifiable token position,
a small set of tokens receives excess probability mass — and coherent alternatives sit
immediately below them in the same distribution. That locality is what the method exploits:
instead of retraining behavior over full sequences (SFT on gold answers, sequence-level DPO),
the correction can be surgical — *at this position, do not pick that token; pick any of these
plausible alternatives instead* — leaving the rest of the model's behavior untouched.

## Pipeline overview

Three stages, shared by both applications:

1. **Elicit and detect.** Generate completions under conditions that surface the pathology
   (pathology-prone prompt mix; low temperature for loops). Detect occurrences — a
   periodic-repetition scanner for loops, or corpus-level frequency analysis against human
   baselines for lexical patterns.
2. **Localize and build single-position preference rows.** Identify the exact token that
   initiates each pathological continuation (the **rejected** token), collect filtered
   alternatives from the model's own top-k at that position (the **chosen** tokens), and
   regularize the resulting dataset so no single token dominates either side.
3. **Train with FTPO.** A single-position, multi-chosen preference objective in logit space
   with a two-tier tether to the reference model, trained as a LoRA adapter and merged.

There is also an **inference-time-only variant** — a suppression sampler that enforces ban
lists during decoding by backtracking — which serves both as a deployable mitigation with no
weight changes and as the preference-data generator for the lexical application (see
[the inference-time variant](#the-inference-time-variant-suppression-sampling-with-backtracking)).

## Stage 1: detection

### Periodic repetition scanner (repetition pipeline)

The loop detector is a cheap fingerprint-and-verify scan over the decoded completion text
(prompt excluded), all at character level:

- Take a 16-character fingerprint every 128 characters (`sample_len=16`,
  `sample_interval=128`).
- Find the fingerprint's nearest other occurrence (forward, then backward). The distance
  between occurrences is a candidate period, accepted if between `min_period=1` and
  `max_period=1024` characters.
- Verify by counting how many times the candidate period-length pattern tiles the text
  contiguously, extending in both directions from the match.
- Flag a loop if the pattern repeats **≥ 4 times** (`min_repeats=4`) with **≥ 60 characters
  of total repeated text** (`min_total_repeated=60`). A period-1 loop therefore needs ≥ 60
  repeats; a period-20 loop needs only the minimum 4.

Only contiguous exact repeats count (one intervening character breaks the chain), and the
sparse fingerprint sampling is a deliberate speed/recall tradeoff. Each completion yields at
most one detection (the first verified hit).

**Elicitation settings** (shipped repetition-pipeline configuration): temperature 0.01–0.1
(the published quick-start uses 0.01), `max_new_tokens: 4000` (loops need room to manifest),
top-20 logprobs recorded per position (these become the chosen-token candidates),
`min_p: 0.01`, one completion per prompt per temperature, generation stopping once 20k
preference rows are collected. Each prompt gets a fixed template appended asking the model to
think step by step and end with `Answer: <...>`; custom prompt mixes are supplied as a local
JSONL or Hugging Face dataset with a designated prompt field (plain string, OpenAI-style, or
ShareGPT-style messages).

### Corpus-level frequency analysis (lexical pipeline)

For lexical patterns, detection is statistical rather than per-completion:

- **Generate a baseline corpus** (recommended 1,000–2,000 prompts, creative-writing prompts in
  the published work) with no interventions.
- **Compare against human baselines.** The operative statistic is the over-representation
  ratio ρ(p) = f_model(p) / f_human(p) for pattern p. Two baselines are used: the `wordfreq`
  package's general English frequencies for single words, and an n-gram profile built from a
  large human-written corpus (Reddit writing-prompt responses; the paper adds Project
  Gutenberg). Both sides are normalized per unit of text (occurrences per 100k characters).
  Two refinements matter for reimplementation: word ranking uses a frequency-modulated ratio
  (ratio boosted by corpus-frequency^0.75 and attenuated by baseline-frequency^0.75, so rare
  one-off ratios don't dominate), and patterns entirely absent from the human baseline
  (infinite ratio — invented names, nonce words) are ranked by their model-side frequency
  under a separate quota.
- **Rank and select into ban lists** under per-iteration quotas, split by n-gram order and by
  presence/absence in the human baseline. Over-represented *words* and exact multi-word
  *phrases* go to a string ban list; content-word *n-grams* to an n-gram ban list; structural
  templates are expressed as user-supplied *regex* patterns. N-gram matching strips stopwords
  and punctuation first, so a banned bigram like "deep breath" also fires on "took a deep
  breath" — but an intervening content word ("deep, calming breath") breaks the match. A
  whitelist protects tokenizer special tokens, chat-template scaffold text, and
  user-specified terms.
- **Iterate.** The loop runs a fixed number of iterations; the published default is two
  (baseline + one suppressed pass), which catches most of the mass. Each suppressed pass runs
  under the accumulated ban lists (cumulative set-unions minus the whitelist), analysis
  re-runs on its output, and newly surfaced patterns are appended — suppression exposes the
  *next tier* of over-represented patterns previously shadowed by the first tier. Note that
  training consumes the final iteration's harvested rows only; earlier suppressed passes'
  rows are not concatenated.

## Stage 2: preference-row construction

Each training row is built at one token position and contains:

- **context**: the full prompt (chat template included) plus the generated text up to but
  *not including* the rejected token — identical context for rejected and chosen;
- **one rejected token**: the single token that initiates the pathological continuation;
- **multiple chosen tokens** (up to 20): plausible alternatives at that same position, drawn
  from the model's own distribution;
- metadata (source prompt, detected span or violated rule, positions).

Both the rejected and every chosen entry must round-trip through the training tokenizer as
**exactly one token** (rows are re-tokenized from decoded surfaces, not generation-time ids).
The repetition pipeline prunes multi-token chosen surfaces individually; the lexical loader
drops the entire row if any chosen (or the rejected) surface is multi-token. Overlength
contexts are discarded, never truncated — the shipped training budget is
`max_seq_length: 6000` tokens (prompts capped at 2,000; completions at 4,000), so every
training row is a ≤ 6k-token context.

### Locating the rejected token (repetition pipeline)

The character-level detection is refined in token space so the correction lands exactly where
the failure begins — the first token of the span's *first repetition* (its second occurrence):

- map the detected span boundaries to token indices and convert the character period to a
  token-count period, then slide the boundary left while the token one token-period earlier
  decodes to the same surface (finding the true earliest token-aligned loop start);
- skip punctuation/whitespace-only boundary tokens (up to 4 tokens / 12 chars) so the rejected
  token is a *readable* token;
- a sentence-restart heuristic: when the period boundary falls mid-sentence but the repeated
  material starts a new sentence with a connective from a fixed restart-word list (`wait`,
  `so`, `but`, `alternatively`, `however`, …), move the target past the sentence-final
  punctuation onto that restart word — the token that actually launches each round of the
  loop.

In the lexical pipeline the rejected token needs no refinement: it is simply the first token
of the banned sequence at the position where the sampler detected it.

### Choosing the chosen tokens

The alternatives come from the top-k log-probabilities the inference engine already recorded
at the rejection position (k = 20 in both pipelines). Selection, in order:

- renormalize the candidate probabilities at the generation temperature of the pass
  (p ∝ exp(logprob)^(1/T), where T is the temperature the completion was sampled at);
- exclude the rejected token itself and its case variants (case-insensitive match on the
  decoded surface; leading-whitespace variants are *not* normalized away and can survive);
- apply a **min-p floor** (keep candidates with p ≥ min_p · p_max, default min_p = 0.01) so
  only coherent continuations qualify — this is the coherence filter that makes the model's
  own alternatives safe to promote;
- apply surface filters (minimum decoded length, alphanumeric requirements, substring skips —
  the exact set differs by pipeline); the lexical pipeline additionally rejects candidates
  that would *begin another banned sequence*, via a precomputed banned-prefix set, so
  suppression is not just pushed one token later;
- keep up to 20, iterating from the low-probability end of the survivors (the lexical sampler
  explicitly harvests this tail; in the repetition pipeline the cap rarely binds — top-20
  capture minus the rejected token leaves at most 20 — so effectively all survivors are kept);
- if nothing survives, the event is discarded and no row is written.

Because the renormalization uses the elicitation temperature, very low temperatures sharpen
the candidate distribution and push more alternatives below the min-p floor — sweeping the
elicitation temperature down raises the detection rate but cuts row yield and per-row chosen
counts. Monitor rows-per-detection and chosen-per-row when picking it.

### Dataset regularization

The raw event stream is badly unbalanced: a small set of initiator tokens accounts for most
rejected slots, and a small set of safe continuations accounts for most chosen slots. Training
on the raw distribution over-suppresses the frequent rejected tokens (degrading reasoning,
since those tokens are also legitimate connectives) and over-promotes the frequent chosen
tokens (which can then become loop initiators themselves). Two flattening passes address
this:

- **Rejected-token flattening** (`rejected_regularisation_strength`; 0.3 in the repetition
  pipeline, 0.8 in the lexical one): token counts above the median are pulled toward the
  median by a power transform — target count ∝ count · (median/count)^strength — and rows are
  greedily sampled to match the flattened distribution. The repetition pipeline adds a
  source-dataset share tiebreak so no prompt source dominates; the lexical loader simply
  fills whichever rejected token is furthest below its target.
- **Chosen-token flattening** (`chosen_regularisation_strength`, 0.5 in the repetition
  pipeline): chosen-token counts above a reference (95th-percentile count, floor 50) are
  pulled toward it; this pass first de-duplicates each row's chosen list by normalized
  surface form, then prunes individual chosen entries out of rows, and only drops a row when
  its last chosen entry is removed. **This pass is functional only in the repetition
  pipeline**: the released lexical implementation computes and logs the chosen-side quotas
  but never applies them, so its config knob is inert — verify against current code before
  relying on it.

**Volume guidance** (repetition pipeline): the README advises starting from ≥ 15k prompts and
aiming for 15–20k preference rows, with `max_train_examples` at roughly 60–70% of the
generated pool (e.g. 12k rows from a 17–20k pool) so the flattening passes have slack to
cull. Plan prompt volume from the yield arithmetic: at most one row per completion, one
completion per prompt per temperature, so expected rows ≈ loop rate × prompts × temperature
passes — at a 10–20% loop rate, 15–20k rows implies on the order of 10⁵ prompts or a
multi-temperature sweep (the published prompt mix has 478k rows). Related knobs:
`filter_rejected_stop_words` stays **false** for repetition work — common words genuinely
initiate loops, and flattening is the safe way to handle their frequency (the lexical
pipeline, by contrast, hard-filters stopword rejected tokens, since a stopword is noise
there, and requires several chosen tokens per row — default 4 — where the repetition
pipeline accepts 1).

## Stage 3: the FTPO objective

FTPO is a preference-optimization loss that constrains updates to the final-token position of
a fixed context. Each row costs one policy forward pass plus one no-grad reference pass (the
same weights with the LoRA adapter disabled) — no per-token rollout; contexts are left-padded
so the last position is the prediction site, and only the logits at that position enter the
loss. Let `z` be the policy's final-position logits, `z_ref` the reference model's, `r` the
rejected token, and `C = {c_1 … c_n}` the chosen tokens.

**Preference term** — for every chosen token, a hinged margin loss on the *logit gap*
`Δ_c = z_c − z_r` with margin `ε` (`clip_epsilon_logits`, default 2.0):

```
per-token loss  = softplus(ε − Δ_c) · w_c,   w_c = clamp((ε − Δ_c)/ε, 0, 1)
preference loss = mean over rows of ( Σ_c per-token loss / |C| )
```

A chosen token already beating the rejected token by the full margin contributes nothing
(`w_c = 0`) — weights stop moving once the preference is won, a self-limiting property that is
load-bearing for stability (see the ablations below). The taper is differentiated through
(`w_c` is not detached). Averaging over multiple chosen tokens spreads the recovered
probability mass across *many* plausible continuations instead of collapsing it onto one
designated alternative. One normalization caveat: both released implementations divide by
`|C|` as shown, while the paper's formulation divides by `Σ_c w_c` (a weighted mean that keeps
full pressure on still-losing chosen tokens as others win) — the variants differ in
late-training dynamics, so match whichever codebase you are reproducing.

**Two-tier reference tether** — MSE in logit space, not KL:

```
tether = λ_nontarget · mean over non-target vocab of (z − z_ref)²
       + λ_target    · mean over targets of max(|z − z_ref| − τ, 0)²
total loss = preference + tether
```

where *targets* = `{r} ∪ C` and *non-target* = the rest of the vocabulary. Defaults:
`λ_nontarget = 0.4` (tight — the untouched vocabulary must stay where it was), `λ_target =
0.05` with a penalty-free dead zone `τ = 0.5` logits (loose — the targets *must* move
substantially relative to each other, since the rejected token typically leads by a wide
margin; that lead **is** the pathology). MSE on raw logits is deliberate: a KL/softmax-based
tether couples every logit through the normalizer and spreads compensatory gradient pressure
across the whole vocabulary, defeating the point of a surgical update.

### How this differs from DPO

1. **Single-position, not sequence-level.** Only the trailing token of a mid-generation
   context is trained; DPO backpropagates through entire chosen/rejected continuations.
2. **Many chosen tokens per row.** Probability mass removed from the rejected token is
   distributed across up to 20 alternatives; DPO updates one chosen continuation per sample.
3. **Logit-space objective.** Margin loss and tether operate on raw logits; there is no β·log
   probability ratio and no softmax term creating coupled pressure on unrelated tokens. (DPO's
   β is a coarse instrument here: high β impairs learning, low β permits large divergence.)
4. **Two-tier regularization with margin deactivation.** Target tokens move freely inside a
   dead zone while the remaining vocabulary is pinned; the preference loss switches itself off
   once won.

The published head-to-head (same preference pairs, matched early stopping): **FTPO reaches
~90% suppression of targeted patterns with < 1% loss on a 100-point judged writing rubric and
lexical diversity at 95–102% of baseline; DPO reaches only ~80% suppression while losing 6–15
rubric points and collapsing diversity to 74–92%.** GSM8K and MMLU stay within 1–3% of
baseline under FTPO. FTPO can be trained to nearly 100% preference accuracy with minimal
degradation, where DPO degrades substantially once preference accuracy exceeds ~40%
(preference accuracy is the `chosen_win` metric of the recipe table). The loss ablations
localize why: removing the margin cutoff collapses the model (rubric quality 67.9 → 19.6);
removing the target-token tether drops quality to 39.7 (the paper's prose quotes a 71%
degradation; its own ablation table gives 67.9 → 39.7, ~41%); over-tightening the target
tether (λ_target = 0.4) caps achievable preference accuracy at 74% and cuts suppression from
~85% to ~56%.

### Training recipe

| knob | repetition pipeline | lexical pipeline | notes |
|---|---|---|---|
| adapter | LoRA r=256, α=128, dropout 0 | r=128–256 recommended (shipped recipes span 128–512), α typically = r, dropout 0.05 | high rank consistently helps: better preference accuracy with less degradation |
| target modules | all of q/k/v/o/gate/up/down_proj + `lm_head`, no layer freezing | same full set with only the last 3–10 layers unfrozen by default; large-model recipes restrict to `up_proj, down_proj, lm_head`; `lm_head`-only for the most fragile | repetition work found full-layer training preferable; lexical work leans on freezing/restriction |
| learning rate | 1e-5 – 2e-5 for ~12k rows (blog: 4e-6 – 2e-5) | manual 1e-6, or an auto-LR rule scaling with batch size, LoRA rank, and dataset size | overtraining can *increase* the failure rate — treat LR as a tuned quantity |
| epochs | 1, with early stopping | 1, with early stopping | |
| early stop on `chosen_win` | **0.35–0.4** (strong loop reduction appears by 0.15–0.3; > 0.5 risks overtraining) | **0.8–0.9** (default 0.85; lower for fragile models) | `chosen_win` = mean fraction of a row's chosen tokens whose logprob beats the rejected token's. The two applications sit at very different points — do not transplant the threshold |
| batch | 4 × grad-accum 4, bf16, AdamW (paged 32-bit), linear schedule, warmup 0.1, max grad norm 2.5 | 1–3 × grad-accum 5–16, optional 4-bit base | |
| loss constants | ε=2.0, λ_nontarget=0.4, λ_target=0.05, τ=0.5 | same | stable across both applications |
| after training | merge the LoRA adapter | same | |

The repetition-pipeline column reproduces the values of the tool's *shipped YAML config*. Its
in-code fallback defaults differ materially (early stop 0.85, stop-word filtering on, layer
freezing on, lr 1e-6, rejected flattening 0.8, chosen flattening off) and apply silently when
the YAML is not found — always pass the config file explicitly.

Compute reference point (LFM2.5-2.6B): ~1 hour of data generation on 8× MI325 (stops at 20k
rows) and 1–2 hours of training on a single MI325 — the full cycle is an hours-scale
intervention. Per-row training cost scales with context length (each row is two forward passes
over its full context).

## The inference-time variant: suppression sampling with backtracking

The lexical pipeline's generator enforces ban lists *during* decoding against an unmodified
model served over any OpenAI-compatible `/v1/completions` endpoint that returns top-k
logprobs. Mechanics:

- **Detect at string level, correct at token level.** Validators scan the decoded text after
  each generated chunk: exact banned strings (with word-boundary semantics), stopword-robust
  content-word n-grams, and regex templates. A ban only triggers once the *entire* sequence
  has appeared — unlike naive token banning, which fires on the first token of any banned
  string and collaterally bans every word sharing that prefix.
- **Backtrack and resample from cache.** On a violation, generation is rewound to the token
  where the banned sequence began. A replacement is resampled *locally* from the top-k
  logprob list the API already returned for that position — no extra model call — after
  down-weighting the banned token by the ban-strength rule `p_new = p_old · 10^(−10s)`
  (s ∈ [0,1]; s = 1 is a hard ban; the soft-ban rule applies to phrase and n-gram bans only —
  regex violations are always hard-banned), then re-applying temperature/min-p/top-p/top-k.
  Already-tried tokens are never retried at the same position; candidates are validated
  against the ban lists before acceptance.
- **Give up gracefully.** If no coherent alternative survives, that specific violation
  instance is suppressed (ignored at that position only) so generation can proceed — the ban
  lists stay in force elsewhere. An optional forced-backtrack mode instead relaxes the
  sampling constraints progressively (temperature, then min-p, then top-p, then top-k) until
  a non-banned candidate exists. The give-up rule also handles the case where a user
  *explicitly asks* for a banned phrase: the request concentrates probability on it, no
  alternative survives the coherence filter, and the phrase is permitted — at moderate ban
  strength (s ≈ 0.4), patterns are suppressed in ~90% of ordinary generation yet fully
  permitted on explicit request.
- **Cost.** Backtracking plus full-text re-validation is expensive: measured throughput drops
  of 69% (1k-entry ban list) to 96% (8k entries) versus unconstrained vLLM serving. This cost
  is the argument for distilling the correction into the weights with FTPO.

Two roles for the sampler:

- **Deployable mitigation without training** — ship the ban lists with the serving config.
  With forced backtracking, suppression of listed patterns is complete (100%) while judged
  output quality *improves* over baseline; with the shipped default (forced backtracking
  off), banned patterns survive wherever no coherent alternative passes the filters. Scales
  to 8,000+ banned patterns, where logit-bias token banning becomes unusable by ~2,000
  entries and collapses to 28/100 rubric quality at 8,000.
- **Preference-data generator** — every backtrack event is exactly an FTPO row in the making:
  identical context, a rejected token (first token of the banned sequence), and the surviving
  validated alternatives as chosen tokens. The lexical pipeline harvests its training rows
  this way during the suppressed iterations. The repetition pipeline does *not* backtrack —
  it detects loops post-hoc in unconstrained completions and reads alternatives from the
  logprobs recorded at generation time; same row format, different harvest. The lexical
  pipeline additionally discards rows harvested from refusals (a generation that is a refusal
  would otherwise teach the model to prefer refusal-flavored continuations at the correction
  point), screening them with a small trained classifier; the repetition pipeline has no
  refusal handling.

## What to expect (published results)

- **Repetition rates** (Liquid AI, hard math/coding prompt mix): LFM2.5-2.6B early checkpoint
  **10.2% → 1.4%**; Qwen3.5-4B **22.9% → 1%** under greedy sampling — with evaluation scores
  *improving* across the board rather than trading off, and the largest gains at low
  temperatures. Blog rule of thumb: stopping at chosen_win ≈ 0.35 typically cuts loop rates
  from 20–30% to 1–2% with minimal degradation.
- **A measurement confound:** the common recommendation that reasoning models need higher
  sampling temperature appears to be confounded with the dominant effect of repetition. Once
  loops were trained out, the best performance appeared at near-greedy temperatures (and the
  trained model's advantage shrinks toward T=1.0, where loops were already rare). If a
  temperature sweep on a small reasoning model shows monotone gains with temperature, measure
  the loop rate before believing the trend.
- **Lexical suppression** (ICLR paper, gemma-3-12b primary): FTPO 83–92% suppression across
  2k/4k/8k-pattern ban lists at ≤ 1% writing-quality change; details and DPO comparison in
  [Stage 3](#how-this-differs-from-dpo). Fine-tuned models move measurably toward the human
  end of the pattern-frequency spectrum (the FTPO fine-tune clustered closer to human authors
  than any tested model on the paper's profile-distance analysis).
- **Multiple rounds compose.** One round can expose new failure points — other tokens begin
  initiating loops, or a second tier of over-used patterns surfaces. Re-run detection after
  training and budget for iteration.
- **Measuring the fix yourself:** neither codebase ships an evaluation harness. The published
  loop-rate method is simply a fresh generation pass on held-out prompts against the merged
  model, counting completions the detector flags (the generation stage's status counts give
  exactly this). Two caveats: detection events whose chosen candidates all fail filtering are
  logged as if no loop occurred, so counts-based loop rates are a lower bound; and
  `chosen_win` is only visible in the training logs (emitted every few steps), with early
  stopping firing on the first log crossing the threshold.

## Pitfalls checklist

- **Overtraining is the dominant failure.** Watch `chosen_win`; stop early. Overtrained runs
  degrade the model *and can raise the repetition rate above baseline*. The margin cutoff is
  what makes extended training survivable — do not disable it (`clip_epsilon_logits` → large
  is equivalent to disabling and collapses the model).
- **Never train on the unregularized event stream.** Frequency flattening is what separates a
  targeted correction from broad suppression of legitimate connective tokens. Remember that
  the released lexical implementation applies it on the rejected side only.
- **Don't stop-word-filter rejected tokens in repetition work.** `the` really does initiate
  loops; flattening, not exclusion, is the safe control. (Lexical work does the opposite —
  know which regime you are in.)
- **Use the shipped config, not the in-code defaults.** The repetition tool's fallback
  defaults are lexical-flavored (early stop 0.85, stop-word filter on, layer freezing on) and
  take effect silently when no config file is passed — exactly the transplanted-threshold
  mistake this checklist warns against.
- **Single-token discipline.** Rejected/chosen surfaces must be exactly one token under the
  *training* tokenizer; multi-token surfaces silently vanish (individually in the repetition
  pipeline; whole-row in the lexical loader) — count your filter drops.
- **Expect pathology migration.** Suppressing tier-1 patterns surfaces tier-2 patterns;
  promoting alternatives can mint new attractors. The chosen-side flattening and the
  multi-chosen objective are the guards; iteration is the backstop.
- **Evaluate at deployment temperature.** The failure and the fix are both
  temperature-dependent; a benchmark run at T=0.7 can hide a loop problem that dominates at
  T=0.01.
- **Hold out prompts for measurement.** Loop-rate and over-representation metrics computed on
  the training prompt mix overstate the fix; the published evaluations exclude training
  prompts and confirm on out-of-distribution prompt sets.
- **Calibrate thresholds to the application.** Early-stop chosen_win 0.35–0.4 for repetition
  vs 0.8–0.9 for lexical; rejected-side flattening 0.3 vs 0.8; layer freezing off vs on. The
  two published recipes differ wherever the pathology differs — port the machinery, not the
  numbers.
- **Do not generalize pathology suppression into capability steering without a locality
  gate.** C52 round 1 showed that pairwise FTPO on near-parity outcome pivots disrupts
  think flow; round 2 restored confident-outlier geometry and still found 0.229-logit
  median non-target drift under demotion. Positive-only +0.5 uplift reduced this to 0.145
  and retained some real-label signal, but held-out capability stayed below base. A
  single-position loss is not a single-context model edit. Audit exact logits on every row
  and stop before downstream evaluation when neighboring drift exceeds the frozen bar.
- **Entropy and varentropy route; they do not label correctness or prescribe pressure.** In
  C52 round 2, low entropy plus nonzero varentropy found focused confident wrong turns, but
  uplift safety was non-monotone across varentropy quartiles: the lowest quartile was
  cleanest and Q3 worst. Do not assume higher varentropy means a more fruitful token to pull
  up or down; any new band needs independent confirmation.

## Relevance to this corpus

- **The published Qwen3.5-4B result is directly about our model**: 22.9% of completions
  looping on hard math/coding under greedy sampling, trained down to 1% with score
  improvements.
- **The corpus has already hit this wall.** The
  [verified-macro long-context rerun](../experiments/qwen35_4b_verified_macro_long_context_rerun/reports/report.md)
  found exact periodic loops dominating unfinished reasoning tails at think@32k+ (81/144),
  and the [model playbook](model_playbook.md) mandates separate gating of exact periodic
  loops, with a loop-control protocol branch when a preregistered ladder exhausts its
  terminal rung with exact loops dominant. FTPO is the obvious candidate lever for that
  branch. One open transfer question to design for: the published fix is trained entirely on
  ≤ 6k-token contexts (the shipped `max_seq_length`), while our loops manifest at 32k+ —
  whether short-context FTPO training transfers to the long-context loop regime is exactly
  what the experiment should measure (raising `max_seq_length` is possible but training cost
  scales with context length).
- **Compatible with the capability-source constraint, with one adaptation.** The chosen
  tokens are the model's *own* alternatives at the rejection position; the training signal is
  the model's own distribution plus a rule-based detector — no larger model, no distillation,
  no gold labels. The one component to avoid importing: the lexical pipeline's optional
  refusal screen is a separate trained classifier, which the one-model rule forbids; the
  repetition pipeline has no such component, and a rule-based screen (or none — refusals are
  rare on reasoning prompt mixes) keeps any adaptation compliant. (The published prompt mix is a 478k-row prompts-only mixture of
  math/code/QA/instruction sources with answers and rationales deliberately excluded; an
  analogous mix can be assembled from our own task substrates.)
- **The deployed-budget pivot adaptation has now failed twice (C52).** Near-parity
  outcome-conditioned forks produced shuffled-control-equivalent think-flow harm. Filtering
  to failed argmax tokens with focused entropy/non-degenerate varentropy and switching from
  demotion to positive-only uplift preserved a local true-label advantage, but did not beat
  base on fresh whitebox or a hidden-tested repository agent. This narrows FTPO's current
  corpus warrant to confident pathology suppression; capability elicitation needs a more
  context-local parameterization before a larger harvest.
- **Cheap enough to run here.** Detection is a text scan over completions we already log;
  data generation is one low-temperature pass over a prompt mix; training is a 1–2 GPU-hour
  LoRA. The corpus-wide scanner step is complete: loops are ~0.1% at deployed budgets but
  dominate many 32k+ contacts. The remaining natural long-context experiment is to mine the
  initiator-token statistics, run a single loop-specific FTPO round, and re-run the
  long-context ladder that was previously loop-censored — with a
  matched-compute control arm per the [quality gates](quality_gates.md). Note the published
  generation stage seeds its sampler non-deterministically, so archive the generated pairs
  and generations JSONL files as the reproducibility anchor, not the generation config.

## References

- Paech, S., Roush, A., Goldfeder, J., Shwartz-Ziv, R. — the framework paper introducing the
  suppression sampler, the profiling pipeline, and FTPO. ICLR 2026.
  https://arxiv.org/abs/2510.15061
- Liquid AI (Paech et al.) — blog post introducing the repetition application of FTPO
  (July 2026). https://www.liquid.ai/blog/antidoom — repetition pipeline:
  https://github.com/Liquid4All/antidoom — prompt mix:
  https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0
- Lexical pipeline: https://github.com/sam-paech/auto-antislop — suppression sampler:
  https://github.com/sam-paech/antislop-vllm (original prototype:
  https://github.com/sam-paech/antislop-sampler) — frequency-forensics toolkit:
  https://github.com/sam-paech/slop-forensics
- Nguyen et al. — *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM
  Outputs* (arXiv:2407.01082) — the coherence filter used to qualify chosen-token
  alternatives.
- Context on the repetition phenomenon: Holtzman et al., *The Curious Case of Neural Text
  Degeneration* (2020); Welleck et al., *Neural Text Generation with Unlikelihood Training*
  (2020) — an earlier sequence-level training approach to repetition; Pipis et al., *"Wait,
  Wait, Wait... Why Do Reasoning Models Loop?"* (2025); Duan et al., *Circular Reasoning:
  Understanding Self-Reinforcing Loops in Large Reasoning Models* (2026).
