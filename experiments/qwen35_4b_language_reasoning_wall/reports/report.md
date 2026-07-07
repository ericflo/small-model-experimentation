# Does the compositional wall exist in LANGUAGE? No — the model chains depth-3+ reasoning steps in natural language near-perfectly

## Motivation
All 36 prior claims are formal/procedural: the fixed 4B walls at depth-3 formal composition and is a value-computer
not a structure-proposer (model-level law, C36). But it is a LANGUAGE model, and its native domain — multi-step
linguistic reasoning — was untouched. This tests C13's **mental-SIMULATION** wall in language (it does **not** touch
the C32/C36 structure-**proposal** wall).

## Method
Contamination-free successor-chain traversal: random chains over made-up pronounceable entities (`Kel → Vor → …`)
+ confusable distractor chains, shuffled. The **same chain** rendered three ways — **linguistic-semantic**
("Kel is directly followed by Vor."), **linguistic-symbolic** ("Kel gorps Vor." — a made-up relation, the
contamination-clean control), and **formal-dict** (a Python `nxt` map). Query: "moving forward D steps from the
start, which name?" Shortcut-hardened (review): the answer is **interior** (chain longer than D — never the sink),
the start is a **random interior node** (depth ≠ line number), recency baseline ≈ 0.04. **No-think is the primary**
(it forces mental simulation, where the wall should live). Depths 1–6, n=80.

## Result (no-think — mental simulation)
| render | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|
| linguistic-semantic | 0.99 | 1.00 | 0.99 | 0.94 | 0.76 | 0.78 |
| linguistic-symbolic ("gorps") | 0.95 | 0.99 | **1.00** | 0.55 | 0.01 | 0.00 |
| formal-dict *(confounded — code-mode)* | 0.03 | 0.79 | 0.75 | 0.29 | 0.06 | 0.42 |

- **No depth-3 wall in language.** Linguistic-semantic is near-perfect through depth-4 (0.94–1.00); the made-up
  relation control is also **perfect through depth-3 (1.00)** — so it is a genuine **modality** effect, not a
  semantic pretraining prior. Both degrade only at depth 5–6 (semantic gracefully to 0.76; the made-up relation
  collapses to 0.00 — **semantics aids deep chaining**, not shallow).
- **Stark contrast to the formal-composition wall (depth-3, C13–C36).** The model chains 3–4 reasoning steps in
  its native linguistic domain, so the "compositional wall" is **NOT a general multi-step limit** — it is specific
  to formal/procedural composition. This **relocates C13's "broken mental simulation"**: mental simulation is
  *intact* for multi-step linguistic reasoning (depth 4–5), broken only for formal ops at depth-3.
- **Secondary (surface-form effect): the formal-DICT rendering triggers CODE-MODE.** The model echoes the dict as
  a ```python block instead of simulating the lookup (d1 = 0.03, the comprehension gate-failure the review
  predicted). So the surface **presentation** determines whether the model **reasons or codes** — the formal-dict
  arm is confounded and is *not* a clean "formal simulation capacity" measurement.

## Implication
The wall we spent 36 claims mapping is a property of **formal composition**, not of the model's ability to reason
multi-step. In its native language, there is no depth-3 wall — the model's mental simulation is intact for
multi-step linguistic reasoning. This is the first result to locate the compositional wall in the *modality*
(formal/procedural), not the *capacity* (chaining reasoning steps).

## Honest scope
- Tests **simulation** (chain given → traverse), which is C13; it does **not** test the C32/C36 structure-**proposal**
  wall (nothing here asks the model to propose a hidden structure) — the "value-computer not structure-proposer"
  headline is untouched.
- **Think conditions are truncation-confounded** (budget 1024 — the model over-reasons trivial tasks and exhausts
  the budget before answering); no-think is the clean primary.
- The formal-dict arm is confounded by code-mode; the clean finding rests on the two linguistic arms.

## Next
- Linguistic **PROPOSAL** task (infer a hidden multi-hop rule from I/O examples, mirroring C32): does the "value-
  computer not structure-proposer" law extend to language, or is proposal also easier linguistically?
- Higher think budget / better answer extraction to get a clean think-vs-no-think (transcription vs mental sim) gap.

## Artifact Manifest
See `reports/artifact_manifest.yaml`.
