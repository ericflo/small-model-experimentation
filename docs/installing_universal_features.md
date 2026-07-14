# Installing Universal Features: A Research Doctrine

*A concept + handoff document. Written 2026-07-13, mid-session, so a fresh agent
can continue the program without losing the thread. Read this, then
`knowledge/claims/claim_ledger.json` (C49–C59), then the "How to run it" section.*

---

## 0. The one-sentence thesis

We are not trying to *beat a benchmark*. We are trying to **install genuinely new,
universal capability into the fixed Qwen3.5‑4B** — and the most promising lever is
**engineered training data**: programmatically designed, synthetically generated
curricula that build *generic, transferable circuits*, iterated on *fast*. Not
rejection sampling. Not benchmark-shaped data. Not slow.

Everything below expands why each of those words is load‑bearing.

---

## 1. The reframe: there is no proven "capacity boundary"

Earlier this session the gauntlet program concluded that "+0.32 on both menagerie
tiers is a proven capacity boundary of the 4B" (claims C54–C56). **That was an
overclaim, and it has been retracted.** The honest statement is: *a dozen recipes
(breadth, compression, oracle‑injection, episode‑mastery, exploration) did not
clear +0.32.* That is emphatically **not** "no training‑data sequence can."

The mistake was epistemic — treating absence of evidence (across a tiny, slowly‑
sampled set of recipes) as evidence of a hard limit. The real limiter was never
the model. **It was iteration speed:** each recipe cost ~1–2 h (harvest → train
~50 min → merge 9 GB → eval), so we sampled ~a dozen darts and called the dartboard
empty.

Corollary that should govern all future work: **do not declare capability walls
from slow, sparse curriculum search.** A wall claim requires either a fast, dense
search of the data‑design space that comes up empty, or a mechanistic argument.
Neither existed for +0.32.

## 2. Why NOT harvest / rejection‑sampling

The install attempts that stalled all shared a flaw: the training signal was
*bounded by the current model or by answer‑narration*.

- **Self‑harvest / expert‑iteration** (sample the model, keep verified successes,
  SFT, repeat) can only reinforce **what the model already produces**. This is
  literally C11's coverage‑boundedness, one level up: you cannot reject‑sample a
  capability the model does not yet have. On the hardest axes (composed induction),
  the model succeeds ~0% at harvest time, so there is *nothing to harvest*.
- **Oracle‑answer traces** (narrate the gym solver's reasoning for a specific item)
  train to loss ≈ 0 / score 1.0 yet **deploy at ~0** on composed induction (C56).
  They demonstrate *an answer for one instance*; they do not build a reusable
  circuit. They also skip the hard part — they *assume* the rule and verify it,
  rather than teaching the model how to **search for** it.

Both are downstream of the same limitation: **the data is a shadow of the model or
of a single answer, not an engineered lesson.**

## 3. The doctrine: design and synthesize the circuits

We have total control over the gym — we *wrote* the families, the operators, the
verifiers. So we can synthesize **arbitrary correct, truth‑blind training text**
(every value computed by running the operator, never copied from a stored answer).
Use that power to *design the circuits we want*:

1. **Teach primitives in isolation.** One lesson per operator (shift / reverse /
   swap / overwrite / conditional‑rotate), worked end‑to‑end → a clean, addressable
   feature per primitive.
2. **Teach the meta‑operation explicitly.** For single‑rule induction: *try each
   operator, check it against every example, keep the one that fits.* This builds
   the **verify** feature.
3. **Teach composition as a SEARCH, not an assertion.** The circuit C56 was missing:
   *output = step2(step1(input)); to find both, FIX a candidate first step, apply it
   to every probe INPUT to get the intermediates, then find the single second step
   that maps all intermediates to the outputs.* Show a real **dead end** and the
   **found** decomposition, with the intermediates written out. This is the
   **decomposition** circuit — the thing that lets the model *propose* a composition
   it has never seen (C39/C56 say it currently cannot).
4. **Sequence it as a curriculum.** Primitives → single‑rule → 2‑composition →
   deeper, interleaved.

The unit of iteration is no longer "sampling weights over a harvest pool." It is
**curriculum design**.

## 4. The keystone: universal features, not benchmark shapes

*This is the real thing we are chasing.*

The training data **should look nothing like the eval.** If we teach the
decomposition circuit on glyph‑string probe logs and then test on glyph‑string
probe logs, a "win" is indistinguishable from learning the surface format — we've
taught the benchmark, not a capability. Worse, format‑matched data actively
*discourages* a generic feature because the surface shortcut is always available.

Instead: teach the **same abstract circuit on many surfaces that are deliberately
disjoint from any benchmark** — digits, letters, roman numerals, invented word‑
tokens, greek syllables; sequence lengths and alphabet sizes varied. Then
**evaluate by TRANSFER to the held‑out benchmark itself — the menagerie, which we
never train on or read** (the firewall already guarantees this). If the circuits
lift the menagerie having trained only on abstract surfaces the menagerie does not
use, the feature is **universal** — it binds to *structure*, not tokens. That
transfer *is* the proof, and it is the whole point: a universal feature lifts
*every* boat (every menagerie axis that shares the circuit), a memorized shape
lifts one. (Note: an early version of this pointed the transfer test at a single
narrow gym family, `glyphgate`. That was abandoned — a narrow, slow‑to‑eval target
that fought the speed doctrine. The right held‑out target is the broad menagerie.)

Design principles that fall out of this:
- **Surface‑agnostic rendering.** The circuit operates on abstract indices; surfaces
  are just re‑renderings. (See `qwen35_4b_universal_curriculum/scripts/gen_curriculum.py`:
  the same op logic rendered over 5 alphabets.)
- **Maximize variation the feature must be invariant to** (alphabet, length, order,
  vocabulary) and minimize any spurious cue the eval shares.
- **Transfer is the metric.** Same‑surface performance is only an *upper‑bound
  control* ("is the circuit installable at all?"), never the claim.
- **Generality over specificity, always.** Prefer a lesson that teaches *why the
  decomposition works* over one that teaches *this rule's answer*.

## 5. Iteration speed IS the research budget

To search the curriculum‑design space we must make one cycle cheap. Current status
(built this session, verified):

- **Harvest once, remix fast.** `build_pool.py` merges all cached SFT components
  into one deduplicated, tagged pool (15,706 rows; by kind × family × level). A
  harvest‑based "recipe" (`build_recipe.py`) is then a *weighted sample* — seconds.
  (This is the *fallback* lever; the doctrine above prefers synthesis.)
- **Synthesize fast.** `qwen35_4b_universal_curriculum/scripts/gen_curriculum.py`
  (the real generator: multi‑skill, surface‑agnostic, truth‑blind) writes a designed
  curriculum in seconds, CPU‑only. It builds on the `generic_curriculum.py` prototype
  in `gauntlet_frontier` (same abstract‑index op logic over 5 alphabets).
- **Do not trust runtime LoRA.** vLLM runtime‑LoRA is a silent no‑op on this model
  (C49). Use direct PEFT only for a fresh, non-benchmark synthetic installability
  screen. A candidate that reaches the benchmark must be explicitly merged into the
  full composite model, with nonzero LoRA application and weight hashes authenticated.
- **Keep the benchmark comparison paired.** Run base, the strongest frozen control,
  and candidate on the same fresh seed through `scripts/run_benchmark_aggregate.py`.
  Every arm uses the canonical `qwen_vllm` backend and tier budget; a once-measured HF
  base is not a valid comparator. Raw suite output remains inside the gateway's private
  temporary directory. Confirm promoted quick results on independent seeds and medium.

Net: synthesize and truth-audit every arm in seconds, fast-train and reject locally bad
arms before paying merge/benchmark cost, then reserve fresh aggregate-only benchmark
events for promoted candidates. This retains a dense local search loop without trading
away backend parity or the benchmark firewall.

**Do not reduce the local gate to exact accuracy.** In the first parent factorial,
designed-only warm continuation and from-base designed-plus-replay both scored 0.6923
on their respective fresh 26-task screens. The former parsed 0.9615 with 1 cap contact;
the latter parsed only 0.8462 with 4 cap contacts and failed before benchmark. Track
accuracy, parse rate, cap contacts, and semantic failure modes prospectively; equal
accuracy can hide materially different installed policies.

**Replay is an active capability intervention, not a neutral retention control.**
In the result-separated replay-anchor successor, replacing 400 of 1,520 replay rows
with designed lessons passed the fresh local screen but scored 0.4238 on paired
quick@1,024: 0.0172 below the mature `blend` policy and 0.0613 below replay-only.
Replay-only continuation reached 0.4851, improved eight of ten public families, tied
two, and regressed none. It had 17.3% more forward-token exposure despite matched
optimizer steps, so this event rejects the candidate but does not isolate the content
effect behind the full gap. A designed-data experiment must beat a replay continuation
matched on both steps and token exposure, not merely retain the pre-refresh checkpoint.
The next useful dose search begins from the authenticated replay-refreshed policy and
reduces designed density sharply; it does not reinterpret the replay control as a
universal win.

**Speed is not a nicety here; it is the enabling condition for the whole thesis.**
A universal feature is found by *many* fast, varied curriculum experiments — see
which designs transfer, double down, ablate what carries the transfer.

## 6. Where this connects to the corpus laws

- **C11** (coverage‑boundedness) → why harvest can't install what the model lacks.
- **C39** (the 4B is an executor/retriever of pretrained structure, not an inducer)
  → the capability we most want to *install* is the inducer/decomposition circuit;
  the doctrine is the attack on exactly this wall.
- **C56** (oracle‑answer SFT trains 1.0, deploys ~0 on composed induction; also:
  *exploration installs and transfers*) → answer‑narration ≠ circuit; but note
  exploration (a genuine executable procedure) *did* install and transfer, which is
  encouraging evidence that the *right* generic procedure can be installed.
- **C59** (serial compute crosses the induction wall only via reasoning *content* —
  not latent recurrence, not filler tokens) → the training text's **content** is
  what matters; this is why *designed* content, not more compute, is the lever. And
  the two‑wall‑regime bound (content crosses shift but not affine) tells us affine‑
  hard composition is where the real test lives.
- **Skin‑shuffle** (C54, fresh pseudo‑vocab per row) → an early gesture at
  surface‑invariance; the generic curriculum generalizes it to *many* surfaces plus
  a held‑out transfer surface.

## 7. How to run it (concrete, for the next agent)

The program lives in its own experiment: **`experiments/qwen35_4b_universal_curriculum/`**.
It reuses the proven train/eval infra prototyped in `qwen35_4b_gauntlet_frontier`
(don't re-derive it).

Environment: repo `.venv` (HF/torch) for train + the HF‑adapter eval; `.venv-vllm`
only if you need vLLM (its `PATH` must include `.venv-vllm/bin` for ninja, and any
vLLM script needs an `if __name__ == "__main__"` guard — spawn). One RTX 4090;
single‑tenant the GPU. Qwen3.5‑4B only (hard rule), pinned revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Scripts:
- `qwen35_4b_universal_curriculum/scripts/gen_curriculum.py` — **the data generator**:
  a generic, MULTI‑SKILL, surface‑agnostic curriculum spanning induction, execution,
  selection, tracing, verification, counting, repair, optimization, abstention, state
  carry, ordering, probe choice, and routing. It renders over six abstract surfaces
  disjoint from the held-out benchmark, and its executable specifications are
  truth/minimum-depth audited. Output: `data/sft_universal.jsonl`.
- `qwen35_4b_gauntlet_frontier/scripts/train_think.py` — QLoRA think‑channel SFT
  (custom per‑token loss; `--rank`, `--epochs`, `--w-think`, `--warm-start`). Fast
  search tiers must freeze row count, token dose, skipped-row count, and package receipt;
  promoted winners are retrained at a registered full dose.
- `scripts/run_benchmark_aggregate.py --tier quick --model <merged-dir>` — **the
  transfer metric**. It exposes only aggregate and public per-family scores from the
  canonical vLLM event. Lifting aggregate without a negative family is the pilot gate;
  strict positive replicated family deltas are required for universality.
- Harvest‑remix fallback (if you want to search *selections* of real data instead
  of synthesizing): `qwen35_4b_gauntlet_frontier/scripts/{build_pool,build_recipe,
  fast_search}.py`.

The experiment loop:
1. `gen_curriculum.py` → `data/sft_universal.jsonl` (seconds).
2. Fast‑train (from base, replay union, or a preregistered warm start) → an adapter in
   `large_artifacts/`; reject it unless zero rows were skipped and fresh synthetic
   installability/retention gates pass.
3. Explicitly merge each promoted arm, then run paired base/control/candidate quick
   events through the aggregate gateway on a fresh seed. **Pilot success = positive
   aggregate with no negative family**, after training on zero benchmark-shaped data.
   Confirm strict wins on independent quick seeds and medium before codifying a claim.
4. Iterate the *curriculum design* fast: which skills / surfaces / lesson framings
   carry the transfer, and what "lifts all boats"? Ablate; add skills; loop.
   Every result-bearing adaptive change moves to a successor experiment so the search
   history and benchmark exposure remain auditable.

## 8. The north star, and the open frontier

The mission (see memory `unearth-latent-capability-mission`) has two halves: *elicit*
latent capability and *install* new capability, in the one 4B, with only provenance
(no larger model) as a constraint. The install half is wide open, and this doctrine
is the current best bet at it: **engineer the training distribution so that the
model builds universal, composable circuits it did not have — and prove it by
transfer to surfaces it never saw.**

Open questions worth a fresh mind:
- What is the *minimal* curriculum that installs a transferable decomposition
  circuit? (primitives alone? the search‑pedagogy is essential?)
- Does teaching the circuit on N surfaces make it *more* universal monotonically, or
  is there a "critical diversity" after which transfer snaps on?
- Can we install the circuit as a feature that composes with *pretrained* structure
  (so it helps natural‑language reasoning too, per C37/C39), not just the gym?
- Curriculum ORDER: does easy→hard, primitive→composite matter, or is the mix
  order‑invariant at this scale?
- The affine‑hard regime (C59): does any curriculum let the base *propose* affine
  compositions in a forward pass, or is that genuinely serial‑compute‑bound?

The framing that should survive into the next session: **we are searching the
space of training distributions for the ones that install universal features; speed
is what makes the search real; transfer to held‑out surfaces is what makes a feature
universal; and "capacity boundary" is a claim we have not earned.**
