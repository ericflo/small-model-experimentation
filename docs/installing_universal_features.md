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
- **Kill the 9 GB merge.** vLLM runtime‑LoRA is a silent no‑op on this model (C49),
  which forced a full merged checkpoint per recipe (~5 min + disk). **The HF `qwen`
  backend applies a PEFT adapter directly** — no merge. Verified: apex adapter at
  budget 1024 lifts menagerie‑quick 0.128 → 0.436. So eval an *adapter*, not a merge.
- **Fast eval proxy.** Menagerie‑quick at a *low* think budget (1024) via the HF
  backend, adapter arm only vs a once‑measured base baseline (~3–4 min). Or a direct
  gym eval for the specific circuit under test. Confirm winners at 8192 + medium.

Net: ~8–15 min per curriculum (fast‑train + adapter eval) vs the old ~90 min —
roughly **10×**, and a real *search* becomes possible. The driver
`fast_search.py` runs a grid of recipes and logs each score.

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
  a generic, MULTI‑SKILL, surface‑agnostic curriculum (INDUCT = composed‑rule
  induction via decomposition search; EXECUTE = apply a stated multi‑step
  procedure; SELECT = pick the item satisfying a constraint conjunction), rendered
  over digits/letters/romans/invented‑words/greek — disjoint from any menagerie
  surface. Truth‑blind. Output `data/sft_universal.jsonl`. Add skills here (repair,
  abstain, longer‑horizon exploration) as new lesson types.
- `qwen35_4b_gauntlet_frontier/scripts/train_think.py` — QLoRA think‑channel SFT
  (custom per‑token loss; `--rank`, `--epochs`, `--w-think`; **no `--warm-start` =
  co‑train from base**). Fast tier for search: `--rank 16 --epochs 1 --batch-size 2
  --grad-accum 4 --max-length 2048` (~5 min). Confirm winners at r32 / 2–3 epochs.
- `qwen35_4b_gauntlet_frontier/scripts/bench.py --backend qwen --adapter <dir>
  --think-budget 1024` — **the transfer metric AND the search fitness function**:
  menagerie‑quick with the adapter applied in HF (no 9 GB merge; vLLM LoRA no‑ops,
  C49). ~4 min/arm. base@1024 ≈ 0.128; apex reference ≈ 0.436. Lifting the menagerie
  aggregate here = the universal feature transferred to the held‑out benchmark.
- Harvest‑remix fallback (if you want to search *selections* of real data instead
  of synthesizing): `qwen35_4b_gauntlet_frontier/scripts/{build_pool,build_recipe,
  fast_search}.py`.

The experiment loop:
1. `gen_curriculum.py` → `data/sft_universal.jsonl` (seconds).
2. Fast‑train (co‑train from base) → an adapter in `large_artifacts/`.
3. `bench.py --backend qwen --adapter … --think-budget 1024`, base vs adapter,
   fresh seed. **Success = the menagerie aggregate lifts** having trained on zero
   menagerie‑shaped data. Confirm real wins at 8192 + medium; codify as a claim.
4. Iterate the *curriculum design* fast: which skills / surfaces / lesson framings
   carry the transfer, and what "lifts all boats"? Ablate; add skills; loop.
   `train_eval_chain.sh` runs steps 2–3 end‑to‑end.

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
