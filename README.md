# Small Model Experimentation

**How much capability is already inside a small model that ordinary use leaves on the table?**

This repository is a research log built around a single question: take *one* fixed [Qwen3.5-4B](https://qwen.ai) — never scaled, never distilled from a larger teacher, never turned into a different model — and find out how much latent capability can be *elicited* from the weights it already has. The bar every method has to beat is the cheapest baseline there is: **just sample more.**

📊 **[Read the findings-first research site →](https://ericflo.github.io/small-model-experimentation/)** — every experiment rendered in full, with native interactive charts, an evidence-linked claim ledger, and the latest results first.

---

## Why this exists

The field's default answer to "make the model better" is *make the model bigger* — more parameters, more data, a stronger teacher to distill from. Those levers work, but they say nothing about a question that matters just as much for anyone deploying a small model on real hardware: **given the model you already have, how much of its capability are you actually using?**

So the constraint here is deliberate and absolute:

- **One model.** Qwen3.5-4B, frozen. No scaling. No distillation. No larger teacher anywhere in the loop — not even to generate training data.
- **Elicitation, not acquisition.** Every method must draw out capability that is *already in the weights*: structured intermediates, verification, self-training on the model's own verified outputs, tool-augmented search, thinking budgets, context orchestration, activation-level probes.
- **A real bar to beat.** "Sample more" is free and strong. A method only counts if it beats matched-compute sampling on held-out, contamination-controlled tasks.

The result is **179 experiments across 12 research programs**, distilled into a **machine-checkable claim ledger** where every claim points at the experiments that support or challenge it, and **792 charts** rendered directly from each experiment's own result files.

## What we found

The corpus tells a connected story. A curated path through it:

- **Structured intermediates are a real lever, but selection is often the wall.** Executable or structured intermediate representations reliably improve small-model reliability (**C1**). But candidate *generation* is usually easier than deployable *selection* — a model's sample pool often contains a correct answer it can't reliably pick out (**C2**).

- **That selection wall is plumbing, not a capability limit.** A frozen 4B's own zero-training thinking-verifier selects best-of-8 well enough to close **~75%** of the pass@1→oracle gap — and when a cheap visible test exists, a free no-think verifier captures **83%** of it. The bottleneck was tooling, not intelligence (**C10**).

- **Native "thinking" is an unused deployable lever.** Turning on the model's reasoning channel is worth **+15pp** on MBPP greedy decoding — and controls prove it's *coherent reasoning content*, not just extra compute: irrelevant thinking collapses accuracy, contentless filler ≈ no-think, and coherent content is the entire gain (**C9**).

- **A small model can teach itself — no teacher required.** On a fresh, contamination-free program-synthesis substrate, test-time execution feedback does *not* beat sampling — but QLoRA-SFT on the model's **own** verified solutions banks capability into single-shot deployment (+42% held-out greedy@1), and iterating it compounds into a flywheel (**C11**). Tool-augmented search then extends the frontier *past* the sampling ceiling and banks that too (**C12**).

- **The compositional "wall" has a precise mechanism.** Where the fixed model fails at multi-step composition, the failure is **broken multi-step mental simulation / hypothesis identification** — *not* execution. Given the plan, the model executes at 0.90–1.00 through depth 4; left to identify the plan itself, it runs barely above chance. It stays a reliable compiler starved of search (**C13**). Capability turns out to be organized by input→output *format* rather than shared internal primitives (**C14**), and deployable capability factors as **module × interface × procedure** (**C15**).

The recipe that emerges from all of it: **tools generate, context orchestrates, the model simulates and transcribes.**

> The full arc runs to **C27** and is still growing. The [live claim ledger](https://ericflo.github.io/small-model-experimentation/claims/) and [cross-program synthesis](https://ericflo.github.io/small-model-experimentation/notebook/synthesis/) carry every claim with its evidence.

## What's in here

| | |
|---|---|
| **179** experiments | each self-contained: README, code, data, runs, analysis, report |
| **12** research programs | durable lines of inquiry, each with a charter, backlog, and evidence |
| **27** evidence-linked claims | a shared belief ledger; every claim cites its experiments |
| **792** result charts | rendered natively from each experiment's own data files |
| **1** fixed model | Qwen3.5-4B — the entire point |

```text
research_programs/<program-id>/   durable research lines
  charter.md · backlog.md · evidence.md

experiments/<experiment-id>/      one self-contained experiment
  README.md  metadata.yaml
  src/ scripts/ configs/          code and run scaffolding
  data/ runs/ analysis/ reports/  inputs, logs, and written-up results

knowledge/    the claim ledger, synthesis, catalogs, and future queue
docs/         operating guidance for contributors
scripts/      indexing, validation, and static-site generation
templates/    starting points for new experiments and programs
```

The 12 programs span **structured execution & compilers, evidence-conditioned selection, active evidence acquisition, algorithmic memory & retrieval, operator & skill inventories, posttraining & adaptation, process control & tool use, benchmark generalization, interpretability & diagnostics, reliability & safety, test-time reasoning budget,** and the meta-program of **collective experimentation infrastructure** — the repository treats itself as a research instrument.

## Navigating the corpus

The fastest way in is the **[research site](https://ericflo.github.io/small-model-experimentation/)**. To read the source directly:

- [`knowledge/synthesis.md`](knowledge/synthesis.md) — the living cross-program synthesis and strategic read.
- [`knowledge/claims/index.md`](knowledge/claims/index.md) — the claim ledger with evidence links.
- [`knowledge/experiment_catalog.md`](knowledge/experiment_catalog.md) — the full experiment inventory.
- [`research_programs/README.md`](research_programs/README.md) — the durable research lines.
- [`knowledge/future_experiment_queue.md`](knowledge/future_experiment_queue.md) — scored, protocol-ready next experiments.

## Reproducing an experiment

Each experiment folder is self-contained. Its `README.md` states the question and result; `reports/` holds the write-up; `data/` and `runs/` hold the inputs and logs. Large regenerable artifacts (adapters, checkpoints) are excluded by design — see each experiment's `reports/artifact_manifest.yaml` for how to regenerate them. Environment and GPU setup are documented in [`docs/compute_environment.md`](docs/compute_environment.md).

## Building the site locally

The research site is generated from the repository by stdlib-only Python (no dependencies to install):

```bash
python3 scripts/build_site.py --out site   # writes the static site
python3 scripts/check_site.py site          # verifies every internal link
python3 -m http.server 8000 --directory site
```

## A note on provenance

This corpus was produced through an intensive, largely autonomous experimentation loop. Individual experiments are working research artifacts, not polished libraries — the value is in the *aggregate*: the reusable patterns, the controlled comparisons, and the evidence-linked claims that make each next experiment less likely to repeat an old mistake. The repository is built to keep growing: every experiment either advances an existing research program or justifies a new one, and results connect upward into shared program evidence and the claim ledger.

## License

No license is currently declared, so default copyright applies (all rights reserved). If you'd like others to reuse this work, add a `LICENSE` file.
