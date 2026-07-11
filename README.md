# Small Model Experimentation

[![Validate](https://github.com/ericflo/small-model-experimentation/actions/workflows/validate.yml/badge.svg)](https://github.com/ericflo/small-model-experimentation/actions/workflows/validate.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ericflo/small-model-experimentation/blob/main/LICENSE)
[![Live site](https://img.shields.io/badge/live%20site-online-brightgreen)](https://ericflo.github.io/small-model-experimentation/)

### How far one Qwen3.5-4B goes without a bigger teacher

**How much capability can you elicit from — and install into — one small model, without ever borrowing from a larger one?**

This repository is a research log built around that single question: take *one* [Qwen3.5-4B](https://qwen.ai) and see how far it can be pushed — drawing out what its weights already hold and installing what they don't — **without ever importing capability from a larger model**: no scaling to something bigger, no distillation from a stronger teacher, no bigger model anywhere in the loop. *How* new capability gets installed is deliberately open: training on the model's own verified solutions, on its failures, on tool-found data, on token patterns that build generalized skills — anything qualifies so long as the signal doesn't come from a bigger model. The bar every method has to beat is the cheapest baseline there is: **just sample more.**

![One example: held-out single-shot accuracy climbs across three rounds of the model training on its own verified solutions.](experiments/qwen35_4b_neurosymbolic_repl_substrate/analysis/ei_trajectory.png)

*One thread from the corpus: the model fine-tuned on its own verified solutions climbs on unseen tasks across three self-training rounds — no larger model anywhere in the loop.*

**→ The [research site](https://ericflo.github.io/small-model-experimentation/) is the living index of everything here** — every experiment, every claim with its current status and evidence, every chart rendered natively from the experiment's own data, latest findings first. This README deliberately sticks to what doesn't change: the question, the rules, and how the corpus is organized. (There's a gate for that: `make check` fails if claim-specific references or hardcoded corpus counts creep back into this file.)

---

## Why this exists

The field's default answer to "make the model better" is *make the model bigger* — more parameters, more data, a stronger teacher to distill from. Those levers work, but they say nothing about a question that matters just as much for anyone deploying a small model on real hardware: **how much more can this exact model do — both by using its weights better and by improving them with what it can generate and gather itself — before you reach for a bigger one?**

So the constraint here is about the *source* of capability, and it is absolute:

- **One model, always Qwen3.5-4B.** Never a larger or different model — no scaling, no distillation, no stronger teacher anywhere in the loop, not even to generate training data. The weights are free to change; the capability *source* is not.
- **Provenance, not a weight freeze — and not one recipe.** The rule is not "don't change the weights" — it's "don't import intelligence from a bigger model." Both levers are in scope: *eliciting* latent capability at test time (structured intermediates, verification, tool-augmented search, thinking budgets, context orchestration, activation probes) **and** *installing* new capability by training. The install signal is itself an open research variable: self-training on verified or tool-found solutions is the best-studied so far, but training on failures, on token patterns that build generalized skills, or on anything else the 4B and its environment can produce is equally in-bounds. Only a larger/other model and plain scaling are off-limits.
- **A real bar to beat.** "Sample more" is free and strong. A method only counts if it beats matched-compute sampling on held-out, contamination-controlled tasks.

## The operating principles

The corpus is produced by an agent-driven loop running many experiments. These are the rules that keep it compounding instead of sprawling — they change rarely, which is why they live here:

- **One narrow question per experiment**, self-contained in its own directory with code, data, runs, and report — narrow enough that a single result can move belief.
- **Controls live next to the result**: shuffled/random baselines, frozen-vs-trained comparisons, and matched-compute sampling arms ship inside the same experiment, because controls are the difference between a result and a story.
- **Deployable and oracle numbers never mix.** What ships (single-shot, no oracle, no hidden-test peeking) is headlined separately from ceilings (oracle selection, best-of-k coverage), which are labeled non-deployable.
- **Contamination discipline.** Capability-gain claims require fresh or procedurally generated substrates; saturated public benchmarks are for measurement studies only.
- **Negative results are first-class.** Ruled-out levers are recorded with the same care as wins, so the next experiment doesn't re-run a dead end.
- **Durable beliefs live in a machine-checkable claim ledger.** Every claim cites the experiments that support or challenge it, carries a status that moves with evidence, and is validated by CI — the ledger, not any narrative, is the corpus's memory.
- **Everything is gated.** `make check` validates catalogs, claims, links, charts, and plain-language briefs before anything lands; the site regenerates from the repository on every push.

For what the corpus has actually *found*, read the living [synthesis](https://ericflo.github.io/small-model-experimentation/notebook/synthesis/) and the [claim ledger](https://ericflo.github.io/small-model-experimentation/claims/) — they are regenerated with the corpus and always current.

## Explore the corpus

The site is the best way in; the source files are one click deeper. Start anywhere:

| Destination | Live site | In the repo |
|---|---|---|
| **Latest findings** — what the corpus discovered most recently | [browse](https://ericflo.github.io/small-model-experimentation/) | — |
| **All experiments** — searchable, filterable, charted | [browse](https://ericflo.github.io/small-model-experimentation/experiments/) | [`experiment_catalog.md`](knowledge/experiment_catalog.md) |
| **Claim ledger** — each claim pointing at its experiments | [browse](https://ericflo.github.io/small-model-experimentation/claims/) | [`claims/index.md`](knowledge/claims/index.md) |
| **Research programs** — the durable lines of inquiry | [browse](https://ericflo.github.io/small-model-experimentation/programs/) | [`research_programs/README.md`](research_programs/README.md) |
| **Synthesis** — the living cross-program read | [browse](https://ericflo.github.io/small-model-experimentation/notebook/synthesis/) | [`synthesis.md`](knowledge/synthesis.md) |
| **What's next** — scored, protocol-ready future experiments | [browse](https://ericflo.github.io/small-model-experimentation/queue/) | [`future_experiment_queue.md`](knowledge/future_experiment_queue.md) |

## What's in here

Every experiment is self-contained (README, code, data, runs, analysis, report); every research program has a charter, backlog, and evidence file; the claims form a shared belief ledger where every claim cites its experiments; and every chart on the site is rendered natively from an experiment's own data files.

```text
research_programs/<program-id>/   durable research lines
  charter.md · backlog.md · evidence.md

experiments/<experiment-id>/      one self-contained experiment
  README.md  metadata.yaml
  src/ scripts/ configs/          code and run scaffolding
  data/ runs/ analysis/ reports/  inputs, logs, and written-up results

knowledge/    the claim ledger, synthesis, catalogs, and future queue
docs/         operating guidance and lifecycle
scripts/      indexing, validation, and static-site generation
templates/    starting points for new experiments and programs
```

**One model throughout — Qwen3.5-4B.** Its weights may be improved by anything the model, its tools, and its environment can produce, but a larger or different model is never in the loop — that constraint is the entire point.

## Reproducing an experiment

Each experiment folder is self-contained. Its `README.md` states the question and result; `reports/` holds the write-up; `data/` and `runs/` hold the inputs and logs. Large regenerable artifacts (adapters, checkpoints) are excluded by design — see each experiment's `reports/artifact_manifest.yaml` for how to regenerate them.

**Environment at a glance:** the current RunPod has one **NVIDIA L40 (48 GB)**. Bulk generation uses the pinned vLLM stack in a `uv`-managed `.venv-vllm`; training and measurements that require model internals use a separately locked `.venv`. Both paths run only **Qwen3.5-4B**. Full setup and historical hardware notes are in [`docs/compute_environment.md`](docs/compute_environment.md).

## Building the site locally

The research site is generated from the repository by stdlib-only Python (no dependencies to install):

```bash
python3 scripts/build_site.py --out site   # writes the static site
python3 scripts/check_site.py site          # verifies every internal link
python3 -m http.server 8000 --directory site
```

## Contributing

This is a largely agent-driven research loop rather than an open PR queue, but the conventions that keep results compounding are documented: [`CONTRIBUTING.md`](CONTRIBUTING.md), the [`docs/agent_handbook.md`](docs/agent_handbook.md), the [`docs/experiment_lifecycle.md`](docs/experiment_lifecycle.md), and the [`docs/quality_gates.md`](docs/quality_gates.md). Every new experiment either advances an existing research program or justifies a new one, and its results connect upward into shared program evidence and the claim ledger.

## A note on provenance

This corpus was produced through an intensive, largely agent-driven experimentation loop — the experiments, reports, and evidence-linked claims were generated by an AI agent workflow, not hand-authored by a human researcher. That orchestration agent drives the loop; the Qwen3.5-4B remains the object of study, never the author. Individual experiments are working research artifacts, not polished libraries — the value is in the *aggregate*: the reusable patterns, the controlled comparisons, and the evidence-linked claims that make each next experiment less likely to repeat an old mistake. The repository is built to keep growing, with every result connecting upward into shared program evidence and the claim ledger.

## How to cite

```bibtex
@misc{florenzano2026smallmodel,
  author       = {Florenzano, Eric},
  title        = {Small Model Experimentation: Eliciting and Installing Capability in a Single Qwen3.5-4B Without a Larger Teacher},
  year         = {2026},
  howpublished = {\url{https://ericflo.github.io/small-model-experimentation/}},
  note         = {Research log}
}
```

A [`CITATION.cff`](CITATION.cff) at the repo root backs GitHub's "Cite this repository" button with identical metadata.

## License

Licensed under the [Apache License 2.0](LICENSE). You're free to use, modify, and build on this work, including commercially, provided you retain the license and attribution.
