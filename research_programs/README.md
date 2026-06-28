# Research Programs

This directory is the repository's forward-looking spine.

The imported experiments are seed evidence, not the boundary of the project. A research program is a durable line of inquiry that can accumulate many experiments over time, branch into sublines, retire failed assumptions, and invite new people or agents to add work without collapsing everything into one idea.

## How Programs Work

Each program directory contains:

- `charter.md`: why the line exists, what would count as progress, and what evidence currently anchors it.
- `backlog.md`: concrete next experiments, including controls and success/failure criteria.
- `evidence.md`: links to experiments, claims, negative results, and open questions.

Programs are intentionally broader than the two imported tracks. The current corpus is mapped into these programs as prototype evidence, and future experiments should either attach to an existing program or create a new program with a charter.

## Current Programs

- [Active Evidence Acquisition](active_evidence_acquisition/charter.md)
- [Algorithmic Memory And Retrieval](algorithmic_memory_and_retrieval/charter.md)
- [Benchmark Generalization](benchmark_generalization/charter.md)
- [Collective Experimentation Infrastructure](collective_experimentation_infrastructure/charter.md)
- [Evidence-Conditioned Selection](evidence_conditioned_selection/charter.md)
- [Interpretability And Diagnostics](interpretability_and_diagnostics/charter.md)
- [Operator And Skill Inventories](operator_and_skill_inventories/charter.md)
- [Posttraining And Adaptation](posttraining_and_adaptation/charter.md)
- [Process Control And Tool Use](process_control_and_tool_use/charter.md)
- [Reliability And Safety](reliability_and_safety/charter.md)
- [Structured Execution And Compilers](structured_execution_and_compilers/charter.md)

## Adding A Program

Use the scaffold command:

```bash
make new-program PROGRAM=<new_program_id> TITLE="<Title>" FOCUS="<one-sentence focus>"
```

Then fill in the charter, backlog, and evidence files. A new program should explain why it is not merely a small variant of an existing program.
