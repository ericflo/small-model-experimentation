# Research Program Lifecycle

## Purpose

A research program is a durable line of inquiry, not a single experiment. It gives future people and agents a place to accumulate evidence, spawn many experiments, retire weak assumptions, and notice when a new line deserves to branch.

## Create A Program

Create a new program when:

- several experiments would naturally belong under the line,
- the line has its own progress signals,
- existing programs would hide the key uncertainty,
- the line can produce both positive and negative knowledge.

Do not create a program for a one-off variant. Use tags or an experiment note instead.

## Required Files

Every program has:

- `charter.md`: why the line exists, what counts as progress, boundaries.
- `backlog.md`: next experiments, controls, stop conditions.
- `evidence.md`: seed evidence, current read, contradictions.

Use the scaffold command:

```bash
make new-program PROGRAM=<program-id> TITLE="<Title>" FOCUS="<one-sentence focus>"
```

Then fill in the generated files and run:

```bash
make catalog
make validate
```

## Maintain A Program

After each result:

- add the experiment to `evidence.md` if it changes the line,
- update `backlog.md` if priorities changed,
- mark negative findings explicitly,
- split the program if it becomes too broad,
- retire hypotheses that repeated controls contradict.

## Program Quality Bar

A good program helps answer:

- What should be tried next?
- Which prior results matter?
- What would falsify the current direction?
- What should a new agent avoid repeating?
- What would make this line obsolete or complete?
