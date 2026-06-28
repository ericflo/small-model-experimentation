# Qwen VM-Agent ECHO QLoRA

This standalone experiment trains a Qwen-attached VM agent. The agent receives a
natural-language task, an initial typed bytecode program, and VM observations.
At each private turn it emits one textual edit action or `STOP`; the VM executes
the edit and returns the updated program and execution trace as text.

The main comparison is:

- action-only QLoRA: train on agent action tokens only;
- ECHO QLoRA: train on both action tokens and VM observation tokens.

## Main Run

The main run is:

```text
runs/main_vm_agent_echo_blank_a512_stoprule/
```

It used blank-program initialization, 512 training tasks, 32 examples per eval
split, K in `{0, 2, 4, 8}`, and a native direct-answer Qwen baseline.

Read these first:

```text
reports/main_vm_agent_echo_blank_a512_stoprule/report.html
reports/main_vm_agent_echo_blank_a512_stoprule/report.md
reports/main_vm_agent_echo_blank_a512_stoprule/summary_k8.csv
```

The HTML report embeds its charts and is the most convenient single file to
open. The Markdown report references the PNGs in `figures/`.

## Directory Layout

```text
src/        experiment, native-baseline, and report-generation scripts
runs/       per-run CSV metrics, manifests, rollout samples, and JSON results
reports/    standalone Markdown/HTML reports, summary tables, and figures
```

Large adapter and checkpoint files are stored under:

```text
large_artifacts/qwen_vm_agent_echo_qlora/checkpoints/
```

The checkpoint manifest maps run names to large artifact directories:

```text
checkpoint_manifest.csv
```
