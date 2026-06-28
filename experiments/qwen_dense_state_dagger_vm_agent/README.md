# Qwen Dense-State DAgger VM Agent

This standalone experiment trains `Qwen/Qwen3-4B` as one recurrent transition
inside a typed bytecode VM loop.

The model receives a natural-language task and dense projected VM-state tokens.
It predicts one VM edit action or `STOP`; the VM executes that action, returns a
new structured state, and the same model is run again for the next step.

The main interventions are:

- encoderless dense VM-state injection through `inputs_embeds`;
- direct action and value heads instead of textual action generation;
- DAgger over the model's own visited VM states;
- value-gated stopping to reduce false `STOP`.

Large checkpoints are stored outside this directory:

```text
large_artifacts/qwen_dense_state_dagger_vm_agent/checkpoints/
```

Read `experiment_log.md` for the iteration record. Final reports are written
under `reports/`.

Main completed run:

```text
runs/main_joint_action_calibrated_s256_r2/
```

Primary reports:

```text
reports/report.md
reports/report.html
```
