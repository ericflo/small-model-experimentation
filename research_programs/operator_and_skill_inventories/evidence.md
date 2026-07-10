# Evidence

## Seed Experiments

- [qwen35_4b_operator_inventory_search_pilot](../../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)
- [qwen35_4b_operator_inventory_scaling_stress](../../experiments/qwen35_4b_operator_inventory_scaling_stress/reports/qwen35_4b_operator_inventory_scaling_stress_report.md)
- [qwen35_4b_inventory_shortlister_training](../../experiments/qwen35_4b_inventory_shortlister_training/README.md)
- [qwen35_4b_joint_shortlister_ladder](../../experiments/qwen35_4b_joint_shortlister_ladder/reports/qwen35_4b_joint_shortlister_ladder_report.md)

## Stopped Prototype and Corrected Interface

- [qwen35_4b_verified_macro_invention](../../experiments/qwen35_4b_verified_macro_invention/reports/report.md)

The verified-macro experiment stopped before its fresh induction smoke and full comparison. Smoke
v1 used an under-sized 192-token thinking budget: all 1,440 solver samples force-closed and 607
answer stages truncated, so it measured an unusable generation interface rather than macro quality.

Task-independent, plan-given probes then separated surface use from induction. With budgeted
thinking, all 16 samples again force-closed and 12/16 spilled into the answer cap. With thinking
off, truncation disappeared and 16/16 outputs were syntactically valid and macro-using, but only
3/16 samples were exact and only 1/4 supplied plans had even one exact optimal transcription. All
13 failures over-aliased beyond depth five. Those probes established only that the low-compute
interfaces failed; they did not establish a model-level alias-placement limit.

The independent long-context follow-up
[`qwen35_4b_verified_macro_long_context_rerun`](../../experiments/qwen35_4b_verified_macro_long_context_rerun/)
corrected that diagnosis before scoring induction. Under vLLM with think@16,384, the disjoint
plan-given gate passed 16/16 records: 63/64 samples were strict valid macro-using rewrites, all 12
cap contacts were classified by the frozen exact-token periodic-tail detector, no cap remained
unresolved, and no answer truncated. At the registered K=4 record-level gate, adequately budgeted
reasoning therefore made the existing free-form alias interface usable when the primitive plan was
supplied; this is not a per-sample-perfect or induction claim. The fresh induction workload is much
harder to provision: its first base arm had 131/144 unresolved contacts at 16,384 and is being
escalated without scoring. No mined-versus-random-versus-designed capability result exists yet.

## Current Read

Inventory search can recover held-out targets when the right primitives exist. The next bottleneck
is scalable shortlisting and disambiguation. Composite aliases are no longer blocked at the
registered plan-given K=4 record-level gate. The open question is now the intended one—whether a
mined inventory helps *induce* fresh programs beyond matched-compute base sampling, literal hints,
random inventories, and a designed ceiling.
