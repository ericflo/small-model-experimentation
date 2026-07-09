# Evidence

## Seed Experiments

- [qwen35_4b_operator_inventory_search_pilot](../../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)
- [qwen35_4b_operator_inventory_scaling_stress](../../experiments/qwen35_4b_operator_inventory_scaling_stress/reports/qwen35_4b_operator_inventory_scaling_stress_report.md)
- [qwen35_4b_inventory_shortlister_training](../../experiments/qwen35_4b_inventory_shortlister_training/README.md)
- [qwen35_4b_joint_shortlister_ladder](../../experiments/qwen35_4b_joint_shortlister_ladder/reports/qwen35_4b_joint_shortlister_ladder_report.md)

## Stopped Interface Result

- [qwen35_4b_verified_macro_invention](../../experiments/qwen35_4b_verified_macro_invention/reports/report.md)

The verified-macro experiment stopped before its fresh induction smoke and full comparison. Smoke
v1 used an under-sized 192-token thinking budget: all 1,440 solver samples force-closed and 607
answer stages truncated, so it measured an unusable generation interface rather than macro quality.

Task-independent, plan-given probes then separated surface use from induction. With budgeted
thinking, all 16 samples again force-closed and 12/16 spilled into the answer cap. With thinking
off, truncation disappeared and 16/16 outputs were syntactically valid and macro-using, but only
3/16 samples were exact and only 1/4 supplied plans had even one exact optimal transcription. All
13 failures over-aliased beyond depth five. Thus Qwen can emit the alias syntax,
but this free-form interface does not reliably select and place the right alias even when the
primitive plan is given. No mined-versus-random-versus-designed induction comparison ran, so the
quality or value of verified macro abstractions remains unresolved and this result supports no
capability claim.

## Current Read

Inventory search can recover held-out targets when the right primitives exist. The next bottleneck
is scalable shortlisting and disambiguation. Composite entries also need an interface that cleanly
separates choosing an abstraction from deciding whether, where, and how many times to call one;
free-form macro emission currently confounds those questions.
