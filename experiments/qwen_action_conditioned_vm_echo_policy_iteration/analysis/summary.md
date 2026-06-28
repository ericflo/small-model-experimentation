# Analysis Summary

Main run: `main_action_vm_echo_s192_thr070`
Validation candidate selector: base 10.2%, learned 10.9%, oracle 36.7%.
Fresh paired direct: seed 18.0%, learned distill 22.7%.
Fresh paired search: seed 51.6%, learned distill 64.1%.
Hard composition search: seed 35.2%, learned distill 52.3%.

Figures:
- `analysis/figures/main_accuracy_by_phase.png`
- `analysis/figures/learned_rerank_gap.png`
- `analysis/figures/target_selection.png`
- `analysis/figures/consequence_training.png`
- `analysis/figures/pilot_threshold_comparison.png`

Main target selection:
 source_examples  targets  oracle_found_rate  selected_correct_rate selected_valid_rate  changed_rate  mean_selected_score  min_score                   phase                             run
            1024      470           0.407227               0.289362            0.995745      1.000000             0.884126        0.7  learned_policy_targets main_action_vm_echo_s192_thr070
            1024      417           0.407227               1.000000                 1.0      0.625899                  NaN        NaN answer_verified_targets main_action_vm_echo_s192_thr070
