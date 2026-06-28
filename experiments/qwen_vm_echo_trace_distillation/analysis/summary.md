# Analysis Summary

Main run: `main_vm_echo_s192_w003`
Fresh paired full-supervised direct: baseline 84.4%, VM-ECHO 84.4%.
Fresh paired full-supervised search: baseline 93.0%, VM-ECHO 96.1%.
Fresh paired trace-top observation: baseline 0.7%, VM-ECHO 43.1%.

Figures:
- `analysis/figures/main_phase_curves.png`
- `analysis/figures/full_supervised_split_bars.png`
- `analysis/figures/echo_observation_accuracy.png`
- `analysis/figures/expert_target_rates.png`
- `analysis/figures/pilot_weight_sweep.png`

Main expert targets:
 source_examples  targets  found_rate  changed_rate  mean_candidates  candidate_valid_rate      arm  round                   phase                    run
            1024      381    0.372070      0.648294       241.000000              0.630932 baseline      1 baseline_expert_round_1 main_vm_echo_s192_w003
            1024      456    0.445312      0.708333       241.000000              0.634154 baseline      2 baseline_expert_round_2 main_vm_echo_s192_w003
            1024      398    0.388672      0.678392       241.216797              0.636584  vm_echo      1  vm_echo_expert_round_1 main_vm_echo_s192_w003
            1024      436    0.425781      0.681193       241.000000              0.643441  vm_echo      2  vm_echo_expert_round_2 main_vm_echo_s192_w003
