# End-to-End Structured Slot Executor Analysis Summary

## Runs

| phase | modulus | variant | slot_capacity | init_mode | transition_mode | supervision | train_steps | length | k | n |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 31 | main_generic_primitive_router | 31 | generic_mlp | primitive_router | full_belief | 1400 | 24 | 24 | 256 |
| main | 31 | main_oracle_exact | 31 | oracle | exact | full_belief | 0 | 24 | 24 | 256 |
| main | 31 | main_oracle_primitive_router | 31 | oracle | primitive_router | full_belief | 800 | 24 | 24 | 256 |
| main | 31 | main_sinkhorn_exact | 31 | sinkhorn_cyclic | exact | full_belief | 1800 | 24 | 24 | 256 |
| main | 31 | main_sinkhorn_mlp | 31 | sinkhorn_cyclic | mlp | full_belief | 1600 | 24 | 24 | 256 |
| main | 31 | main_sinkhorn_primitive_router | 31 | sinkhorn_cyclic | primitive_router | full_belief | 2400 | 24 | 24 | 256 |
| pilot | 11 | pilot_generic_primitive_router | 11 | generic_mlp | primitive_router | full_belief | 1000 | 12 | 12 | 256 |
| pilot | 11 | pilot_oracle_exact | 11 | oracle | exact | full_belief | 0 | 12 | 12 | 256 |
| pilot | 11 | pilot_oracle_primitive_router | 11 | oracle | primitive_router | full_belief | 600 | 12 | 12 | 256 |
| pilot | 11 | pilot_sinkhorn_exact | 11 | sinkhorn_cyclic | exact | full_belief | 900 | 12 | 12 | 256 |
| pilot | 11 | pilot_sinkhorn_mlp | 11 | sinkhorn_cyclic | mlp | full_belief | 1000 | 12 | 12 | 256 |
| pilot | 11 | pilot_sinkhorn_primitive_router | 11 | sinkhorn_cyclic | primitive_router | full_belief | 1400 | 12 | 12 | 256 |
| scale | 97 | scale_oracle_exact | 97 | oracle | exact | full_belief | 0 | 24 | 24 | 64 |
| scale | 97 | scale_sinkhorn_primitive_router | 97 | sinkhorn_cyclic | primitive_router | full_belief | 1800 | 24 | 24 | 64 |
| smoke | 7 | smoke_generic_primitive_router | 7 | generic_mlp | primitive_router | full_belief | 700 | 3 | 3 | 128 |
| smoke | 7 | smoke_oracle_exact | 7 | oracle | exact | full_belief | 0 | 3 | 3 | 128 |
| smoke | 7 | smoke_oracle_primitive_router | 7 | oracle | primitive_router | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_sinkhorn_cyclic_mixer | 7 | sinkhorn_cyclic | cyclic_mixer | full_belief | 700 | 3 | 3 | 128 |
| smoke | 7 | smoke_sinkhorn_cyclic_mixer_lr01 | 7 | sinkhorn_cyclic | cyclic_mixer | full_belief | 900 | 3 | 3 | 128 |
| smoke | 7 | smoke_sinkhorn_exact | 7 | sinkhorn_cyclic | exact | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_sinkhorn_mlp | 7 | sinkhorn_cyclic | mlp | full_belief | 700 | 3 | 3 | 128 |
| smoke | 7 | smoke_sinkhorn_primitive_router | 7 | sinkhorn_cyclic | primitive_router | full_belief | 700 | 3 | 3 | 128 |
| smoke | 7 | smoke_sinkhorn_primitive_router_lr01 | 7 | sinkhorn_cyclic | primitive_router | full_belief | 900 | 3 | 3 | 128 |

## First K >= L Summary

### main modulus 31

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main_generic_primitive_router | 4 | 4 | 80.2% | 52.8% | 58.6% | 85.3% | 63.4% | 0.044 | 0.626 | 1.000 | 1.845 |
| main_generic_primitive_router | 8 | 8 | 70.1% | 47.5% | 58.6% | 85.3% | 63.3% | 0.044 | 0.512 | 1.000 | 1.405 |
| main_generic_primitive_router | 12 | 12 | 62.2% | 42.8% | 58.4% | 85.0% | 63.2% | 0.044 | 0.420 | 1.000 | 1.020 |
| main_generic_primitive_router | 16 | 16 | 55.8% | 38.5% | 58.9% | 84.9% | 63.6% | 0.043 | 0.335 | 1.000 | 0.870 |
| main_generic_primitive_router | 24 | 24 | 43.0% | 28.7% | 58.6% | 84.9% | 63.4% | 0.044 | 0.229 | 1.000 | 0.656 |
| main_oracle_exact | 4 | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.476 | -1.000 | 2.128 |
| main_oracle_exact | 8 | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.309 | -1.000 | 1.527 |
| main_oracle_exact | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.208 | -1.000 | 0.986 |
| main_oracle_exact | 16 | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.175 | -1.000 | 0.741 |
| main_oracle_exact | 24 | 24 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.141 | -1.000 | 0.331 |
| main_oracle_primitive_router | 4 | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.525 | 1.000 | 2.128 |
| main_oracle_primitive_router | 8 | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.413 | 1.000 | 1.528 |
| main_oracle_primitive_router | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.325 | 1.000 | 0.986 |
| main_oracle_primitive_router | 16 | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.283 | 1.000 | 0.741 |
| main_oracle_primitive_router | 24 | 24 | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% | 0.000 | 0.216 | 1.000 | 0.332 |
| main_sinkhorn_exact | 4 | 4 | 99.7% | 99.4% | 99.6% | 100.0% | 100.0% | 0.000 | 0.777 | -1.000 | 2.141 |
| main_sinkhorn_exact | 8 | 8 | 99.5% | 99.1% | 99.6% | 100.0% | 100.0% | 0.000 | 0.607 | -1.000 | 1.547 |
| main_sinkhorn_exact | 12 | 12 | 99.3% | 98.9% | 99.6% | 100.0% | 100.0% | 0.000 | 0.466 | -1.000 | 1.011 |
| main_sinkhorn_exact | 16 | 16 | 99.0% | 98.5% | 99.6% | 100.0% | 100.0% | 0.000 | 0.371 | -1.000 | 0.769 |
| main_sinkhorn_exact | 24 | 24 | 97.7% | 96.8% | 99.6% | 100.0% | 100.0% | 0.000 | 0.248 | -1.000 | 0.368 |
| main_sinkhorn_mlp | 4 | 4 | 65.7% | 19.0% | 58.9% | 100.0% | 93.5% | 0.014 | 0.173 | -1.000 | 2.177 |
| main_sinkhorn_mlp | 8 | 8 | 50.7% | 13.9% | 58.9% | 100.0% | 93.5% | 0.014 | 0.178 | -1.000 | 1.683 |
| main_sinkhorn_mlp | 12 | 12 | 38.4% | 10.0% | 58.9% | 100.0% | 93.5% | 0.014 | 0.177 | -1.000 | 1.283 |
| main_sinkhorn_mlp | 16 | 16 | 30.4% | 7.8% | 58.9% | 100.0% | 93.5% | 0.014 | 0.177 | -1.000 | 1.102 |
| main_sinkhorn_mlp | 24 | 24 | 19.9% | 5.0% | 58.9% | 100.0% | 93.5% | 0.014 | 0.174 | -1.000 | 0.834 |
| main_sinkhorn_primitive_router | 4 | 4 | 99.8% | 99.6% | 99.7% | 100.0% | 100.0% | 0.000 | 0.844 | 1.000 | 2.136 |
| main_sinkhorn_primitive_router | 8 | 8 | 99.7% | 99.5% | 99.7% | 100.0% | 100.0% | 0.000 | 0.698 | 1.000 | 1.540 |
| main_sinkhorn_primitive_router | 12 | 12 | 99.6% | 99.3% | 99.7% | 100.0% | 100.0% | 0.000 | 0.559 | 1.000 | 1.003 |
| main_sinkhorn_primitive_router | 16 | 16 | 99.4% | 99.1% | 99.7% | 100.0% | 100.0% | 0.000 | 0.454 | 1.000 | 0.759 |
| main_sinkhorn_primitive_router | 24 | 24 | 98.7% | 98.1% | 99.7% | 100.0% | 100.0% | 0.000 | 0.308 | 1.000 | 0.356 |

### pilot modulus 11

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot_generic_primitive_router | 3 | 3 | 82.4% | 58.6% | 60.6% | 88.7% | 62.1% | 0.130 | 0.810 | 1.000 | 1.100 |
| pilot_generic_primitive_router | 6 | 6 | 76.1% | 58.8% | 60.7% | 88.4% | 62.1% | 0.130 | 0.758 | 1.000 | 0.818 |
| pilot_generic_primitive_router | 9 | 9 | 72.9% | 60.0% | 60.7% | 88.6% | 62.1% | 0.130 | 0.701 | 1.000 | 0.550 |
| pilot_generic_primitive_router | 12 | 12 | 69.5% | 59.5% | 60.4% | 88.2% | 61.8% | 0.132 | 0.655 | 1.000 | 0.410 |
| pilot_oracle_exact | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.679 | -1.000 | 1.491 |
| pilot_oracle_exact | 6 | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.526 | -1.000 | 1.046 |
| pilot_oracle_exact | 9 | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.431 | -1.000 | 0.621 |
| pilot_oracle_exact | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.395 | -1.000 | 0.438 |
| pilot_oracle_primitive_router | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.690 | 1.000 | 1.491 |
| pilot_oracle_primitive_router | 6 | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.560 | 1.000 | 1.046 |
| pilot_oracle_primitive_router | 9 | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.486 | 1.000 | 0.621 |
| pilot_oracle_primitive_router | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.458 | 1.000 | 0.438 |
| pilot_sinkhorn_exact | 3 | 3 | 99.4% | 98.4% | 98.8% | 100.0% | 100.0% | 0.001 | 0.783 | -1.000 | 1.514 |
| pilot_sinkhorn_exact | 6 | 6 | 99.0% | 98.0% | 98.8% | 100.0% | 100.0% | 0.001 | 0.647 | -1.000 | 1.080 |
| pilot_sinkhorn_exact | 9 | 9 | 98.6% | 97.7% | 98.8% | 100.0% | 100.0% | 0.001 | 0.548 | -1.000 | 0.664 |
| pilot_sinkhorn_exact | 12 | 12 | 98.2% | 97.3% | 98.8% | 100.0% | 100.0% | 0.001 | 0.498 | -1.000 | 0.483 |
| pilot_sinkhorn_mlp | 3 | 3 | 93.4% | 82.0% | 49.2% | 100.0% | 90.9% | 0.051 | 0.798 | -1.000 | 1.530 |
| pilot_sinkhorn_mlp | 6 | 6 | 88.6% | 77.6% | 49.2% | 100.0% | 90.9% | 0.051 | 0.805 | -1.000 | 1.100 |
| pilot_sinkhorn_mlp | 9 | 9 | 83.5% | 75.0% | 49.2% | 100.0% | 90.9% | 0.051 | 0.807 | -1.000 | 0.709 |
| pilot_sinkhorn_mlp | 12 | 12 | 78.4% | 70.2% | 49.2% | 100.0% | 90.9% | 0.051 | 0.810 | -1.000 | 0.521 |
| pilot_sinkhorn_primitive_router | 3 | 3 | 99.7% | 99.3% | 99.5% | 100.0% | 100.0% | 0.001 | 0.827 | 1.000 | 1.503 |
| pilot_sinkhorn_primitive_router | 6 | 6 | 99.6% | 99.1% | 99.5% | 100.0% | 100.0% | 0.001 | 0.705 | 1.000 | 1.063 |
| pilot_sinkhorn_primitive_router | 9 | 9 | 99.4% | 99.0% | 99.5% | 100.0% | 100.0% | 0.001 | 0.605 | 1.000 | 0.642 |
| pilot_sinkhorn_primitive_router | 12 | 12 | 99.2% | 98.9% | 99.5% | 100.0% | 100.0% | 0.001 | 0.549 | 1.000 | 0.460 |

### scale modulus 97

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scale_oracle_exact | 4 | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.451 | -1.000 | 3.162 |
| scale_oracle_exact | 8 | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.211 | -1.000 | 2.187 |
| scale_oracle_exact | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.135 | -1.000 | 1.730 |
| scale_oracle_exact | 16 | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.112 | -1.000 | 1.255 |
| scale_oracle_exact | 24 | 24 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.058 | -1.000 | 0.702 |
| scale_sinkhorn_primitive_router | 4 | 4 | 99.6% | 99.0% | 99.4% | 100.0% | 100.0% | 0.000 | 0.679 | 1.000 | 3.176 |
| scale_sinkhorn_primitive_router | 8 | 8 | 99.2% | 98.7% | 99.4% | 100.0% | 100.0% | 0.000 | 0.446 | 1.000 | 2.213 |
| scale_sinkhorn_primitive_router | 12 | 12 | 98.6% | 97.6% | 99.4% | 100.0% | 100.0% | 0.000 | 0.299 | 1.000 | 1.762 |
| scale_sinkhorn_primitive_router | 16 | 16 | 97.8% | 96.6% | 99.4% | 100.0% | 100.0% | 0.000 | 0.226 | 1.000 | 1.302 |
| scale_sinkhorn_primitive_router | 24 | 24 | 96.2% | 94.9% | 99.4% | 100.0% | 100.0% | 0.000 | 0.113 | 1.000 | 0.746 |

### smoke modulus 7

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke_generic_primitive_router | 2 | 2 | 90.4% | 73.3% | 75.5% | 85.5% | 75.8% | 0.091 | 0.834 | 1.000 | 1.258 |
| smoke_generic_primitive_router | 3 | 3 | 87.9% | 72.8% | 75.5% | 85.9% | 75.7% | 0.089 | 0.820 | 1.000 | 1.095 |
| smoke_oracle_exact | 2 | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.814 | -1.000 | 1.421 |
| smoke_oracle_exact | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.736 | -1.000 | 1.197 |
| smoke_oracle_primitive_router | 2 | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.851 | 1.000 | 1.421 |
| smoke_oracle_primitive_router | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.824 | 1.000 | 1.197 |
| smoke_sinkhorn_cyclic_mixer | 2 | 2 | 89.0% | 68.1% | 72.6% | 100.0% | 100.0% | 0.046 | 0.592 | 1.000 | 1.576 |
| smoke_sinkhorn_cyclic_mixer | 3 | 3 | 86.5% | 67.6% | 72.6% | 100.0% | 100.0% | 0.046 | 0.557 | 1.000 | 1.421 |
| smoke_sinkhorn_cyclic_mixer_lr01 | 2 | 2 | 94.3% | 85.4% | 89.0% | 100.0% | 100.0% | 0.018 | 0.792 | 0.763 | 1.249 |
| smoke_sinkhorn_cyclic_mixer_lr01 | 3 | 3 | 91.5% | 82.5% | 89.0% | 100.0% | 100.0% | 0.018 | 0.752 | 0.762 | 1.064 |
| smoke_sinkhorn_exact | 2 | 2 | 98.7% | 96.2% | 96.9% | 100.0% | 100.0% | 0.005 | 0.836 | -1.000 | 1.453 |
| smoke_sinkhorn_exact | 3 | 3 | 98.4% | 96.0% | 96.9% | 100.0% | 100.0% | 0.005 | 0.778 | -1.000 | 1.242 |
| smoke_sinkhorn_mlp | 2 | 2 | 94.4% | 80.9% | 61.0% | 100.0% | 100.0% | 0.065 | 0.806 | -1.000 | 1.504 |
| smoke_sinkhorn_mlp | 3 | 3 | 90.6% | 77.0% | 61.0% | 100.0% | 100.0% | 0.065 | 0.799 | -1.000 | 1.300 |
| smoke_sinkhorn_primitive_router | 2 | 2 | 89.5% | 69.6% | 73.9% | 100.0% | 100.0% | 0.044 | 0.605 | 1.000 | 1.575 |
| smoke_sinkhorn_primitive_router | 3 | 3 | 87.1% | 69.1% | 73.9% | 100.0% | 100.0% | 0.044 | 0.567 | 1.000 | 1.419 |
| smoke_sinkhorn_primitive_router_lr01 | 2 | 2 | 99.6% | 98.7% | 98.9% | 100.0% | 100.0% | 0.002 | 0.856 | 1.000 | 1.434 |
| smoke_sinkhorn_primitive_router_lr01 | 3 | 3 | 99.4% | 98.6% | 98.9% | 100.0% | 100.0% | 0.002 | 0.796 | 1.000 | 1.215 |

## Final Training Rows

| phase | modulus | variant | step | loss | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 31 | main_generic_primitive_router | 1400 | 3.484 | 79.8% | 51.6% | 59.0% | 85.5% | 63.7% | 0.043 | 0.619 | 1.000 |
| main | 31 | main_oracle_primitive_router | 800 | 2.415 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.483 | 1.000 |
| main | 31 | main_sinkhorn_exact | 1800 | 2.851 | 99.7% | 99.3% | 99.6% | 100.0% | 100.0% | 0.000 | 0.760 | -1.000 |
| main | 31 | main_sinkhorn_mlp | 1600 | 4.550 | 60.8% | 18.3% | 58.9% | 100.0% | 93.5% | 0.014 | 0.168 | -1.000 |
| main | 31 | main_sinkhorn_primitive_router | 2400 | 2.915 | 99.8% | 99.6% | 99.7% | 100.0% | 100.0% | 0.000 | 0.821 | 1.000 |
| pilot | 11 | pilot_generic_primitive_router | 1000 | 2.572 | 80.9% | 60.0% | 60.5% | 87.4% | 62.0% | 0.127 | 0.791 | 1.000 |
| pilot | 11 | pilot_oracle_primitive_router | 600 | 1.748 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.673 | 1.000 |
| pilot | 11 | pilot_sinkhorn_exact | 900 | 2.046 | 99.3% | 98.4% | 98.8% | 100.0% | 100.0% | 0.001 | 0.776 | -1.000 |
| pilot | 11 | pilot_sinkhorn_mlp | 1000 | 2.409 | 92.4% | 82.0% | 49.2% | 100.0% | 90.9% | 0.051 | 0.795 | -1.000 |
| pilot | 11 | pilot_sinkhorn_primitive_router | 1400 | 1.961 | 99.7% | 99.3% | 99.5% | 100.0% | 100.0% | 0.001 | 0.798 | 1.000 |
| scale | 97 | scale_sinkhorn_primitive_router | 1800 | 4.104 | 99.6% | 99.2% | 99.4% | 100.0% | 100.0% | 0.000 | 0.638 | 1.000 |
| smoke | 7 | smoke_generic_primitive_router | 700 | 2.063 | 89.5% | 71.1% | 74.7% | 85.6% | 74.9% | 0.097 | 0.826 | 1.000 |
| smoke | 7 | smoke_oracle_primitive_router | 500 | 1.615 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.856 | 1.000 |
| smoke | 7 | smoke_sinkhorn_cyclic_mixer | 700 | 2.217 | 89.1% | 68.8% | 72.6% | 100.0% | 100.0% | 0.046 | 0.599 | 1.000 |
| smoke | 7 | smoke_sinkhorn_cyclic_mixer_lr01 | 900 | 2.177 | 94.1% | 85.3% | 89.0% | 100.0% | 100.0% | 0.018 | 0.789 | 0.786 |
| smoke | 7 | smoke_sinkhorn_exact | 500 | 1.854 | 98.9% | 96.4% | 96.9% | 100.0% | 100.0% | 0.005 | 0.838 | -1.000 |
| smoke | 7 | smoke_sinkhorn_mlp | 700 | 2.215 | 93.8% | 81.3% | 61.0% | 100.0% | 100.0% | 0.065 | 0.791 | -1.000 |
| smoke | 7 | smoke_sinkhorn_primitive_router | 700 | 2.191 | 89.6% | 70.1% | 73.9% | 100.0% | 100.0% | 0.044 | 0.612 | 1.000 |
| smoke | 7 | smoke_sinkhorn_primitive_router_lr01 | 900 | 1.822 | 99.6% | 98.8% | 98.9% | 100.0% | 100.0% | 0.002 | 0.838 | 1.000 |