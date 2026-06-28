# Structured Slot Initializer Ladder Analysis Summary

## Runs

| phase | modulus | variant | slot_capacity | init_mode | transition_mode | supervision | train_steps | length | k | n |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 31 | main_factorized_cyclic_overlap | 31 | factorized_cyclic | exact | full_belief | 2400 | 24 | 24 | 256 |
| main | 31 | main_factorized_free_b_reg | 31 | factorized_free_b | exact | full_belief | 1600 | 24 | 24 | 256 |
| main | 31 | main_generic_mlp_final_query | 31 | generic_mlp | exact | final_query | 1400 | 24 | 24 | 256 |
| main | 31 | main_generic_mlp_full_belief | 31 | generic_mlp | exact | full_belief | 1400 | 24 | 24 | 256 |
| main | 31 | main_oracle_ceiling | 31 | oracle | exact | full_belief | 0 | 24 | 24 | 256 |
| main | 31 | main_sinkhorn_cyclic | 31 | sinkhorn_cyclic | exact | full_belief | 2400 | 24 | 24 | 256 |
| pilot | 11 | pilot_factorized_cyclic_overlap | 11 | factorized_cyclic | exact | full_belief | 1400 | 12 | 12 | 256 |
| pilot | 11 | pilot_factorized_free_b_reg | 11 | factorized_free_b | exact | full_belief | 1100 | 12 | 12 | 256 |
| pilot | 11 | pilot_generic_mlp_final_query | 11 | generic_mlp | exact | final_query | 1100 | 12 | 12 | 256 |
| pilot | 11 | pilot_generic_mlp_full_belief | 11 | generic_mlp | exact | full_belief | 1100 | 12 | 12 | 256 |
| pilot | 11 | pilot_oracle_ceiling | 11 | oracle | exact | full_belief | 0 | 12 | 12 | 256 |
| pilot | 11 | pilot_sinkhorn_cyclic | 11 | sinkhorn_cyclic | exact | full_belief | 800 | 12 | 12 | 256 |
| scale | 97 | scale_oracle_init_only | 97 | oracle | exact | full_belief | 0 | 0 | 0 | 64 |
| scale | 97 | scale_sinkhorn_cyclic_init_only | 97 | sinkhorn_cyclic | exact | full_belief | 1200 | 0 | 0 | 64 |
| smoke | 7 | smoke_factorized_cyclic_overlap | 7 | factorized_cyclic | exact | full_belief | 900 | 3 | 3 | 128 |
| smoke | 7 | smoke_factorized_cyclic_plain | 7 | factorized_cyclic | exact | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_factorized_cyclic_reg | 7 | factorized_cyclic | exact | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_factorized_free_b_reg | 7 | factorized_free_b | exact | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_generic_mlp | 7 | generic_mlp | exact | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_generic_mlp_final_query | 7 | generic_mlp | exact | final_query | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_indexed_cyclic | 7 | indexed_cyclic | exact | full_belief | 0 | 3 | 3 | 64 |
| smoke | 7 | smoke_oracle_ceiling | 7 | oracle | exact | full_belief | 0 | 3 | 3 | 64 |
| smoke | 7 | smoke_sinkhorn_cyclic | 7 | sinkhorn_cyclic | exact | full_belief | 500 | 3 | 3 | 128 |

## First K >= L Summary

### main modulus 31

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main_factorized_cyclic_overlap | 4 | 4 | 94.6% | 90.1% | 90.0% | 100.0% | 90.3% | 0.007 | 0.754 | -1.000 | 2.171 |
| main_factorized_cyclic_overlap | 8 | 8 | 94.0% | 91.5% | 90.0% | 100.0% | 90.3% | 0.007 | 0.604 | -1.000 | 1.622 |
| main_factorized_cyclic_overlap | 12 | 12 | 94.5% | 92.7% | 90.0% | 100.0% | 90.3% | 0.007 | 0.472 | -1.000 | 1.102 |
| main_factorized_cyclic_overlap | 16 | 16 | 94.0% | 92.5% | 90.0% | 100.0% | 90.3% | 0.007 | 0.382 | -1.000 | 0.855 |
| main_factorized_cyclic_overlap | 24 | 24 | 93.6% | 92.2% | 90.0% | 100.0% | 90.3% | 0.007 | 0.261 | -1.000 | 0.461 |
| main_factorized_free_b_reg | 4 | 4 | 85.1% | 72.1% | 74.2% | 88.9% | 83.9% | 0.015 | 0.548 | -1.000 | 2.312 |
| main_factorized_free_b_reg | 8 | 8 | 82.0% | 73.5% | 74.2% | 89.1% | 83.9% | 0.015 | 0.426 | -1.000 | 1.838 |
| main_factorized_free_b_reg | 12 | 12 | 81.0% | 74.7% | 74.2% | 88.9% | 83.9% | 0.015 | 0.338 | -1.000 | 1.419 |
| main_factorized_free_b_reg | 16 | 16 | 80.9% | 75.7% | 74.2% | 89.1% | 83.9% | 0.015 | 0.274 | -1.000 | 1.181 |
| main_factorized_free_b_reg | 24 | 24 | 78.9% | 73.3% | 74.2% | 89.0% | 83.9% | 0.015 | 0.192 | -1.000 | 0.776 |
| main_generic_mlp_final_query | 4 | 4 | 63.7% | 26.6% | 29.6% | 53.1% | 48.2% | 0.047 | 0.281 | -1.000 | 2.524 |
| main_generic_mlp_final_query | 8 | 8 | 53.1% | 25.8% | 29.5% | 52.7% | 48.2% | 0.047 | 0.233 | -1.000 | 2.215 |
| main_generic_mlp_final_query | 12 | 12 | 45.0% | 23.6% | 29.5% | 53.2% | 48.2% | 0.047 | 0.205 | -1.000 | 1.976 |
| main_generic_mlp_final_query | 16 | 16 | 38.8% | 20.8% | 29.8% | 52.5% | 48.7% | 0.046 | 0.169 | -1.000 | 1.827 |
| main_generic_mlp_final_query | 24 | 24 | 30.5% | 16.7% | 29.4% | 52.6% | 48.2% | 0.047 | 0.129 | -1.000 | 1.709 |
| main_generic_mlp_full_belief | 4 | 4 | 70.8% | 47.3% | 47.0% | 85.3% | 50.9% | 0.114 | 0.752 | -1.000 | 2.090 |
| main_generic_mlp_full_belief | 8 | 8 | 66.5% | 52.2% | 47.0% | 84.8% | 50.9% | 0.115 | 0.705 | -1.000 | 1.801 |
| main_generic_mlp_full_belief | 12 | 12 | 65.9% | 55.8% | 46.6% | 85.1% | 50.6% | 0.116 | 0.664 | -1.000 | 1.471 |
| main_generic_mlp_full_belief | 16 | 16 | 66.6% | 58.2% | 46.9% | 84.9% | 50.9% | 0.115 | 0.603 | -1.000 | 1.273 |
| main_generic_mlp_full_belief | 24 | 24 | 62.8% | 55.5% | 46.4% | 84.5% | 50.4% | 0.119 | 0.482 | -1.000 | 0.928 |
| main_oracle_ceiling | 4 | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.476 | -1.000 | 2.128 |
| main_oracle_ceiling | 8 | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.309 | -1.000 | 1.527 |
| main_oracle_ceiling | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.208 | -1.000 | 0.986 |
| main_oracle_ceiling | 16 | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.175 | -1.000 | 0.741 |
| main_oracle_ceiling | 24 | 24 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.141 | -1.000 | 0.331 |
| main_sinkhorn_cyclic | 4 | 4 | 99.8% | 99.6% | 99.7% | 100.0% | 100.0% | 0.000 | 0.845 | -1.000 | 2.137 |
| main_sinkhorn_cyclic | 8 | 8 | 99.7% | 99.4% | 99.7% | 100.0% | 100.0% | 0.000 | 0.697 | -1.000 | 1.541 |
| main_sinkhorn_cyclic | 12 | 12 | 99.5% | 99.3% | 99.7% | 100.0% | 100.0% | 0.000 | 0.558 | -1.000 | 1.003 |
| main_sinkhorn_cyclic | 16 | 16 | 99.4% | 99.1% | 99.7% | 100.0% | 100.0% | 0.000 | 0.452 | -1.000 | 0.760 |
| main_sinkhorn_cyclic | 24 | 24 | 98.6% | 98.1% | 99.7% | 100.0% | 100.0% | 0.000 | 0.305 | -1.000 | 0.357 |

### pilot modulus 11

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot_factorized_cyclic_overlap | 3 | 3 | 99.7% | 99.4% | 99.5% | 100.0% | 100.0% | 0.000 | 0.789 | -1.000 | 1.501 |
| pilot_factorized_cyclic_overlap | 6 | 6 | 99.6% | 99.2% | 99.5% | 100.0% | 100.0% | 0.000 | 0.663 | -1.000 | 1.061 |
| pilot_factorized_cyclic_overlap | 9 | 9 | 99.4% | 99.1% | 99.5% | 100.0% | 100.0% | 0.000 | 0.570 | -1.000 | 0.640 |
| pilot_factorized_cyclic_overlap | 12 | 12 | 99.3% | 98.9% | 99.5% | 100.0% | 100.0% | 0.000 | 0.520 | -1.000 | 0.458 |
| pilot_factorized_free_b_reg | 3 | 3 | 88.7% | 72.0% | 76.3% | 92.5% | 81.8% | 0.039 | 0.524 | -1.000 | 1.521 |
| pilot_factorized_free_b_reg | 6 | 6 | 83.9% | 70.1% | 76.3% | 92.7% | 81.8% | 0.039 | 0.424 | -1.000 | 1.167 |
| pilot_factorized_free_b_reg | 9 | 9 | 79.9% | 69.6% | 76.3% | 92.5% | 81.8% | 0.039 | 0.375 | -1.000 | 0.848 |
| pilot_factorized_free_b_reg | 12 | 12 | 77.9% | 69.0% | 76.3% | 92.5% | 81.8% | 0.039 | 0.345 | -1.000 | 0.687 |
| pilot_generic_mlp_final_query | 3 | 3 | 86.9% | 65.0% | 68.4% | 89.2% | 69.8% | 0.084 | 0.773 | -1.000 | 1.272 |
| pilot_generic_mlp_final_query | 6 | 6 | 80.3% | 62.6% | 68.4% | 89.0% | 69.8% | 0.084 | 0.712 | -1.000 | 0.932 |
| pilot_generic_mlp_final_query | 9 | 9 | 74.6% | 61.5% | 68.3% | 89.5% | 69.6% | 0.085 | 0.682 | -1.000 | 0.670 |
| pilot_generic_mlp_final_query | 12 | 12 | 71.5% | 60.8% | 68.3% | 89.2% | 69.6% | 0.085 | 0.649 | -1.000 | 0.539 |
| pilot_generic_mlp_full_belief | 3 | 3 | 81.4% | 55.5% | 59.5% | 79.1% | 64.3% | 0.140 | 0.762 | -1.000 | 1.135 |
| pilot_generic_mlp_full_belief | 6 | 6 | 75.1% | 56.7% | 60.1% | 79.4% | 64.8% | 0.139 | 0.717 | -1.000 | 0.865 |
| pilot_generic_mlp_full_belief | 9 | 9 | 70.5% | 56.5% | 59.7% | 79.2% | 64.5% | 0.139 | 0.685 | -1.000 | 0.605 |
| pilot_generic_mlp_full_belief | 12 | 12 | 66.9% | 55.0% | 59.2% | 79.9% | 64.2% | 0.142 | 0.651 | -1.000 | 0.463 |
| pilot_oracle_ceiling | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.679 | -1.000 | 1.491 |
| pilot_oracle_ceiling | 6 | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.526 | -1.000 | 1.046 |
| pilot_oracle_ceiling | 9 | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.431 | -1.000 | 0.621 |
| pilot_oracle_ceiling | 12 | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 0.395 | -1.000 | 0.438 |
| pilot_sinkhorn_cyclic | 3 | 3 | 99.2% | 98.0% | 98.5% | 100.0% | 100.0% | 0.001 | 0.778 | -1.000 | 1.519 |
| pilot_sinkhorn_cyclic | 6 | 6 | 98.7% | 97.5% | 98.5% | 100.0% | 100.0% | 0.001 | 0.642 | -1.000 | 1.087 |
| pilot_sinkhorn_cyclic | 9 | 9 | 98.3% | 97.2% | 98.5% | 100.0% | 100.0% | 0.001 | 0.544 | -1.000 | 0.672 |
| pilot_sinkhorn_cyclic | 12 | 12 | 97.8% | 96.7% | 98.5% | 100.0% | 100.0% | 0.001 | 0.494 | -1.000 | 0.492 |

### scale modulus 97

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scale_oracle_init_only | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 | 1.000 | -1.000 | 4.575 |
| scale_sinkhorn_cyclic_init_only | 0 | 0 | 99.7% | 98.7% | 98.7% | 100.0% | 100.0% | 0.000 | 0.987 | -1.000 | 4.575 |

### smoke modulus 7

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke_factorized_cyclic_overlap | 2 | 2 | 94.2% | 83.3% | 85.3% | 100.0% | 85.7% | 0.048 | 0.781 | -1.000 | 1.434 |
| smoke_factorized_cyclic_overlap | 3 | 3 | 92.7% | 83.0% | 85.3% | 100.0% | 85.7% | 0.048 | 0.736 | -1.000 | 1.251 |
| smoke_factorized_cyclic_plain | 2 | 2 | 87.7% | 65.2% | 69.3% | 100.0% | 71.4% | nan | 0.574 | -1.000 | 1.214 |
| smoke_factorized_cyclic_plain | 3 | 3 | 85.4% | 64.4% | 69.3% | 100.0% | 71.4% | nan | 0.544 | -1.000 | 1.059 |
| smoke_factorized_cyclic_reg | 2 | 2 | 88.0% | 66.0% | 70.3% | 100.0% | 71.4% | nan | 0.670 | -1.000 | 1.408 |
| smoke_factorized_cyclic_reg | 3 | 3 | 85.7% | 65.1% | 70.3% | 100.0% | 71.4% | nan | 0.629 | -1.000 | 1.242 |
| smoke_factorized_free_b_reg | 2 | 2 | 87.1% | 63.1% | 67.0% | 91.3% | 85.7% | nan | 0.554 | -1.000 | 1.548 |
| smoke_factorized_free_b_reg | 3 | 3 | 83.8% | 62.5% | 67.0% | 91.1% | 85.7% | nan | 0.515 | -1.000 | 1.429 |
| smoke_generic_mlp | 2 | 2 | 94.9% | 84.5% | 85.8% | 90.0% | 88.2% | nan | 0.867 | -1.000 | 1.347 |
| smoke_generic_mlp | 3 | 3 | 93.3% | 84.7% | 85.4% | 90.8% | 88.0% | nan | 0.853 | -1.000 | 1.171 |
| smoke_generic_mlp_final_query | 2 | 2 | 91.5% | 75.1% | 77.5% | 89.7% | 79.2% | 0.079 | 0.842 | -1.000 | 1.361 |
| smoke_generic_mlp_final_query | 3 | 3 | 89.9% | 76.1% | 77.9% | 89.8% | 79.6% | 0.078 | 0.829 | -1.000 | 1.203 |
| smoke_indexed_cyclic | 2 | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | nan | 0.817 | -1.000 | 1.433 |
| smoke_indexed_cyclic | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | nan | 0.742 | -1.000 | 1.218 |
| smoke_oracle_ceiling | 2 | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | nan | 0.817 | -1.000 | 1.433 |
| smoke_oracle_ceiling | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | nan | 0.742 | -1.000 | 1.218 |
| smoke_sinkhorn_cyclic | 2 | 2 | 98.7% | 96.2% | 96.9% | 100.0% | 100.0% | 0.005 | 0.836 | -1.000 | 1.453 |
| smoke_sinkhorn_cyclic | 3 | 3 | 98.4% | 96.0% | 96.9% | 100.0% | 100.0% | 0.005 | 0.778 | -1.000 | 1.242 |

## Final Training Rows

| phase | modulus | variant | step | loss | decoder_query_target_mass | decoder_belief_target_mass | init_belief_target_mass | init_slot_relation_accuracy | init_slot_unique_a_frac | init_slot_a_overlap | mean_slot_purity | mean_route_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 31 | main_factorized_cyclic_overlap | 2400 | 3.840 | 95.1% | 90.9% | 90.0% | 100.0% | 90.3% | 0.007 | 0.730 | -1.000 |
| main | 31 | main_factorized_free_b_reg | 1600 | 4.228 | 85.9% | 74.5% | 74.2% | 88.9% | 83.9% | 0.015 | 0.542 | -1.000 |
| main | 31 | main_generic_mlp_final_query | 1400 | 2.509 | 66.9% | 28.2% | 29.4% | 52.0% | 48.3% | 0.047 | 0.274 | -1.000 |
| main | 31 | main_generic_mlp_full_belief | 1400 | 3.757 | 72.5% | 48.8% | 46.7% | 85.4% | 50.5% | 0.117 | 0.754 | -1.000 |
| main | 31 | main_sinkhorn_cyclic | 2400 | 2.915 | 99.8% | 99.6% | 99.7% | 100.0% | 100.0% | 0.000 | 0.817 | -1.000 |
| pilot | 11 | pilot_factorized_cyclic_overlap | 1400 | 2.438 | 99.7% | 99.3% | 99.5% | 100.0% | 100.0% | 0.000 | 0.757 | -1.000 |
| pilot | 11 | pilot_factorized_free_b_reg | 1100 | 3.126 | 87.6% | 74.1% | 76.3% | 92.9% | 81.8% | 0.039 | 0.528 | -1.000 |
| pilot | 11 | pilot_generic_mlp_final_query | 1100 | 1.369 | 84.7% | 66.5% | 67.4% | 89.4% | 68.6% | 0.089 | 0.770 | -1.000 |
| pilot | 11 | pilot_generic_mlp_full_belief | 1100 | 2.603 | 78.0% | 57.0% | 61.0% | 79.4% | 65.8% | 0.136 | 0.768 | -1.000 |
| pilot | 11 | pilot_sinkhorn_cyclic | 800 | 2.040 | 99.2% | 98.0% | 98.5% | 100.0% | 100.0% | 0.001 | 0.754 | -1.000 |
| scale | 97 | scale_sinkhorn_cyclic_init_only | 1200 | 5.048 | 99.5% | 98.7% | 98.7% | 100.0% | 100.0% | 0.000 | 0.987 | -1.000 |
| smoke | 7 | smoke_factorized_cyclic_overlap | 900 | 2.557 | 94.5% | 84.1% | 85.3% | 100.0% | 85.7% | 0.048 | 0.771 | -1.000 |
| smoke | 7 | smoke_factorized_cyclic_plain | 500 | 2.129 | 89.1% | 64.6% | 69.3% | 100.0% | 71.4% | nan | 0.578 | -1.000 |
| smoke | 7 | smoke_factorized_cyclic_reg | 500 | 2.758 | 89.3% | 65.4% | 70.3% | 100.0% | 71.4% | nan | 0.669 | -1.000 |
| smoke | 7 | smoke_factorized_free_b_reg | 500 | 2.945 | 88.0% | 64.2% | 67.0% | 92.0% | 85.7% | nan | 0.566 | -1.000 |
| smoke | 7 | smoke_generic_mlp | 500 | 1.862 | 95.4% | 84.1% | 84.9% | 90.0% | 87.2% | nan | 0.858 | -1.000 |
| smoke | 7 | smoke_generic_mlp_final_query | 500 | 1.331 | 93.2% | 76.8% | 78.2% | 89.6% | 79.9% | 0.076 | 0.846 | -1.000 |
| smoke | 7 | smoke_sinkhorn_cyclic | 500 | 1.854 | 98.9% | 96.4% | 96.9% | 100.0% | 100.0% | 0.005 | 0.838 | -1.000 |