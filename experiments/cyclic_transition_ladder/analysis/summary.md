# Cyclic Transition Ladder Analysis Summary

## Runs

| phase | modulus | variant | slot_capacity | init_mode | transition_mode | supervision | train_steps | length | k | n |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 31 | main_cyclic_mixer | 31 | oracle | cyclic_mixer | full_belief | 800 | 24 | 24 | 512 |
| main | 31 | main_exact_ceiling | 31 | oracle | exact | full_belief | 0 | 24 | 24 | 512 |
| main | 31 | main_fourier_mlp | 31 | oracle | fourier_mlp | full_belief | 1200 | 24 | 24 | 512 |
| main | 31 | main_mlp | 31 | oracle | mlp | full_belief | 1200 | 24 | 24 | 512 |
| main | 31 | main_primitive_router | 31 | oracle | primitive_router | full_belief | 600 | 24 | 24 | 512 |
| pilot | 11 | pilot_cyclic_mixer | 11 | oracle | cyclic_mixer | full_belief | 600 | 12 | 12 | 512 |
| pilot | 11 | pilot_exact_ceiling | 11 | oracle | exact | full_belief | 0 | 12 | 12 | 512 |
| pilot | 11 | pilot_fourier_mlp | 11 | oracle | fourier_mlp | full_belief | 900 | 12 | 12 | 512 |
| pilot | 11 | pilot_mlp | 11 | oracle | mlp | full_belief | 900 | 12 | 12 | 512 |
| pilot | 11 | pilot_primitive_router | 11 | oracle | primitive_router | full_belief | 500 | 12 | 12 | 512 |
| scale | 97 | scale_cyclic_mixer | 97 | oracle | cyclic_mixer | full_belief | 300 | 24 | 24 | 128 |
| scale | 97 | scale_exact_ceiling | 97 | oracle | exact | full_belief | 0 | 24 | 24 | 128 |
| scale | 97 | scale_primitive_router | 97 | oracle | primitive_router | full_belief | 300 | 24 | 24 | 128 |
| smoke | 7 | smoke_cyclic_mixer | 7 | oracle | cyclic_mixer | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_exact_ceiling | 7 | oracle | exact | full_belief | 0 | 3 | 3 | 128 |
| smoke | 7 | smoke_fourier_mlp | 7 | oracle | fourier_mlp | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_mlp | 7 | oracle | mlp | full_belief | 500 | 3 | 3 | 128 |
| smoke | 7 | smoke_primitive_router | 7 | oracle | primitive_router | full_belief | 500 | 3 | 3 | 128 |

## First K >= L Summary

### main modulus 31

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|
| main_cyclic_mixer | 4 | 4 | 100.0% | 100.0% | 0.539 | 0.945 | 2.125 |
| main_cyclic_mixer | 8 | 8 | 100.0% | 100.0% | 0.386 | 0.944 | 1.456 |
| main_cyclic_mixer | 12 | 12 | 100.0% | 100.0% | 0.296 | 0.945 | 1.009 |
| main_cyclic_mixer | 16 | 16 | 100.0% | 100.0% | 0.235 | 0.940 | 0.681 |
| main_cyclic_mixer | 24 | 24 | 100.0% | 100.0% | 0.170 | 0.943 | 0.349 |
| main_exact_ceiling | 4 | 4 | 100.0% | 100.0% | 0.477 | -1.000 | 2.125 |
| main_exact_ceiling | 8 | 8 | 100.0% | 100.0% | 0.293 | -1.000 | 1.456 |
| main_exact_ceiling | 12 | 12 | 100.0% | 100.0% | 0.218 | -1.000 | 1.009 |
| main_exact_ceiling | 16 | 16 | 100.0% | 100.0% | 0.177 | -1.000 | 0.681 |
| main_exact_ceiling | 24 | 24 | 100.0% | 100.0% | 0.142 | -1.000 | 0.348 |
| main_fourier_mlp | 4 | 4 | 60.2% | 12.5% | 0.081 | -1.000 | 2.044 |
| main_fourier_mlp | 8 | 8 | 43.9% | 9.4% | 0.085 | -1.000 | 1.558 |
| main_fourier_mlp | 12 | 12 | 33.1% | 7.0% | 0.082 | -1.000 | 1.308 |
| main_fourier_mlp | 16 | 16 | 26.1% | 5.4% | 0.082 | -1.000 | 1.108 |
| main_fourier_mlp | 24 | 24 | 18.4% | 3.9% | 0.080 | -1.000 | 0.915 |
| main_mlp | 4 | 4 | 66.4% | 20.4% | 0.192 | -1.000 | 2.104 |
| main_mlp | 8 | 8 | 50.3% | 14.9% | 0.192 | -1.000 | 1.592 |
| main_mlp | 12 | 12 | 39.1% | 11.4% | 0.189 | -1.000 | 1.271 |
| main_mlp | 16 | 16 | 31.4% | 8.4% | 0.188 | -1.000 | 1.052 |
| main_mlp | 24 | 24 | 22.6% | 5.9% | 0.189 | -1.000 | 0.796 |
| main_primitive_router | 4 | 4 | 100.0% | 100.0% | 0.528 | 1.000 | 2.125 |
| main_primitive_router | 8 | 8 | 100.0% | 100.0% | 0.376 | 1.000 | 1.456 |
| main_primitive_router | 12 | 12 | 100.0% | 100.0% | 0.293 | 1.000 | 1.009 |
| main_primitive_router | 16 | 16 | 100.0% | 100.0% | 0.233 | 1.000 | 0.681 |
| main_primitive_router | 24 | 24 | 100.0% | 100.0% | 0.170 | 1.000 | 0.349 |

### pilot modulus 11

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|
| pilot_cyclic_mixer | 3 | 3 | 100.0% | 100.0% | 0.766 | 0.908 | 1.513 |
| pilot_cyclic_mixer | 6 | 6 | 100.0% | 100.0% | 0.663 | 0.910 | 0.997 |
| pilot_cyclic_mixer | 9 | 9 | 100.0% | 100.0% | 0.594 | 0.915 | 0.674 |
| pilot_cyclic_mixer | 12 | 12 | 100.0% | 100.0% | 0.550 | 0.912 | 0.452 |
| pilot_exact_ceiling | 3 | 3 | 100.0% | 100.0% | 0.685 | -1.000 | 1.513 |
| pilot_exact_ceiling | 6 | 6 | 100.0% | 100.0% | 0.505 | -1.000 | 0.997 |
| pilot_exact_ceiling | 9 | 9 | 100.0% | 100.0% | 0.434 | -1.000 | 0.674 |
| pilot_exact_ceiling | 12 | 12 | 100.0% | 100.0% | 0.399 | -1.000 | 0.451 |
| pilot_fourier_mlp | 3 | 3 | 60.6% | 17.1% | 0.111 | -1.000 | 1.254 |
| pilot_fourier_mlp | 6 | 6 | 38.4% | 8.3% | 0.068 | -1.000 | 1.156 |
| pilot_fourier_mlp | 9 | 9 | 28.4% | 6.2% | 0.064 | -1.000 | 1.113 |
| pilot_fourier_mlp | 12 | 12 | 23.1% | 5.1% | 0.068 | -1.000 | 1.133 |
| pilot_mlp | 3 | 3 | 73.7% | 35.7% | 0.315 | -1.000 | 1.374 |
| pilot_mlp | 6 | 6 | 55.1% | 24.2% | 0.300 | -1.000 | 1.033 |
| pilot_mlp | 9 | 9 | 44.8% | 19.1% | 0.301 | -1.000 | 0.825 |
| pilot_mlp | 12 | 12 | 38.2% | 16.4% | 0.309 | -1.000 | 0.694 |
| pilot_primitive_router | 3 | 3 | 100.0% | 100.0% | 0.759 | 1.000 | 1.513 |
| pilot_primitive_router | 6 | 6 | 100.0% | 100.0% | 0.657 | 1.000 | 0.997 |
| pilot_primitive_router | 9 | 9 | 100.0% | 100.0% | 0.592 | 1.000 | 0.674 |
| pilot_primitive_router | 12 | 12 | 100.0% | 100.0% | 0.546 | 1.000 | 0.451 |

### scale modulus 97

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|
| scale_cyclic_mixer | 4 | 4 | 100.0% | 100.0% | 0.505 | 1.000 | 3.066 |
| scale_cyclic_mixer | 8 | 8 | 100.0% | 100.0% | 0.318 | 1.000 | 2.193 |
| scale_cyclic_mixer | 12 | 12 | 100.0% | 99.9% | 0.223 | 1.000 | 1.667 |
| scale_cyclic_mixer | 16 | 16 | 99.9% | 99.9% | 0.161 | 1.000 | 1.183 |
| scale_cyclic_mixer | 24 | 24 | 99.9% | 99.8% | 0.099 | 1.000 | 0.753 |
| scale_exact_ceiling | 4 | 4 | 100.0% | 100.0% | 0.421 | -1.000 | 3.066 |
| scale_exact_ceiling | 8 | 8 | 100.0% | 100.0% | 0.193 | -1.000 | 2.193 |
| scale_exact_ceiling | 12 | 12 | 100.0% | 100.0% | 0.128 | -1.000 | 1.666 |
| scale_exact_ceiling | 16 | 16 | 100.0% | 100.0% | 0.089 | -1.000 | 1.181 |
| scale_exact_ceiling | 24 | 24 | 100.0% | 100.0% | 0.064 | -1.000 | 0.750 |
| scale_primitive_router | 4 | 4 | 100.0% | 100.0% | 0.468 | 1.000 | 3.066 |
| scale_primitive_router | 8 | 8 | 100.0% | 100.0% | 0.273 | 1.000 | 2.193 |
| scale_primitive_router | 12 | 12 | 100.0% | 100.0% | 0.194 | 1.000 | 1.666 |
| scale_primitive_router | 16 | 16 | 100.0% | 100.0% | 0.143 | 1.000 | 1.182 |
| scale_primitive_router | 24 | 24 | 99.9% | 99.9% | 0.088 | 1.000 | 0.751 |

### smoke modulus 7

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_route_accuracy | mean_weight_entropy |
|---|---|---|---|---|---|---|---|
| smoke_cyclic_mixer | 2 | 2 | 100.0% | 100.0% | 0.806 | 0.892 | 1.257 |
| smoke_cyclic_mixer | 3 | 3 | 100.0% | 100.0% | 0.748 | 0.896 | 0.958 |
| smoke_exact_ceiling | 2 | 2 | 100.0% | 100.0% | 0.761 | -1.000 | 1.257 |
| smoke_exact_ceiling | 3 | 3 | 100.0% | 100.0% | 0.651 | -1.000 | 0.958 |
| smoke_fourier_mlp | 2 | 2 | 73.0% | 36.5% | 0.268 | -1.000 | 0.977 |
| smoke_fourier_mlp | 3 | 3 | 56.8% | 24.5% | 0.218 | -1.000 | 0.791 |
| smoke_mlp | 2 | 2 | 83.9% | 57.0% | 0.461 | -1.000 | 0.943 |
| smoke_mlp | 3 | 3 | 70.3% | 45.3% | 0.427 | -1.000 | 0.716 |
| smoke_primitive_router | 2 | 2 | 100.0% | 100.0% | 0.806 | 1.000 | 1.257 |
| smoke_primitive_router | 3 | 3 | 100.0% | 100.0% | 0.752 | 1.000 | 0.958 |

## Final Training Rows

| phase | modulus | variant | step | loss | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_route_accuracy |
|---|---|---|---|---|---|---|---|---|
| main | 31 | main_cyclic_mixer | 800 | 2.538 | 100.0% | 100.0% | 0.546 | 0.942 |
| main | 31 | main_fourier_mlp | 1200 | 4.415 | 58.4% | 13.1% | 0.089 | -1.000 |
| main | 31 | main_mlp | 1200 | 4.015 | 64.8% | 22.9% | 0.221 | -1.000 |
| main | 31 | main_primitive_router | 600 | 2.562 | 100.0% | 100.0% | 0.541 | 1.000 |
| pilot | 11 | pilot_cyclic_mixer | 600 | 1.747 | 100.0% | 100.0% | 0.740 | 0.924 |
| pilot | 11 | pilot_fourier_mlp | 900 | 3.342 | 60.7% | 21.2% | 0.148 | -1.000 |
| pilot | 11 | pilot_mlp | 900 | 2.739 | 72.5% | 37.6% | 0.337 | -1.000 |
| pilot | 11 | pilot_primitive_router | 500 | 1.780 | 100.0% | 100.0% | 0.753 | 1.000 |
| scale | 97 | scale_cyclic_mixer | 300 | 3.708 | 100.0% | 100.0% | 0.538 | 1.000 |
| scale | 97 | scale_primitive_router | 300 | 3.708 | 100.0% | 100.0% | 0.514 | 1.000 |
| smoke | 7 | smoke_cyclic_mixer | 500 | 1.547 | 100.0% | 100.0% | 0.801 | 0.904 |
| smoke | 7 | smoke_fourier_mlp | 500 | 2.347 | 74.7% | 43.3% | 0.337 | -1.000 |
| smoke | 7 | smoke_mlp | 500 | 2.023 | 83.4% | 60.3% | 0.513 | -1.000 |
| smoke | 7 | smoke_primitive_router | 500 | 1.547 | 100.0% | 100.0% | 0.804 | 1.000 |