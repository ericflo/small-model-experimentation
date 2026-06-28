# Learned Sparse Slot Analysis Summary

## Runs

| phase | modulus | variant | slot_capacity | init_mode | transition_mode | supervision | train_steps | length | k | n |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 31 | main_learned_init_exact_final_query | 31 | learned | exact | final_query | 900 | 24 | 24 | 512 |
| main | 31 | main_oracle_init_exact_ceiling | 31 | oracle | exact | full_belief | 0 | 24 | 24 | 512 |
| main | 31 | main_oracle_init_neural_full_belief | 31 | oracle | neural | full_belief | 1200 | 24 | 24 | 512 |
| pilot | 11 | pilot_learned_init_exact_final_query | 11 | learned | exact | final_query | 900 | 12 | 12 | 512 |
| pilot | 11 | pilot_learned_init_exact_full_belief | 11 | learned | exact | full_belief | 1400 | 12 | 12 | 512 |
| pilot | 11 | pilot_learned_init_neural_full_belief | 11 | learned | neural | full_belief | 1100 | 12 | 12 | 512 |
| pilot | 11 | pilot_oracle_init_neural_full_belief | 11 | oracle | neural | full_belief | 900 | 12 | 12 | 512 |
| smoke | 7 | smoke_learned_init_exact_final_query | 7 | learned | exact | final_query | 700 | 3 | 3 | 128 |
| smoke | 7 | smoke_learned_init_exact_transition | 7 | learned | exact | full_belief | 200 | 3 | 3 | 128 |
| smoke | 7 | smoke_learned_init_neural_transition | 7 | learned | neural | full_belief | 700 | 3 | 3 | 128 |
| smoke | 7 | smoke_oracle_init_exact_transition | 7 | oracle | exact | full_belief | 0 | 3 | 3 | 64 |
| smoke | 7 | smoke_oracle_init_neural_transition | 7 | oracle | neural | full_belief | 500 | 3 | 3 | 128 |

## First K >= L Summary

### main modulus 31

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_weight_entropy |
|---|---|---|---|---|---|---|
| main_learned_init_exact_final_query | 4 | 4 | 84.8% | 60.0% | 0.614 | 1.905 |
| main_learned_init_exact_final_query | 8 | 8 | 74.6% | 54.5% | 0.503 | 1.415 |
| main_learned_init_exact_final_query | 12 | 12 | 66.4% | 49.0% | 0.413 | 1.109 |
| main_learned_init_exact_final_query | 16 | 16 | 59.3% | 44.2% | 0.347 | 0.896 |
| main_learned_init_exact_final_query | 24 | 24 | 49.1% | 36.5% | 0.254 | 0.704 |
| main_oracle_init_exact_ceiling | 4 | 4 | 100.0% | 100.0% | 0.477 | 2.125 |
| main_oracle_init_exact_ceiling | 8 | 8 | 100.0% | 100.0% | 0.293 | 1.456 |
| main_oracle_init_exact_ceiling | 12 | 12 | 100.0% | 100.0% | 0.218 | 1.009 |
| main_oracle_init_exact_ceiling | 16 | 16 | 100.0% | 100.0% | 0.177 | 0.681 |
| main_oracle_init_exact_ceiling | 24 | 24 | 100.0% | 100.0% | 0.142 | 0.348 |
| main_oracle_init_neural_full_belief | 4 | 4 | 74.1% | 34.4% | 0.308 | 2.162 |
| main_oracle_init_neural_full_belief | 8 | 8 | 60.3% | 27.5% | 0.314 | 1.569 |
| main_oracle_init_neural_full_belief | 12 | 12 | 49.2% | 21.3% | 0.307 | 1.196 |
| main_oracle_init_neural_full_belief | 16 | 16 | 41.2% | 17.2% | 0.308 | 0.926 |
| main_oracle_init_neural_full_belief | 24 | 24 | 30.7% | 12.3% | 0.309 | 0.636 |

### pilot modulus 11

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_weight_entropy |
|---|---|---|---|---|---|---|
| pilot_learned_init_exact_final_query | 3 | 3 | 89.3% | 73.0% | 0.809 | 1.328 |
| pilot_learned_init_exact_final_query | 6 | 6 | 82.7% | 69.5% | 0.735 | 0.925 |
| pilot_learned_init_exact_final_query | 9 | 9 | 80.1% | 69.7% | 0.694 | 0.696 |
| pilot_learned_init_exact_final_query | 12 | 12 | 78.9% | 70.2% | 0.662 | 0.521 |
| pilot_learned_init_exact_full_belief | 3 | 3 | 84.1% | 60.9% | 0.763 | 1.204 |
| pilot_learned_init_exact_full_belief | 6 | 6 | 75.3% | 57.6% | 0.694 | 0.846 |
| pilot_learned_init_exact_full_belief | 9 | 9 | 72.3% | 58.1% | 0.666 | 0.644 |
| pilot_learned_init_exact_full_belief | 12 | 12 | 70.2% | 58.1% | 0.639 | 0.499 |
| pilot_learned_init_neural_full_belief | 3 | 3 | 84.2% | 58.5% | 0.561 | 1.504 |
| pilot_learned_init_neural_full_belief | 6 | 6 | 72.4% | 49.7% | 0.552 | 1.112 |
| pilot_learned_init_neural_full_belief | 9 | 9 | 64.9% | 43.6% | 0.556 | 0.881 |
| pilot_learned_init_neural_full_belief | 12 | 12 | 57.8% | 39.3% | 0.560 | 0.722 |
| pilot_oracle_init_neural_full_belief | 3 | 3 | 98.4% | 95.6% | 0.929 | 1.512 |
| pilot_oracle_init_neural_full_belief | 6 | 6 | 97.6% | 95.5% | 0.931 | 0.992 |
| pilot_oracle_init_neural_full_belief | 9 | 9 | 97.2% | 95.4% | 0.932 | 0.667 |
| pilot_oracle_init_neural_full_belief | 12 | 12 | 97.1% | 95.5% | 0.931 | 0.448 |

### smoke modulus 7

| variant | length | first_k_ge_l | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity | mean_weight_entropy |
|---|---|---|---|---|---|---|
| smoke_learned_init_exact_final_query | 2 | 2 | 91.2% | 77.3% | 0.841 | 1.224 |
| smoke_learned_init_exact_final_query | 3 | 3 | 88.0% | 76.8% | 0.830 | 0.982 |
| smoke_learned_init_exact_transition | 2 | 2 | 90.0% | 75.7% | 0.848 | 1.199 |
| smoke_learned_init_exact_transition | 3 | 3 | 87.5% | 75.8% | 0.843 | 0.980 |
| smoke_learned_init_neural_transition | 2 | 2 | 90.5% | 72.8% | 0.665 | 1.096 |
| smoke_learned_init_neural_transition | 3 | 3 | 83.6% | 68.3% | 0.647 | 0.842 |
| smoke_oracle_init_exact_transition | 2 | 2 | 100.0% | 100.0% | 0.743 | 1.207 |
| smoke_oracle_init_exact_transition | 3 | 3 | 100.0% | 100.0% | 0.679 | 1.018 |
| smoke_oracle_init_neural_transition | 2 | 2 | 93.9% | 84.4% | 0.752 | 1.146 |
| smoke_oracle_init_neural_transition | 3 | 3 | 90.8% | 82.2% | 0.734 | 0.829 |

## Final Training Rows

| phase | modulus | variant | step | loss | decoder_query_target_mass | decoder_belief_target_mass | mean_slot_purity |
|---|---|---|---|---|---|---|---|
| main | 31 | main_learned_init_exact_final_query | 900 | 2.032 | 82.2% | 59.3% | 0.620 |
| main | 31 | main_oracle_init_neural_full_belief | 1200 | 3.611 | 71.7% | 35.4% | 0.339 |
| pilot | 11 | pilot_learned_init_exact_final_query | 900 | 1.447 | 86.2% | 69.5% | 0.791 |
| pilot | 11 | pilot_learned_init_exact_full_belief | 1400 | 2.521 | 81.4% | 60.2% | 0.745 |
| pilot | 11 | pilot_learned_init_neural_full_belief | 1100 | 2.561 | 83.5% | 59.5% | 0.576 |
| pilot | 11 | pilot_oracle_init_neural_full_belief | 900 | 1.814 | 98.2% | 95.5% | 0.932 |
| smoke | 7 | smoke_learned_init_exact_final_query | 700 | 1.159 | 90.3% | 77.2% | 0.840 |
| smoke | 7 | smoke_learned_init_exact_transition | 200 | 2.036 | 91.0% | 73.3% | 0.842 |
| smoke | 7 | smoke_learned_init_neural_transition | 700 | 1.867 | 90.2% | 73.8% | 0.674 |
| smoke | 7 | smoke_oracle_init_neural_transition | 500 | 1.671 | 93.8% | 84.8% | 0.770 |