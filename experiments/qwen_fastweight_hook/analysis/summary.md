# Latent Fast-Weight Experiment Summary

## Large Retests

| run                  | checkpoint                                                 | split   |   k |   k_num |   accuracy |   n |   ci_low |   ci_high |
|:---------------------|:-----------------------------------------------------------|:--------|----:|--------:|-----------:|----:|---------:|----------:|
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | val     |   0 |       0 |      0.208 | 250 | 0.162293 |  0.262545 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | val     |   1 |       1 |      0.192 | 250 | 0.147983 |  0.245339 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | val     |   2 |       2 |      0.184 | 250 | 0.140875 |  0.23669  |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | val     |   4 |       4 |      0.192 | 250 | 0.147983 |  0.245339 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | val     |   8 |       8 |      0.2   | 250 | 0.155123 |  0.253957 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | hard    |   0 |       0 |      0.26  | 250 | 0.209549 |  0.317715 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | hard    |   1 |       1 |      0.24  | 250 | 0.191248 |  0.296622 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | hard    |   2 |       2 |      0.236 | 250 | 0.187606 |  0.292385 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | hard    |   4 |       4 |      0.236 | 250 | 0.187606 |  0.292385 |
| Full step 200 retest | runs/main_qwen35_hook_full_seed7/latent_adapter_step200.pt | hard    |   8 |       8 |      0.244 | 250 | 0.194896 |  0.300852 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | val     |   0 |       0 |      0.176 | 250 | 0.133799 |  0.228008 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | val     |   2 |       2 |      0.152 | 250 | 0.112787 |  0.201746 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | val     |   4 |       4 |      0.168 | 250 | 0.126758 |  0.219291 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | val     |   8 |       8 |      0.156 | 250 | 0.116265 |  0.206147 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | hard    |   0 |       0 |      0.212 | 250 | 0.165889 |  0.266828 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | hard    |   2 |       2 |      0.228 | 250 | 0.180341 |  0.283892 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | hard    |   4 |       4 |      0.204 | 250 | 0.158705 |  0.258255 |
| Aux final retest     | runs/main_qwen35_hook_aux02_seed7/latent_adapter.pt        | hard    |   8 |       8 |      0.228 | 250 | 0.180341 |  0.283892 |

## K Gain Summary

| run              | split   | step          | best_k_gt0_gain_pp                   |
|:-----------------|:--------|:--------------|:-------------------------------------|
| Full recurrent   | val     | 0             | 0.0                                  |
| Full recurrent   | val     | 50            | 3.0000000000000027                   |
| Full recurrent   | val     | 100           | 2.0000000000000018                   |
| Full recurrent   | val     | 150           | 6.000000000000003                    |
| Full recurrent   | val     | 200           | 4.000000000000001                    |
| Full recurrent   | val     | 250           | 6.0                                  |
| Full recurrent   | val     | 300           | -2.0000000000000018                  |
| Full recurrent   | val     | best observed | best accuracy 28.0% at step 150, K=2 |
| Full recurrent   | hard    | 0             | 0.9999999999999953                   |
| Full recurrent   | hard    | 50            | 2.0000000000000018                   |
| Full recurrent   | hard    | 100           | -1.0000000000000009                  |
| Full recurrent   | hard    | 150           | 3.0000000000000027                   |
| Full recurrent   | hard    | 200           | 5.000000000000002                    |
| Full recurrent   | hard    | 250           | 0.0                                  |
| Full recurrent   | hard    | 300           | 0.0                                  |
| Full recurrent   | hard    | best observed | best accuracy 32.0% at step 150, K=1 |
| K=0-only control | val     | 100           | 0.0                                  |
| K=0-only control | val     | 200           | 1.9999999999999991                   |
| K=0-only control | val     | 300           | 0.0                                  |
| K=0-only control | val     | best observed | best accuracy 25.0% at step 100, K=0 |
| K=0-only control | hard    | 100           | 3.0                                  |
| K=0-only control | hard    | 200           | 3.9999999999999982                   |
| K=0-only control | hard    | 300           | 1.0000000000000009                   |
| K=0-only control | hard    | best observed | best accuracy 30.0% at step 200, K=4 |
| Aux value loss   | val     | 100           | 5.000000000000002                    |
| Aux value loss   | val     | 200           | 0.0                                  |
| Aux value loss   | val     | 300           | 3.0                                  |
| Aux value loss   | val     | best observed | best accuracy 23.0% at step 100, K=2 |
| Aux value loss   | hard    | 100           | 0.0                                  |
| Aux value loss   | hard    | 200           | -2.0000000000000018                  |
| Aux value loss   | hard    | 300           | 5.000000000000002                    |
| Aux value loss   | hard    | best observed | best accuracy 28.0% at step 200, K=0 |
