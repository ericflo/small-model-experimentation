# On-Policy Repair-to-Compiler Analysis Summary

Primary run: `main_onpolicy_repair_s256`

## Final Metrics

| Split                  | Compiler | Local repair ceiling | Program exact | State prefix | Repair found |
| ---------------------- | -------- | -------------------- | ------------- | ------------ | ------------ |
| val_len24              | 96.9%    | 100.0%               | 96.9%         | 98.9%        | 100.0%       |
| fresh_standard_len24   | 99.6%    | 100.0%               | 99.6%         | 99.6%        | 100.0%       |
| fresh_paraphrase_len24 | 97.3%    | 100.0%               | 97.3%         | 98.2%        | 100.0%       |
| fresh_paired_len24     | 99.2%    | 100.0%               | 99.2%         | 99.6%        | 100.0%       |

## Baseline To Final

| Split                  | Baseline compiler | Final compiler | Baseline repair ceiling | Final repair ceiling |
| ---------------------- | ----------------- | -------------- | ----------------------- | -------------------- |
| val_len24              | 26.6%             | 96.9%          | 69.5%                   | 100.0%               |
| fresh_standard_len24   | 28.5%             | 99.6%          | 67.2%                   | 100.0%               |
| fresh_paraphrase_len24 | 28.9%             | 97.3%          | 66.8%                   | 100.0%               |
| fresh_paired_len24     | 29.1%             | 99.2%          | 64.5%                   | 100.0%               |

## Controls

| Run                       | Fresh paired compiler | Fresh paired repair ceiling | Program exact |
| ------------------------- | --------------------- | --------------------------- | ------------- |
| main_onpolicy_repair_s256 | 99.2%                 | 100.0%                      | 99.2%         |
| control_gold_only_s256    | 99.2%                 | 100.0%                      | 99.2%         |
| control_repair_only_s256  | 91.0%                 | 99.2%                       | 91.0%         |

