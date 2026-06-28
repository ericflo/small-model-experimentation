# Qwen Hidden VM On-Policy Canonical Repair Analysis Summary

Primary run: `main_repair_or_gold_s512`

## Final Splits

| Split                   | Direct | Hidden VM | Repair | Program exact | Repair exact | State prefix | Repair found |
| ----------------------- | ------ | --------- | ------ | ------------- | ------------ | ------------ | ------------ |
| val_mixed               | 13.9%  | 58.3%     | 95.1%  | 41.7%         | 83.3%        | 70.8%        | 93.1%        |
| fresh_standard_mixed    | 7.3%   | 72.9%     | 96.4%  | 56.8%         | 87.5%        | 76.3%        | 94.8%        |
| fresh_paraphrase_mixed  | 12.5%  | 62.0%     | 91.7%  | 46.9%         | 80.7%        | 74.5%        | 90.1%        |
| fresh_paired_mixed      | 8.6%   | 60.9%     | 89.1%  | 44.5%         | 73.4%        | 72.8%        | 88.3%        |
| hard_standard_mixed     | 9.9%   | 53.1%     | 85.4%  | 32.3%         | 64.1%        | 67.5%        | 83.9%        |
| hard_paraphrase_mixed   | 12.0%  | 27.6%     | 64.1%  | 8.3%          | 39.6%        | 60.5%        | 61.5%        |
| harder_standard_mixed   | 9.4%   | 31.8%     | 65.6%  | 11.5%         | 39.6%        | 60.7%        | 60.4%        |
| harder_paraphrase_mixed | 10.4%  | 16.7%     | 33.9%  | 2.1%          | 10.9%        | 49.9%        | 26.0%        |
| domain_arithmetic       | 3.1%   | 34.4%     | 90.6%  | 34.4%         | 90.6%        | 62.5%        | 90.6%        |
| domain_calendar         | 12.5%  | 46.9%     | 100.0% | 37.5%         | 75.0%        | 72.4%        | 100.0%       |
| domain_unit             | 0.0%   | 37.5%     | 84.4%  | 37.5%         | 84.4%        | 66.7%        | 84.4%        |
| domain_list             | 0.0%   | 81.2%     | 93.8%  | 46.9%         | 53.1%        | 82.8%        | 90.6%        |
| domain_boolean          | 46.9%  | 93.8%     | 100.0% | 53.1%         | 93.8%        | 72.9%        | 96.9%        |
| domain_lookup           | 0.0%   | 71.9%     | 93.8%  | 43.8%         | 62.5%        | 71.4%        | 90.6%        |

## Headline

- Fresh paired direct logits: 8.6%
- Fresh paired hidden VM: 60.9% (+52.3 pp vs direct)
- Fresh paired verified repair: 89.1% (+28.1 pp repair headroom)
- Program exact: 44.5%

## Fresh Paired Domain Breakdown

| Domain     | n     | Direct | Hidden VM | Repair |
| ---------- | ----- | ------ | --------- | ------ |
| arithmetic | 44.00 | 0.0%   | 40.9%     | 79.5%  |
| calendar   | 44.00 | 4.5%   | 45.5%     | 86.4%  |
| unit       | 42.00 | 0.0%   | 61.9%     | 85.7%  |
| list       | 42.00 | 2.4%   | 81.0%     | 100.0% |
| boolean    | 42.00 | 45.2%  | 85.7%     | 95.2%  |
| lookup     | 42.00 | 0.0%   | 52.4%     | 88.1%  |

## Run Summary

| Run                                          | Variant | Direct | Hidden VM | Repair | Program exact | State prefix |
| -------------------------------------------- | ------- | ------ | --------- | ------ | ------------- | ------------ |
| main_gold_control_s512                       | trace   | 8.6%   | 60.9%     | 89.1%  | 44.5%         | 72.8%        |
| main_repair_only_s512                        | trace   | 8.6%   | 60.9%     | 89.1%  | 44.5%         | 72.8%        |
| main_repair_or_gold_s512                     | trace   | 8.6%   | 60.9%     | 89.1%  | 44.5%         | 72.8%        |
| main_trace_control_s512                      | trace   | 8.6%   | 59.0%     | 89.5%  | 44.1%         | 71.0%        |
| pilot_onpolicy_canonical_headonly_cap24_s192 | trace   | 8.3%   | 60.4%     | 86.5%  | 42.7%         | 71.0%        |
| pilot_onpolicy_canonical_headonly_s192       | trace   | 8.3%   | 63.5%     | 91.7%  | 46.9%         | 72.0%        |
| pilot_onpolicy_canonical_repair_s192         | trace   | 8.3%   | 50.0%     | 96.9%  | 37.5%         | 69.3%        |
| pilot_onpolicy_canonical_stable_s192         | trace   | 8.3%   | 47.9%     | 91.7%  | 34.4%         | 68.8%        |

