# Qwen Hidden VM Curriculum Repair Analysis Summary

Primary run: `main_curriculum_repair_s512`

## Final Splits

| Split                   | Direct | Hidden VM | Repair | Program exact | Repair exact | State prefix | Repair found |
| ----------------------- | ------ | --------- | ------ | ------------- | ------------ | ------------ | ------------ |
| val_mixed               | 12.5%  | 20.1%     | 97.9%  | 4.9%          | 36.8%        | 68.1%        | 97.9%        |
| fresh_standard_mixed    | 9.9%   | 28.1%     | 98.4%  | 15.6%         | 48.4%        | 72.7%        | 98.4%        |
| fresh_paraphrase_mixed  | 12.0%  | 25.0%     | 98.4%  | 8.9%          | 30.2%        | 64.8%        | 98.4%        |
| fresh_paired_mixed      | 10.9%  | 35.2%     | 99.6%  | 18.0%         | 47.3%        | 72.1%        | 99.6%        |
| hard_standard_mixed     | 10.9%  | 10.4%     | 94.8%  | 0.0%          | 4.7%         | 56.0%        | 94.8%        |
| hard_paraphrase_mixed   | 14.1%  | 14.6%     | 97.9%  | 0.0%          | 2.6%         | 52.0%        | 97.9%        |
| harder_standard_mixed   | 10.9%  | 9.4%      | 96.4%  | 0.0%          | 0.0%         | 44.1%        | 96.4%        |
| harder_paraphrase_mixed | 9.4%   | 8.9%      | 88.0%  | 0.0%          | 0.0%         | 40.6%        | 88.0%        |
| domain_arithmetic       | 0.0%   | 12.5%     | 100.0% | 9.4%          | 50.0%        | 66.7%        | 100.0%       |
| domain_calendar         | 9.4%   | 40.6%     | 100.0% | 31.2%         | 50.0%        | 78.1%        | 100.0%       |
| domain_unit             | 3.1%   | 9.4%      | 100.0% | 6.2%          | 50.0%        | 69.8%        | 100.0%       |
| domain_list             | 3.1%   | 53.1%     | 100.0% | 0.0%          | 6.2%         | 69.3%        | 100.0%       |
| domain_boolean          | 40.6%  | 37.5%     | 100.0% | 21.9%         | 50.0%        | 68.2%        | 100.0%       |
| domain_lookup           | 0.0%   | 12.5%     | 84.4%  | 3.1%          | 18.8%        | 61.5%        | 84.4%        |

## Headline

- Fresh paired direct logits: 10.9%
- Fresh paired hidden VM: 35.2% (+24.2 pp vs direct)
- Fresh paired verified repair: 99.6% (+64.5 pp repair headroom)
- Program exact: 18.0%

## Fresh Paired Domain Breakdown

| Domain     | n     | Direct | Hidden VM | Repair |
| ---------- | ----- | ------ | --------- | ------ |
| arithmetic | 44.00 | 0.0%   | 22.7%     | 97.7%  |
| calendar   | 44.00 | 18.2%  | 40.9%     | 100.0% |
| unit       | 42.00 | 0.0%   | 26.2%     | 100.0% |
| list       | 42.00 | 4.8%   | 69.0%     | 100.0% |
| boolean    | 42.00 | 42.9%  | 31.0%     | 100.0% |
| lookup     | 42.00 | 0.0%   | 21.4%     | 100.0% |

## Run Summary

| Run                                  | Variant | Direct | Hidden VM | Repair | Program exact | State prefix |
| ------------------------------------ | ------- | ------ | --------- | ------ | ------------- | ------------ |
| main_curriculum_repair_s512          | trace   | 10.9%  | 35.2%     | 99.6%  | 18.0%         | 72.1%        |
| main_curriculum_trace_s512           | trace   | 10.9%  | 71.9%     | 98.8%  | 55.9%         | 77.3%        |
| pilot_curriculum_repair_keep_l6_s256 | trace   | 12.5%  | 61.5%     | 100.0% | 45.8%         | 70.8%        |
| pilot_curriculum_repair_l6_s256      | trace   | 13.5%  | 45.8%     | 100.0% | 18.8%         | 55.0%        |

