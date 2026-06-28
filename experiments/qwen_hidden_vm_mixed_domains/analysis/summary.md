# Qwen Hidden VM Mixed-Domain Analysis Summary

Primary run: `main_hidden_vm_trace_s512`

## Final Splits

| Split                  | Direct | Hidden VM | Program exact | State prefix | Pair both-correct |
| ---------------------- | ------ | --------- | ------------- | ------------ | ----------------- |
| val_mixed              | 13.2%  | 70.8%     | 63.9%         | 81.8%        | n/a               |
| fresh_standard_mixed   | 12.0%  | 72.9%     | 62.5%         | 79.4%        | n/a               |
| fresh_paraphrase_mixed | 13.0%  | 75.5%     | 63.5%         | 81.4%        | n/a               |
| fresh_paired_mixed     | 14.8%  | 77.7%     | 63.7%         | 81.0%        | 68.8%             |
| hard_standard_mixed    | 10.9%  | 50.0%     | 33.3%         | 68.6%        | n/a               |
| hard_paraphrase_mixed  | 11.5%  | 35.9%     | 19.3%         | 64.9%        | n/a               |
| domain_arithmetic      | 0.0%   | 65.6%     | 65.6%         | 80.5%        | n/a               |
| domain_calendar        | 28.1%  | 56.2%     | 56.2%         | 78.9%        | n/a               |
| domain_unit            | 3.1%   | 71.9%     | 71.9%         | 83.6%        | n/a               |
| domain_list            | 3.1%   | 71.9%     | 43.8%         | 80.5%        | n/a               |
| domain_boolean         | 46.9%  | 90.6%     | 81.2%         | 86.7%        | n/a               |
| domain_lookup          | 0.0%   | 87.5%     | 68.8%         | 79.7%        | n/a               |

## Headline

- Fresh paired direct logits: 14.8%
- Fresh paired trace hidden VM: 77.7% (+62.9 pp vs direct)
- Fresh paired answer-only hidden VM: 60.2% (+17.6 pp trace margin)
- Program exact: 63.7% trace vs 34.4% answer-only

## Fresh Paired Domain Breakdown

| Domain     | n     | Direct | Hidden VM |
| ---------- | ----- | ------ | --------- |
| arithmetic | 44.00 | 4.5%   | 84.1%     |
| calendar   | 44.00 | 22.7%  | 45.5%     |
| unit       | 42.00 | 0.0%   | 66.7%     |
| list       | 42.00 | 0.0%   | 88.1%     |
| boolean    | 42.00 | 61.9%  | 97.6%     |
| lookup     | 42.00 | 0.0%   | 85.7%     |

## Run Summary

| Run                                    | Variant     | Direct | Hidden VM | Program exact | State prefix |
| -------------------------------------- | ----------- | ------ | --------- | ------------- | ------------ |
| control_hidden_vm_answer_only_s512     | answer_only | 12.1%  | 60.2%     | 34.4%         | 58.1%        |
| main_hidden_vm_trace_s512              | trace       | 14.8%  | 77.7%     | 63.7%         | 81.0%        |
| pilot_hidden_vm_trace_balanced_l4_s256 | trace       | 14.6%  | 55.2%     | 49.0%         | 70.6%        |
| pilot_hidden_vm_trace_s160             | trace       | 14.1%  | 15.6%     | 7.8%          | 41.4%        |

