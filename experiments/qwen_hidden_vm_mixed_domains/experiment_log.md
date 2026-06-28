# Experiment Log

## 2026-06-23

Created standalone mixed-domain hidden VM experiment.

Initial design:

- Use a fixed typed VM with ten operations: add, subtract, multiply, calendar add/subtract modulo 7, set, max, min, xor, and greater-than.
- Generate exact hidden traces for six task families: arithmetic chains, calendar shifts, unit-style transforms, list aggregation, boolean thresholding, and lookup/adjust rules.
- Train Qwen 3 4B with QLoRA plus a compiler head to emit VM slots from prompts.
- Compare trace-supervised hidden VM training against answer-only executor training.
- Evaluate direct answer logits, compiler execution accuracy, exact program match, state-prefix accuracy, per-domain accuracy, fresh paraphrases, paired standard/paraphrase prompts, and longer/harder programs.

Next steps:

- Implement the standalone trainer and analyzer.
- Run a tiny smoke test.
- If the smoke passes, run a pilot to tune training steps and loss weights.
- Run main and answer-only control runs.
- Generate charts, markdown, HTML, and artifact manifest.

Smoke and first pilot notes:

- `smoke_hidden_vm_mixed` passed end-to-end after adding a guard that `max_steps` must cover train, eval, and hard program lengths.
- `pilot_hidden_vm_trace_s160` did not improve fresh mixed-domain executor accuracy. The useful signal was diagnostic rather than positive: initialization was learned, operation slots partially learned, and argument slots were weak.
- A lookup-domain bug was found: the selected table value could appear only before the `SET` operation while the compiler's argument reader is intentionally local after each operation token. The renderer now repeats the selected value as the explicit `SET` argument.
- Dataset construction now balances domains by cycling through the configured domain list before shuffling, so pilots cannot accidentally under-sample a hard domain.

Repaired pilot plan:

```bash
python experiments/qwen_hidden_vm_mixed_domains/src/qwen_hidden_vm_mixed_domains_experiment.py \
  --run_name pilot_hidden_vm_trace_balanced_l4_s256 --variant trace \
  --train_examples 256 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 6 --train_max_len 4 --eval_length 4 --hard_length 6 --max_length 512 \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 25 --seed 94 --eval_seed 94001
```

Repaired pilot result:

- `pilot_hidden_vm_trace_balanced_l4_s256` completed in 184.9 seconds.
- Fresh paired hidden-VM executor accuracy improved from 24.0% at initialization to 55.2%; direct numeric-token accuracy finished at 14.6%.
- Fresh paired program-exact accuracy reached 49.0%; state-prefix fraction reached 70.6%.
- Per-domain fresh paired hidden-VM accuracy: arithmetic 62.5%, calendar 31.2%, unit 56.2%, list 100.0%, boolean 50.0%, lookup 31.2%.
- Hard length-6 transfer remained weak: 18.1% on standard prompts and 29.2% on paraphrases.

Main trace run plan:

```bash
python experiments/qwen_hidden_vm_mixed_domains/src/qwen_hidden_vm_mixed_domains_experiment.py \
  --run_name main_hidden_vm_trace_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 520 --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 6 --train_max_len 4 --eval_length 4 --hard_length 6 --max_length 512 \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 65 --seed 95 --eval_seed 95001
```

Main trace result:

- `main_hidden_vm_trace_s512` completed in 333.2 seconds.
- Fresh paired hidden-VM executor accuracy reached 77.7%; direct numeric-token accuracy finished at 14.8%.
- Fresh paired program-exact accuracy reached 63.7%; state-prefix fraction reached 81.0%.
- Fresh standard and paraphrase hidden-VM executor accuracy reached 72.9% and 75.5%.
- Hard length-6 transfer improved to 50.0% on standard prompts and 35.9% on paraphrases.
- Per-domain mixed hidden-VM accuracy: arithmetic 65.6%, calendar 56.2%, unit 71.9%, list 71.9%, boolean 90.6%, lookup 87.5%.

Matched answer-only control plan:

```bash
python experiments/qwen_hidden_vm_mixed_domains/src/qwen_hidden_vm_mixed_domains_experiment.py \
  --run_name control_hidden_vm_answer_only_s512 --variant answer_only \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 520 --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 6 --train_max_len 4 --eval_length 4 --hard_length 6 --max_length 512 \
  --lr 5e-5 --executor_loss_weight 1.0 --state_loss_weight 0.0 --direct_answer_loss_weight 0.0 \
  --log_every 65 --seed 96 --eval_seed 95001
```

Matched answer-only control result:

- `control_hidden_vm_answer_only_s512` completed in 358.5 seconds.
- Fresh paired hidden-VM executor accuracy reached 60.2%; direct numeric-token accuracy finished at 12.1%.
- Fresh paired program-exact accuracy reached 34.4%; state-prefix fraction reached 58.1%.
- Hard length-6 transfer reached 36.5% on standard prompts and 37.0% on paraphrases.
- The control is materially better than chance, so final-answer gradients can discover a useful executable policy. Trace supervision still adds +17.6 percentage points on fresh paired execution, +29.3 percentage points on program exactness, and +22.9 percentage points on state-prefix recovery.

Analysis artifacts:

- Generated `analysis/summary.md`.
- Generated `reports/qwen_hidden_vm_mixed_domains_paper.md`.
- Generated `reports/qwen_hidden_vm_mixed_domains_paper.html`.
- Generated figures: `split_accuracy.png`, `domain_accuracy.png`, `training_curve.png`, and `run_summary.png`.
- Generated `checkpoint_manifest.csv` pointing to large checkpoint artifacts under `large_artifacts/qwen_hidden_vm_mixed_domains/checkpoints/`.

Current interpretation:

This is a positive result for the Qwen-attached hidden-executor direction. The trace-supervised hidden VM improves fresh paired accuracy from 14.8% direct next-token answering to 77.7% executable hidden-program answering. The answer-only executable control reaches 60.2%, which is also important: the runtime architecture itself is useful, but dense trace supervision makes the compiled programs much more reliable and inspectable.

The remaining weakness is length-generalization. Training on length 1-4 transfers to length 6, but not cleanly enough: 50.0% on standard hard prompts and 35.9% on paraphrased hard prompts.
