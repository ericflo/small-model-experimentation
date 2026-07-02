# Qwen3.5-4B Context Composition Experiment Log

## Design

Third installation mechanism (context), pre-registered. Same 120 verified tasks + identical 2AFC decoys
(seed 4242) as the keystone. Conditions: {base, SIM adapter} x {plain@1024, orchestrated, ICL} for 2AFC;
orchestrated generate-and-test for identification. SIM adapter regenerated (train_loss 0.021-equivalent).

## Results

2AFC raw/parse/parse-conditional: base plain .74/.94/.79; base orch .83/1.00/.83; base ICL .78/.97/.81;
SIM plain .46/.53/.87; SIM orch .51/.53/.95. Identification: base gen&test 0.08 (=bare), SIM 0.13 (bare .09).

- Context COMPOSES discrimination (procedure lifts base to 0.83, flat to d4 -- discrimination only needs
  partial simulation of the differing op).
- The trained module IS accessible in-context (+12pp parse-conditional over base under the same procedure)
  but FORMAT CAPTURE gates the interface (parse 0.53) -> raw 0.51. Module composes; interface captured.
- Hypothesis GENERATION un-composable: no context strategy moves identification (0.08-0.13).
- RETRO-CORRECTION: keystone P12 "thinking-2AFC at chance" was budget-512 + weak first-char parser; at
  1024 + strict format base = 0.74-0.79 ~ no-think logit. "Thinking hurts" retracted -> "doesn't help
  without a procedure".
- P-C1 refuted on raw / split by module-vs-interface; P-C2 direction confirmed (gain does NOT shrink with
  depth); P-C3 confirmed (+0.04); P-C4 confirmed; P-C5 raw negative / conditional positive.

Ops: smoke revealed answer-truncation (answer_max 700 -> 1100). SIM ident smoke 0.38 was a cell artifact
(first 8 tasks = d2k0). Claim C15; C13 P12 clause + C14 sealed-modules clause corrected.
