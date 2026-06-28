# Backlog

## Next Experiments

- Replicate the strongest structural compiler results across seeds, lengths, and operator mixes.
- Run a direct-text-program versus typed-bytecode versus latent-slot comparison on one shared task suite.
- Add adversarial paraphrase and compositional splits where direct prompt cues fail.
- Measure whether state-prefix supervision, final-answer supervision, or program-token supervision is the causal lift.
- Build a small diagnostic suite that every compiler-style experiment can run before claiming generalization.

## Required Controls

- Direct answer baseline.
- Same model and data with unstructured output.
- Shuffled or corrupted state supervision when state traces are used.
- Length and family holdouts.

## Stop Conditions

Retire a variant when it improves train or IID accuracy but cannot survive harder length/family/paraphrase splits after two controlled attempts.
