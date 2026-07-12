# Terminal analysis summary

The context-local intervention produced perfect semantic transport, but the
frozen terminal decision is `INVALID_CONTROL`.

| confirmation arm | direct target | mapped target digit |
| --- | ---: | ---: |
| baseline | 0/48 | 0/48 |
| full donor | 48/48 | 48/48 |
| all-24 J clamp | 48/48 | 48/48 |
| pair J clamp | 48/48 | 47/48 |
| wrong-donor J | 0/48 target; 48/48 wrong | 0/48 target; 48/48 wrong |
| concept logit lens | 0/48 | 0/48 |
| random orthogonal control | 0/48 | 0/48 |

One of 96 random-control rows missed the exact realized-norm tolerance:
`confirm-0046` consequence, 1.155e-5 versus the 1e-5 maximum. Therefore the
perfect J-minus-random effect cannot be promoted. Replicate on fresh mappings
with simultaneous post-bf16 norm and span-orthogonality constraints.
