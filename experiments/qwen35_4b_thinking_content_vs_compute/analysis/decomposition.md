| rung | full-pass | visible-pass | best-layer probe AUC | what it adds vs the rung below |
| --- | ---: | ---: | ---: | --- |
| no_think | 0.764 | 0.802 | 0.682 | (baseline) |
| foreign | 0.043 | 0.043 | 0.994 | compute + scaffold (irrelevant thinking) |
| shuffle | 0.739 | 0.776 | 0.636 | token-presence / relevance (relevant tokens, scrambled) |
| real | 0.859 | 0.895 | 0.676 | coherent order |

## Decomposition of the behavioral thinking gain (full-pass)
- compute + scaffold (foreign - no_think): -0.721
- token-presence / relevance (shuffle - foreign): +0.696
- coherent order (real - shuffle): +0.120
- total (real - no_think): +0.095
