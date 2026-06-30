| rung | full-pass | visible-pass | best-layer probe AUC | description |
| --- | ---: | ---: | ---: | --- |
| no_think | 0.749 | 0.806 | 0.646 | no thinking (baseline) |
| filler | 0.744 | 0.789 | 0.703 | pure compute + scaffold (contentless '.' tokens) |
| foreign | 0.040 | 0.041 | 0.987 | off-ladder: a DIFFERENT task's thinking (misleading content) |
| shuffle | 0.739 | 0.781 | 0.645 | relevant tokens, scrambled order |
| real | 0.861 | 0.902 | 0.722 | relevant tokens, coherent order |

## Decomposition (additive ladder no_think -> filler -> shuffle -> real; full-pass)
- pure compute + scaffold (filler - no_think): -0.005
- token-presence / relevance (shuffle - filler): -0.005
- coherent order (real - shuffle): +0.122
- total (real - no_think): +0.112
- [off-ladder] misleading content (foreign - no_think): -0.709
