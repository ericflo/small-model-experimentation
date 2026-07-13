# Source

Experiment-local routing, replay, target-cache, loss, and training helpers.

Completion-target sequence fitting is deliberately centralized in
`training_units.fit_prompt_around_completion`. The entire generated completion
and its registered natural target positions are preserved; if the concatenated
sequence exceeds the frozen training `max_length`, only the oldest prompt
tokens are removed. Every cache and training receipt records the cut. A
completion that leaves no causal prompt token remains a hard failure.
