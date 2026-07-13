# Source

Experiment-local routing, replay, target-cache, loss, and training helpers.

Completion-target sequence fitting is deliberately centralized in
`training_units.fit_prompt_around_completion`. The entire generated completion
and its registered natural target positions are preserved; if the concatenated
sequence exceeds the frozen training `max_length`, only the oldest prompt
tokens are removed and the cut is receipted. That fitted representation may be
cached for a matched control, but `capability` and `anchor` cache creation fails
before policy scoring if either prefix was shortened. Every trainer independently
rejects shortened samples selected by its arm. A completion that leaves no
causal prompt token remains a hard failure.

All-policy cache provenance is checked at creation, orchestration resume, and
training. The receipt must bind the frozen config/top-k and the resolved quick,
deep, and soup model paths, model configs, and merge receipts.
