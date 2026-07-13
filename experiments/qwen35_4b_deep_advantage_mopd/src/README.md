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

The non-advantage-route control has an additional full-prefix overlay path.
`route_control_matching.py` defines the frozen tier/identity order, while
`control_rematch.py` replays that matcher after filtering candidates solely by
whether the complete observed prefix and completion fit `max_length`. The
unfiltered replay must reproduce the original manifest, and the filtered replay
may change only controls whose cached prompt was shortened. The derived cache
keeps the primary manifest/cache immutable, copies every unaffected sample at
the same index, and rescales no target: only replacement states are scored
again under the same quick/deep/soup policies. Selection and cache receipts
bind all candidate artifacts, tokenizer files, source hashes, replacement
identities, copied-sample semantic hashes, and the zero-truncation inventory.
