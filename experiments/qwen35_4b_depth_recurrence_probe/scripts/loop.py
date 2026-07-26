"""Cache-safe layer looping for Qwen3.5-4B: weight-shared depth that works with KV/state caching.

WHY THIS MODULE EXISTS. The first looping implementation (recur.py) duplicates entries of
`model.model.layers`, which is correct for a SINGLE forward pass with `use_cache=False` but cannot
generate. Qwen3.5 indexes per-layer cache state by an integer stored on the layer's inner module:
full-attention layers call `past_key_values.update(k, v, self.self_attn.layer_idx)`, and gated-delta-net
layers read/write `cache_params.layers[self.linear_attn.layer_idx].{conv,recurrent}_states`. Duplicating
a layer object reuses its index, so the second pass would overwrite the first pass's cache entry — the
model would silently generate from corrupted state.

That restriction is what confined the probe to forced single-token reads. Lifting it unlocks the three
things that decide whether looping is real rather than a toy-substrate artifact: generation-mode
evaluation (does looping STACK with chain-of-thought or merely substitute for it?), standard coding
benchmarks, and eventually a servable path.

HOW. Each duplicated position gets a SHALLOW module copy with a fresh `layer_idx`, sharing every
Parameter with the original — so k=3 over an 8-layer block costs no extra weights, only cache slots.
`copy.copy` on an nn.Module gives a new object whose `__dict__` is a fresh dict but whose `_modules`,
`_parameters` and `_buffers` are the SAME dict objects; assigning a submodule on the copy would
therefore mutate the original. `_shallow_copy` rebinds those containers first, which is the subtle part.

`config.num_hidden_layers` must also grow, because the decoder iterates
`enumerate(self.layers[: self.config.num_hidden_layers])` and silently truncates otherwise (that bug
produced a fake +0.125 and a 6-nat coherence collapse before it was caught), and `config.layer_types`
must grow in lockstep because masks are selected per position from it.
"""
from __future__ import annotations

import copy

from torch import nn


def _shallow_copy(mod: nn.Module) -> nn.Module:
    """A new module object sharing all parameters/buffers/children, with independent containers.

    Every mutable container nn.Module keeps in __dict__ must be rebound, not just the obvious three.
    `copy.copy` gives the new object a fresh __dict__ whose VALUES are the originals, so a shared
    `_forward_hooks` dict means a hook registered on one clone fires for every clone that shares it --
    measured: instrumentation counted 64 layer executions where 40 ran, which would have corrupted the
    depth gate and any future hook-based probe (steering, activation capture, damping).
    """
    new = copy.copy(mod)
    for attr in ("_parameters", "_buffers", "_modules", "_forward_hooks", "_forward_pre_hooks",
                 "_backward_hooks", "_backward_pre_hooks", "_state_dict_hooks",
                 "_load_state_dict_pre_hooks", "_state_dict_pre_hooks",
                 "_load_state_dict_post_hooks", "_forward_hooks_with_kwargs",
                 "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called"):
        cur = getattr(mod, attr, None)
        if isinstance(cur, dict):
            setattr(new, attr, type(cur)(cur) if type(cur) is not dict else dict(cur))
        elif isinstance(cur, set):
            setattr(new, attr, set(cur))
    new._non_persistent_buffers_set = set(mod._non_persistent_buffers_set)
    return new


def _clone_layer(layer: nn.Module, new_idx: int) -> nn.Module:
    """Clone a decoder layer for a new stack position: shared weights, fresh cache index."""
    new = _shallow_copy(layer)
    for attr in ("self_attn", "linear_attn"):
        inner = getattr(layer, attr, None)
        if inner is not None:
            inner_copy = _shallow_copy(inner)
            inner_copy.layer_idx = new_idx
            setattr(new, attr, inner_copy)
    if hasattr(layer, "layer_idx"):
        new.layer_idx = new_idx
    return new


class CacheSafeLoop:
    """Context manager: loop layers[a:b] k times with correct cache indexing.

    Restores the layer list, `layer_types` and `num_hidden_layers` on exit; leaving any of the three
    modified would silently contaminate every later arm in the same process.
    """

    def __init__(self, model, a: int, b: int, k: int):
        self.model, self.a, self.b, self.k = model, a, b, k
        self.inner = (model.model.language_model
                      if hasattr(model.model, "language_model") else model.model)
        self.cfg = (model.config.get_text_config()
                    if hasattr(model.config, "get_text_config") else model.config)
        self.orig_layers = self.inner.layers
        self.orig_types = list(getattr(self.cfg, "layer_types", []) or [])
        self.orig_n = getattr(self.cfg, "num_hidden_layers", None)

    def __enter__(self):
        if self.k <= 1:
            return self
        layers = list(self.orig_layers)
        types = self.orig_types
        a, b = self.a, self.b
        new_layers = list(layers[:b])
        new_types = list(types[:b]) if types else []
        for _ in range(self.k - 1):
            for j in range(a, b):
                new_layers.append(_clone_layer(layers[j], len(new_layers)))
                if types:
                    new_types.append(types[j])
        # Tail layers keep their ORIGINAL modules but move to higher positions, so their cache index
        # must be re-pointed too or they would collide with the clones inserted before them.
        for j in range(b, len(layers)):
            new_layers.append(_clone_layer(layers[j], len(new_layers)))
            if types:
                new_types.append(types[j])
        self.inner.layers = nn.ModuleList(new_layers)
        if types:
            self.cfg.layer_types = new_types
        self.cfg.num_hidden_layers = len(new_layers)
        return self

    def __exit__(self, *exc):
        self.inner.layers = self.orig_layers
        if self.orig_types:
            self.cfg.layer_types = self.orig_types
        if self.orig_n is not None:
            self.cfg.num_hidden_layers = self.orig_n
        return False

    @property
    def depth(self) -> int:
        return len(self.orig_layers) + (self.b - self.a) * max(0, self.k - 1)
