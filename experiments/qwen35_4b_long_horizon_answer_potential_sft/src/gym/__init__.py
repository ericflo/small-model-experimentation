"""Experiment-local, firewall-clean procedural atom generators.

Training and held-family registries are deliberately separate.  Code used by
calibration, harvesting, selection, and SFT imports only ``gym.families``;
``gym.heldout_families`` is an evaluation-only namespace.
"""
