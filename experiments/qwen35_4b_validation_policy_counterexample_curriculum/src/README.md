# Source notes

This directory is a self-contained copy of the predecessor's procedural
repository environment, looping JSON-tool agent, and transition bank. The new
task builders create near-correct validation-policy states over bundle-map,
record, and tuple representations. `content_digest` fingerprints only public
issue/files and deliberately excludes task ID and split for independence
checks.

`bank.py` keeps seven state→action strata. The build script uses only each
policy task's `diagnosis_to_changed_patch` row; all other treatment rows are
copied from the frozen predecessor bank. Hidden tests and patch objects remain
host-only and `assert_firewall_clean` rejects private-field serialization.
