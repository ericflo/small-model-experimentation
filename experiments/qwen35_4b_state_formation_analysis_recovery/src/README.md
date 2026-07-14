# Recovery source

`recovery.py` imports the immutable producer source under an isolated package
name, verifies source-contract v11 and all relevant file/config identities, and
installs one temporary exact-prefix path seam. It neither copies nor changes the
producer's scientific summarizers. The seam is restored even when analysis
raises.

The recovery smoke hashes this source, its tests, config, runner, and adversarial
review. Any later change invalidates analysis until a new reviewed smoke is
published.
