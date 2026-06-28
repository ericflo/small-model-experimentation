# Quality Gates

`make check` is the repository release gate. Run it before committing and expect GitHub Actions to run the same gate on pushes and pull requests.

## What It Checks

- Regenerates catalogs and verifies generated files are committed.
- Validates experiment, program, artifact, and adapter invariants.
- Compiles repository maintenance scripts without writing cache files.
- Checks local markdown links in navigation and knowledge surfaces.
- Scans for stale repository framing and temporary scaffold residue.

## Why This Exists

The repository is meant to grow through many independent experiments and research programs. The quality gate protects that shape: experiments stay self-contained, programs stay explicit, generated indexes stay current, and old source-track provenance does not become the repository boundary again.

## Commands

```bash
make check
make new-program PROGRAM=<program_id> TITLE="<Title>" FOCUS="<one-sentence focus>"
make new-experiment EXPERIMENT=<experiment_id> PROGRAM=<program_id> TITLE="<Title>"
```
