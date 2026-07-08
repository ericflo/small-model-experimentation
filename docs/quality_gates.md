# Quality Gates

`make check` is the repository release gate. Run it before committing and expect GitHub Actions to run the same gate on pushes and pull requests.

## What It Checks

- Regenerates catalogs and verifies generated files are committed.
- Regenerates the experiment readiness matrix so curation gaps stay visible.
- Regenerates and validates the future experiment queue so every program keeps a launchpad for new work.
- Regenerates the structured claim index and verifies claim references.
- Validates experiment, program, artifact, and adapter invariants.
- Validates standard artifact-manifest presence for new experiments and documentation for local adapter directories.
- Compiles repository maintenance scripts without writing cache files.
- Keeps the related-work discovery script available for idea routing.
- Keeps the queue-to-experiment scaffold command available.
- Builds and validates the generated static research atlas.
- Checks local markdown links in navigation and knowledge surfaces.
- Scans for stale repository framing and temporary scaffold residue.
- Requires GitHub workflow, issue, and pull request templates to stay present.
- Requires idea-intake, decision-record, and program-scorecard surfaces to stay present.

## Why This Exists

The repository is meant to grow through many independent experiments and research programs. The quality gate protects that shape: experiments stay self-contained, programs stay explicit, generated indexes stay current, and old source-track provenance does not become the repository boundary again.

## Commands

```bash
make check
make new-program PROGRAM=<program_id> TITLE="<Title>" FOCUS="<one-sentence focus>"
make new-experiment EXPERIMENT=<experiment_id> PROGRAM=<program_id> TITLE="<Title>"
```

## Landing a new experiment (exact order)

The registration steps have ordering dependencies; doing them out of order produces confusing
gate failures:

1. Author the experiment content (README, report, log, `reports/artifact_manifest.yaml`).
2. Add the practitioner brief to `knowledge/experiment_brief.json` (nested under
   `"experiments"`; required field `plain_answer`) and ≥1 native chart spec to
   `knowledge/experiment_viz.json` (see `docs/site_maintenance.md`).
3. Add the claim to `knowledge/claims/claim_ledger.json`.
4. **Commit** the experiment + knowledge edits (the date filler reads git history, so the
   commit must exist first).
5. `make catalog` — the date filler only covers experiments already in the catalog.
6. `.venv/bin/python scripts/extract_experiment_dates.py --apply` — git-fills the date entry.
7. `make catalog` twice (manifest fixpoint), then `make check`; amend the regenerated files
   into the commit. `git status --short | wc -l` must be 0 afterward (determinism check).
8. Push, then **check `gh run list`** — local `make check` can pass while CI diverges (below).

## Common failures and fixes

- **`generated-clean` red in CI but green locally.** The gate regenerates catalogs in CI's
  fresh checkout and diffs against the commit; anything that differs between your filesystem
  and a fresh clone breaks it. Known causes:
  - **Empty directories** — git can't track them, so they exist locally but not in CI, and
    the catalog's `top_level_dirs`/file counts diverge. `make validate` now fails on empty
    dirs under `experiments/` before this reaches CI; delete the dir.
  - **Local-vs-UTC date stamps** — `generated_on` stamps must use UTC
    (`build_knowledgebase.py` does since 2026-07-03); an evening PDT commit with a local-date
    stamp reddens CI. If a date-shaped diff appears, check for a new non-UTC stamp site.
  - **Stale fixpoint** — `make catalog` must run twice; the manifest reaches a fixpoint on
    the second pass.
  - To debug a red run: `gh run view <id> --log-failed` and read the embedded `diff --git`.
- **`validate` fails on files that aren't tracked.** The validator scans the *working tree*
  (pycache dirs, `*.pyc`, Zone.Identifier files, >100 MB files) — the offender doesn't need
  to be committed, or even trackable. Delete it. Importing `build_site.py` writes
  `scripts/__pycache__`; delete it before `make validate` if you imported the module.
- **`briefs-gate` fails for a new experiment.** It needs BOTH a date entry and a brief. The
  date fill requires the experiment committed *and* in the catalog first (steps 4–6 above).
  Verify with `python3 scripts/site_content_status.py`.
- **In-place edits mysteriously reverted** before a launch (observed with sed on training
  scripts): use a real editor/Edit tool and re-grep the file before trusting a run.
