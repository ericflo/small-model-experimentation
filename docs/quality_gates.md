# Quality Gates

`make check` is the repository release gate. Run it before committing and expect GitHub Actions to run the same gate on pushes and pull requests.

## What It Checks

- Regenerates catalogs and verifies generated files are committed.
- Prints the complete canonical in-progress roster so every release reviews active lifecycle state.
- Proves catalog output is invariant to gitignored/external experiment artifacts.
- Regenerates the experiment readiness matrix so curation gaps stay visible.
- Regenerates and validates the future experiment queue so every program keeps a launchpad for new work.
- Regenerates the structured claim index and verifies claim references.
- Validates experiment, program, artifact, and adapter invariants.
- Enforces the benchmarks firewall: no Python file under experiments/ may import or reference benchmark family internals — held-out suites are run-only (see benchmarks/README.md).
- Validates standard artifact-manifest presence for new experiments and documentation for local adapter directories.
- Compiles repository maintenance scripts without writing cache files.
- Keeps the related-work discovery script available for idea routing.
- Keeps the queue-to-experiment scaffold command available.
- Builds and validates the generated static research atlas.
- Checks local markdown links in navigation and knowledge surfaces.
- Scans for stale repository framing and temporary scaffold residue.
- Keeps the root README principles-only: it fails on claim-specific anchors (`claims/#cN`) or hardcoded corpus counts, which go stale with every pipeline commit — findings and counts live on the generated site.
- Requires GitHub workflow, issue, and pull request templates to stay present.
- Requires idea-intake, decision-record, and program-scorecard surfaces to stay present.

## Why This Exists

The repository is meant to grow through many independent experiments and research programs. The quality gate protects that shape: experiments stay self-contained, programs stay explicit, generated indexes stay current, and old source-track provenance does not become the repository boundary again.

## Commands

```bash
make check
make active-experiments
make new-program PROGRAM=<program_id> TITLE="<Title>" FOCUS="<one-sentence focus>"
make new-experiment EXPERIMENT=<experiment_id> PROGRAM=<program_id> TITLE="<Title>"
```

## The standalone-reproducibility gate (owner directive, 2026-07-15)

Every experiment must be reproducible from its OWN directory plus the pinned
base model revision — never by walking another experiment's directory,
receipts, or `large_artifacts/` tree. Concretely, any cell that trains from,
merges onto, or even just EVALUATES a non-base checkpoint must contain:

- `data/lineage/` — the complete ordered SFT datasets, copied in as
  `stage01_<name>.jsonl`, `stage02_<name>.jsonl`, … (byte-identical copies of
  every dataset in the composite's training history, in training order);
- `data/lineage/lineage_manifest.json` — per stage: the dataset file + sha256,
  base (always the pinned model revision at stage 1), adapter rank/alpha,
  full hyperparameters, the FIXED training seed, and the merge step; plus the
  expected tree/weights sha256 of each stage's output as a verification aid;
- `scripts/rebuild_lineage.py` — replays the stages deterministically (same
  datasets, same order, same seeds → same checkpoints) and verifies each
  stage's output hash against the manifest.

Cross-experiment SHAs are allowed only as verification aids (asserting a
rebuilt artifact matches what was measured); they are never the reproduction
path. Historical experiments predating this gate are grandfathered but new
cells must comply, including eval-only cells.

Scope boundary (owner clarification, same directive): shared MEASUREMENT
instruments are repo-level infrastructure and are referenced in place, not
copied — `benchmarks/` suites and the trusted aggregate gateway
(`scripts/run_benchmark_aggregate.py`) in particular. The standalone
requirement covers the model-reproduction path only: datasets, training
recipes and seeds, adapter/merge steps, and the scripts that execute them
(trainer and merger are copied into the cell).

## Landing a new experiment (exact order)

The registration steps have ordering dependencies; doing them out of order produces confusing
gate failures:

1. Author the experiment content (README, report, log, `reports/artifact_manifest.yaml`).
2. Add the practitioner brief to `knowledge/experiment_brief.json` (nested under
   `"experiments"`; required field `plain_answer`) and ≥1 native chart spec to
   `knowledge/experiment_viz.json` (see `docs/site_maintenance.md`).
3. If and only if the result changes a durable corpus belief, add or update the
   supported claim in `knowledge/claims/claim_ledger.json`; design-only work and
   non-strategic results do not manufacture a claim.
4. **Commit** the experiment + knowledge edits (the date filler reads git history, so the
   commit must exist first).
5. `make catalog` — the date filler only covers experiments already in the catalog.
6. `python3 scripts/extract_experiment_dates.py --apply` — git-fills the date entry.
7. `make catalog` twice (manifest fixpoint), then `make check`; amend the regenerated files
   into the commit. `git status --short | wc -l` must be 0 afterward (determinism check).
8. Push, then **check `gh run list`** — local `make check` can pass while CI diverges (below).

## Common failures and fixes

- **Concurrent `make check` runs corrupt the generated `site/` tree.** `make site`
  removes and rebuilds the same working-tree directory, so overlapping checks in one
  checkout can produce missing assets/pages or `Directory not empty` errors even when
  repository validation is green. Confirm no other check is running, then rerun one
  `make check` to completion; do not diagnose the resulting site errors as content
  failures.
- **`generated-clean` red in CI but green locally.** The gate regenerates catalogs in CI's
  fresh checkout and diffs against the commit; anything that differs between your filesystem
  and a fresh clone breaks it. Known causes:
  - **Gitignored run data leaked into catalog inventory** — catalogs enumerate Git-visible
    files (`git ls-files --cached --others --exclude-standard`), so deterministic corpora,
    adapters, caches, and external rows cannot change file counts or artifact indexes merely
    by existing locally. `make catalog-test` creates an ignored sentinel and verifies every
    generated catalog byte is unchanged. If a file should be discoverable, do not ignore it;
    track its small manifest or receipt and describe omitted payloads in
    `reports/artifact_manifest.yaml`.
  - **Empty directories** — git can't track them, so they exist locally but not in CI, and
    the catalog's `top_level_dirs`/file counts diverge. `make validate` now fails on empty
    dirs under `experiments/` before this reaches CI; delete the dir.
  - **Wall-clock stamps** — tracked generated files must be a pure function of repo
    content: no `datetime.now()`/`date.today()` output may reach them. The old
    `generated_on` stamps did exactly that and reddened every branch at UTC midnight
    (even after the 2026-07-03 UTC fix), so they were removed on 2026-07-09. If a
    date-shaped diff appears, a new wall-clock stamp site has crept into a generator —
    remove the stamp rather than re-dating the files.
  - **Stale fixpoint** — `make catalog` must run twice; the manifest reaches a fixpoint on
    the second pass.
  - **Scaffold placeholders** — `make validate` fails if template filler prose ("Fill this
    after the run.", "What changed after this result? ...", etc.) survives in an
    experiment's `README.md` or `reports/*.md`. The site publishes those files verbatim,
    and on 2026-07-10 a placeholder Results section shipped to the site while the real
    results sat in `reports/report.md` — fill the README sections, don't rely on
    remembering to.
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
- **A preregistration/design-commit guard fails after rebase.** Rebasing changes commit IDs even when
  frozen design files are byte-identical. Re-anchor the configured design commit to the rebased commit,
  verify every frozen-file digest is unchanged, and record both old and new IDs in the experiment log.
  Never weaken the ancestry/digest guard or silently substitute the current `HEAD`.
- **A lock receipt invalidates its own frozen metadata.** Do not include generated `metadata.yaml`
  or repository catalog/index outputs in an experiment's frozen design-file set. Adding the tracked
  lock receipt changes file counts and therefore legitimately regenerates those surfaces; freeze the
  scientific config, code, tests, intake, preregistration, and design review instead.
