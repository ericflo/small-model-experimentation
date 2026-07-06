# Keeping the research site current

The published site (`scripts/build_site.py` → GitHub Pages) is regenerated from
the repository on every push, so **prose, catalogs, claims, and figures are
always live**. Three *curated* content layers are richer than what a plain build
can derive and are stored as committed artifacts:

| Artifact | What it powers | How it is produced |
|---|---|---|
| `knowledge/experiment_dates.json` | true run windows / chronology | **deterministic** — git history |
| `knowledge/experiment_viz.json` | native result charts | agent enrichment pass |
| `knowledge/experiment_brief.json` | plain-language practitioner brief (top of each experiment page) | agent enrichment pass |

The build **degrades gracefully**: an experiment with no entry in any of these
just shows less (no chart section, no brief, a git-derived or "in progress"
date). Nothing breaks. So the site never *regresses* as experiments land — it
only lacks the enriched layer until the passes below run.

## The maintenance loop

After new experiments are added (or on a schedule), run:

```bash
make site-content      # 1. auto-fill dates from git, then report coverage
```

`site-content` runs `extract_experiment_dates.py --apply` (fills a git-derived
run window for every **post-import** experiment — the 2026-06-28 import is
excluded so its date-collapse can't recur) and then `site_content_status.py`,
which prints coverage and lists exactly which experiments still lack **charts**
or a **brief**:

```
site content coverage over N experiments:
  dates   N/N
  charts  N/N
  briefs  M/N
- K no practitioner brief → run the brief enrichment pass
    exp_a, exp_b, …
```

Charts and briefs need a model in the loop, so they are filled by the two
enrichment workflows kept in [`scripts/enrichment/`](../scripts/enrichment):

- `extract_charts.workflow.js` — reads each experiment's own result files and
  emits verified chart specs (every plotted number must appear verbatim in a
  cited source; unverifiable charts are dropped).
- `generate_briefs.workflow.js` — authors the plain-language brief per
  experiment (verdict, plain question/answer, why-it-matters, KPI numbers, and
  per-chart "how to read"/"takeaway"), under a strict no-jargon ban-list.

Both are [Workflow tool](../CONTRIBUTING.md) scripts run by the orchestrating
agent; both are **resumable** and only need to cover experiments the status
report flags as missing. After a pass, re-check with `make site-content` and
commit the updated `knowledge/experiment_*.json` artifacts.

## Guardrails

- `make site-content` is safe to run any time; dates auto-fill is idempotent.
- `python3 scripts/site_content_status.py --strict` exits non-zero when charts
  or briefs are missing — use it to gate a release if you want full coverage.
- CI runs `site_content_status.py` as an **informational** step (it never fails
  the build), so drift is visible in the Actions log without blocking deploys.
- Never hand-edit numbers into the artifacts; the enrichment passes verify every
  figure against the experiment's own data. Keep the human-authored knowledge
  (`synthesis.md`, the claim ledger) as the source of truth for framing.
