# GitHub Sync

The repository is intended to be private on GitHub under `ericflo/small-model-experimentation`.

Routine sync:

```bash
make check
git status --short
git add README.md AGENTS.md CONTRIBUTING.md Makefile .gitignore .gitattributes docs knowledge research_programs scripts templates experiments
git commit -m "Update experiment knowledgebase"
git push
```

Before pushing large new artifacts, run:

```bash
find experiments -type f -size +100M -print
git ls-files 'experiments/**/reports/adapters/**'
git lfs status
```

No file may exceed GitHub's 100 MB non-LFS hard limit.

The interactive research atlas is published by GitHub Pages from `.github/workflows/pages.yml`. The workflow rebuilds the site from repository data with `make site-check`, uploads the generated `site/` bundle, and deploys it through GitHub Pages.

GitHub issue and pull request templates are part of the operating system. Keep them aligned with `docs/experiment_lifecycle.md`, `docs/research_program_lifecycle.md`, and `docs/quality_gates.md` when the workflow changes.
