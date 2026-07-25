"""Registry of candidate real Python repos for the stub-a-function instrument.

Selection criteria (all four required, enforced by fetch_repos.py rather than trusted here):
  1. PURE PYTHON — no compiled extension needed to run the suite (a C build failure on this box is
     an instrument outage, not a research signal).
  2. Runnable pytest suite that is GREEN at the pinned commit, in under ~180s.
  3. Permissive license (BSD/MIT/Apache).
  4. Library-shaped, i.e. top-level functions/methods with real behavior covered by real tests —
     the stub-a-function task only exists where removing a body breaks a test.

`src` globs point at the modules whose functions become candidate tasks. Keep them narrow enough
that we do not stub test helpers, `setup.py`, or vendored code.

The train/test FIREWALL is at REPO granularity (see split_tasks.py): a held-out task lives in a repo
whose code was never in any training corpus, so a win cannot come from having memorized that repo's
conventions during harvest. `group` is the pre-registered side of the firewall, fixed HERE before any
task is generated or any number is measured, so the split cannot be tuned to a result later.

Commits are NOT pinned in this file: fetch_repos.py records the resolved SHA of the default branch
at clone time into data/repo_manifest.json, which is committed. That manifest is the reproduction
path (the standalone-experiment directive) — re-fetch with --manifest to reproduce byte-identical
checkouts.
"""

# group: "train" repos may be harvested/trained on; "test" repos are EVAL-ONLY, forever.
REPOS = [
    # ---------------- train side ----------------
    dict(name="toolz", url="https://github.com/pytoolz/toolz",
         src=["toolz/itertoolz.py", "toolz/functoolz.py", "toolz/dicttoolz.py"], group="train"),
    dict(name="funcy", url="https://github.com/Suor/funcy",
         src=["funcy/seqs.py", "funcy/colls.py", "funcy/strings.py", "funcy/funcs.py",
              "funcy/calc.py", "funcy/tree.py", "funcy/types.py"], group="train"),
    dict(name="boltons", url="https://github.com/mahmoud/boltons",
         src=["boltons/strutils.py", "boltons/iterutils.py", "boltons/dictutils.py",
              "boltons/mathutils.py", "boltons/listutils.py", "boltons/setutils.py",
              "boltons/urlutils.py", "boltons/funcutils.py"], group="train"),
    dict(name="sortedcontainers", url="https://github.com/grantjenks/python-sortedcontainers",
         src=["src/sortedcontainers/sortedlist.py", "src/sortedcontainers/sorteddict.py",
              "src/sortedcontainers/sortedset.py"], group="train"),
    dict(name="jmespath", url="https://github.com/jmespath/jmespath.py",
         src=["jmespath/functions.py", "jmespath/lexer.py", "jmespath/parser.py",
              "jmespath/visitor.py"], group="train"),
    dict(name="jsonpatch", url="https://github.com/stefankoegl/python-json-patch",
         src=["jsonpatch.py"], group="train"),
    dict(name="inflection", url="https://github.com/jpvanhal/inflection",
         src=["inflection/__init__.py"], group="train"),
    dict(name="tabulate", url="https://github.com/astanin/python-tabulate",
         src=["tabulate/__init__.py"], group="train"),
    dict(name="natsort", url="https://github.com/SethMMorton/natsort",
         src=["natsort/natsort.py", "natsort/utils.py", "natsort/ns_enum.py"], group="train"),
    dict(name="schema", url="https://github.com/keleshev/schema",
         src=["schema/__init__.py"], group="train"),
    dict(name="wcwidth", url="https://github.com/jquast/wcwidth",
         src=["wcwidth/wcwidth.py"], group="train"),
    dict(name="mergedeep", url="https://github.com/clarketm/mergedeep",
         src=["mergedeep/mergedeep.py"], group="train"),
    dict(name="pathspec", url="https://github.com/cpburnz/python-pathspec",
         src=["pathspec/pathspec.py", "pathspec/util.py", "pathspec/patterns/gitwildmatch.py"],
         group="train"),

    # ---------------- test side (EVAL ONLY — never harvested, never trained on) ----------------
    dict(name="more-itertools", url="https://github.com/more-itertools/more-itertools",
         src=["more_itertools/more.py", "more_itertools/recipes.py"], group="test"),
    dict(name="glom", url="https://github.com/mahmoud/glom",
         src=["glom/core.py", "glom/reduction.py", "glom/matching.py", "glom/streaming.py"],
         group="test"),
    dict(name="bidict", url="https://github.com/jab/bidict",
         src=["bidict/_base.py", "bidict/_bidict.py", "bidict/_orderedbase.py"], group="test"),
    dict(name="dpath", url="https://github.com/dpath-maintainers/dpath-python",
         src=["dpath/__init__.py", "dpath/util.py", "dpath/segments.py"], group="test"),
    dict(name="semver", url="https://github.com/python-semver/python-semver",
         src=["src/semver/version.py", "src/semver/_deprecated.py"], group="test"),
    dict(name="humanize", url="https://github.com/python-humanize/humanize",
         src=["src/humanize/number.py", "src/humanize/time.py", "src/humanize/filesize.py",
              "src/humanize/i18n.py"], group="test"),
    # upstream moved the single module into a package; the flat `parse.py` no longer exists
    dict(name="parse", url="https://github.com/r1chardj0n3s/parse",
         src=["parse/__init__.py"], group="test"),
    dict(name="python-slugify", url="https://github.com/un33k/python-slugify",
         src=["slugify/slugify.py"], group="test"),
    dict(name="voluptuous", url="https://github.com/alecthomas/voluptuous",
         src=["voluptuous/validators.py", "voluptuous/schema_builder.py", "voluptuous/util.py"],
         group="test"),
    dict(name="jsonpointer", url="https://github.com/stefankoegl/python-json-pointer",
         src=["jsonpointer.py"], group="test"),
    dict(name="cerberus", url="https://github.com/pyeve/cerberus",
         src=["cerberus/validator.py", "cerberus/utils.py", "cerberus/schema.py"], group="test"),
]

BY_NAME = {r["name"]: r for r in REPOS}
