---
name: codex-exec
description: Delegate coding, analysis, and review tasks to the OpenAI Codex CLI (`codex exec`, GPT-5.6-sol at max reasoning) running non-interactively. Use when you want a second autonomous coding agent — parallel workers, structured JSON extraction from a codebase, an independent code review, or a self-contained implementation task. Covers exact flags, sandbox modes, sessions, JSONL events, exit-code semantics, and the prompt style the GPT-5.x agents respond to.
---

# Driving `codex exec` (GPT-5.6-sol, max)

`codex exec` runs OpenAI's Codex agent headlessly: one prompt in, agent works (reads files, runs commands, edits), final message out. The configured default here is **gpt-5.6-sol at `model_reasoning_effort=max`** — an exceptionally literal, precise instruction-follower. Treat every prompt as a spec, not a wish: it will do exactly what you say, resolve what you leave open without asking, and verify what you tell it to verify. (The behavioral findings in this skill were measured on gpt-5.5 xhigh and spot-verified to carry over to 5.6-sol; the interface is identical.)

**5.6 prompting delta:** GPT-5.6 measurably rewards *shorter* prompts — OpenAI reports 10–15% quality gains while cutting prompt tokens 41–66%. Keep the precision doctrine below (byte-exact constraints, decide every ambiguity, fence the scope) but trim scaffolding: state priorities instead of exhaustive rule lists, skip generic "be careful/be thorough" filler, keep structure guidance lightweight.

Health check if anything seems off: `codex doctor` (auth via ChatGPT account; config in `~/.codex/config.toml`).

## Golden template

```bash
codex exec -C <workdir> --skip-git-repo-check "<PROMPT>" </dev/null
```

- **Always append `</dev/null`.** exec unconditionally reads non-TTY stdin and appends it to the prompt as a `<stdin>` block; an open pipe can hang the run. (The stderr line "Reading additional input from stdin..." appears even with the redirect — that's normal, not a hang.) To *intentionally* pipe a prompt: `echo "prompt" | codex exec -`.
- `-C <dir>` sets the agent's working root — prefer it over `cd`.
- `--skip-git-repo-check` is required outside a git repo; inside one, omit it.
- Add `--yolo` (alias for `--dangerously-bypass-approvals-and-sandbox`) **only** when the task needs network or writes outside the workspace — plain `codex exec` is already fully headless (`approval: never`), just sandboxed. See the matrix.
- **Run it in a clean directory.** Codex inventories the workdir (`rg --files`) and reads *every* file it finds — your logs, baselines, even its own in-progress `-o`/event files — and lets them shape its solution. Keep orchestration artifacts (event streams, output captures, fixture backups) *outside* `-C`. Corollary: a runnable test/sanity script left in the workdir gets discovered and executed unprompted — free verification.

## Output plumbing (what lands where)

| Stream/flag | Content |
|---|---|
| **stdout** | the final agent message ONLY |
| **stderr** | everything else: header (model/sandbox/session id), streamed transcript, token counts |
| `-o FILE` / `--output-last-message FILE` | final message also written to FILE (write it outside the workdir) |
| `--json` | stdout becomes JSONL events instead (see below) |

Cleanest capture:

```bash
answer=$(codex exec --skip-git-repo-check "<PROMPT>" </dev/null 2>/dev/null)
```

### `--json` event stream (audit trail)

One JSON object per line on stdout:

```
{"type":"thread.started","thread_id":"<session-id>"}     ← capture to resume later
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"agent_message","text":"..."}}          ← progress narration + final message
{"type":"item.completed","item":{"type":"command_execution","command":"...","exit_code":0,"aggregated_output":"..."}}
{"type":"item.completed","item":{"type":"file_change","changes":[{"path":"..."}]}}
{"type":"item.completed","item":{"type":"web_search","query":"..."}}
{"type":"turn.completed","usage":{"input_tokens":..,"cached_input_tokens":..,"output_tokens":..,"reasoning_output_tokens":..}}
```

Parsing notes (all verified):
- Items are emitted twice — `item.started` (exit_code null) then `item.completed`. **Dedupe on `item.completed`.**
- A multi-file edit arrives as ONE atomic `file_change` item listing every add/delete/update — one event = the whole diff surface.
- `command_execution` items carry the exact command and exit code — a complete, replayable audit of what codex did.
- **No `reasoning` items ever appear**, even at xhigh; chain-of-thought surfaces only as `reasoning_output_tokens` in `turn.completed`. Progress narration comes as interstitial `agent_message` items (only in `--json`; `-o` gets just the final one).
- Nonzero exit codes in the command stream are normal exploration, not failures — codex probes with `git status` (128 in a non-repo), tries `python` before `python3` (127), then adapts silently.

### Structured output (`--output-schema`)

Force the final message to match a JSON Schema — the reliable way to get machine-readable results:

```bash
codex exec --skip-git-repo-check --sandbox read-only \
  --output-schema /outside/schema.json -o /outside/out.json "<PROMPT>" </dev/null
# out.json is raw, fence-free JSON — json.load it directly
```

Verified behavior:
- Strict schemas are safe and effective: `"additionalProperties": false` + `required` on every object never caused a failure — no invented keys, no omissions. Use them.
- **Design the schema to express everything you want back.** Conflicts resolve field-by-field: content with no schema home (a requested prose summary) is *silently dropped*; semi-fitting extras (severity ratings) get smuggled into existing string fields. Add a `summary` field or `severity` enum instead of asking in prose.
- **Trap: with `--output-schema`, EVERY `agent_message` in the `--json` stream is schema-coerced** — intermediate narration arrives as degenerate instances (empty arrays, fake entries like `{"file":"","line":0,"description":"I'll scan..."}`). Use only the LAST `agent_message`, or just read the `-o` file.
- Facts are earned, not guessed: line numbers came back 10/10 exact because codex reads numbered source (`nl -ba`) and self-verifies with AST passes. Still state conventions ("1-indexed line of the `def` statement").
- Open-ended lists vary run-to-run (5 vs 2 suspected bugs on identical code) — the schema stabilizes *shape*, not editorial judgment. Pin inclusion criteria ("flag every division whose denominator is unchecked") for reproducible coverage.
- Pair with `--sandbox read-only` for zero-mutation analysis: commands still run (rg, nl, python heredocs), writes are provably blocked, and the `-o` file is still written (the CLI writes it outside the sandbox). Keep `schema.json`/`out.json` outside the analyzed dir so they don't pollute the inventory.

### Exit codes — the trap

**Exit 0 means the CLI ran, NOT that the task succeeded.** A run where the agent was blocked (sandbox denial, missing infrastructure, impossible ask) still exits 0 with an honest explanatory final message. Non-zero = CLI/API failure only (bad flag, invalid model, auth). **Never gate automation on exit code — verify outcomes yourself** (run the tests, check the file, diff the repo) or demand a parseable sentinel: *"If you cannot, say BLOCKED and explain in one sentence."*

## Sandbox & approval matrix

| Invocation | Writes | Network | Use for |
|---|---|---|---|
| `codex exec` (default) | workdir + `/tmp` + `$TMPDIR` | ✗ (DNS fails) | most code changes — already headless |
| `codex exec --sandbox read-only` | none | ✗ | analysis, inventory, Q&A — guarantees no mutation |
| `codex exec --add-dir <dir2>` | + dir2 | ✗ | multi-root edits |
| `codex --yolo exec` | everywhere | ✓ | needs network (pip install, git push) or out-of-workspace writes |

Network without full yolo: `-c sandbox_workspace_write.network_access=true`. When blocked, codex says so honestly in the final message rather than faking success — but exit code is still 0 (see the trap).

## Key flags & config

- `-m gpt-5.6-sol` — the configured default and OpenAI's flagship (alias `gpt-5.6`; 1.05M context; same $5/$30 MTok price as gpt-5.5 but reaches the same quality with fewer output tokens). Cheaper tiers for mechanical bulk work: `gpt-5.6-terra` ($2.50/$15), `gpt-5.6-luna` ($1/$6). Legacy: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`.
- `-c model_reasoning_effort=none|low|medium|high|xhigh|max` — default max (OpenAI: reserve `max` for the hardest quality-first workloads; `medium` is the balanced starting point). **Effort scales with task difficulty, not a fixed tax**: trivial Q&A returns in ~3 s at every level, and high effort spent only ~1.2k reasoning tokens on a small refactor. Don't downgrade for real work; consider terra/luna + lower effort for trivial format probes.
- Cost note: prompt-cache *writes* bill at 1.25× uncached input on 5.6; cached reads are 10× cheaper than fresh input.
- `codex --search exec ...` — live web search. **`--search` must come BEFORE `exec`** (exec rejects it after).
- `-i img.png` — attach image(s).
- `--ephemeral` — no session persisted (not resumable).
- Any `~/.codex/config.toml` key can be overridden per call with `-c key=value`.

**Latency/cost budget (xhigh, verified):** answer-only prompts ~3 s; any file-touching task has a ~25–45 s floor (exploration + self-verification dominate); small multi-file refactor ~90 s. Each call is **stateless** and re-explores the directory from scratch — batch related steps into one prompt, or use `resume` for sequential steps. Precision is also a cost lever: on identical code, a vague ask cost ~73% more wall time and ~74% more output tokens than a scoped one.

## Sessions (multi-turn)

Session id is in the stderr header and the `thread.started` event.

```bash
codex exec resume --last "follow-up instruction" </dev/null      # continue most recent
codex exec resume <SESSION_ID> "follow-up" </dev/null            # continue specific one
```

Resumed sessions retain full context (verified: recalls prior-turn facts). For parallel workers capture each `thread_id` — `--last` races. Concurrent codex processes are safe; sessions are independent.

## AGENTS.md is live

Codex auto-loads `AGENTS.md` from the working root and **obeys it literally** (verified: a formatting rule planted there was followed with no mention in the prompt). Check the target directory's AGENTS.md before delegating — it silently shapes behavior. It's also a lever: put persistent constraints there instead of repeating them per prompt.

## Prompting GPT-5.5 xhigh — the doctrine

### What it does by default (don't waste prompt words on these)

- **Minimal-diff surgery.** "Fix the bug in X()" produced a 1-line diff even with four planted cleanup temptations and *no* scope restriction — identical diff with and without the fence. It won't opportunistically rename, reformat, or delete dead code.
- **Explores before editing** (file listing, reads, import-graph greps) — never say "look at the files first".
- **Self-verifies unprompted**: runs discovered test/sanity scripts, re-reads files it just wrote, re-greps for stale imports, `git diff`s its own blast radius, and machine-checks fussy constraints (`nl`, `ast`) — then reports what it verified, accurately.
- **Adapts to the environment silently and discloses**: `python`→`python3`, pytest missing→file's `__main__` runner, and the final message states which fallback it used. It even cleans up side effects (`__pycache__` from its own `py_compile`).
- **Honest about missing infrastructure**: asked to run tests where none exist, it tried three escalating runners, quoted their exit codes, and truthfully reported none found — no fabrication.
- **Backward-compat instinct**: even a vague "make it better" rewrite kept old function names as wrapper shims rather than break callers.

### What you must say explicitly (it won't guess — or will guess without asking)

- **It NEVER asks questions in exec mode.** It states assumptions in stream messages and proceeds. Contradictory instructions get resolved by a creative third reading (told "add type hints but do not modify the file", it produced a sibling `.pyi` stub satisfying both). If you want a hard stop instead: *"If any instructions conflict, stop and report the conflict instead of acting."*
- **Vague asks trigger maximal rewrites.** "Make app.py better" nearly doubled the file (renames, TypedDicts, type hints, restructured entry point) in one shot. Say exactly which improvements you want.
- **Fence the scope anyway**: end scoped asks with *"Change NOTHING else — no renames, no comment fixes, no cleanup, no formatting."* At xhigh it's cheap insurance that observably tightens process (it reads less and self-audits with `git diff` before finishing).
- **Decide ambiguities in the prompt.** "Split X into Y and Z" ⇒ X gets **deleted** (disclosed, but done); want a compat shim, say so. It ends on whatever git branch it created; want to end on main, say so.
- **Enumerate exactly — plurals are interpreted by intent, not letter.** Asked to fix "the unused import" when two existed, it removed both without comment.
- **Byte-exact constraints work — state them byte-exactly.** Exact file/function names, exact docstring text, exact blank-line counts (even PEP8-violating), exact commit-message format: all honored and self-verified. Quote literal text and say "EXACTLY".
- **Phrase acceptance criteria as executable one-liners** — *"`from pkg import load, transform, save` must still work"* — it runs the literal line as a dedicated post-check.
- **Demand a baseline when the repo might already be red**: it does NOT run tests before editing on its own — *"run the tests first and report the baseline, then make the change."*
- **It won't editorialize.** Stale comments, factual errors in docstrings, design smells adjacent to the instructed change are neither fixed nor flagged. Add *"also update any now-stale comments"* or request a review pass if you want that.
- **Verification-tool absence is non-fatal by default**: it shipped an unvalidated `.pyi` when mypy/pyright were absent, mentioning it only implicitly. Add *"fail loudly if verification tooling is unavailable"* when validation matters.
- **It won't attest to what it left untouched** unless asked — the final message reports what changed + verification, not restraint. Ask for an explicit "confirm you did not change X" if you need it.

## Git tasks

Git hygiene at xhigh is textbook (verified): `git status` first, branch created *before* editing, minimal diff, then self-verifies with `git log`/`git status`/`git rev-parse` before reporting. Commit messages come out **byte-exact** when you quote them and say "EXACTLY this message" (it uses `git commit -m <subject> -m <body>`). It honors negative constraints (no push, don't touch main, leave tree clean) and its final state summary was fully accurate. Notes:

- Commits use the repo's configured git identity, no co-author/tool trailers.
- Batch branch+fix+commit into ONE prompt — each exec call re-pays the ~40 s exploration floor.
- Commit fixtures/baselines to git before delegating so you can `git diff` the blast radius after.

## Code review mode

```bash
codex exec review --uncommitted            # review staged+unstaged+untracked
codex exec review --base main              # review branch vs base
codex exec review --commit <SHA>           # review one commit
codex exec review --uncommitted "focus on error handling"
```

Requires a git repo. Output: prioritized findings (`[P1]`, `[P2]`…) with `file:line` anchors and concrete failure scenarios; caught a planted logic bug even at low effort. Great independent second opinion on your own diffs.

## Recipes

**Delegate a self-contained change, verify yourself:**
```bash
codex exec -C /path/to/proj "Fix the off-by-one in pager.py:paginate — the last page drops one item. Add a regression test test_last_page_full to tests/test_pager.py. Run the tests first and report the baseline, then fix, then run them again and report. Change NOTHING else." </dev/null 2>/dev/null
cd /path/to/proj && python3 -m pytest tests/test_pager.py   # trust but verify
```

**Structured codebase inventory (zero mutation risk):**
```bash
codex exec --sandbox read-only -C /path/to/proj \
  --output-schema /outside/schema.json -o /outside/inventory.json \
  "Inventory every public function: file, name, line. Flag suspected bugs with file, line, description." </dev/null
```

**Parallel workers on independent modules:**
```bash
for m in parser lexer emitter; do
  codex exec -C "$PWD/src/$m" "Add type hints to every function in this directory. Run 'python3 -m mypy .' if available and report; fail loudly if mypy is unavailable. Change NOTHING else." </dev/null 2>"/outside/log_$m.txt" &
done; wait
```

**Multi-turn: plan → approve → execute:**
```bash
sid=$(codex exec --json --sandbox read-only -C proj "Propose a plan to X. Do not change anything yet." </dev/null 2>/dev/null | python3 -c "import sys,json;print(next(json.loads(l)['thread_id'] for l in sys.stdin if 'thread.started' in l))")
# ... inspect the plan ...
codex exec resume "$sid" "The plan is approved. Execute step 1 only, then stop." </dev/null
```

## Gotchas checklist

- `codex exec review --base <branch>` rejects a custom prompt alongside it (despite the usage string suggesting otherwise) — use plain `--base` for branch reviews; custom review instructions only work with `--uncommitted`-style default scope.

- `</dev/null` on every scripted call — stdin is read unconditionally; the redirect prevents hangs (the stderr "Reading additional input..." line still appears; ignore it).
- Keep the workdir clean — codex reads every file in it, including your orchestration artifacts, and they shape its behavior. `-o`/logs/schemas go outside `-C`.
- `--search` goes before `exec`, not after.
- Exit 0 ≠ task success — verify outcomes or demand a BLOCKED sentinel.
- Default sandbox has **no network** (DNS itself fails); `--yolo` or the config key to enable.
- Outside a git repo, exec errors without `--skip-git-repo-check`; inside one, omit it (codex probes git itself regardless).
- stdout is only the final message; transcript on stderr — don't grep stdout for progress.
- `AGENTS.md` in the workdir is silently loaded and obeyed.
- Each exec call is stateless and re-explores — batch steps or `resume`; ~40 s floor per file-touching call at xhigh.
- Dedupe `--json` items on `item.completed`; no reasoning items ever stream.
