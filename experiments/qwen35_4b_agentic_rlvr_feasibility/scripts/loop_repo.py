"""Real-repo agentic episodes: copy a repo, stub a tested function, let the base implement it.

Reuses loop_raw's raw-completions loop, Qwen-XML tool parsing, and thinking capture, but the
"project" is a real repo checkout with a function body deleted, graded by the repo's test command
run through the repo's own venv (isolated via cwd-insertion of the local copy).
"""
import ast, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import loop_raw as LR


def stub_function(repo_root: Path, rel_file: str, func_name: str) -> tuple[str, str] | None:
    """Replace func body with `raise NotImplementedError`. Returns (original_src, stubbed_src) or None."""
    p = repo_root / rel_file
    src = p.read_text()
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            target = node
            break
    if target is None:
        return None
    lines = src.split("\n")
    body = target.body
    # keep a leading docstring if present (helps the agent), stub the rest
    first = body[0]
    keep_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str)
    start = (body[1].lineno - 1) if (keep_doc and len(body) > 1) else (first.lineno - 1)
    end = body[-1].end_lineno
    col = target.body[0].col_offset
    indent = " " * col
    original_body = "\n".join(lines[start:end])
    lines[start:end] = [indent + "raise NotImplementedError  # TODO: implement"]
    p.write_text("\n".join(lines))
    return original_body, "\n".join(lines)


def _env_for(python_path: str, shim: Path = None):
    # Prepend the venv's REAL bin dir (has python/python3/pytest/pip with correct venv
    # resolution). A symlink to the venv python breaks site-packages detection.
    venv_bin = str(Path(python_path).parent)
    return {**os.environ, "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": str(Path(python_path).parent.parent),
            "PYTHONDONTWRITEBYTECODE": "1"}


def run_repo_episode(task, temperature=0.7, step_cap=20):
    """task: {id, repo_dir, python, rel_file, func_name, test_cmd, prompt}. Returns result + transcript."""
    project = Path(tempfile.mkdtemp(prefix="repoep_"))
    try:
        # copy repo (skip .git and venvs for speed)
        shutil.copytree(task["repo_dir"], project, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", "*.pyc"))
        stub = stub_function(project, task["rel_file"], task["func_name"])
        if stub is None:
            return {"id": task["id"], "passed": False, "error": "stub_failed"}
        env = _env_for(task["python"], project.parent / (project.name + "_shim"))
        # agentic loop (reuse loop_raw internals with our env)
        history = [{"role": "system", "content": LR.SYSTEM}, {"role": "user", "content": task["prompt"]}]
        transcript = []
        n_tool_calls = 0
        stopped = "cap"
        for step in range(step_cap):
            prompt = LR.tok().apply_chat_template(history, tokenize=False, add_generation_prompt=True, tools=LR.TOOLS)
            try:
                text, finish = LR.generate(prompt, temperature)
            except Exception as e:
                stopped = f"api_error:{type(e).__name__}"; break
            reasoning, natural, calls = LR.parse_completion(text)
            asst = {"role": "assistant", "content": natural}
            if calls:
                asst["tool_calls"] = [{"id": f"c{step}_{i}", "type": "function",
                                       "function": {"name": c["name"], "arguments": c["arguments"]}}
                                      for i, c in enumerate(calls)]
            history.append(asst)
            transcript.append({"step": step, "reasoning": reasoning, "natural": natural, "calls": calls, "finish": finish})
            if not calls:
                stopped = "model_done"; break
            for i, c in enumerate(calls):
                n_tool_calls += 1
                result = _exec_tool(project, c["name"], c["arguments"], env)
                history.append({"role": "tool", "tool_call_id": f"c{step}_{i}", "content": result})
        chk = subprocess.run(task["test_cmd"], shell=True, cwd=str(project), text=True, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        passed = chk.returncode == 0
        produced = _target_changed(project, task)
        return {"id": task["id"], "passed": passed, "produced_edit": produced, "stopped": stopped,
                "n_steps": len(transcript), "n_tool_calls": n_tool_calls,
                "think_steps": sum(1 for t in transcript if t["reasoning"]),
                "check_tail": (chk.stdout or "")[-300:], "transcript": transcript, "history": history}
    finally:
        shutil.rmtree(project, ignore_errors=True)
        shutil.rmtree(project.parent / (project.name + "_shim"), ignore_errors=True)


def _safe(project, path):
    p = (project / (path or ".")).resolve()
    if not str(p).startswith(str(project.resolve())):
        raise ValueError("path escapes project")
    return p


def _exec_tool(project, name, args, env):
    """Thread-safe tool exec (env passed explicitly, no module global)."""
    try:
        if name == "list_dir":
            d = _safe(project, args.get("path", ".") or ".")
            if not d.exists():
                return "(no such directory)"
            return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())) or "(empty)"
        if name == "read_file":
            f = _safe(project, args.get("path", ""))
            return f.read_text(encoding="utf-8", errors="replace")[:LR.MAX_TOOL_OUTPUT] if f.exists() else "(no such file)"
        if name == "write_file":
            f = _safe(project, args.get("path", ""))
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(args.get("content", ""), encoding="utf-8")
            return f"wrote {args.get('path')} ({len(args.get('content',''))} bytes)"
        if name == "run":
            r = subprocess.run(args.get("command", ""), shell=True, cwd=str(project), text=True, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=LR.RUN_TIMEOUT)
            return f"[exit {r.returncode}]\n{(r.stdout or '')[:LR.MAX_TOOL_OUTPUT]}"
        return f"(unknown tool {name})"
    except subprocess.TimeoutExpired:
        return f"[timeout after {LR.RUN_TIMEOUT}s]"
    except Exception as e:
        return f"[tool error: {type(e).__name__}: {e}]"


def _target_changed(project, task):
    p = project / task["rel_file"]
    try:
        return "raise NotImplementedError  # TODO: implement" not in p.read_text()
    except Exception:
        return False
