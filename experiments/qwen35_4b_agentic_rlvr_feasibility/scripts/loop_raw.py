"""Agentic loop over raw /v1/completions with manual Qwen-XML tool parsing + thinking capture.

Mirrors the pi-coding-agent 4-tool schema (read_file/write_file/list_dir/run). Renders the
multi-turn history with the model's own chat template (tools + primed <think>), samples raw text,
parses <think>reasoning</think> and <tool_call> blocks itself, executes tools in an isolated temp
project, grades with the scenario check. Captures per-turn reasoning (harvest source for SFT).

Why raw completions instead of the OpenAI chat API: the chat path drops the primed-<think> reasoning
(the deepseek_r1 parser only fills reasoning_content when the completion re-emits <think>), and
multi-turn tool_calls must carry DICT arguments (a JSON string breaks the Qwen template render).
"""
import json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import requests
from transformers import AutoTokenizer

BASE_MODEL = os.environ.get(
    "AGENTIC_BASE_MODEL",
    "/home/ericflo/Development/small-model-experimentation/large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized",
)
SERVER = os.environ.get("AGENTIC_SERVER", "http://localhost:1234/v1/completions")
SERVED_MODEL = os.environ.get("AGENTIC_SERVED_MODEL", "base")
STEP_CAP = 20
MAX_TOOL_OUTPUT = 12000
RUN_TIMEOUT = 30
MAX_TOKENS = 4096

_SHIM = Path(tempfile.gettempdir()) / "agentic_pyshim"
_SHIM.mkdir(exist_ok=True)
if not (_SHIM / "python").exists():
    (_SHIM / "python").symlink_to(shutil.which("python3") or sys.executable)
_ENV = {**os.environ, "PATH": str(_SHIM) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1"}

TOOLS = [
    {"type": "function", "function": {"name": "list_dir",
        "description": "List names in a directory (project-relative), dirs marked with trailing /.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "read_file",
        "description": "Read a UTF-8 text file (project-relative); returns full content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
        "description": "Write (overwrite) a UTF-8 text file with the full new content (project-relative).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run",
        "description": "Run a shell command from the project root; returns combined stdout/stderr and exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

SYSTEM = ("You are an expert Python coding agent working inside a project directory. Think step by "
          "step, then use the tools to inspect files, write code, and run the tests until they pass. "
          "Always run the tests to verify before finishing. When the tests pass, stop calling tools "
          "and give a one-line summary.")

_tok = None
def tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    return _tok


def parse_completion(text):
    """Return (reasoning, natural_content, tool_calls[{name, arguments:dict}])."""
    reasoning = ""
    rest = text
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", rest, re.DOTALL):
        fm = re.search(r"<function=([^>\s]+)\s*>(.*?)</function>", m.group(1), re.DOTALL)
        if not fm:
            continue
        args = {}
        for pm in re.finditer(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", fm.group(2), re.DOTALL):
            args[pm.group(1).strip()] = pm.group(2)
        calls.append({"name": fm.group(1).strip(), "arguments": args})
    natural = re.sub(r"<tool_call>.*?</tool_call>", "", rest, flags=re.DOTALL).strip()
    return reasoning, natural, calls


def _safe(project, path):
    p = (project / (path or ".")).resolve()
    if not str(p).startswith(str(project.resolve())):
        raise ValueError("path escapes project")
    return p


def exec_tool(project, name, args, env=None):
    env = env or _ENV
    try:
        if name == "list_dir":
            d = _safe(project, args.get("path", ".") or ".")
            if not d.exists():
                return "(no such directory)"
            return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())) or "(empty)"
        if name == "read_file":
            f = _safe(project, args.get("path", ""))
            return f.read_text(encoding="utf-8", errors="replace")[:MAX_TOOL_OUTPUT] if f.exists() else "(no such file)"
        if name == "write_file":
            f = _safe(project, args.get("path", ""))
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(args.get("content", ""), encoding="utf-8")
            return f"wrote {args.get('path')} ({len(args.get('content',''))} bytes)"
        if name == "run":
            r = subprocess.run(args.get("command", ""), shell=True, cwd=str(project), text=True, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT)
            return f"[exit {r.returncode}]\n{(r.stdout or '')[:MAX_TOOL_OUTPUT]}"
        return f"(unknown tool {name})"
    except subprocess.TimeoutExpired:
        return f"[timeout after {RUN_TIMEOUT}s]"
    except Exception as e:
        return f"[tool error: {type(e).__name__}: {e}]"


def generate(prompt, temperature):
    r = requests.post(SERVER, json={"model": SERVED_MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
                                    "temperature": temperature, "top_p": 0.95,
                                    "stop": ["<|im_end|>"], "seed": 0}, timeout=600).json()
    return r["choices"][0]["text"], r["choices"][0]["finish_reason"]


def run_episode(scenario, project, temperature=0.7, step_cap=STEP_CAP):
    """scenario: {id, prompt, files:{path:content}, check:'cmd', hidden_files?:{}}."""
    for rel, content in scenario.get("files", {}).items():
        f = _safe(project, rel); f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": scenario["prompt"]}]
    transcript, n_tool_calls, stopped = [], 0, "cap"
    for step in range(step_cap):
        prompt = tok().apply_chat_template(history, tokenize=False, add_generation_prompt=True, tools=TOOLS)
        try:
            text, finish = generate(prompt, temperature)
        except Exception as e:
            stopped = f"api_error:{type(e).__name__}"; break
        reasoning, natural, calls = parse_completion(text)
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
            history.append({"role": "tool", "tool_call_id": f"c{step}_{i}",
                            "content": exec_tool(project, c["name"], c["arguments"])})
    for rel, content in scenario.get("hidden_files", {}).items():
        f = _safe(project, rel); f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    chk = subprocess.run(scenario["check"], shell=True, cwd=str(project), text=True, env=_ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    return {"id": scenario["id"], "passed": chk.returncode == 0, "stopped": stopped,
            "n_steps": len(transcript), "n_tool_calls": n_tool_calls,
            "check_out": (chk.stdout or "")[-400:], "transcript": transcript, "history": history}


def with_temp_project(scenario, fn):
    d = Path(tempfile.mkdtemp(prefix="agentic_"))
    try:
        return fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
