"""Enumerate valid stub-a-function tasks in a repo: functions whose removal breaks the test suite."""
import argparse, ast, json, shutil, subprocess, sys, tempfile
from pathlib import Path


def top_functions(src_file: Path):
    tree = ast.parse(src_file.read_text())
    out = []
    for node in tree.body:  # top-level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            nlines = (node.body[-1].end_lineno - node.body[0].lineno)
            out.append((node.name, nlines))
    return out


def stub_and_test(repo_dir: Path, python: str, rel_file: str, func: str, test_cmd: str, timeout=120):
    tmp = Path(tempfile.mkdtemp(prefix="taskgen_"))
    try:
        shutil.copytree(repo_dir, tmp, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", "*.pyc"))
        sys.path.insert(0, str(Path(__file__).parent))
        import loop_repo
        if loop_repo.stub_function(tmp, rel_file, func) is None:
            return None
        r = subprocess.run(test_cmd, shell=True, cwd=str(tmp), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           env={"PATH": str(Path(python).parent) + ":/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
        # count failures
        import re
        m = re.search(r"(\d+) failed", r.stdout or "")
        nfail = int(m.group(1)) if m else (0 if r.returncode == 0 else 1)
        return {"returncode": r.returncode, "nfail": nfail}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--src", nargs="+", required=True, help="source files (repo-relative)")
    ap.add_argument("--test-cmd", default="python -m pytest -q")
    ap.add_argument("--min-body-lines", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo)
    tasks = []
    for rel in a.src:
        for func, nlines in top_functions(repo / rel):
            if nlines < a.min_body_lines:
                continue
            res = stub_and_test(repo, a.python, rel, func, a.test_cmd)
            if res and res["returncode"] != 0 and res["nfail"] >= 1:
                tasks.append({"repo": repo.name, "rel_file": rel, "func_name": func,
                              "body_lines": nlines, "n_failing_when_stubbed": res["nfail"]})
                print(f"  TASK {rel}::{func} (body {nlines} lines, {res['nfail']} tests fail)", flush=True)
            else:
                print(f"  skip {rel}::{func} (not test-covered)", flush=True)
    Path(a.out).write_text(json.dumps(tasks, indent=1))
    print(f"\n{len(tasks)} valid tasks -> {a.out}")


if __name__ == "__main__":
    main()
