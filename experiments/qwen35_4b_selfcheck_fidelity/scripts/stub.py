"""Qualified-name-aware AST function stubber + candidate enumeration.

Extends the prior harness (which matched functions by bare name at top level only) in two ways:
  * METHODS are eligible, addressed by qualified name ("Class.method"), which roughly triples the
    task supply and covers class-heavy repos (bidict, cerberus, semver) that a top-level-only
    enumerator sees as empty.
  * Bare-name matching is REPLACED by an exact qualified-name walk. Matching by bare name is unsafe
    once methods are in scope: `toolz.functoolz.Compose.__call__` and a module-level `__call__`
    would collide, and `ast.walk` returns whichever comes first — silently stubbing the wrong body
    and producing a task whose prompt points at a different function than the one that was broken.

STUB_MARKER is the single source of truth for "the agent has not written an implementation yet"; the
episode runner's engagement gate greps for exactly this string, so it must never be paraphrased.
"""
import ast

STUB_MARKER = "raise NotImplementedError  # TODO: implement"

# Decorators that make a stubbed body meaningless or the task ill-posed.
SKIP_DECORATORS = {"abstractmethod", "abstractproperty", "overload", "singledispatch",
                   "singledispatchmethod", "deprecated"}


def _dec_names(node):
    out = set()
    for d in node.decorator_list:
        cur = d.func if isinstance(d, ast.Call) else d
        while isinstance(cur, ast.Attribute):
            out.add(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            out.add(cur.id)
    return out


def _is_docstring(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant)
            and isinstance(stmt.value.value, str))


def _body_span(node):
    """(start_line_0indexed, end_line_1indexed, indent) of the body EXCLUDING a leading docstring."""
    body = node.body
    keep_doc = _is_docstring(body[0])
    if keep_doc and len(body) == 1:
        return None                       # docstring-only body: nothing to implement
    first = body[1] if keep_doc else body[0]
    return (first.lineno - 1, body[-1].end_lineno, " " * first.col_offset)


def candidates(src_file, rel_file, min_body_lines=3, max_body_lines=120):
    """Enumerate stubbable functions/methods in a file as task candidates.

    Filters (each one exists because the alternative produces an unanswerable or degenerate task):
      * private/dunder names — not part of the documented surface the tests describe;
      * docstring-only or trivial bodies (< min_body_lines) — nothing to implement;
      * huge bodies (> max_body_lines) — beyond a 4B's single-episode budget, and they dominate
        wall-clock without discriminating;
      * abstract/overload decorators — see SKIP_DECORATORS.
    """
    try:
        tree = ast.parse(src_file.read_text(errors="replace"))
    except SyntaxError:
        return []
    out = []

    def visit(node, prefix):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
                qual = f"{prefix}{name}"
                if name.startswith("_"):
                    continue
                if _dec_names(child) & SKIP_DECORATORS:
                    continue
                span = _body_span(child)
                if span is None:
                    continue
                nlines = span[1] - span[0]
                if not (min_body_lines <= nlines <= max_body_lines):
                    continue
                has_doc = _is_docstring(child.body[0])
                out.append({"rel_file": rel_file, "qual_name": qual, "func_name": name,
                            "body_lines": nlines, "is_method": bool(prefix),
                            "has_docstring": has_doc, "n_args": len(child.args.args),
                            "is_async": isinstance(child, ast.AsyncFunctionDef)})
            elif isinstance(child, ast.ClassDef):
                if not child.name.startswith("_"):
                    visit(child, f"{prefix}{child.name}.")

    visit(tree, "")
    return out


def find_node(tree, qual_name):
    """Resolve an exact dotted qualified name to its AST node (None if absent)."""
    parts = qual_name.split(".")
    scope = tree
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        found = None
        for child in getattr(scope, "body", []):
            if last and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == part:
                found = child
                break
            if not last and isinstance(child, ast.ClassDef) and child.name == part:
                found = child
                break
        if found is None:
            return None
        scope = found
    return scope


def stub_function(root, rel_file, qual_name):
    """Replace a function body with STUB_MARKER, keeping signature + docstring. True on success."""
    path = root / rel_file
    try:
        src = path.read_text(errors="replace")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return False
    node = find_node(tree, qual_name)
    if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    span = _body_span(node)
    if span is None:
        return False
    start, end, indent = span
    lines = src.split("\n")
    lines[start:end] = [indent + STUB_MARKER]
    new_src = "\n".join(lines)
    try:
        ast.parse(new_src)          # never hand a syntactically broken file to an episode
    except SyntaxError:
        return False
    path.write_text(new_src)
    return True


def is_stubbed(root, rel_file):
    try:
        return STUB_MARKER in (root / rel_file).read_text(errors="replace")
    except OSError:
        return False
