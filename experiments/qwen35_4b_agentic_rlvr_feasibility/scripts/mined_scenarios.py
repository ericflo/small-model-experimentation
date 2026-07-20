"""Self-contained coding scenarios MINED FROM THE SHAPES of duet-eval — re-authored, not copied.

Why: `synth_scenarios.py` only has 12 tasks and the base clears most of them, so a GRPO group has no
within-group variance to learn from. RLVR needs (a) many more tasks and (b) tasks the model passes
SOMETIMES. These 40 are drawn from the *archetypes* of `~/Development/duet-eval/eval/scenarios`
(885 scenarios) and biased medium/hard for exactly that reason.

TRAIN/TEST FIREWALL — read this before touching the file.
Nothing here is copied from duet-eval. No prompt text, no test body, no stub, no file name, and no
identifier was lifted. What was taken is the *archetype*: "there exists a family of real tasks about
prefix tries with counts", "...about largest-remainder money allocation". Every contract, every
signature, every assertion below was written fresh in this file, and where an archetype was too close
to reproduce safely the task was deliberately rotated onto a different API (see the DIVERGENCE notes).
duet-eval remains a clean held-out eval; training on these must not contaminate it.

The firewall was audited, not just asserted: 8-gram shingles of every stub + test file here were
intersected with the prompts and file bodies of all 884 duet scenarios. Four shingles collide in
shipped material and all four are generic idiom — a fixture list `a = [1, 2, 2, 2, 3] assert`, the
JSON Schema keyword names `enum / minimum / maximum / minLength / maxLength`, and `h = MinHeap()
assert len(h) == 0`. The REFERENCE solutions add ~22 more, all inside textbook algorithm bodies (the
binary-search loop, `find(a) == find(b)`), which no correct solution can avoid. Re-run that audit if
you add tasks.

Archetype provenance (duet scenario family -> our re-authored task, with the divergence taken):
  gen4-pyalgo-trie-prefix-counts, gen4-pyalgo-union-find-path-compression -> `union_find`
      DIVERGENCE: trie dropped entirely (too identifiable); DSU rotated onto a groups()/count() API.
  gen4-pyalgo-dijkstra-adjacency-dict            -> `widest_path`   (min-of-max bottleneck, not sum)
  gen4-pyalgo-bisect-bounds-range-count          -> `binsearch_bounds` (renamed API + insert_sorted)
  gen4-pyalgo-binary-min-heap-class              -> `min_heap`      (adds pushpop/replace/heapsort)
  gen4-pyalgo-lru-cache-ordereddict              -> `lfu_cache`     (LFU w/ LRU tiebreak, not LRU)
  gen4-pyalgo-topo-sort-cycle-detect,
    gen4-graph2-kahn-lexicographic-toposort,
    hard-build-wave-scheduler-toposort           -> `topo_lex`      (lexicographic order + waves)
  gen4-graph2-bipartite-two-coloring-witness     -> `bipartite`     (witness checked structurally)
  gen4-graph2-bellman-ford-negative-cycle        -> `bellman_ford`
  gen4-graph2-flood-fill-region-count            -> `flood_fill`
  gen4-pydata-dotted-path-get-set-delete         -> `dotted_path`   (list indices in the path)
  gen4-pydata-deep-merge-lists-by-index          -> `deep_merge`    (adds a DELETE sentinel)
  gen4-pydata-stable-multikey-order-by           -> `order_by`      (sign-prefixed keys, not tuples)
  gen4-pydata-group-by-aggregate                 -> `group_stats`
  gen4-pydata-mustache-lite-template-renderer,
    hard-mustache-context-stack-engine           -> `mustache`
  gen4-pydata-type-inferring-csv-reader,
    gen4-fmt-csv-rfc4180-roundtrip,
    hard-csv-rfc4180-quoted-parser-fix           -> `csv_rfc4180`   (parse+write roundtrip, no typing)
  gen4-fmt-base32-rfc4648-roundtrip              -> `base32`
  gen4-fmt-percent-encoding-utf8-roundtrip       -> `percent_codec`
  gen4-fmt-iso8601-duration-parse                -> `iso_duration`  (adds the format direction)
  gen4-interp-ini-section-parser                 -> `ini_parser`
  gen4-num-integer-sqrt                          -> `isqrt`
  gen4-geo-polygon-shoelace-orientation          -> `shoelace`
  gen4-geo-point-in-polygon-onedge               -> `point_in_polygon`
  gen4-diff-lcs-length-edit-distance,
    gen4-diff-longest-common-substring           -> `lcs_edit`      (all three in one module)
  gen4-diff-apply-patch-conflict-detection,
    gen4-diff-invert-line-patch-roundtrip        -> `patch_apply`   (explicit hunk dicts, not text)
  gen4-fsm-token-bucket-refill                   -> `token_bucket`  (injected clock)
  gen4-fsm-circuit-breaker                       -> `circuit_breaker`
  gen4-valid-json-schema-validator               -> `schema_lite`   (returns failing paths, not msgs)
  gen4-result-core-map-flatmap-match,
    gen4-result-combine-all-sequence             -> `result_type`
  gen4-iter-dedup-keeping-last                   -> `dedup`
  gen4-iter-chunk-lazy-zip-longest               -> `chunking`
  hard-glob-path-segment-matcher                 -> `glob_match`
  hard-brace-expansion-glob                      -> `brace_expand`
  hard-rfc6901-pointer-get-set-engine            -> `json_pointer`
  hard-largest-remainder-cent-allocator          -> `allocate`
  hard-semver-max-satisfying-resolver            -> `semver`
  hard-unicode-width-truncate-ellipsis,
    r2-cli-truncate-nearest-space-boundary       -> `truncate`      (ASCII width, word boundary)
  r2-markdown-toc-slugify-space-join,
    gen-implement-slugify                        -> `slugify`
  r2-camelcase-implement-conversion              -> `case_convert`
  r2-pretty-bytes-signed-option                  -> `pretty_bytes`
  r2-bytes-parse-implement                       -> `parse_bytes`

Format is identical to synth_scenarios.py (id / prompt / files / check), plus an additive
`difficulty` key ("easy" | "medium" | "hard") for curriculum sampling — existing consumers index by
name (harvest.py, rlvr_band.py, calibrate_difficulty.py) so the extra key is inert.

Every test file uses plain asserts in functions named t0..tN called from __main__, ordered
easiest-first, because the reward function reads partial credit off which tN the runner died in.
stdlib only, no network, no third-party imports.

REFERENCE holds a working solution for every scenario, keyed by scenario id. It is not shipped to the
model; it exists to prove solvability and to support later failure analysis. Every scenario was
gated on a TWO-SIDED check before landing — write the files to a temp dir and run `check`: it must
exit non-zero with the shipped stub and exit 0 (printing ALL PASS) with REFERENCE[id] dropped in as
solution.py. Re-run that check on any edit to this file; a task that only passes one side is broken
(an unsolvable task poisons the group, a stub-passable task hands out free reward).
"""

PROMPT = ("Implement the solution in `solution.py` so that the tests pass. The specification is defined "
          "by the assertions in `test_solution.py` — read that file first. Check your work by running "
          "`python3 test_solution.py` (it prints ALL PASS and exits 0 when correct). Only edit "
          "`solution.py`. Iterate until the tests pass.")
CHECK = "python3 test_solution.py"


def _runner(asserts, imports):
    fns = [f"def t{i}():\n" + "\n".join("    " + l for l in a) for i, a in enumerate(asserts)]
    calls = "\n".join(f"    t{i}()" for i in range(len(asserts)))
    return imports + "\n\n" + "\n\n".join(fns) + "\n\nif __name__ == '__main__':\n" + calls + "\n    print('ALL PASS')\n"


def _sc(sid, diff, stub, imports, asserts):
    return {"id": sid, "difficulty": diff, "prompt": PROMPT, "check": CHECK,
            "files": {"solution.py": stub, "test_solution.py": _runner(asserts, imports)}}


# ---------------------------------------------------------------------------------------------
# EASY (11) — one obvious function or two, no hidden edge cases beyond the ones spelled out.
# ---------------------------------------------------------------------------------------------
EASY = [
    _sc("slugify", "easy",
        ("def slugify(text):\n"
         "    # lowercase; every run of non-alphanumeric ASCII becomes ONE hyphen; no leading/trailing hyphen\n"
         "    raise NotImplementedError\n"),
        "from solution import slugify",
        [["assert slugify('Hello World') == 'hello-world'"],
         ["assert slugify('  Multiple   Spaces  ') == 'multiple-spaces'"],
         ["assert slugify('Foo/Bar_Baz!') == 'foo-bar-baz'"],
         ["assert slugify('') == ''", "assert slugify('---') == ''", "assert slugify('!?') == ''"],
         ["assert slugify('a1B2') == 'a1b2'", "assert slugify('v2.0 Release') == 'v2-0-release'"]]),

    _sc("pretty_bytes", "easy",
        ("UNITS = ['B', 'kB', 'MB', 'GB', 'TB']\n\n"
         "def pretty_bytes(n):\n"
         "    # decimal units (factor 1000), at most 2 decimals with trailing zeros stripped, '-' for negatives\n"
         "    raise NotImplementedError\n"),
        "from solution import pretty_bytes",
        [["assert pretty_bytes(0) == '0 B'", "assert pretty_bytes(999) == '999 B'"],
         ["assert pretty_bytes(1000) == '1 kB'", "assert pretty_bytes(1500) == '1.5 kB'"],
         ["assert pretty_bytes(1536) == '1.54 kB'"],
         ["assert pretty_bytes(-1500) == '-1.5 kB'", "assert pretty_bytes(-1) == '-1 B'"],
         ["assert pretty_bytes(1500000) == '1.5 MB'", "assert pretty_bytes(10 ** 12) == '1 TB'",
          "assert pretty_bytes(10 ** 15) == '1000 TB'"]]),

    _sc("parse_bytes", "easy",
        ("def parse_bytes(s):\n"
         "    # inverse of a decimal byte formatter: '1.5 MB' -> 1500000. Unit optional (bytes).\n"
         "    # Case-insensitive, spaces ignored, result rounded to int. ValueError on junk.\n"
         "    raise NotImplementedError\n"),
        "from solution import parse_bytes",
        [["assert parse_bytes('0') == 0", "assert parse_bytes('120') == 120"],
         ["assert parse_bytes('1kB') == 1000", "assert parse_bytes('1 KB') == 1000"],
         ["assert parse_bytes('1.5 MB') == 1500000", "assert parse_bytes('2gb') == 2000000000"],
         ["assert parse_bytes('-2gb') == -2000000000", "assert parse_bytes('+3b') == 3"],
         ["for bad in ['abc', '', 'kb', '1 zb', '1.2.3kb']:",
          "    try:",
          "        parse_bytes(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"]]),

    _sc("chunking", "easy",
        ("def chunks(xs, n):\n"
         "    # consecutive blocks of n; the last block may be short. ValueError if n < 1.\n"
         "    raise NotImplementedError\n\n"
         "def windows(xs, n):\n"
         "    # every contiguous slice of length exactly n, left to right. ValueError if n < 1.\n"
         "    raise NotImplementedError\n"),
        "from solution import chunks, windows",
        [["assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"],
         ["assert chunks([], 3) == []", "assert chunks([1, 2, 3], 5) == [[1, 2, 3]]",
          "assert chunks([1, 2, 3], 1) == [[1], [2], [3]]"],
         ["assert windows([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]",
          "assert windows([1, 2, 3], 3) == [[1, 2, 3]]"],
         ["assert windows([1, 2], 5) == []", "assert windows([], 1) == []"],
         ["for f in (chunks, windows):",
          "    try:",
          "        f([1, 2], 0)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError from %s' % f.__name__)"]]),

    _sc("dedup", "easy",
        ("def dedup_first(xs):\n"
         "    # drop later duplicates, keep each value at its FIRST position\n"
         "    raise NotImplementedError\n\n"
         "def dedup_last(xs):\n"
         "    # drop earlier duplicates, keep each value at its LAST position\n"
         "    raise NotImplementedError\n"),
        "from solution import dedup_first, dedup_last",
        [["assert dedup_first([1, 2, 1, 3]) == [1, 2, 3]", "assert dedup_first([]) == []"],
         ["assert dedup_last([1, 2, 1, 3]) == [2, 1, 3]", "assert dedup_last([]) == []"],
         ["assert dedup_first([3, 3, 3]) == [3]", "assert dedup_last([3, 3, 3]) == [3]"],
         ["assert dedup_first(['a', 'b', 'a', 'b']) == ['a', 'b']",
          "assert dedup_last(['a', 'b', 'a', 'b']) == ['a', 'b']"],
         ["src = [1, 2, 3, 2, 1]", "assert dedup_first(src) == [1, 2, 3]", "assert dedup_last(src) == [3, 2, 1]",
          "assert src == [1, 2, 3, 2, 1]"]]),

    _sc("isqrt", "easy",
        ("def isqrt(n):\n"
         "    # floor of the square root, exact for huge ints. Do NOT use math.isqrt or floats.\n"
         "    raise NotImplementedError\n\n"
         "def is_perfect_square(n):\n"
         "    raise NotImplementedError\n"),
        "from solution import isqrt, is_perfect_square",
        [["assert isqrt(0) == 0", "assert isqrt(1) == 1", "assert isqrt(4) == 2"],
         ["assert isqrt(15) == 3", "assert isqrt(16) == 4", "assert isqrt(17) == 4"],
         ["assert is_perfect_square(49) is True", "assert is_perfect_square(50) is False",
          "assert is_perfect_square(0) is True"],
         ["assert isqrt(10 ** 18) == 10 ** 9", "assert isqrt(10 ** 18 - 1) == 10 ** 9 - 1"],
         ["for bad in (-1, -100):",
          "    try:",
          "        isqrt(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError')"],
         ["assert all(isqrt(k * k) == k and isqrt(k * k + k) == k for k in range(1, 300))"]]),

    _sc("flood_fill", "easy",
        ("def count_regions(grid):\n"
         "    # grid: list of equal-length strings, '#' is filled and '.' is empty.\n"
         "    # Count connected groups of '#' using 4-way (up/down/left/right) adjacency.\n"
         "    raise NotImplementedError\n"),
        "from solution import count_regions",
        [["assert count_regions([]) == 0", "assert count_regions(['...']) == 0"],
         ["assert count_regions(['#']) == 1", "assert count_regions(['#.#']) == 2"],
         ["assert count_regions(['##', '##']) == 1", "assert count_regions(['#.', '.#']) == 2"],
         ["assert count_regions(['#.#', '.#.', '#.#']) == 5"],
         ["g = ['##..#', '#..##', '..#..', '#...#']", "assert count_regions(g) == 5"]]),

    _sc("shoelace", "easy",
        ("def area(poly):\n"
         "    # absolute polygon area from its vertices (floats), 0.0 for degenerate input\n"
         "    raise NotImplementedError\n\n"
         "def orientation(poly):\n"
         "    # 'ccw' | 'cw' | 'degenerate'\n"
         "    raise NotImplementedError\n"),
        "from solution import area, orientation",
        [["assert area([(0, 0), (4, 0), (4, 3), (0, 3)]) == 12.0"],
         ["assert area([(0, 0), (4, 0), (0, 3)]) == 6.0", "assert area([(0, 0), (1, 1)]) == 0.0",
          "assert area([]) == 0.0"],
         ["assert orientation([(0, 0), (1, 0), (1, 1)]) == 'ccw'"],
         ["assert orientation([(0, 0), (1, 1), (1, 0)]) == 'cw'"],
         ["assert orientation([(0, 0), (1, 1), (2, 2)]) == 'degenerate'", "assert orientation([(0, 0)]) == 'degenerate'"],
         ["p = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]", "assert area(p) == 12.0",
          "assert area(list(reversed(p))) == 12.0", "assert orientation(list(reversed(p))) == 'cw'"]]),

    _sc("group_stats", "easy",
        ("def group_stats(rows, key_field, value_field):\n"
         "    # rows: list of dicts. Group by rows[key_field]; per group return\n"
         "    # {'count': int, 'sum': number, 'min': number, 'max': number, 'avg': float}.\n"
         "    # A row missing either field raises KeyError.\n"
         "    raise NotImplementedError\n"),
        "from solution import group_stats",
        [["assert group_stats([], 'c', 'v') == {}"],
         ["assert group_stats([{'c': 'a', 'v': 5}], 'c', 'v') == "
          "{'a': {'count': 1, 'sum': 5, 'min': 5, 'max': 5, 'avg': 5.0}}"],
         ["rows = [{'c': 'a', 'v': 1}, {'c': 'a', 'v': 3}, {'c': 'b', 'v': 5}]",
          "assert group_stats(rows, 'c', 'v') == {",
          "    'a': {'count': 2, 'sum': 4, 'min': 1, 'max': 3, 'avg': 2.0},",
          "    'b': {'count': 1, 'sum': 5, 'min': 5, 'max': 5, 'avg': 5.0}}"],
         ["rows = [{'c': 'x', 'v': -2}, {'c': 'x', 'v': 2}, {'c': 'x', 'v': 3}]",
          "assert group_stats(rows, 'c', 'v')['x'] == {'count': 3, 'sum': 3, 'min': -2, 'max': 3, 'avg': 1.0}"],
         ["try:",
          "    group_stats([{'c': 'a'}], 'c', 'v')",
          "except KeyError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected KeyError')"]]),

    _sc("truncate", "easy",
        ("def truncate(s, width, ellipsis='...'):\n"
         "    # Return s unchanged if len(s) <= width. Otherwise return a string of at most `width`\n"
         "    # chars ending in `ellipsis`. Cut at the last space inside the kept region when the cut\n"
         "    # would land mid-word, then strip trailing spaces. ValueError if width < len(ellipsis).\n"
         "    raise NotImplementedError\n"),
        "from solution import truncate",
        [["assert truncate('hello world', 20) == 'hello world'", "assert truncate('abc', 3) == 'abc'"],
         ["assert truncate('abcdefgh', 5) == 'ab...'", "assert truncate('abcd', 3) == '...'"],
         ["assert truncate('hello world', 8) == 'hello...'"],
         ["assert truncate('hello world', 9) == 'hello...'", "assert truncate('hello world', 10) == 'hello...'"],
         ["assert truncate('one two three', 9, '..') == 'one two..'"],
         ["try:",
          "    truncate('abcdef', 2)",
          "except ValueError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ValueError')"]]),

    _sc("case_convert", "easy",
        ("def to_snake(s):\n"
         "    # 'helloWorld' -> 'hello_world'; acronym runs stay together ('HTTPServer' -> 'http_server');\n"
         "    # hyphens and spaces are separators too\n"
         "    raise NotImplementedError\n\n"
         "def to_camel(s):\n"
         "    raise NotImplementedError\n\n"
         "def to_pascal(s):\n"
         "    raise NotImplementedError\n"),
        "from solution import to_snake, to_camel, to_pascal",
        [["assert to_snake('helloWorld') == 'hello_world'", "assert to_snake('hello_world') == 'hello_world'"],
         ["assert to_snake('HTTPServer') == 'http_server'", "assert to_snake('parseHTTPResponse') == 'parse_http_response'"],
         ["assert to_snake('foo-bar baz') == 'foo_bar_baz'", "assert to_snake('  spaced  out ') == 'spaced_out'"],
         ["assert to_camel('hello_world') == 'helloWorld'", "assert to_camel('foo-bar baz') == 'fooBarBaz'"],
         ["assert to_pascal('hello_world') == 'HelloWorld'", "assert to_pascal('') == ''", "assert to_camel('') == ''"],
         ["for s in ['helloWorld', 'parseHTTPResponse', 'foo_bar_baz']:",
          "    assert to_snake(to_camel(s)) == to_snake(s)",
          "    assert to_snake(to_pascal(s)) == to_snake(s)"]]),
]

# ---------------------------------------------------------------------------------------------
# MEDIUM (14) — multi-step logic, or one function with edge cases the obvious solution misses.
# ---------------------------------------------------------------------------------------------
MEDIUM = [
    _sc("percent_codec", "medium",
        ("UNRESERVED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'\n\n"
         "def percent_encode(s):\n"
         "    # UTF-8 encode, then %XX (UPPERCASE hex) every byte outside UNRESERVED\n"
         "    raise NotImplementedError\n\n"
         "def percent_decode(s):\n"
         "    # inverse; %xx is case-insensitive. ValueError on a truncated/invalid escape or bad UTF-8.\n"
         "    raise NotImplementedError\n"),
        "from solution import percent_encode, percent_decode",
        [["assert percent_encode('abc') == 'abc'", "assert percent_encode('a-b._~') == 'a-b._~'",
          "assert percent_encode('') == ''"],
         ["assert percent_encode('a b') == 'a%20b'", "assert percent_encode('a/b?c=d') == 'a%2Fb%3Fc%3Dd'"],
         ["assert percent_encode('\\u00e9') == '%C3%A9'", "assert percent_encode('\\u20ac') == '%E2%82%AC'"],
         ["assert percent_decode('abc') == 'abc'", "assert percent_decode('a%20b') == 'a b'",
          "assert percent_decode('%c3%a9') == '\\u00e9'"],
         ["for bad in ['%zz', '%A', '%', 'ok%2']:",
          "    try:",
          "        percent_decode(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"],
         ["for s in ['', 'plain', 'a b/c', '\\u00e9\\u20ac', '100% sure?', '~-._']:",
          "    assert percent_decode(percent_encode(s)) == s"]]),

    _sc("base32_crockford", "medium",
        ("ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'  # Crockford: no I, L, O or U\n\n"
         "def b32_encode(data):\n"
         "    # data: bytes -> str. Big-endian bit packing, 5 bits per char, RIGHT-padded with zero\n"
         "    # bits to fill the last char. NO '=' padding: exactly ceil(8*len(data)/5) chars.\n"
         "    raise NotImplementedError\n\n"
         "def b32_decode(s):\n"
         "    # str -> bytes. Case-insensitive; '-' separators ignored; 'I'/'L' read as '1' and 'O' as '0'.\n"
         "    # ValueError on an unknown character or if the trailing pad bits are not zero.\n"
         "    raise NotImplementedError\n"),
        "from solution import b32_encode, b32_decode",
        [["assert b32_encode(b'') == ''", "assert b32_decode('') == b''"],
         ["assert b32_encode(b'f') == 'CR'", "assert b32_encode(b'\\x00') == '00'",
          "assert b32_encode(b'\\xff') == 'ZW'"],
         ["assert b32_encode(b'foo') == 'CSQPY'", "assert b32_encode(b'foobar') == 'CSQPYRK1E8'"],
         ["assert b32_decode('CR') == b'f'", "assert b32_decode('csqpy') == b'foo'",
          "assert b32_decode('CSQP-YRK1-E8') == b'foobar'"],
         ["assert b32_decode('CR') == b32_decode('cr')", "assert b32_decode('IC') == b32_decode('1C')",
          "assert b32_decode('LC') == b32_decode('1C')", "assert b32_decode('CO') == b32_decode('C0')"],
         ["for bad in ['CU', 'C!', 'CZ']:",
          "    try:",
          "        b32_decode(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"],
         ["for n in range(0, 24):",
          "    data = bytes((i * 37 + 11) % 256 for i in range(n))",
          "    assert b32_decode(b32_encode(data)) == data"]]),

    _sc("ini_parser", "medium",
        ("def parse_ini(text):\n"
         "    # -> {section_name: {key: value}}. Keys before any [section] live under ''.\n"
         "    # ';' and '#' start whole-line comments; blank lines are skipped; keys and values are\n"
         "    # stripped; only the FIRST '=' splits; a duplicate key takes the last value; a section\n"
         "    # header with no keys still appears (empty dict). ValueError on a malformed line.\n"
         "    raise NotImplementedError\n"),
        "from solution import parse_ini",
        [["assert parse_ini('') == {}", "assert parse_ini('\\n\\n  \\n') == {}"],
         ["assert parse_ini('a=1\\nb = 2') == {'': {'a': '1', 'b': '2'}}"],
         ["assert parse_ini('[s]\\nx=1\\n; note\\n# note2\\n\\n[t]\\ny=2') == {'s': {'x': '1'}, 't': {'y': '2'}}",
          "assert parse_ini('[empty]') == {'empty': {}}"],
         ["assert parse_ini('url=http://x?a=b') == {'': {'url': 'http://x?a=b'}}",
          "assert parse_ini('k=') == {'': {'k': ''}}"],
         ["assert parse_ini('[s]\\nk=1\\nk=2') == {'s': {'k': '2'}}",
          "assert parse_ini('top=0\\n[s]\\nk=1') == {'': {'top': '0'}, 's': {'k': '1'}}"],
         ["for bad in ['garbage', '[unclosed', '=5', '[]']:",
          "    try:",
          "        parse_ini(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"]]),

    _sc("iso_duration", "medium",
        ("def parse_duration(s):\n"
         "    # ISO-8601 duration -> whole seconds. Supported: PnW, PnD, and PT with nH/nM/nS in that\n"
         "    # order; any subset may be omitted but at least one must be present. Years and months\n"
         "    # are NOT supported. ValueError on anything else (including a bare 'P' or a trailing 'T').\n"
         "    raise NotImplementedError\n\n"
         "def format_duration(seconds):\n"
         "    # inverse, in canonical days/hours/minutes/seconds form (never weeks). 0 -> 'PT0S'.\n"
         "    raise NotImplementedError\n"),
        "from solution import parse_duration, format_duration",
        [["assert parse_duration('PT1H') == 3600", "assert parse_duration('PT0S') == 0",
          "assert parse_duration('P1D') == 86400"],
         ["assert parse_duration('PT1H30M') == 5400", "assert parse_duration('PT90M') == 5400",
          "assert parse_duration('P1DT2H3M4S') == 93784"],
         ["assert parse_duration('P2W') == 1209600"],
         ["for bad in ['1H', 'P1Y', 'P', 'PT', '', 'P1M2D', 'PT1S1H']:",
          "    try:",
          "        parse_duration(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"],
         ["assert format_duration(0) == 'PT0S'", "assert format_duration(3600) == 'PT1H'",
          "assert format_duration(86400) == 'P1D'", "assert format_duration(93784) == 'P1DT2H3M4S'",
          "assert format_duration(61) == 'PT1M1S'"],
         ["for n in [0, 1, 59, 60, 61, 3599, 3600, 86399, 86400, 93784, 1209600]:",
          "    assert parse_duration(format_duration(n)) == n"]]),

    _sc("binsearch_bounds", "medium",
        ("def lower_bound(a, x):\n"
         "    # leftmost index i with a[i] >= x (so x would insert BEFORE any equal elements)\n"
         "    raise NotImplementedError\n\n"
         "def upper_bound(a, x):\n"
         "    # leftmost index i with a[i] > x (so x would insert AFTER any equal elements)\n"
         "    raise NotImplementedError\n\n"
         "def count_between(a, lo, hi):\n"
         "    # how many elements satisfy lo <= v <= hi, INCLUSIVE; 0 when lo > hi\n"
         "    raise NotImplementedError\n\n"
         "def insert_sorted(a, x):\n"
         "    # NEW list with x placed at upper_bound(a, x); `a` is not mutated\n"
         "    raise NotImplementedError\n"),
        "from solution import lower_bound, upper_bound, count_between, insert_sorted",
        [["assert lower_bound([], 5) == 0", "assert upper_bound([], 5) == 0", "assert count_between([], 1, 2) == 0"],
         ["a = [1, 2, 2, 2, 3]", "assert lower_bound(a, 2) == 1", "assert upper_bound(a, 2) == 4",
          "assert upper_bound(a, 2) - lower_bound(a, 2) == 3"],
         ["a = [1, 2, 2, 2, 3]", "assert lower_bound(a, 0) == 0", "assert lower_bound(a, 9) == 5",
          "assert upper_bound(a, 0) == 0", "assert upper_bound(a, 9) == 5",
          "assert lower_bound(a, 3) == 4", "assert upper_bound(a, 3) == 5"],
         ["a = [1, 2, 2, 2, 3]", "assert count_between(a, 2, 3) == 4", "assert count_between(a, 0, 9) == 5",
          "assert count_between(a, 4, 9) == 0", "assert count_between(a, 3, 2) == 0"],
         ["b = [-5, -5, -5]", "assert lower_bound(b, -5) == 0", "assert upper_bound(b, -5) == 3",
          "assert count_between(b, -5, -5) == 3", "assert count_between(b, -6, -6) == 0"],
         ["src = [1, 2, 2, 3]", "assert insert_sorted(src, 2) == [1, 2, 2, 2, 3]", "assert src == [1, 2, 2, 3]",
          "assert insert_sorted([], 1) == [1]", "assert insert_sorted([2, 3], 1) == [1, 2, 3]"],
         ["a = [0, 1, 1, 3, 5, 5, 5, 8]",
          "for x in range(-2, 11):",
          "    assert lower_bound(a, x) == sum(1 for v in a if v < x)",
          "    assert upper_bound(a, x) == sum(1 for v in a if v <= x)",
          "    for y in range(-2, 11):",
          "        assert count_between(a, x, y) == sum(1 for v in a if x <= v <= y)"]]),

    _sc("dotted_path", "medium",
        ("def get_path(obj, path, default=None):\n"
         "    # 'a.b.1.c' walks dicts by key and lists by int index (negatives allowed).\n"
         "    # Any miss (absent key, out-of-range index, or a scalar in the middle) returns `default`.\n"
         "    raise NotImplementedError\n\n"
         "def set_path(obj, path, value):\n"
         "    # mutate obj in place, creating intermediate dicts as needed; return obj\n"
         "    raise NotImplementedError\n\n"
         "def del_path(obj, path):\n"
         "    # remove the leaf; True if something was removed, False if it was already absent\n"
         "    raise NotImplementedError\n"),
        "from solution import get_path, set_path, del_path",
        [["assert get_path({'a': {'b': 1}}, 'a.b') == 1", "assert get_path({'a': 1}, 'a') == 1"],
         ["assert get_path({'a': [10, 20]}, 'a.1') == 20", "assert get_path({'a': [10, 20]}, 'a.-1') == 20",
          "assert get_path({'a': [{'b': 7}]}, 'a.0.b') == 7"],
         ["assert get_path({}, 'x.y', 'z') == 'z'", "assert get_path({'a': 1}, 'a.b') is None",
          "assert get_path({'a': [1]}, 'a.5', 'd') == 'd'", "assert get_path({'a': [1]}, 'a.q', 'd') == 'd'"],
         ["d = {}", "assert set_path(d, 'a.b.c', 1) is d", "assert d == {'a': {'b': {'c': 1}}}",
          "d2 = {'a': [1, 2]}", "set_path(d2, 'a.0', 9)", "assert d2 == {'a': [9, 2]}",
          "set_path(d2, 'a', 5)", "assert d2 == {'a': 5}"],
         ["d = {'a': {'b': 1, 'c': 2}}", "assert del_path(d, 'a.b') is True", "assert d == {'a': {'c': 2}}",
          "assert del_path(d, 'a.zz') is False", "assert del_path(d, 'q.r') is False",
          "assert d == {'a': {'c': 2}}"],
         ["d = {'a': [1, 2, 3]}", "assert del_path(d, 'a.1') is True", "assert d == {'a': [1, 3]}",
          "for f, args in [(get_path, ({}, '')), (set_path, ({}, '', 1)), (del_path, ({}, ''))]:",
          "    try:",
          "        f(*args)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError from %s' % f.__name__)"]]),

    _sc("deep_merge", "medium",
        ("DELETE = object()  # sentinel, already provided: an override value of DELETE removes the key\n\n\n"
         "def deep_merge(base, override):\n"
         "    # Return a NEW structure. Dicts merge key-by-key, recursively. Lists merge BY INDEX (the\n"
         "    # override's element i wins, recursively; whichever list is longer supplies the tail).\n"
         "    # Anything else: override wins. Neither argument may be mutated, and the result must not\n"
         "    # share any mutable object with either input.\n"
         "    raise NotImplementedError\n"),
        "from solution import deep_merge, DELETE",
        [["assert deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}",
          "assert deep_merge({'a': 1}, {'a': 2}) == {'a': 2}", "assert deep_merge({}, {}) == {}"],
         ["assert deep_merge({'a': {'x': 1, 'y': 2}}, {'a': {'y': 3}}) == {'a': {'x': 1, 'y': 3}}",
          "assert deep_merge({'a': {'x': 1}}, {'a': 5}) == {'a': 5}"],
         ["assert deep_merge({'l': [1, 2, 3]}, {'l': [9]}) == {'l': [9, 2, 3]}",
          "assert deep_merge({'l': [1]}, {'l': [9, 8]}) == {'l': [9, 8]}",
          "assert deep_merge({'l': []}, {'l': [1]}) == {'l': [1]}"],
         ["assert deep_merge({'l': [{'a': 1}]}, {'l': [{'b': 2}]}) == {'l': [{'a': 1, 'b': 2}]}"],
         ["assert deep_merge({'a': 1, 'b': 2}, {'b': DELETE}) == {'a': 1}",
          "assert deep_merge({'a': {'b': 1, 'c': 2}}, {'a': {'b': DELETE}}) == {'a': {'c': 2}}",
          "assert deep_merge({'a': 1}, {'zz': DELETE}) == {'a': 1}"],
         ["b = {'a': {'x': 1}, 'l': [1, 2]}", "o = {'a': {'y': 2}}", "r = deep_merge(b, o)",
          "assert r == {'a': {'x': 1, 'y': 2}, 'l': [1, 2]}",
          "assert b == {'a': {'x': 1}, 'l': [1, 2]} and o == {'a': {'y': 2}}",
          "r['a']['x'] = 99", "r['l'].append(3)",
          "assert b == {'a': {'x': 1}, 'l': [1, 2]}"]]),

    _sc("order_by", "medium",
        ("def order_by(rows, keys):\n"
         "    # rows: list of dicts. keys: list of field names; a leading '-' means DESCENDING (a\n"
         "    # leading '+' is an explicit ascending). Sorting is stable and lexicographic across keys.\n"
         "    # A field that is missing or None sorts LAST under BOTH directions. Returns a NEW list;\n"
         "    # `rows` is not mutated. ValueError if a key has no field name after the sign.\n"
         "    raise NotImplementedError\n"),
        "from solution import order_by",
        [["assert order_by([], ['a']) == []"],
         ["rows = [{'a': 2}, {'a': 1}, {'a': 3}]",
          "assert order_by(rows, ['a']) == [{'a': 1}, {'a': 2}, {'a': 3}]",
          "assert rows == [{'a': 2}, {'a': 1}, {'a': 3}]",
          "out = order_by(rows, [])", "assert out == rows and out is not rows"],
         ["rows = [{'a': 2}, {'a': 1}, {'a': 3}]",
          "assert order_by(rows, ['-a']) == [{'a': 3}, {'a': 2}, {'a': 1}]",
          "assert order_by(rows, ['+a']) == [{'a': 1}, {'a': 2}, {'a': 3}]"],
         ["rows = [{'a': 1, 'b': 2}, {'a': 1, 'b': 1}, {'a': 0, 'b': 9}]",
          "assert order_by(rows, ['a', 'b']) == [{'a': 0, 'b': 9}, {'a': 1, 'b': 1}, {'a': 1, 'b': 2}]",
          "assert order_by(rows, ['a', '-b']) == [{'a': 0, 'b': 9}, {'a': 1, 'b': 2}, {'a': 1, 'b': 1}]"],
         ["rows = [{'a': 1, 'i': 0}, {'i': 1}, {'a': None, 'i': 2}, {'a': 2, 'i': 3}]",
          "assert [r['i'] for r in order_by(rows, ['a'])] == [0, 3, 1, 2]",
          "assert [r['i'] for r in order_by(rows, ['-a'])] == [3, 0, 1, 2]"],
         ["rows = [{'k': 'b', 'i': 0}, {'k': 'a', 'i': 1}, {'k': 'b', 'i': 2}, {'k': 'a', 'i': 3}]",
          "assert [r['i'] for r in order_by(rows, ['k'])] == [1, 3, 0, 2]",
          "assert [r['i'] for r in order_by(rows, ['-k'])] == [0, 2, 1, 3]"],
         ["for bad in ['-', '+', '']:",
          "    try:",
          "        order_by([{'a': 1}], [bad])",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"]]),

    _sc("token_bucket", "medium",
        ("class TokenBucket:\n"
         "    def __init__(self, capacity, refill_per_sec):\n"
         "        # starts FULL. The clock is injected: every call passes the current time.\n"
         "        raise NotImplementedError\n\n"
         "    def allow(self, now, n=1):\n"
         "        # refill by (now - last_seen) * refill_per_sec, capped at capacity, then try to spend\n"
         "        # n tokens. True if spent, False if not (and then nothing is spent).\n"
         "        # ValueError if n < 1, if n > capacity, or if `now` moves backwards.\n"
         "        raise NotImplementedError\n"),
        "from solution import TokenBucket",
        [["b = TokenBucket(2, 1.0)", "assert b.allow(0.0) is True", "assert b.allow(0.0) is True",
          "assert b.allow(0.0) is False"],
         ["b = TokenBucket(2, 1.0)", "b.allow(0.0)", "b.allow(0.0)",
          "assert b.allow(0.5) is False", "assert b.allow(1.0) is True"],
         ["b = TokenBucket(2, 1.0)", "b.allow(0.0)", "b.allow(0.0)",
          "assert b.allow(100.0) is True", "assert b.allow(100.0) is True", "assert b.allow(100.0) is False"],
         ["b = TokenBucket(5, 1.0)", "assert b.allow(0.0, 3) is True", "assert b.allow(0.0, 3) is False",
          "assert b.allow(0.0, 2) is True", "assert b.allow(0.0, 1) is False"],
         ["b = TokenBucket(1, 0.5)", "assert b.allow(0.0) is True", "assert b.allow(1.0) is False",
          "assert b.allow(2.0) is True"],
         ["b = TokenBucket(2, 1.0)", "b.allow(5.0)",
          "for args in [(5.0, 0), (5.0, 3), (4.0, 1)]:",
          "    try:",
          "        b.allow(*args)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % (args,))"]]),

    _sc("lcs_edit", "medium",
        ("def lcs_length(a, b):\n"
         "    # length of the longest common SUBSEQUENCE\n"
         "    raise NotImplementedError\n\n"
         "def edit_distance(a, b):\n"
         "    # Levenshtein: insert / delete / substitute, all cost 1\n"
         "    raise NotImplementedError\n\n"
         "def longest_common_substring(a, b):\n"
         "    # the longest CONTIGUOUS run present in both. On a tie, the one starting earliest in `a`.\n"
         "    raise NotImplementedError\n"),
        "from solution import lcs_length, edit_distance, longest_common_substring",
        [["assert lcs_length('', 'abc') == 0", "assert lcs_length('abc', '') == 0", "assert lcs_length('abc', 'abc') == 3"],
         ["assert lcs_length('abcde', 'ace') == 3", "assert lcs_length('abc', 'xyz') == 0",
          "assert lcs_length('aggtab', 'gxtxayb') == 4"],
         ["assert edit_distance('', 'abc') == 3", "assert edit_distance('abc', '') == 3",
          "assert edit_distance('abc', 'abc') == 0"],
         ["assert edit_distance('kitten', 'sitting') == 3", "assert edit_distance('flaw', 'lawn') == 2",
          "assert edit_distance('sunday', 'saturday') == 3"],
         ["assert longest_common_substring('abcdef', 'zabcy') == 'abc'",
          "assert longest_common_substring('abc', 'xyz') == ''",
          "assert longest_common_substring('banana', 'ananas') == 'anana'"],
         ["assert longest_common_substring('aXbYc', 'abc') == 'a'",
          "assert longest_common_substring('', 'abc') == ''"],
         ["for a, b in [('abcde', 'ace'), ('kitten', 'sitting'), ('', ''), ('xx', 'xx')]:",
          "    assert edit_distance(a, b) == edit_distance(b, a)",
          "    assert lcs_length(a, b) == lcs_length(b, a)",
          "    assert edit_distance(a, b) >= abs(len(a) - len(b))"]]),

    _sc("schema_lite", "medium",
        ("def validate(schema, value):\n"
         "    # Return the SORTED list of paths that fail, '$' for the root, '$.k' for a property and\n"
         "    # '$[i]' for an array element. [] means valid. Supported keywords:\n"
         "    #   type: object|array|string|integer|number|boolean|null  (bool is NOT an integer/number)\n"
         "    #   required: [key, ...]      -> a missing key reports '$.key'\n"
         "    #   properties: {key: schema} -> only checked when the key is present\n"
         "    #   items: schema             -> applied to every element\n"
         "    #   enum / minimum / maximum / minLength / maxLength\n"
         "    # A wrong type short-circuits: do not also report the constraints below it.\n"
         "    raise NotImplementedError\n"),
        "from solution import validate",
        [["assert validate({'type': 'string'}, 'ok') == []", "assert validate({}, 12345) == []",
          "assert validate({'type': 'null'}, None) == []"],
         ["assert validate({'type': 'string'}, 5) == ['$']", "assert validate({'type': 'integer'}, True) == ['$']",
          "assert validate({'type': 'number'}, '5') == ['$']", "assert validate({'type': 'number'}, 5) == []",
          "assert validate({'type': 'boolean'}, True) == []"],
         ["s = {'type': 'object', 'required': ['a', 'b'], 'properties': {'a': {'type': 'integer'}}}",
          "assert validate(s, {'a': 1, 'b': 'x'}) == []", "assert validate(s, {'a': 'x'}) == ['$.a', '$.b']"],
         ["s = {'type': 'object', 'properties': {'a': {'type': 'object', 'properties': {'b': {'type': 'integer'}}}}}",
          "assert validate(s, {'a': {'b': 'no'}}) == ['$.a.b']", "assert validate(s, {'a': {'b': 3}}) == []",
          "assert validate(s, {}) == []"],
         ["s = {'type': 'array', 'items': {'type': 'number'}}",
          "assert validate(s, [1, 2.5]) == []", "assert validate(s, [1, 'x', 3, 'y']) == ['$[1]', '$[3]']",
          "assert validate(s, 'nope') == ['$']"],
         ["assert validate({'type': 'integer', 'minimum': 1, 'maximum': 10}, 0) == ['$']",
          "assert validate({'type': 'integer', 'minimum': 1, 'maximum': 10}, 5) == []",
          "assert validate({'type': 'string', 'minLength': 2, 'maxLength': 3}, 'a') == ['$']",
          "assert validate({'enum': ['a', 'b']}, 'c') == ['$']", "assert validate({'enum': ['a', 'b']}, 'a') == []"],
         ["s = {'type': 'object', 'required': ['xs'],",
          "     'properties': {'xs': {'type': 'array', 'items': {'type': 'object',",
          "                                                      'required': ['n'],",
          "                                                      'properties': {'n': {'type': 'integer',",
          "                                                                           'minimum': 0}}}}}}",
          "assert validate(s, {'xs': [{'n': 1}, {'n': -1}, {}]}) == ['$.xs[1].n', '$.xs[2].n']",
          "assert validate(s, {}) == ['$.xs']"]]),

    _sc("point_in_polygon", "medium",
        ("def classify(poly, pt):\n"
         "    # poly: list of (x, y) vertices in order, implicitly closed. Return 'inside', 'outside'\n"
         "    # or 'boundary' (the point lies exactly on an edge or vertex). Works for concave\n"
         "    # polygons and must not be fooled by the ray passing through a vertex.\n"
         "    # Fewer than 3 vertices is degenerate: always 'outside'.\n"
         "    raise NotImplementedError\n"),
        "from solution import classify",
        [["sq = [(0, 0), (4, 0), (4, 4), (0, 4)]", "assert classify(sq, (2, 2)) == 'inside'",
          "assert classify(sq, (5, 2)) == 'outside'", "assert classify(sq, (-1, -1)) == 'outside'"],
         ["sq = [(0, 0), (4, 0), (4, 4), (0, 4)]", "assert classify(sq, (2, 0)) == 'boundary'",
          "assert classify(sq, (0, 0)) == 'boundary'", "assert classify(sq, (4, 2)) == 'boundary'",
          "assert classify(sq, (0, 4)) == 'boundary'"],
         ["tri = [(0, 0), (4, 0), (2, 4)]", "assert classify(tri, (2, 1)) == 'inside'",
          "assert classify(tri, (2, 4)) == 'boundary'", "assert classify(tri, (0, 4)) == 'outside'",
          "assert classify(tri, (2, 5)) == 'outside'"],
         ["arrow = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]",
          "assert classify(arrow, (2, 1)) == 'inside'", "assert classify(arrow, (2, 3)) == 'outside'",
          "assert classify(arrow, (2, 2)) == 'boundary'"],
         ["assert classify([], (0, 0)) == 'outside'", "assert classify([(0, 0), (1, 1)], (0, 0)) == 'outside'"],
         ["sq = [(0, 0), (4, 0), (4, 4), (0, 4)]",
          "for x in range(-1, 6):",
          "    for y in range(-1, 6):",
          "        got = classify(sq, (x, y))",
          "        on = (0 <= x <= 4 and y in (0, 4)) or (0 <= y <= 4 and x in (0, 4))",
          "        want = 'boundary' if on else ('inside' if 0 < x < 4 and 0 < y < 4 else 'outside')",
          "        assert got == want, (x, y, got, want)"]]),

    _sc("semver", "medium",
        ("def parse(v):\n"
         "    # 'MAJOR.MINOR.PATCH' -> (major, minor, patch). Exactly three numeric parts, no\n"
         "    # pre-release or build metadata. ValueError otherwise.\n"
         "    raise NotImplementedError\n\n"
         "def compare(a, b):\n"
         "    # -1 / 0 / 1, comparing each part NUMERICALLY (so 1.10.0 > 1.2.3)\n"
         "    raise NotImplementedError\n\n"
         "def satisfies(v, rng):\n"
         "    # rng is one of: '*', an exact version, '>=x', '>x', '<=x', '<x', '^x' or '~x'.\n"
         "    #   ^1.2.3 -> >=1.2.3 and <2.0.0    ^0.2.3 -> >=0.2.3 and <0.3.0    ^0.0.3 -> >=0.0.3 and <0.0.4\n"
         "    #   ~1.2.3 -> >=1.2.3 and <1.3.0\n"
         "    raise NotImplementedError\n\n"
         "def max_satisfying(versions, rng):\n"
         "    # the highest version in `versions` that satisfies rng, or None\n"
         "    raise NotImplementedError\n"),
        "from solution import parse, compare, satisfies, max_satisfying",
        [["assert parse('1.2.3') == (1, 2, 3)", "assert parse('0.0.0') == (0, 0, 0)",
          "assert parse('10.20.30') == (10, 20, 30)"],
         ["for bad in ['1.2', '1.2.3.4', 'v1.2.3', '1.2.x', '', '1.2.3-rc1']:",
          "    try:",
          "        parse(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"],
         ["assert compare('1.2.3', '1.10.0') == -1", "assert compare('1.10.0', '1.2.3') == 1",
          "assert compare('2.0.0', '2.0.0') == 0", "assert compare('2.0.0', '10.0.0') == -1"],
         ["assert satisfies('1.2.3', '*') is True", "assert satisfies('1.2.3', '1.2.3') is True",
          "assert satisfies('1.2.4', '1.2.3') is False", "assert satisfies('1.2.4', '>=1.2.3') is True",
          "assert satisfies('1.2.3', '>1.2.3') is False", "assert satisfies('1.2.2', '<1.2.3') is True",
          "assert satisfies('1.2.3', '<=1.2.3') is True"],
         ["assert satisfies('1.2.5', '^1.2.3') is True", "assert satisfies('1.9.9', '^1.2.3') is True",
          "assert satisfies('2.0.0', '^1.2.3') is False", "assert satisfies('1.2.2', '^1.2.3') is False"],
         ["assert satisfies('0.2.9', '^0.2.3') is True", "assert satisfies('0.3.0', '^0.2.3') is False",
          "assert satisfies('0.0.3', '^0.0.3') is True", "assert satisfies('0.0.4', '^0.0.3') is False"],
         ["assert satisfies('1.2.9', '~1.2.3') is True", "assert satisfies('1.3.0', '~1.2.3') is False",
          "assert satisfies('1.2.2', '~1.2.3') is False"],
         ["vs = ['1.0.0', '1.2.4', '1.3.0', '2.0.0', '1.10.1']",
          "assert max_satisfying(vs, '^1.2.0') == '1.10.1'", "assert max_satisfying(vs, '~1.2.0') == '1.2.4'",
          "assert max_satisfying(vs, '*') == '2.0.0'", "assert max_satisfying(vs, '^3.0.0') is None",
          "assert max_satisfying([], '*') is None"]]),

    _sc("allocate", "medium",
        ("def allocate(total, weights):\n"
         "    # Split `total` (an int, may be negative) into len(weights) ints that sum EXACTLY to\n"
         "    # total, in proportion to the weights. Use the largest-remainder method: give everyone\n"
         "    # their floored share, then hand the leftover units out one at a time, largest remainder\n"
         "    # first, ties broken by lowest index. Use integer arithmetic only — no floats.\n"
         "    # ValueError on an empty weights list, a negative weight, or weights summing to zero.\n"
         "    raise NotImplementedError\n"),
        "from solution import allocate",
        [["assert allocate(100, [1, 1]) == [50, 50]", "assert allocate(0, [1, 2]) == [0, 0]",
          "assert allocate(5, [1]) == [5]"],
         ["assert allocate(100, [1, 1, 1]) == [34, 33, 33]"],
         ["assert allocate(7, [3, 1]) == [5, 2]", "assert allocate(10, [0, 0, 1]) == [0, 0, 10]"],
         ["assert allocate(100, [1] * 6) == [17, 17, 17, 17, 16, 16]"],
         ["assert allocate(-100, [1, 1, 1]) == [-33, -33, -34]", "assert sum(allocate(-100, [1, 1, 1])) == -100"],
         ["for bad in [(10, []), (10, [1, -1]), (10, [0, 0])]:",
          "    try:",
          "        allocate(*bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % (bad,))"],
         ["ws = [3, 1, 4, 1, 5, 9, 2, 6]",
          "for total in range(-20, 40):",
          "    out = allocate(total, ws)",
          "    assert len(out) == len(ws) and sum(out) == total",
          "    assert all(isinstance(x, int) for x in out)"]]),
]

# ---------------------------------------------------------------------------------------------
# HARD (15) — several cooperating functions, a stateful class, or one genuinely subtle algorithm.
# ---------------------------------------------------------------------------------------------
HARD = [
    _sc("glob_match", "hard",
        ("def match(pattern, path):\n"
         "    # Path-segment glob over '/'-separated strings.\n"
         "    #   '*'  matches any run of characters WITHIN one segment (never crosses '/')\n"
         "    #   '?'  matches exactly one character within one segment\n"
         "    #   '**' as a WHOLE segment matches zero or more consecutive segments\n"
         "    raise NotImplementedError\n"),
        "from solution import match",
        [["assert match('a/b', 'a/b') is True", "assert match('a/b', 'a/c') is False",
          "assert match('a', 'a/b') is False"],
         ["assert match('a/*', 'a/b') is True", "assert match('a/*', 'a/b/c') is False",
          "assert match('a/*', 'a') is False", "assert match('*', 'a') is True"],
         ["assert match('*.py', 'main.py') is True", "assert match('*.py', 'src/main.py') is False",
          "assert match('*.py', 'main.pyc') is False"],
         ["assert match('a/?/c', 'a/b/c') is True", "assert match('a/?/c', 'a/bb/c') is False",
          "assert match('a/?/c', 'a//c') is False"],
         ["assert match('a/**', 'a/b/c') is True", "assert match('a/**', 'a') is True",
          "assert match('a/**', 'b/c') is False"],
         ["assert match('**/x', 'x') is True", "assert match('**/x', 'a/b/x') is True",
          "assert match('**/x', 'a/x/b') is False"],
         ["assert match('a/**/d', 'a/d') is True", "assert match('a/**/d', 'a/b/c/d') is True",
          "assert match('a/**/d', 'a/b/c') is False", "assert match('src/**/*.py', 'src/x/y/z.py') is True"]]),

    _sc("brace_expand", "hard",
        ("def expand(s):\n"
         "    # Shell-style brace expansion, in order, as a list of strings.\n"
         "    #   '{a,b}'      alternation (at least two alternatives), nestable\n"
         "    #   '{1..4}'     inclusive numeric range, descending if the first bound is larger\n"
         "    #   '{a..c}'     inclusive single-letter range\n"
         "    #   '{1..9..3}'  range with a step (the step's sign is ignored)\n"
         "    # A brace group that is neither (no top-level comma, not a range) stays literal, and so\n"
         "    # does an unmatched '{'. Groups combine as a cross product, left group varying slowest.\n"
         "    raise NotImplementedError\n"),
        "from solution import expand",
        [["assert expand('a') == ['a']", "assert expand('') == ['']"],
         ["assert expand('a{b,c}d') == ['abd', 'acd']", "assert expand('{a,b}') == ['a', 'b']"],
         ["assert expand('{a,b}{1,2}') == ['a1', 'a2', 'b1', 'b2']"],
         ["assert expand('a{b,{c,d}}e') == ['abe', 'ace', 'ade']",
          "assert expand('{a,b{c,d}}') == ['a', 'bc', 'bd']"],
         ["assert expand('{1..4}') == ['1', '2', '3', '4']", "assert expand('x{1..3}y') == ['x1y', 'x2y', 'x3y']",
          "assert expand('{3..1}') == ['3', '2', '1']"],
         ["assert expand('{1..9..3}') == ['1', '4', '7']", "assert expand('{a..c}') == ['a', 'b', 'c']",
          "assert expand('{c..a}') == ['c', 'b', 'a']"],
         ["assert expand('a{b') == ['a{b']", "assert expand('{}') == ['{}']", "assert expand('{a}') == ['{a}']",
          "assert expand('a}b') == ['a}b']"]]),

    _sc("union_find", "hard",
        ("class DisjointSet:\n"
         "    def __init__(self):\n"
         "        # `count` is a plain attribute: the number of distinct components right now\n"
         "        self.count = 0\n"
         "        raise NotImplementedError\n\n"
         "    def add(self, x):\n"
         "        # True if x was new, False if it was already known\n"
         "        raise NotImplementedError\n\n"
         "    def find(self, x):\n"
         "        # representative of x's component; KeyError if x is unknown\n"
         "        raise NotImplementedError\n\n"
         "    def union(self, a, b):\n"
         "        # adds either element if unknown; True if two components were merged, False if they\n"
         "        # were already the same\n"
         "        raise NotImplementedError\n\n"
         "    def connected(self, a, b):\n"
         "        raise NotImplementedError\n\n"
         "    def size_of(self, x):\n"
         "        # how many elements are in x's component\n"
         "        raise NotImplementedError\n\n"
         "    def groups(self):\n"
         "        # list of components, each sorted, the list itself sorted by each group's first member\n"
         "        raise NotImplementedError\n"),
        "from solution import DisjointSet",
        [["ds = DisjointSet()", "assert ds.count == 0", "assert ds.add('a') is True",
          "assert ds.add('a') is False", "assert ds.count == 1", "assert ds.groups() == [['a']]"],
         ["ds = DisjointSet()", "for x in 'abc':", "    ds.add(x)", "assert ds.count == 3",
          "assert ds.union('a', 'b') is True", "assert ds.count == 2",
          "assert ds.union('a', 'b') is False", "assert ds.count == 2"],
         ["ds = DisjointSet()", "ds.union('a', 'b')", "ds.union('b', 'c')",
          "assert ds.connected('a', 'c') is True", "assert ds.count == 1", "assert ds.size_of('a') == 3",
          "assert ds.find('a') == ds.find('c')"],
         ["ds = DisjointSet()", "ds.union('a', 'b')", "ds.union('x', 'y')", "ds.add('m')",
          "assert ds.count == 3", "assert ds.connected('a', 'x') is False",
          "assert ds.groups() == [['a', 'b'], ['m'], ['x', 'y']]",
          "assert ds.size_of('m') == 1"],
         ["ds = DisjointSet()", "ds.add('a')",
          "for call in [lambda: ds.find('zz'), lambda: ds.connected('a', 'zz'), lambda: ds.size_of('zz')]:",
          "    try:",
          "        call()",
          "    except KeyError:",
          "        continue",
          "    raise AssertionError('expected KeyError')"],
         ["ds = DisjointSet()", "names = ['n%03d' % i for i in range(200)]",
          "for i in range(1, 200):", "    assert ds.union(names[i - 1], names[i]) is True",
          "assert ds.count == 1", "assert ds.size_of(names[0]) == 200", "assert len(ds.groups()) == 1",
          "assert ds.groups()[0] == sorted(names)"]]),

    _sc("widest_path", "hard",
        ("def max_bottleneck(graph, src, dst):\n"
         "    # graph: {node: {neighbor: capacity}}, DIRECTED, capacities are positive numbers.\n"
         "    # Over every src->dst path take the SMALLEST capacity on that path; return the LARGEST\n"
         "    # such value. 0 if dst is unreachable. float('inf') when src == dst.\n"
         "    # KeyError if src is not a node of the graph.\n"
         "    raise NotImplementedError\n\n"
         "def reachable_within(graph, src, min_cap):\n"
         "    # sorted list of nodes reachable from src (src included) using only edges with\n"
         "    # capacity >= min_cap\n"
         "    raise NotImplementedError\n"),
        "from solution import max_bottleneck, reachable_within",
        [["import math", "g = {'a': {'b': 5}, 'b': {}}", "assert max_bottleneck(g, 'a', 'b') == 5",
          "assert math.isinf(max_bottleneck(g, 'a', 'a'))"],
         ["g = {'a': {'b': 5, 'c': 2}, 'b': {'d': 3}, 'c': {'d': 9}, 'd': {}}",
          "assert max_bottleneck(g, 'a', 'd') == 3", "assert max_bottleneck(g, 'a', 'c') == 2",
          "assert max_bottleneck(g, 'd', 'a') == 0"],
         ["g = {'s': {'t': 1, 'u': 10}, 'u': {'v': 10}, 'v': {'t': 10}, 't': {}}",
          "assert max_bottleneck(g, 's', 't') == 10"],
         ["g = {'a': {}, 'b': {}}", "assert max_bottleneck(g, 'a', 'b') == 0",
          "assert max_bottleneck(g, 'a', 'nope') == 0"],
         ["g = {'a': {'b': 5, 'c': 2}, 'b': {'d': 3}, 'c': {'d': 9}, 'd': {}}",
          "assert reachable_within(g, 'a', 3) == ['a', 'b', 'd']",
          "assert reachable_within(g, 'a', 1) == ['a', 'b', 'c', 'd']",
          "assert reachable_within(g, 'a', 6) == ['a']"],
         ["g = {'a': {'b': 5}, 'b': {}}",
          "for call in [lambda: max_bottleneck(g, 'zz', 'a'), lambda: reachable_within(g, 'zz', 1)]:",
          "    try:",
          "        call()",
          "    except KeyError:",
          "        continue",
          "    raise AssertionError('expected KeyError')"]]),

    _sc("topo_lex", "hard",
        ("def topo_sort(graph):\n"
         "    # graph: {node: [dependencies]} — every dependency must come BEFORE the node, and every\n"
         "    # node (even isolated ones) is a key. Return the LEXICOGRAPHICALLY SMALLEST valid order:\n"
         "    # whenever several nodes are ready, take the smallest.\n"
         "    # ValueError on a cycle, KeyError if a dependency is not a key of the graph.\n"
         "    raise NotImplementedError\n\n"
         "def build_waves(graph):\n"
         "    # the same graph as parallel batches: wave 0 is everything with no dependencies, wave k\n"
         "    # is everything whose dependencies all landed in earlier waves. Each wave sorted.\n"
         "    raise NotImplementedError\n"),
        "from solution import topo_sort, build_waves",
        [["assert topo_sort({}) == []", "assert build_waves({}) == []",
          "assert topo_sort({'a': []}) == ['a']"],
         ["assert topo_sort({'a': [], 'b': ['a']}) == ['a', 'b']",
          "assert topo_sort({'c': [], 'b': [], 'a': []}) == ['a', 'b', 'c']"],
         ["d = {'d': ['b', 'c'], 'b': ['a'], 'c': ['a'], 'a': []}",
          "assert topo_sort(d) == ['a', 'b', 'c', 'd']",
          "assert build_waves(d) == [['a'], ['b', 'c'], ['d']]"],
         ["g = {'x': [], 'a': ['x'], 'b': []}", "assert topo_sort(g) == ['b', 'x', 'a']",
          "assert build_waves(g) == [['b', 'x'], ['a']]"],
         ["for bad in [{'a': ['b'], 'b': ['a']}, {'a': ['a']}, {'a': ['b'], 'b': ['c'], 'c': ['a']}]:",
          "    try:",
          "        topo_sort(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"],
         ["try:",
          "    topo_sort({'a': ['ghost']})",
          "except KeyError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected KeyError')"],
         ["g = {'app': ['lib', 'ui'], 'ui': ['lib'], 'lib': ['core'], 'core': [], 'docs': []}",
          "order = topo_sort(g)",
          "assert order == ['core', 'docs', 'lib', 'ui', 'app']",
          "assert build_waves(g) == [['core', 'docs'], ['lib'], ['ui'], ['app']]"]]),

    _sc("bipartite", "hard",
        ("def two_color(graph):\n"
         "    # graph: {node: [neighbours]}, undirected (the adjacency is symmetric) and every node is\n"
         "    # a key. Returns (True, colors) with colors mapping every node to 0 or 1 such that no\n"
         "    # edge joins equal colors — the SMALLEST node of each component gets 0. If the graph is\n"
         "    # not bipartite, returns (False, cycle) where cycle is a list of distinct nodes of ODD\n"
         "    # length in which every node is adjacent to the next, and the last to the first.\n"
         "    # KeyError if a neighbour is not a key of the graph.\n"
         "    raise NotImplementedError\n"),
        "from solution import two_color",
        [["assert two_color({}) == (True, {})", "assert two_color({'a': []}) == (True, {'a': 0})"],
         ["g = {'a': ['b'], 'b': ['a', 'c'], 'c': ['b']}",
          "assert two_color(g) == (True, {'a': 0, 'b': 1, 'c': 0})"],
         ["g = {'a': ['b', 'd'], 'b': ['a', 'c'], 'c': ['b', 'd'], 'd': ['c', 'a']}",
          "ok, colors = two_color(g)", "assert ok is True",
          "for u in g:", "    for v in g[u]:", "        assert colors[u] != colors[v]"],
         ["g = {'b': ['c'], 'c': ['b'], 'a': []}", "assert two_color(g) == (True, {'a': 0, 'b': 0, 'c': 1})"],
         ["g = {'a': ['b', 'c'], 'b': ['a', 'c'], 'c': ['a', 'b']}", "ok, cyc = two_color(g)",
          "assert ok is False", "assert len(cyc) % 2 == 1 and len(cyc) >= 3",
          "assert len(set(cyc)) == len(cyc)",
          "for i in range(len(cyc)):", "    assert cyc[(i + 1) % len(cyc)] in g[cyc[i]]"],
         ["g = {'a': ['b', 'e'], 'b': ['a', 'c'], 'c': ['b', 'd'], 'd': ['c', 'e'], 'e': ['d', 'a']}",
          "ok, cyc = two_color(g)", "assert ok is False", "assert len(cyc) % 2 == 1 and len(cyc) >= 3",
          "assert len(set(cyc)) == len(cyc)",
          "for i in range(len(cyc)):", "    assert cyc[(i + 1) % len(cyc)] in g[cyc[i]]"],
         ["assert two_color({'a': ['a']})[0] is False",
          "try:",
          "    two_color({'a': ['ghost']})",
          "except KeyError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected KeyError')"]]),

    _sc("bellman_ford", "hard",
        ("def shortest_paths(n, edges, src):\n"
         "    # Nodes are 0..n-1. edges: list of (u, v, w) DIRECTED, w may be negative.\n"
         "    # Return a list of n distances from src, None where a node is unreachable.\n"
         "    # ValueError if a negative cycle is REACHABLE FROM src (an unreachable one is fine).\n"
         "    raise NotImplementedError\n\n"
         "def has_negative_cycle(n, edges):\n"
         "    # True if the graph contains a negative cycle ANYWHERE, reachable from src or not\n"
         "    raise NotImplementedError\n"),
        "from solution import shortest_paths, has_negative_cycle",
        [["assert shortest_paths(1, [], 0) == [0]", "assert shortest_paths(3, [(0, 1, 1), (1, 2, 2)], 0) == [0, 1, 3]"],
         ["assert shortest_paths(3, [(0, 1, 1)], 0) == [0, 1, None]",
          "assert shortest_paths(3, [], 2) == [None, None, 0]"],
         ["assert shortest_paths(3, [(0, 1, 4), (0, 2, 5), (1, 2, -3)], 0) == [0, 4, 1]",
          "assert shortest_paths(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 10)], 0) == [0, 1, 2, 3]"],
         ["try:",
          "    shortest_paths(3, [(0, 1, 1), (1, 2, -1), (2, 1, -1)], 0)",
          "except ValueError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ValueError')"],
         ["assert shortest_paths(4, [(0, 1, 1), (2, 3, -1), (3, 2, -1)], 0) == [0, 1, None, None]"],
         ["assert has_negative_cycle(4, [(0, 1, 1), (2, 3, -1), (3, 2, -1)]) is True",
          "assert has_negative_cycle(3, [(0, 1, 1), (1, 2, 2)]) is False",
          "assert has_negative_cycle(2, [(0, 1, 5), (1, 0, -5)]) is False",
          "assert has_negative_cycle(1, [(0, 0, -1)]) is True"]]),

    _sc("min_heap", "hard",
        ("class MinHeap:\n"
         "    def __init__(self, items=None):\n"
         "        # build in place from `items` (which must NOT be mutated); do not import heapq\n"
         "        raise NotImplementedError\n\n"
         "    def __len__(self):\n"
         "        raise NotImplementedError\n\n"
         "    def push(self, x):\n"
         "        raise NotImplementedError\n\n"
         "    def peek(self):\n"
         "        # smallest element without removing it; IndexError when empty\n"
         "        raise NotImplementedError\n\n"
         "    def pop(self):\n"
         "        # remove and return the smallest; IndexError when empty\n"
         "        raise NotImplementedError\n\n"
         "    def pushpop(self, x):\n"
         "        # push then pop, but cheaper: if x is already <= the minimum, return x untouched\n"
         "        raise NotImplementedError\n\n"
         "    def replace(self, x):\n"
         "        # pop then push: return the old minimum and leave x in the heap; IndexError when empty\n"
         "        raise NotImplementedError\n\n\n"
         "def heapsort(xs):\n"
         "    # ascending, via MinHeap; `xs` is not mutated\n"
         "    raise NotImplementedError\n"),
        "from solution import MinHeap, heapsort",
        [["h = MinHeap()", "assert len(h) == 0",
          "for call in [h.peek, h.pop, lambda: h.replace(1)]:",
          "    try:",
          "        call()",
          "    except IndexError:",
          "        continue",
          "    raise AssertionError('expected IndexError')"],
         ["h = MinHeap()", "for x in [3, 1, 2]:", "    h.push(x)", "assert len(h) == 3",
          "assert h.peek() == 1", "assert h.pop() == 1", "assert h.pop() == 2", "assert h.pop() == 3",
          "assert len(h) == 0"],
         ["src = [5, 3, 8, 1, 9, 2]", "h = MinHeap(src)", "assert src == [5, 3, 8, 1, 9, 2]",
          "assert [h.pop() for _ in range(6)] == [1, 2, 3, 5, 8, 9]"],
         ["h = MinHeap([4, 4, 1, 4])", "assert [h.pop() for _ in range(4)] == [1, 4, 4, 4]"],
         ["h = MinHeap([2, 4])", "assert h.pushpop(1) == 1", "assert len(h) == 2", "assert h.peek() == 2",
          "assert h.pushpop(5) == 2", "assert sorted([h.pop(), h.pop()]) == [4, 5]"],
         ["h = MinHeap([2, 4])", "assert h.replace(1) == 2", "assert h.peek() == 1", "assert len(h) == 2"],
         ["src = [(i * 37 + 11) % 101 for i in range(60)]", "assert heapsort(src) == sorted(src)",
          "assert src == [(i * 37 + 11) % 101 for i in range(60)]", "assert heapsort([]) == []"]]),

    _sc("lfu_cache", "hard",
        ("class LFUCache:\n"
         "    def __init__(self, capacity):\n"
         "        raise NotImplementedError\n\n"
         "    def get(self, key):\n"
         "        # the value, or None if absent. A hit counts as a use.\n"
         "        raise NotImplementedError\n\n"
         "    def put(self, key, value):\n"
         "        # Inserting when full evicts the LEAST FREQUENTLY used key; ties go to the least\n"
         "        # RECENTLY used of those. A fresh insert starts at frequency 1. Updating an existing\n"
         "        # key overwrites the value and counts as a use. A capacity of 0 (or less) stores\n"
         "        # nothing at all.\n"
         "        raise NotImplementedError\n"),
        "from solution import LFUCache",
        [["c = LFUCache(0)", "c.put(1, 1)", "assert c.get(1) is None",
          "c2 = LFUCache(-3)", "c2.put(1, 1)", "assert c2.get(1) is None"],
         ["c = LFUCache(2)", "assert c.get(9) is None", "c.put(1, 1)", "assert c.get(1) == 1",
          "c.put(1, 10)", "assert c.get(1) == 10"],
         ["c = LFUCache(2)", "c.put(1, 1)", "c.put(2, 2)", "assert c.get(1) == 1", "c.put(3, 3)",
          "assert c.get(2) is None", "assert c.get(1) == 1", "assert c.get(3) == 3"],
         ["c = LFUCache(2)", "c.put(1, 1)", "c.put(2, 2)", "assert c.get(1) == 1", "c.put(3, 3)",
          "assert c.get(3) == 3", "c.put(4, 4)",
          "assert c.get(1) is None", "assert c.get(3) == 3", "assert c.get(4) == 4"],
         ["c = LFUCache(2)", "c.put(1, 1)", "c.put(2, 2)", "c.put(1, 10)", "c.put(3, 3)",
          "assert c.get(2) is None", "assert c.get(1) == 10", "assert c.get(3) == 3"],
         ["c = LFUCache(3)", "for k in [1, 2, 3]:", "    c.put(k, k * 10)",
          "for _ in range(3):", "    c.get(1)", "c.get(2)", "c.put(4, 40)",
          "assert c.get(3) is None", "assert c.get(1) == 10", "assert c.get(2) == 20",
          "assert c.get(4) == 40"]]),

    _sc("circuit_breaker", "hard",
        ("class CircuitBreaker:\n"
         "    def __init__(self, fail_threshold, reset_timeout, half_open_successes=1):\n"
         "        # The clock is injected: every method takes the current time.\n"
         "        raise NotImplementedError\n\n"
         "    def state(self, now):\n"
         "        # 'closed' | 'open' | 'half_open'. An open breaker becomes half-open on its own once\n"
         "        # reset_timeout has elapsed since it opened.\n"
         "        raise NotImplementedError\n\n"
         "    def allow(self, now):\n"
         "        # False only while open\n"
         "        raise NotImplementedError\n\n"
         "    def record(self, now, ok):\n"
         "        # closed:    fail_threshold CONSECUTIVE failures open it; any success resets the run\n"
         "        # half_open: half_open_successes consecutive successes close it and clear the run;\n"
         "        #            a single failure re-opens it and restarts the timeout from `now`\n"
         "        raise NotImplementedError\n"),
        "from solution import CircuitBreaker",
        [["cb = CircuitBreaker(2, 10.0)", "assert cb.state(0.0) == 'closed'", "assert cb.allow(0.0) is True",
          "cb.record(0.0, True)", "assert cb.state(0.0) == 'closed'"],
         ["cb = CircuitBreaker(2, 10.0)", "cb.record(0.0, False)", "assert cb.state(0.0) == 'closed'",
          "cb.record(1.0, False)", "assert cb.state(1.0) == 'open'", "assert cb.allow(1.0) is False"],
         ["cb = CircuitBreaker(2, 10.0)", "cb.record(0.0, False)", "cb.record(1.0, True)",
          "cb.record(2.0, False)", "assert cb.state(2.0) == 'closed'", "cb.record(3.0, False)",
          "assert cb.state(3.0) == 'open'"],
         ["cb = CircuitBreaker(2, 10.0)", "cb.record(0.0, False)", "cb.record(1.0, False)",
          "assert cb.allow(10.0) is False", "assert cb.state(11.0) == 'half_open'",
          "assert cb.allow(11.0) is True"],
         ["cb = CircuitBreaker(2, 10.0)", "cb.record(0.0, False)", "cb.record(1.0, False)",
          "assert cb.state(11.0) == 'half_open'", "cb.record(11.0, True)",
          "assert cb.state(11.0) == 'closed'", "assert cb.allow(11.0) is True",
          "cb.record(12.0, False)", "assert cb.state(12.0) == 'closed'"],
         ["cb = CircuitBreaker(2, 10.0)", "cb.record(0.0, False)", "cb.record(1.0, False)",
          "assert cb.state(11.0) == 'half_open'", "cb.record(11.0, False)",
          "assert cb.state(11.0) == 'open'", "assert cb.state(20.0) == 'open'",
          "assert cb.state(21.0) == 'half_open'"],
         ["cb = CircuitBreaker(1, 5.0, half_open_successes=2)", "cb.record(0.0, False)",
          "assert cb.state(0.0) == 'open'", "assert cb.state(5.0) == 'half_open'",
          "cb.record(5.0, True)", "assert cb.state(5.0) == 'half_open'", "cb.record(6.0, True)",
          "assert cb.state(6.0) == 'closed'"]]),

    _sc("mustache", "hard",
        ("def render(template, context):\n"
         "    # A small mustache-style renderer over a CONTEXT STACK.\n"
         "    #   {{name}}    lookup, HTML-escaped (& < > \" ')\n"
         "    #   {{{name}}}  and {{&name}}   lookup, raw\n"
         "    #   {{a.b.c}}   dotted lookup;  {{.}} is the current item\n"
         "    #   {{! ... }}  comment, renders nothing\n"
         "    #   {{#k}}..{{/k}}   list -> render once per item with the item pushed on the stack;\n"
         "    #                    other truthy value -> render once (a dict is pushed, anything else\n"
         "    #                    is not); falsy -> render nothing\n"
         "    #   {{^k}}..{{/k}}   the inverse: render only when k is missing or falsy\n"
         "    # A name that resolves nowhere renders as ''. None -> '', True -> 'true', False -> 'false'.\n"
         "    # Lookup walks the stack from the innermost frame outwards. ValueError on an unclosed tag.\n"
         "    raise NotImplementedError\n"),
        "from solution import render",
        [["assert render('no tags', {}) == 'no tags'", "assert render('Hello {{name}}!', {'name': 'World'}) == 'Hello World!'",
          "assert render('{{nope}}', {}) == ''"],
         ["assert render('{{x}}|{{{x}}}|{{&x}}', {'x': '<a&b>'}) == '&lt;a&amp;b&gt;|<a&b>|<a&b>'",
          "assert render('{{q}}', {'q': 'a\"b\\'c'}) == 'a&quot;b&#39;c'"],
         ["assert render('{{a.b.c}}', {'a': {'b': {'c': 'deep'}}}) == 'deep'",
          "assert render('{{a.zz}}', {'a': {}}) == ''",
          "assert render('[{{n}}][{{t}}][{{f}}]', {'n': None, 't': True, 'f': False}) == '[][true][false]'"],
         ["assert render('{{#xs}}[{{.}}]{{/xs}}', {'xs': [1, 2, 3]}) == '[1][2][3]'",
          "assert render('{{#xs}}x{{/xs}}', {'xs': []}) == ''"],
         ["ctx = {'host': 'h', 'users': [{'name': 'a'}, {'name': 'b'}]}",
          "assert render('{{#users}}{{name}}@{{host}};{{/users}}', ctx) == 'a@h;b@h;'"],
         ["assert render('{{^xs}}none{{/xs}}{{! hi }}{{#xs}}some{{/xs}}', {'xs': []}) == 'none'",
          "assert render('{{^xs}}none{{/xs}}', {'xs': [1]}) == ''",
          "assert render('{{^gone}}yes{{/gone}}', {}) == 'yes'"],
         ["assert render('{{#a}}{{b}}{{/a}}', {'a': {'b': 'B'}}) == 'B'",
          "assert render('{{#a}}{{#b}}x{{/b}}{{/a}}', {'a': {'b': True}}) == 'x'",
          "assert render('{{#a}}on{{/a}}', {'a': 0}) == ''"],
         ["for bad in ['{{oops', '{{#a}}no close']:",
          "    try:",
          "        render(bad, {'a': True})",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"]]),

    _sc("csv_rfc4180", "hard",
        ("def parse_csv(text):\n"
         "    # RFC4180-ish. Comma delimited; rows split on '\\n' or '\\r\\n'. A field that STARTS with\n"
         "    # '\"' is quoted: inside it, '\"\"' is a literal quote and commas / newlines are data.\n"
         "    # A single trailing newline does NOT produce an empty final row; '' parses to [].\n"
         "    # ValueError if a quoted field is never closed.\n"
         "    raise NotImplementedError\n\n"
         "def write_csv(rows):\n"
         "    # Inverse. Quote a field exactly when it contains a comma, a quote, CR or LF (doubling\n"
         "    # any quote inside). Rows are joined with '\\r\\n' and there is no trailing newline.\n"
         "    raise NotImplementedError\n"),
        "from solution import parse_csv, write_csv",
        [["assert parse_csv('') == []", "assert parse_csv('a,b\\nc,d') == [['a', 'b'], ['c', 'd']]"],
         ["assert parse_csv('a\\n') == [['a']]", "assert parse_csv('a,,b') == [['a', '', 'b']]",
          "assert parse_csv('\\n') == [['']]", "assert parse_csv('a,b\\n\\nc') == [['a', 'b'], [''], ['c']]"],
         ["assert parse_csv('\"a,b\",c') == [['a,b', 'c']]",
          "assert parse_csv('\"say \"\"hi\"\"\",x') == [['say \"hi\"', 'x']]",
          "assert parse_csv('\"\"') == [['']]"],
         ["assert parse_csv('\"l1\\nl2\",b') == [['l1\\nl2', 'b']]",
          "assert parse_csv('a,b\\r\\nc,d\\r\\n') == [['a', 'b'], ['c', 'd']]"],
         ["assert write_csv([['a', 'b']]) == 'a,b'",
          "assert write_csv([['a', 'b,c'], ['d\"e', 'f\\ng']]) == 'a,\"b,c\"\\r\\n\"d\"\"e\",\"f\\ng\"'",
          "assert write_csv([]) == ''"],
         ["rows = [['a', 'b'], ['x,y', 'q\"r'], ['', 'multi\\nline'], ['plain']]",
          "assert parse_csv(write_csv(rows)) == rows"],
         ["for bad in ['\"abc', 'a,\"bc', '\"a\"\"']:",
          "    try:",
          "        parse_csv(bad)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % bad)"]]),

    _sc("patch_apply", "hard",
        ("class ConflictError(Exception):\n"
         "    pass\n\n\n"
         "def apply_patch(lines, hunks):\n"
         "    # lines: list of strings. Each hunk is {'start': i, 'remove': [...], 'add': [...]} where\n"
         "    # `start` indexes the ORIGINAL `lines` and lines[start:start+len(remove)] must equal\n"
         "    # `remove`. Return a NEW list with every hunk applied; `lines` is not mutated.\n"
         "    # ConflictError if the context does not match; ValueError if hunks are out of order or\n"
         "    # overlap.\n"
         "    raise NotImplementedError\n\n"
         "def invert(hunks):\n"
         "    # hunks that undo `hunks`, with starts in the coordinates of the PATCHED file, so that\n"
         "    # apply_patch(apply_patch(lines, hunks), invert(hunks)) == lines\n"
         "    raise NotImplementedError\n"),
        "from solution import apply_patch, invert, ConflictError",
        [["assert apply_patch(['a', 'b', 'c'], []) == ['a', 'b', 'c']",
          "assert apply_patch([], []) == []",
          "assert apply_patch(['a', 'b', 'c'], [{'start': 1, 'remove': ['b'], 'add': ['B']}]) == ['a', 'B', 'c']"],
         ["assert apply_patch(['a', 'c'], [{'start': 1, 'remove': [], 'add': ['b']}]) == ['a', 'b', 'c']",
          "assert apply_patch(['a'], [{'start': 1, 'remove': [], 'add': ['z']}]) == ['a', 'z']",
          "assert apply_patch(['a'], [{'start': 0, 'remove': [], 'add': ['z']}]) == ['z', 'a']"],
         ["assert apply_patch(['a', 'b', 'c'], [{'start': 1, 'remove': ['b'], 'add': []}]) == ['a', 'c']",
          "src = ['a', 'b', 'c']",
          "apply_patch(src, [{'start': 0, 'remove': ['a'], 'add': ['A', 'A2']}])",
          "assert src == ['a', 'b', 'c']"],
         ["hs = [{'start': 0, 'remove': ['a'], 'add': []}, {'start': 2, 'remove': ['c'], 'add': ['C']}]",
          "assert apply_patch(['a', 'b', 'c', 'd'], hs) == ['b', 'C', 'd']"],
         ["try:",
          "    apply_patch(['a', 'b'], [{'start': 0, 'remove': ['x'], 'add': []}])",
          "except ConflictError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ConflictError')",
          "try:",
          "    apply_patch(['a'], [{'start': 0, 'remove': ['a', 'b'], 'add': []}])",
          "except ConflictError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ConflictError')"],
         ["bad = [{'start': 0, 'remove': ['a', 'b'], 'add': []}, {'start': 1, 'remove': ['b'], 'add': []}]",
          "try:",
          "    apply_patch(['a', 'b', 'c'], bad)",
          "except ValueError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ValueError')"],
         ["cases = [",
          "    (['a', 'b', 'c'], [{'start': 1, 'remove': ['b'], 'add': ['B1', 'B2']}]),",
          "    (['a', 'b', 'c', 'd'], [{'start': 0, 'remove': ['a'], 'add': []},",
          "                            {'start': 2, 'remove': ['c'], 'add': ['C']}]),",
          "    (['x'], [{'start': 1, 'remove': [], 'add': ['y', 'z']}]),",
          "    (['p', 'q'], []),",
          "]",
          "for lines, hs in cases:",
          "    patched = apply_patch(lines, hs)",
          "    assert apply_patch(patched, invert(hs)) == lines"]]),

    _sc("result_type", "hard",
        ("class Ok:\n"
         "    def __init__(self, value):\n"
         "        self.value = value\n"
         "        raise NotImplementedError\n\n\n"
         "class Err:\n"
         "    def __init__(self, error):\n"
         "        self.error = error\n"
         "        raise NotImplementedError\n\n\n"
         "# Both need: is_ok(), map(f), map_err(f), and_then(f), unwrap(), unwrap_or(default) and\n"
         "# value-based ==.  map/and_then are no-ops on Err; map_err is a no-op on Ok.\n"
         "# and_then's callable returns a Result itself.  Ok.unwrap() gives the value, Err.unwrap()\n"
         "# raises ValueError.  Ok holds `.value`, Err holds `.error`.\n\n\n"
         "def combine(results):\n"
         "    # all Ok -> Ok([values...]);  otherwise Err([errors...]) with every error, in order\n"
         "    raise NotImplementedError\n\n\n"
         "def attempt(fn, *args):\n"
         "    # call fn(*args); Ok(result), or Err(str(exception)) if it raises\n"
         "    raise NotImplementedError\n"),
        "from solution import Ok, Err, combine, attempt",
        [["assert Ok(1).is_ok() is True", "assert Err('e').is_ok() is False", "assert Ok(1).value == 1",
          "assert Err('e').error == 'e'"],
         ["assert Ok(1) == Ok(1)", "assert Ok(1) != Ok(2)", "assert Ok(1) != Err(1)",
          "assert Err('a') == Err('a')", "assert Err('a') != Err('b')"],
         ["assert Ok(2).map(lambda x: x * 3) == Ok(6)", "assert Err('e').map(lambda x: x * 3) == Err('e')",
          "assert Err('e').map_err(lambda s: s.upper()) == Err('E')",
          "assert Ok(1).map_err(lambda s: s.upper()) == Ok(1)"],
         ["assert Ok(4).and_then(lambda x: Ok(x + 1)) == Ok(5)",
          "assert Ok(4).and_then(lambda x: Err('bad')) == Err('bad')",
          "assert Err('z').and_then(lambda x: Ok(1)) == Err('z')",
          "assert Ok(1).map(lambda x: x + 1).and_then(lambda x: Ok(x * 10)) == Ok(20)"],
         ["assert Ok(7).unwrap() == 7", "assert Ok(7).unwrap_or(0) == 7", "assert Err('e').unwrap_or(0) == 0",
          "try:",
          "    Err('e').unwrap()",
          "except ValueError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ValueError')"],
         ["assert combine([]) == Ok([])", "assert combine([Ok(1), Ok(2)]) == Ok([1, 2])",
          "assert combine([Ok(1), Err('a'), Err('b')]) == Err(['a', 'b'])",
          "assert combine([Err('only')]) == Err(['only'])"],
         ["assert attempt(int, '12') == Ok(12)", "r = attempt(int, 'x')",
          "assert r.is_ok() is False and isinstance(r.error, str) and r.error != ''",
          "assert attempt(lambda: 1 // 0).is_ok() is False"]]),

    _sc("json_pointer", "hard",
        ("def resolve(doc, pointer):\n"
         "    # RFC6901. '' is the whole document; otherwise the pointer starts with '/' and each\n"
         "    # token is unescaped '~1'->'/' then '~0'->'~'. Dict tokens are keys; list tokens must be\n"
         "    # digits. ValueError for a malformed pointer or a non-numeric list token, KeyError for a\n"
         "    # missing key (or a scalar in the middle), IndexError for an out-of-range list index.\n"
         "    raise NotImplementedError\n\n"
         "def set_at(doc, pointer, value):\n"
         "    # mutate doc in place and return it; missing intermediate DICT keys are created. On a\n"
         "    # list, the token '-' appends and an index == len(list) appends too. ValueError for the\n"
         "    # empty pointer (the whole document cannot be replaced).\n"
         "    raise NotImplementedError\n\n"
         "def remove_at(doc, pointer):\n"
         "    # remove the target and return the removed value; same errors as resolve, plus\n"
         "    # ValueError for the empty pointer\n"
         "    raise NotImplementedError\n"),
        "from solution import resolve, set_at, remove_at",
        [["doc = {'a': 1}", "assert resolve(doc, '') is doc", "assert resolve(doc, '/a') == 1"],
         ["assert resolve({'a': [10, {'b': 2}]}, '/a/1/b') == 2", "assert resolve({'a': [10]}, '/a/0') == 10",
          "assert resolve({'': 5}, '/') == 5"],
         ["d = {'a/b': 1, 'm~n': 2, '~1': 3}", "assert resolve(d, '/a~1b') == 1",
          "assert resolve(d, '/m~0n') == 2", "assert resolve(d, '/~01') == 3"],
         ["for ptr in ['a', '/a/-', '/a/x']:",
          "    try:",
          "        resolve({'a': [1]}, ptr)",
          "    except ValueError:",
          "        continue",
          "    raise AssertionError('expected ValueError for %r' % ptr)",
          "try:",
          "    resolve({}, '/x')",
          "except KeyError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected KeyError')",
          "try:",
          "    resolve({'a': [1]}, '/a/5')",
          "except IndexError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected IndexError')"],
         ["d = {'a': {}}", "assert set_at(d, '/a/b', 1) is d", "assert d == {'a': {'b': 1}}",
          "d2 = {}", "set_at(d2, '/x/y', 2)", "assert d2 == {'x': {'y': 2}}",
          "set_at(d2, '/x/y', 3)", "assert d2 == {'x': {'y': 3}}"],
         ["d = {'l': [1, 2]}", "set_at(d, '/l/0', 9)", "assert d == {'l': [9, 2]}",
          "set_at(d, '/l/-', 3)", "assert d == {'l': [9, 2, 3]}",
          "set_at(d, '/l/3', 7)", "assert d == {'l': [9, 2, 3, 7]}",
          "try:",
          "    set_at(d, '', 1)",
          "except ValueError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ValueError')"],
         ["d = {'a': {'b': 1, 'c': 2}}", "assert remove_at(d, '/a/b') == 1", "assert d == {'a': {'c': 2}}",
          "d2 = {'l': [1, 2, 3]}", "assert remove_at(d2, '/l/1') == 2", "assert d2 == {'l': [1, 3]}",
          "try:",
          "    remove_at({'x': 1}, '')",
          "except ValueError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected ValueError')",
          "try:",
          "    remove_at({}, '/nope')",
          "except KeyError:",
          "    pass",
          "else:",
          "    raise AssertionError('expected KeyError')"]]),
]

SCENARIOS = EASY + MEDIUM + HARD


# ---------------------------------------------------------------------------------------------
# REFERENCE — a working solution.py for every scenario. Not shipped to the model: this exists so
# scripts/verify_mined.py can prove each task is solvable (reference exits 0, stub exits non-zero)
# and so failed rollouts can later be diffed against something known-good.
# ---------------------------------------------------------------------------------------------
REFERENCE = {}

REFERENCE["slugify"] = '''
def slugify(text):
    s = ''.join(ch if (ch.isalnum() and ch.isascii()) else '-' for ch in text.lower())
    return '-'.join(p for p in s.split('-') if p)
'''

REFERENCE["pretty_bytes"] = '''
UNITS = ['B', 'kB', 'MB', 'GB', 'TB']


def pretty_bytes(n):
    sign = '-' if n < 0 else ''
    v = float(abs(n))
    i = 0
    while v >= 1000 and i < len(UNITS) - 1:
        v /= 1000.0
        i += 1
    if i == 0:
        return sign + str(int(v)) + ' B'
    return sign + ('%.2f' % v).rstrip('0').rstrip('.') + ' ' + UNITS[i]
'''

REFERENCE["parse_bytes"] = '''
MULT = {'b': 1, 'kb': 1000, 'mb': 10 ** 6, 'gb': 10 ** 9, 'tb': 10 ** 12}


def parse_bytes(s):
    t = s.strip().lower().replace(' ', '')
    i = 1 if t[:1] in ('+', '-') else 0
    j = i
    while j < len(t) and (t[j].isdigit() or t[j] == '.'):
        j += 1
    if j == i:
        raise ValueError('no number in %r' % s)
    try:
        val = float(t[:j])
    except ValueError:
        raise ValueError('bad number in %r' % s)
    unit = t[j:] or 'b'
    if unit not in MULT:
        raise ValueError('bad unit in %r' % s)
    return int(round(val * MULT[unit]))
'''

REFERENCE["chunking"] = '''
def chunks(xs, n):
    if n < 1:
        raise ValueError('n must be >= 1')
    return [list(xs[i:i + n]) for i in range(0, len(xs), n)]


def windows(xs, n):
    if n < 1:
        raise ValueError('n must be >= 1')
    return [list(xs[i:i + n]) for i in range(0, len(xs) - n + 1)]
'''

REFERENCE["dedup"] = '''
def dedup_first(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def dedup_last(xs):
    return list(reversed(dedup_first(list(reversed(xs)))))
'''

REFERENCE["isqrt"] = '''
def isqrt(n):
    if n < 0:
        raise ValueError('negative')
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n
'''

REFERENCE["flood_fill"] = '''
def count_regions(grid):
    if not grid:
        return 0
    h, w = len(grid), len(grid[0])
    seen = set()
    total = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != '#' or (r, c) in seen:
                continue
            total += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and grid[ny][nx] == '#':
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    return total
'''

REFERENCE["shoelace"] = '''
def _cross2(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def area(poly):
    return abs(_cross2(poly)) / 2.0


def orientation(poly):
    s = _cross2(poly)
    if s > 0:
        return 'ccw'
    if s < 0:
        return 'cw'
    return 'degenerate'
'''

REFERENCE["group_stats"] = '''
def group_stats(rows, key_field, value_field):
    buckets = {}
    for r in rows:
        k = r[key_field]
        buckets.setdefault(k, []).append(r[value_field])
    out = {}
    for k, vs in buckets.items():
        out[k] = {'count': len(vs), 'sum': sum(vs), 'min': min(vs), 'max': max(vs),
                  'avg': sum(vs) / float(len(vs))}
    return out
'''

REFERENCE["truncate"] = '''
def truncate(s, width, ellipsis='...'):
    if width < len(ellipsis):
        raise ValueError('width smaller than the ellipsis')
    if len(s) <= width:
        return s
    keep = width - len(ellipsis)
    head = s[:keep]
    if keep < len(s) and s[keep] != ' ' and ' ' in head:
        head = head[:head.rindex(' ')]
    return head.rstrip() + ellipsis
'''

REFERENCE["case_convert"] = '''
import re


def to_snake(s):
    s = re.sub(r'[-\\s]+', '_', s)
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\\1_\\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s)
    return re.sub(r'_+', '_', s).strip('_').lower()


def to_pascal(s):
    return ''.join(p[:1].upper() + p[1:] for p in to_snake(s).split('_') if p)


def to_camel(s):
    p = to_pascal(s)
    return p[:1].lower() + p[1:]
'''

REFERENCE["percent_codec"] = '''
UNRESERVED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
_HEX = '0123456789abcdefABCDEF'


def percent_encode(s):
    out = []
    for b in s.encode('utf-8'):
        c = chr(b)
        out.append(c if c in UNRESERVED else '%%%02X' % b)
    return ''.join(out)


def percent_decode(s):
    buf = bytearray()
    i = 0
    while i < len(s):
        if s[i] == '%':
            h = s[i + 1:i + 3]
            if len(h) != 2 or h[0] not in _HEX or h[1] not in _HEX:
                raise ValueError('bad escape at %d' % i)
            buf.append(int(h, 16))
            i += 3
        else:
            buf.extend(s[i].encode('utf-8'))
            i += 1
    try:
        return buf.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('invalid utf-8')
'''

REFERENCE["base32_crockford"] = '''
ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
_ALIAS = {'I': '1', 'L': '1', 'O': '0'}


def b32_encode(data):
    n = len(data)
    if n == 0:
        return ''
    nchars = (8 * n + 4) // 5
    acc = int.from_bytes(data, 'big') << (5 * nchars - 8 * n)
    return ''.join(ALPHABET[(acc >> (5 * (nchars - 1 - k))) & 31] for k in range(nchars))


def b32_decode(s):
    t = ''.join(ch for ch in s.upper() if ch != '-')
    if not t:
        return b''
    acc = 0
    for ch in t:
        ch = _ALIAS.get(ch, ch)
        j = ALPHABET.find(ch)
        if j < 0:
            raise ValueError('bad character %r' % ch)
        acc = (acc << 5) | j
    nbytes = (5 * len(t)) // 8
    extra = 5 * len(t) - 8 * nbytes
    if acc & ((1 << extra) - 1):
        raise ValueError('non-zero padding bits')
    return (acc >> extra).to_bytes(nbytes, 'big')
'''

REFERENCE["ini_parser"] = '''
def parse_ini(text):
    out = {}
    section = ''
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in ';#':
            continue
        if line.startswith('['):
            if not line.endswith(']') or len(line) < 3:
                raise ValueError('bad section header: %r' % raw)
            section = line[1:-1].strip()
            out.setdefault(section, {})
            continue
        if '=' not in line:
            raise ValueError('not a key=value line: %r' % raw)
        k, v = line.split('=', 1)
        k = k.strip()
        if not k:
            raise ValueError('empty key: %r' % raw)
        out.setdefault(section, {})[k] = v.strip()
    return out
'''

REFERENCE["iso_duration"] = '''
import re

_RE = re.compile(r'^P(?:(\\d+)W)?(?:(\\d+)D)?(?:T(?:(\\d+)H)?(?:(\\d+)M)?(?:(\\d+)S)?)?$')


def parse_duration(s):
    m = _RE.match(s or '')
    if not m or all(g is None for g in m.groups()):
        raise ValueError('bad duration %r' % s)
    w, d, h, mi, sec = (int(g) if g else 0 for g in m.groups())
    return ((w * 7 + d) * 24 + h) * 3600 + mi * 60 + sec


def format_duration(seconds):
    if seconds < 0:
        raise ValueError('negative')
    if seconds == 0:
        return 'PT0S'
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    out = 'P' + ('%dD' % d if d else '')
    if h or m or s:
        out += 'T' + ('%dH' % h if h else '') + ('%dM' % m if m else '') + ('%dS' % s if s else '')
    return out
'''

REFERENCE["binsearch_bounds"] = '''
def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def count_between(a, lo, hi):
    if lo > hi:
        return 0
    return upper_bound(a, hi) - lower_bound(a, lo)


def insert_sorted(a, x):
    i = upper_bound(a, x)
    return list(a[:i]) + [x] + list(a[i:])
'''

REFERENCE["dotted_path"] = '''
def _steps(path):
    if not path:
        raise ValueError('empty path')
    return path.split('.')


def _index(s):
    return s.lstrip('-').isdigit()


def get_path(obj, path, default=None):
    cur = obj
    for s in _steps(path):
        if isinstance(cur, dict):
            if s not in cur:
                return default
            cur = cur[s]
        elif isinstance(cur, list):
            if not _index(s):
                return default
            i = int(s)
            if i < -len(cur) or i >= len(cur):
                return default
            cur = cur[i]
        else:
            return default
    return cur


def set_path(obj, path, value):
    steps = _steps(path)
    cur = obj
    for s in steps[:-1]:
        if isinstance(cur, list):
            cur = cur[int(s)]
        else:
            if s not in cur or not isinstance(cur[s], (dict, list)):
                cur[s] = {}
            cur = cur[s]
    last = steps[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value
    return obj


def del_path(obj, path):
    steps = _steps(path)
    cur = obj
    for s in steps[:-1]:
        if isinstance(cur, dict):
            if s not in cur:
                return False
            cur = cur[s]
        elif isinstance(cur, list):
            if not _index(s) or int(s) >= len(cur):
                return False
            cur = cur[int(s)]
        else:
            return False
    last = steps[-1]
    if isinstance(cur, dict):
        if last in cur:
            del cur[last]
            return True
        return False
    if isinstance(cur, list) and _index(last) and int(last) < len(cur):
        del cur[int(last)]
        return True
    return False
'''

REFERENCE["deep_merge"] = '''
DELETE = object()


def _copy(v):
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy(x) for x in v]
    return v


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        out = {}
        for k, v in base.items():
            if k in override:
                if override[k] is DELETE:
                    continue
                out[k] = deep_merge(v, override[k])
            else:
                out[k] = _copy(v)
        for k, v in override.items():
            if k in base or v is DELETE:
                continue
            out[k] = _copy(v)
        return out
    if isinstance(base, list) and isinstance(override, list):
        out = []
        for i in range(max(len(base), len(override))):
            if i >= len(override):
                out.append(_copy(base[i]))
            elif i >= len(base):
                out.append(_copy(override[i]))
            elif override[i] is DELETE:
                continue
            else:
                out.append(deep_merge(base[i], override[i]))
        return out
    return _copy(override)
'''

REFERENCE["order_by"] = '''
def order_by(rows, keys):
    specs = []
    for k in keys:
        desc = k[:1] == '-'
        field = k[1:] if k[:1] in ('-', '+') else k
        if not field:
            raise ValueError('key %r has no field name' % k)
        specs.append((field, desc))
    out = list(rows)
    for field, desc in reversed(specs):
        def keyfn(r, f=field, d=desc):
            v = r.get(f)
            if v is None:
                return (0, 0) if d else (1, 0)
            return (1, v) if d else (0, v)
        out.sort(key=keyfn, reverse=desc)
    return out
'''

REFERENCE["token_bucket"] = '''
class TokenBucket:
    def __init__(self, capacity, refill_per_sec):
        self.capacity = capacity
        self.rate = refill_per_sec
        self.tokens = float(capacity)
        self.last = None

    def allow(self, now, n=1):
        if n < 1:
            raise ValueError('n must be >= 1')
        if n > self.capacity:
            raise ValueError('n exceeds capacity')
        if self.last is None:
            self.last = now
        if now < self.last:
            raise ValueError('clock moved backwards')
        self.tokens = min(float(self.capacity), self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
'''

REFERENCE["lcs_edit"] = '''
def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def longest_common_substring(a, b):
    best_len = 0
    best_i = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len = cur[j]
                    best_i = i - cur[j]
        prev = cur
    return a[best_i:best_i + best_len]
'''

REFERENCE["schema_lite"] = '''
def _type_ok(t, v):
    if t == 'object':
        return isinstance(v, dict)
    if t == 'array':
        return isinstance(v, list)
    if t == 'string':
        return isinstance(v, str)
    if t == 'integer':
        return isinstance(v, int) and not isinstance(v, bool)
    if t == 'number':
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == 'boolean':
        return isinstance(v, bool)
    if t == 'null':
        return v is None
    raise ValueError('unknown type %r' % t)


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check(schema, v, path, errs):
    t = schema.get('type')
    if t is not None and not _type_ok(t, v):
        errs.append(path)
        return
    if 'enum' in schema and v not in schema['enum']:
        errs.append(path)
    if _numeric(v):
        if 'minimum' in schema and v < schema['minimum']:
            errs.append(path)
        if 'maximum' in schema and v > schema['maximum']:
            errs.append(path)
    if isinstance(v, str):
        if 'minLength' in schema and len(v) < schema['minLength']:
            errs.append(path)
        if 'maxLength' in schema and len(v) > schema['maxLength']:
            errs.append(path)
    if isinstance(v, dict):
        for k in schema.get('required', []):
            if k not in v:
                errs.append(path + '.' + k)
        for k, sub in schema.get('properties', {}).items():
            if k in v:
                _check(sub, v[k], path + '.' + k, errs)
    if isinstance(v, list) and 'items' in schema:
        for i, item in enumerate(v):
            _check(schema['items'], item, '%s[%d]' % (path, i), errs)


def validate(schema, value):
    errs = []
    _check(schema, value, '$', errs)
    return sorted(set(errs))
'''

REFERENCE["point_in_polygon"] = '''
def _on_segment(a, b, p):
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if cross != 0:
        return False
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def classify(poly, pt):
    n = len(poly)
    if n < 3:
        return 'outside'
    for i in range(n):
        if _on_segment(poly[i], poly[(i + 1) % n], pt):
            return 'boundary'
    x, y = pt
    inside = False
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        if (ay > y) != (by > y):
            xint = ax + (y - ay) * (bx - ax) / float(by - ay)
            if xint > x:
                inside = not inside
    return 'inside' if inside else 'outside'
'''

REFERENCE["semver"] = '''
def parse(v):
    parts = v.split('.')
    if len(parts) != 3:
        raise ValueError('bad version %r' % v)
    out = []
    for p in parts:
        if not p.isdigit():
            raise ValueError('bad version %r' % v)
        out.append(int(p))
    return tuple(out)


def compare(a, b):
    pa, pb = parse(a), parse(b)
    return -1 if pa < pb else (1 if pa > pb else 0)


def _bounds(rng):
    """-> (lo_inclusive, hi_exclusive_or_None, exact_or_None, op_or_None)"""
    for op in ('>=', '<=', '>', '<'):
        if rng.startswith(op):
            return (op, parse(rng[len(op):]))
    if rng[:1] in ('^', '~'):
        return (rng[0], parse(rng[1:]))
    return ('=', parse(rng))


def satisfies(v, rng):
    pv = parse(v)
    if rng == '*':
        return True
    op, base = _bounds(rng)
    if op == '=':
        return pv == base
    if op == '>=':
        return pv >= base
    if op == '>':
        return pv > base
    if op == '<=':
        return pv <= base
    if op == '<':
        return pv < base
    if op == '~':
        return base <= pv < (base[0], base[1] + 1, 0)
    major, minor, patch = base
    if major > 0:
        hi = (major + 1, 0, 0)
    elif minor > 0:
        hi = (0, minor + 1, 0)
    else:
        hi = (0, 0, patch + 1)
    return base <= pv < hi


def max_satisfying(versions, rng):
    ok = [v for v in versions if satisfies(v, rng)]
    if not ok:
        return None
    return max(ok, key=parse)
'''

REFERENCE["allocate"] = '''
def allocate(total, weights):
    if not weights:
        raise ValueError('empty weights')
    if any(w < 0 for w in weights):
        raise ValueError('negative weight')
    sw = sum(weights)
    if sw == 0:
        raise ValueError('weights sum to zero')
    base = []
    rems = []
    for i, w in enumerate(weights):
        q, r = divmod(total * w, sw)
        base.append(q)
        rems.append((-r, i))
    rems.sort()
    for k in range(total - sum(base)):
        base[rems[k][1]] += 1
    return base
'''

REFERENCE["glob_match"] = '''
def _seg_match(pat, seg):
    m, n = len(pat), len(seg)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] and pat[i - 1] == '*'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pat[i - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            elif pat[i - 1] == '?' or pat[i - 1] == seg[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]


def _match(p, s):
    if not p:
        return not s
    if p[0] == '**':
        return any(_match(p[1:], s[k:]) for k in range(len(s) + 1))
    if not s:
        return False
    return _seg_match(p[0], s[0]) and _match(p[1:], s[1:])


def match(pattern, path):
    return _match(pattern.split('/'), path.split('/'))
'''

REFERENCE["brace_expand"] = '''
def _find_group(s):
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0:
                return start, i
    return None


def _split_top(body):
    out = []
    depth = 0
    cur = ''
    for ch in body:
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        cur += ch
    out.append(cur)
    return out


def _range(body):
    parts = body.split('..')
    if len(parts) not in (2, 3):
        return None
    step = 1
    if len(parts) == 3:
        if not parts[2].lstrip('-').isdigit():
            return None
        step = abs(int(parts[2]))
        if step == 0:
            return None
    a, b = parts[0], parts[1]
    if a.lstrip('-').isdigit() and b.lstrip('-').isdigit():
        x, y = int(a), int(b)
        conv = str
    elif len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha():
        x, y = ord(a), ord(b)
        conv = chr
    else:
        return None
    if x <= y:
        return [conv(v) for v in range(x, y + 1, step)]
    return [conv(v) for v in range(x, y - 1, -step)]


def expand(s):
    g = _find_group(s)
    if g is None:
        return [s]
    i, j = g
    pre, body, post = s[:i], s[i + 1:j], s[j + 1:]
    tails = expand(post)
    alts = _range(body)
    if alts is None:
        parts = _split_top(body)
        if len(parts) < 2:
            return [pre + '{' + body + '}' + t for t in tails]
        alts = []
        for p in parts:
            alts.extend(expand(p))
    return [pre + a + t for a in alts for t in tails]
'''

REFERENCE["union_find"] = '''
class DisjointSet:
    def __init__(self):
        self.count = 0
        self._parent = {}
        self._size = {}

    def add(self, x):
        if x in self._parent:
            return False
        self._parent[x] = x
        self._size[x] = 1
        self.count += 1
        return True

    def find(self, x):
        if x not in self._parent:
            raise KeyError(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b):
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        self.count -= 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def size_of(self, x):
        return self._size[self.find(x)]

    def groups(self):
        buckets = {}
        for x in self._parent:
            buckets.setdefault(self.find(x), []).append(x)
        return sorted((sorted(v) for v in buckets.values()), key=lambda v: v[0])
'''

REFERENCE["widest_path"] = '''
def max_bottleneck(graph, src, dst):
    if src not in graph:
        raise KeyError(src)
    if src == dst:
        return float('inf')
    best = {src: float('inf')}
    done = set()
    while True:
        u, bu = None, 0
        for k, v in best.items():
            if k not in done and v > bu:
                u, bu = k, v
        if u is None:
            break
        done.add(u)
        for v, cap in graph.get(u, {}).items():
            nb = min(bu, cap)
            if nb > best.get(v, 0):
                best[v] = nb
    return best.get(dst, 0)


def reachable_within(graph, src, min_cap):
    if src not in graph:
        raise KeyError(src)
    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v, cap in graph.get(u, {}).items():
            if cap >= min_cap and v not in seen:
                seen.add(v)
                stack.append(v)
    return sorted(seen)
'''

REFERENCE["topo_lex"] = '''
import heapq


def _edges(graph):
    indeg = dict((n, 0) for n in graph)
    outs = dict((n, []) for n in graph)
    for n, deps in graph.items():
        for d in deps:
            if d not in graph:
                raise KeyError(d)
            outs[d].append(n)
            indeg[n] += 1
    return indeg, outs


def topo_sort(graph):
    indeg, outs = _edges(graph)
    heap = [n for n in graph if indeg[n] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        n = heapq.heappop(heap)
        out.append(n)
        for m in outs[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, m)
    if len(out) != len(graph):
        raise ValueError('cycle detected')
    return out


def build_waves(graph):
    indeg, outs = _edges(graph)
    ready = sorted(n for n in graph if indeg[n] == 0)
    waves = []
    placed = 0
    while ready:
        waves.append(ready)
        placed += len(ready)
        nxt = []
        for n in ready:
            for m in outs[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    nxt.append(m)
        ready = sorted(nxt)
    if placed != len(graph):
        raise ValueError('cycle detected')
    return waves
'''

REFERENCE["bipartite"] = '''
def _witness(parent, u, v):
    up = []
    x = u
    while x is not None:
        up.append(x)
        x = parent[x]
    idx = dict((n, i) for i, n in enumerate(up))
    down = []
    y = v
    while y not in idx:
        down.append(y)
        y = parent[y]
    return up[:idx[y] + 1] + list(reversed(down))


def two_color(graph):
    color = {}
    parent = {}
    for start in sorted(graph):
        if start in color:
            continue
        color[start] = 0
        parent[start] = None
        stack = [start]
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v not in graph:
                    raise KeyError(v)
                if v not in color:
                    color[v] = 1 - color[u]
                    parent[v] = u
                    stack.append(v)
                elif color[v] == color[u]:
                    return (False, _witness(parent, u, v))
    return (True, color)
'''

REFERENCE["bellman_ford"] = '''
INF = float('inf')


def shortest_paths(n, edges, src):
    dist = [INF] * n
    dist[src] = 0
    for _ in range(max(n - 1, 0)):
        changed = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            raise ValueError('negative cycle reachable from source')
    return [None if d == INF else d for d in dist]


def has_negative_cycle(n, edges):
    dist = [0] * n
    for _ in range(n):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return False
    return any(dist[u] + w < dist[v] for u, v, w in edges)
'''

REFERENCE["min_heap"] = '''
class MinHeap:
    def __init__(self, items=None):
        self._a = [] if items is None else list(items)
        for i in range(len(self._a) // 2 - 1, -1, -1):
            self._down(i)

    def __len__(self):
        return len(self._a)

    def _up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self._a[i] < self._a[p]:
                self._a[i], self._a[p] = self._a[p], self._a[i]
                i = p
            else:
                return

    def _down(self, i):
        n = len(self._a)
        while True:
            small = i
            for c in (2 * i + 1, 2 * i + 2):
                if c < n and self._a[c] < self._a[small]:
                    small = c
            if small == i:
                return
            self._a[i], self._a[small] = self._a[small], self._a[i]
            i = small

    def push(self, x):
        self._a.append(x)
        self._up(len(self._a) - 1)

    def peek(self):
        if not self._a:
            raise IndexError('peek from an empty heap')
        return self._a[0]

    def pop(self):
        if not self._a:
            raise IndexError('pop from an empty heap')
        top = self._a[0]
        last = self._a.pop()
        if self._a:
            self._a[0] = last
            self._down(0)
        return top

    def pushpop(self, x):
        if self._a and self._a[0] < x:
            x, self._a[0] = self._a[0], x
            self._down(0)
        return x

    def replace(self, x):
        if not self._a:
            raise IndexError('replace on an empty heap')
        top = self._a[0]
        self._a[0] = x
        self._down(0)
        return top


def heapsort(xs):
    h = MinHeap(xs)
    return [h.pop() for _ in range(len(h))]
'''

REFERENCE["lfu_cache"] = '''
class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._vals = {}
        self._freq = {}
        self._used = {}
        self._tick = 0

    def _touch(self, key):
        self._tick += 1
        self._used[key] = self._tick
        self._freq[key] = self._freq.get(key, 0) + 1

    def get(self, key):
        if key not in self._vals:
            return None
        self._touch(key)
        return self._vals[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self._vals:
            self._vals[key] = value
            self._touch(key)
            return
        if len(self._vals) >= self.capacity:
            victim = min(self._vals, key=lambda k: (self._freq[k], self._used[k]))
            del self._vals[victim]
            del self._freq[victim]
            del self._used[victim]
        self._vals[key] = value
        self._freq[key] = 0
        self._touch(key)
'''

REFERENCE["circuit_breaker"] = '''
class CircuitBreaker:
    def __init__(self, fail_threshold, reset_timeout, half_open_successes=1):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.half_open_successes = half_open_successes
        self._open = False
        self._opened_at = None
        self._fails = 0
        self._half_ok = 0

    def state(self, now):
        if not self._open:
            return 'closed'
        if now - self._opened_at >= self.reset_timeout:
            return 'half_open'
        return 'open'

    def allow(self, now):
        return self.state(now) != 'open'

    def record(self, now, ok):
        st = self.state(now)
        if st == 'closed':
            if ok:
                self._fails = 0
            else:
                self._fails += 1
                if self._fails >= self.fail_threshold:
                    self._open = True
                    self._opened_at = now
                    self._fails = 0
                    self._half_ok = 0
            return
        if st == 'open':
            self._opened_at = now
            return
        if ok:
            self._half_ok += 1
            if self._half_ok >= self.half_open_successes:
                self._open = False
                self._opened_at = None
                self._fails = 0
                self._half_ok = 0
        else:
            self._opened_at = now
            self._half_ok = 0
'''

REFERENCE["mustache"] = '''
_ESCAPES = (('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;'), ('"', '&quot;'), ("'", '&#39;'))


def _escape(s):
    for a, b in _ESCAPES:
        s = s.replace(a, b)
    return s


def _stringify(v):
    if v is None:
        return ''
    if v is True:
        return 'true'
    if v is False:
        return 'false'
    return str(v)


def _lookup(stack, name):
    if name == '.':
        return stack[-1]
    parts = name.split('.')
    for frame in reversed(stack):
        if isinstance(frame, dict) and parts[0] in frame:
            cur = frame[parts[0]]
            for p in parts[1:]:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur
    return None


def _scan_section(t, i, name):
    depth = 1
    pos = i
    while True:
        j = t.find('{{', pos)
        if j < 0:
            raise ValueError('unclosed section %r' % name)
        if t.startswith('{{{', j):
            k = t.find('}}}', j)
            if k < 0:
                raise ValueError('unclosed tag')
            pos = k + 3
            continue
        k = t.find('}}', j)
        if k < 0:
            raise ValueError('unclosed tag')
        body = t[j + 2:k].strip()
        if body[:1] in ('#', '^') and body[1:].strip() == name:
            depth += 1
        elif body[:1] == '/' and body[1:].strip() == name:
            depth -= 1
            if depth == 0:
                return t[i:j], k + 2
        pos = k + 2


def _render(t, stack):
    out = []
    i = 0
    while i < len(t):
        j = t.find('{{', i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        if t.startswith('{{{', j):
            k = t.find('}}}', j)
            if k < 0:
                raise ValueError('unclosed tag')
            out.append(_stringify(_lookup(stack, t[j + 3:k].strip())))
            i = k + 3
            continue
        k = t.find('}}', j)
        if k < 0:
            raise ValueError('unclosed tag')
        body = t[j + 2:k].strip()
        i = k + 2
        if not body:
            raise ValueError('empty tag')
        sig = body[0] if body[0] in '#^/!&' else ''
        name = body[1:].strip() if sig else body
        if sig == '!':
            continue
        if sig == '&':
            out.append(_stringify(_lookup(stack, name)))
            continue
        if sig == '/':
            raise ValueError('unexpected close tag %r' % name)
        if sig in ('#', '^'):
            inner, i = _scan_section(t, i, name)
            val = _lookup(stack, name)
            if sig == '^':
                if not val:
                    out.append(_render(inner, stack))
                continue
            if isinstance(val, list):
                for item in val:
                    out.append(_render(inner, stack + [item]))
            elif val:
                out.append(_render(inner, (stack + [val]) if isinstance(val, dict) else stack))
            continue
        out.append(_escape(_stringify(_lookup(stack, name))))
    return ''.join(out)


def render(template, context):
    return _render(template, [context])
'''

REFERENCE["csv_rfc4180"] = '''
def parse_csv(text):
    if text == '':
        return []
    rows = []
    row = []
    field = []
    in_quotes = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            field.append(ch)
            i += 1
            continue
        if ch == '"' and not field:
            in_quotes = True
            i += 1
            continue
        if ch == ',':
            row.append(''.join(field))
            field = []
            i += 1
            continue
        if ch == '\\r' and i + 1 < n and text[i + 1] == '\\n':
            row.append(''.join(field))
            field = []
            rows.append(row)
            row = []
            i += 2
            continue
        if ch == '\\n':
            row.append(''.join(field))
            field = []
            rows.append(row)
            row = []
            i += 1
            continue
        field.append(ch)
        i += 1
    if in_quotes:
        raise ValueError('unterminated quoted field')
    row.append(''.join(field))
    rows.append(row)
    if text.endswith('\\n') and rows[-1] == ['']:
        rows.pop()
    return rows


def write_csv(rows):
    lines = []
    for row in rows:
        cells = []
        for c in row:
            if any(x in c for x in (',', '"', '\\r', '\\n')):
                cells.append('"' + c.replace('"', '""') + '"')
            else:
                cells.append(c)
        lines.append(','.join(cells))
    return '\\r\\n'.join(lines)
'''

REFERENCE["patch_apply"] = '''
class ConflictError(Exception):
    pass


def apply_patch(lines, hunks):
    out = []
    pos = 0
    for idx, h in enumerate(hunks):
        start = h['start']
        remove = list(h.get('remove', []))
        if start < pos:
            raise ValueError('hunk %d overlaps or is out of order' % idx)
        if list(lines[start:start + len(remove)]) != remove:
            raise ConflictError('hunk %d does not apply' % idx)
        out.extend(lines[pos:start])
        out.extend(h.get('add', []))
        pos = start + len(remove)
    out.extend(lines[pos:])
    return out


def invert(hunks):
    inverted = []
    delta = 0
    for h in hunks:
        remove = list(h.get('remove', []))
        add = list(h.get('add', []))
        inverted.append({'start': h['start'] + delta, 'remove': add, 'add': remove})
        delta += len(add) - len(remove)
    return inverted
'''

REFERENCE["result_type"] = '''
class Ok:
    def __init__(self, value):
        self.value = value

    def is_ok(self):
        return True

    def map(self, f):
        return Ok(f(self.value))

    def map_err(self, f):
        return self

    def and_then(self, f):
        return f(self.value)

    def unwrap(self):
        return self.value

    def unwrap_or(self, default):
        return self.value

    def __eq__(self, other):
        return isinstance(other, Ok) and other.value == self.value

    def __repr__(self):
        return 'Ok(%r)' % (self.value,)


class Err:
    def __init__(self, error):
        self.error = error

    def is_ok(self):
        return False

    def map(self, f):
        return self

    def map_err(self, f):
        return Err(f(self.error))

    def and_then(self, f):
        return self

    def unwrap(self):
        raise ValueError('unwrap on Err: %r' % (self.error,))

    def unwrap_or(self, default):
        return default

    def __eq__(self, other):
        return isinstance(other, Err) and other.error == self.error

    def __repr__(self):
        return 'Err(%r)' % (self.error,)


def combine(results):
    errors = [r.error for r in results if not r.is_ok()]
    if errors:
        return Err(errors)
    return Ok([r.value for r in results])


def attempt(fn, *args):
    try:
        return Ok(fn(*args))
    except Exception as exc:
        return Err(str(exc) or exc.__class__.__name__)
'''

REFERENCE["json_pointer"] = '''
def _tokens(pointer):
    if pointer == '':
        return []
    if not pointer.startswith('/'):
        raise ValueError('pointer must be empty or start with /')
    return [p.replace('~1', '/').replace('~0', '~') for p in pointer[1:].split('/')]


def _step(cur, tok):
    if isinstance(cur, dict):
        if tok not in cur:
            raise KeyError(tok)
        return cur[tok]
    if isinstance(cur, list):
        if not tok.isdigit():
            raise ValueError('bad array index %r' % tok)
        i = int(tok)
        if i >= len(cur):
            raise IndexError(tok)
        return cur[i]
    raise KeyError(tok)


def resolve(doc, pointer):
    cur = doc
    for tok in _tokens(pointer):
        cur = _step(cur, tok)
    return cur


def set_at(doc, pointer, value):
    toks = _tokens(pointer)
    if not toks:
        raise ValueError('cannot replace the whole document')
    cur = doc
    for tok in toks[:-1]:
        if isinstance(cur, dict) and tok not in cur:
            cur[tok] = {}
        cur = _step(cur, tok)
    last = toks[-1]
    if isinstance(cur, list):
        if last == '-':
            cur.append(value)
            return doc
        if not last.isdigit():
            raise ValueError('bad array index %r' % last)
        i = int(last)
        if i > len(cur):
            raise IndexError(last)
        if i == len(cur):
            cur.append(value)
        else:
            cur[i] = value
        return doc
    if isinstance(cur, dict):
        cur[last] = value
        return doc
    raise KeyError(last)


def remove_at(doc, pointer):
    toks = _tokens(pointer)
    if not toks:
        raise ValueError('cannot remove the whole document')
    cur = doc
    for tok in toks[:-1]:
        cur = _step(cur, tok)
    last = toks[-1]
    if isinstance(cur, dict):
        if last not in cur:
            raise KeyError(last)
        return cur.pop(last)
    if isinstance(cur, list):
        if not last.isdigit():
            raise ValueError('bad array index %r' % last)
        i = int(last)
        if i >= len(cur):
            raise IndexError(last)
        return cur.pop(i)
    raise KeyError(last)
'''
