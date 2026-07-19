"""Self-contained coding scenarios for fast, high-yield loop-discipline harvesting.

The base solves these reliably via the agentic loop, so they yield abundant COMPLETED
read->write->test->iterate trajectories cheaply. Task shape differs from the real-repo RLVR env
(implement solution.py vs stub-a-function-in-a-repo), but the LOOP behaviour and the tool usage are
identical — which is exactly what the warm-start needs to install. Contamination-free (authored
here, nothing mined from any eval).
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


def _sc(sid, stub, imports, asserts):
    return {"id": sid, "prompt": PROMPT, "check": CHECK,
            "files": {"solution.py": stub, "test_solution.py": _runner(asserts, imports)}}


SCENARIOS = [
    _sc("stats2", "def mean(xs):\n    raise NotImplementedError\n\ndef median(xs):\n    raise NotImplementedError\n",
        "from solution import mean, median",
        [["assert mean([2, 4, 6]) == 4", "assert mean([1, 2]) == 1.5"],
         ["assert median([3, 1, 2]) == 2"], ["assert median([4, 1, 3, 2]) == 2.5"], ["assert median([7]) == 7"]]),
    _sc("rpn_calc", "class Calc:\n    def eval(self, expr):\n        raise NotImplementedError\n",
        "from solution import Calc",
        [["assert Calc().eval('1 2 +') == 3"], ["assert Calc().eval('4 5 + 2 *') == 18"],
         ["assert Calc().eval('10 3 -') == 7"], ["assert Calc().eval('7 2 /') == 3.5"], ["assert Calc().eval('42') == 42"]]),
    _sc("wordcount", "def top_words(text, k):\n    raise NotImplementedError\n",
        "from solution import top_words",
        [["assert top_words('a a b', 2) == [('a', 2), ('b', 1)]"],
         ["assert top_words('The the THE cat', 1) == [('the', 3)]"],
         ["assert top_words('hi, hi! bye.', 2) == [('hi', 2), ('bye', 1)]"],
         ["assert top_words('b a c', 3) == [('a', 1), ('b', 1), ('c', 1)]"]]),
    _sc("flatten", "def flatten(d):\n    raise NotImplementedError\n",
        "from solution import flatten",
        [["assert flatten({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}"],
         ["assert flatten({'a': {'b': 1, 'c': 2}}) == {'a.b': 1, 'a.c': 2}"],
         ["assert flatten({'a': {'b': {'c': 3}}}) == {'a.b.c': 3}"],
         ["assert flatten({'x': 1, 'y': {'z': 2}}) == {'x': 1, 'y.z': 2}"]]),
    _sc("interval_merge", "def merge(intervals):\n    raise NotImplementedError\n",
        "from solution import merge",
        [["assert merge([(1, 3), (2, 6)]) == [(1, 6)]"], ["assert merge([(1, 2), (5, 6)]) == [(1, 2), (5, 6)]"],
         ["assert merge([(5, 6), (1, 4), (2, 3)]) == [(1, 4), (5, 6)]"],
         ["assert merge([(1, 2), (2, 3)]) == [(1, 3)]"], ["assert merge([]) == []"]]),
    _sc("expr_eval", "def evaluate(expr):\n    raise NotImplementedError\n",
        "from solution import evaluate",
        [["assert evaluate('1 + 2 * 3') == 7"], ["assert evaluate('(1 + 2) * 3') == 9"],
         ["assert evaluate('2 * 3 + 4 * 5') == 26"], ["assert evaluate('-3 + 5') == 2"],
         ["assert evaluate('10 - 2 - 3') == 5"], ["assert evaluate('((2))') == 2"]]),
    _sc("mini_vm", "def run(program):\n    # program: list of (op, *args). Return final top of stack.\n    raise NotImplementedError\n",
        "from solution import run",
        [["assert run([('push', 5), ('push', 3), ('add',)]) == 8"],
         ["assert run([('push', 10), ('store', 'x'), ('push', 2), ('load', 'x'), ('mul',)]) == 20"],
         ["assert run([('push', 7), ('push', 4), ('sub',)]) == 3"],
         ["assert run([('push', 6), ('dup',), ('mul',)]) == 36"]]),
    _sc("roman", "def to_roman(n):\n    raise NotImplementedError\n\ndef from_roman(s):\n    raise NotImplementedError\n",
        "from solution import to_roman, from_roman",
        [["assert to_roman(4) == 'IV'"], ["assert to_roman(9) == 'IX'"], ["assert to_roman(1994) == 'MCMXCIV'"],
         ["assert from_roman('IV') == 4"], ["assert from_roman('MCMXCIV') == 1994"],
         ["assert all(from_roman(to_roman(x)) == x for x in range(1, 200))"]]),
    _sc("base_convert", "def to_base(n, b):\n    raise NotImplementedError\n\ndef from_base(s, b):\n    raise NotImplementedError\n",
        "from solution import to_base, from_base",
        [["assert to_base(255, 16) == 'ff'"], ["assert to_base(0, 2) == '0'"], ["assert to_base(-10, 2) == '-1010'"],
         ["assert from_base('ff', 16) == 255"], ["assert from_base('-1010', 2) == -10"],
         ["assert all(from_base(to_base(x, b), b) == x for b in range(2, 20) for x in range(-30, 30))"]]),
    _sc("rle", "def encode(s):\n    raise NotImplementedError\n\ndef decode(s):\n    raise NotImplementedError\n",
        "from solution import encode, decode",
        [["assert encode('aaab') == 'a3b1'"], ["assert encode('') == ''"], ["assert encode('a') == 'a1'"],
         ["assert encode('aaaaaaaaaaaa') == 'a12'"], ["assert decode('a3b1') == 'aaab'"],
         ["assert all(decode(encode(w)) == w for w in ['', 'a', 'zzz', 'abcabc', 'mississippi'])"]]),
    _sc("poly", ("def add(p, q):\n    raise NotImplementedError\n\ndef mul(p, q):\n    raise NotImplementedError\n\n"
                 "def evaluate(p, x):\n    raise NotImplementedError\n\ndef derivative(p):\n    raise NotImplementedError\n"),
        "from solution import add, mul, evaluate, derivative",
        [["assert add([1,2],[3,0,4]) == [4,2,4]"], ["assert add([1,2,3],[-1,-2,-3]) == []"],
         ["assert mul([1,1],[1,1]) == [1,2,1]"], ["assert mul([],[1,2]) == []"],
         ["assert evaluate([1,2,3],2) == 17"], ["assert derivative([5,4,3]) == [4,6]"], ["assert derivative([7]) == []"]]),
    _sc("fraction", "class Fraction:\n    def __init__(self, num, den):\n        raise NotImplementedError\n",
        "from solution import Fraction",
        [["assert Fraction(2,4).num == 1 and Fraction(2,4).den == 2"],
         ["assert Fraction(1,-2).num == -1 and Fraction(1,-2).den == 2"],
         ["assert (Fraction(1,2)+Fraction(1,3)) == Fraction(5,6)"],
         ["assert (Fraction(2,3)*Fraction(3,4)) == Fraction(1,2)"],
         ["assert Fraction(0,5).num == 0 and Fraction(0,5).den == 1"]]),
]
