"""Is in-context learning RETRIEVAL of a familiar rule or INDUCTION of a novel one? Substrate: letter-substitution
ciphers (all matched complexity = a bijection over 26 letters), spanning a FAMILIARITY gradient at MATCHED
complexity: caesar (+k, very familiar), atbash (reverse alphabet, a known cipher), affine (a*i+b, a known concept
but arbitrary key), random (arbitrary, no rule). Few-shot examples (plaintext -> ciphertext) whose letters cover
only PART of the alphabet; the query mixes SEEN and UNSEEN letters. The tell = accuracy on UNSEEN query letters
(generalization): a generalizable rule that the model RETRIEVES (caesar) applies to unseen letters; an arbitrary
map (random) cannot. If accuracy on unseen letters tracks FAMILIARITY at matched complexity, ICL = retrieval of
familiar rules, not induction of novel ones."""
from __future__ import annotations
import random, string

AL = string.ascii_lowercase
N = 26


def make_cipher(kind, rng):
    if kind == "caesar":
        k = rng.randint(3, 23); return {c: AL[(i + k) % N] for i, c in enumerate(AL)}, f"caesar+{k}"
    if kind == "atbash":
        return {c: AL[N - 1 - i] for i, c in enumerate(AL)}, "atbash"
    if kind == "affine":
        a = rng.choice([3, 5, 7, 9, 11]); b = rng.randint(1, 25)  # a coprime to 26
        return {c: AL[(a * i + b) % N] for i, c in enumerate(AL)}, f"affine({a},{b})"
    if kind == "random":
        perm = list(AL); rng.shuffle(perm); return {c: perm[i] for i, c in enumerate(AL)}, "random"
    if kind == "scrambled_caesar":
        # THE CRUX (review): shift by k in a RANDOM alphabet ordering (stated). Same 1-param 'shift by k'
        # description-length as caesar, but the order is unfamiliar -> cannot be retrieved as a named skill,
        # yet fully generalizable IF the model applies the shift in the given order.
        order = list(AL); rng.shuffle(order); k = rng.randint(3, 23)
        pos = {c: i for i, c in enumerate(order)}
        cipher = {c: order[(pos[c] + k) % N] for c in AL}
        cipher["__order__"] = "".join(order); cipher["__k__"] = k
        return cipher, f"scrambled_caesar(k={k})"
    raise ValueError(kind)


def gen_task(kind, seed, n_examples=6, str_len=6, seen_letters=13):
    rng = random.Random(seed)
    cipher, name = make_cipher(kind, rng)
    seen = rng.sample(AL, seen_letters)                 # letters allowed in the examples
    unseen = [c for c in AL if c not in seen]

    def rstr(pool, L): return "".join(rng.choice(pool) for _ in range(L))
    examples = [(s, "".join(cipher[c] for c in s)) for s in (rstr(seen, str_len) for _ in range(n_examples))]
    # query: half seen, half unseen letters (so we can score generalization to UNSEEN)
    qletters = rng.sample(seen, str_len // 2) + rng.sample(unseen, str_len - str_len // 2)
    rng.shuffle(qletters); query = "".join(qletters)
    answer = "".join(cipher[c] for c in query)
    seen_set = set("".join(s for s, _ in examples))
    return {"kind": kind, "name": name, "cipher": cipher, "examples": examples, "query": query,
            "answer": answer, "seen_in_examples": seen_set,
            "query_seen_mask": [c in seen_set for c in query]}


def render(t, application_only=False):
    ex = "\n".join(f"{s} -> {c}" for s, c in t["examples"])
    order_note = ""
    if t["kind"] == "scrambled_caesar":
        order_note = (f"The letters are arranged in this fixed circular order: {t['cipher']['__order__']} "
                      f"(after the last letter it wraps back to the first).\n")
    if application_only:
        table = ", ".join(f"{k}->{v}" for k, v in t["cipher"].items() if not k.startswith("__"))
        return (f"{order_note}A cipher replaces each letter using this table: {table}\n\n"
                f"Apply the cipher to: {t['query']}\n"
                f"Answer with the result string on the last line as `Answer: <string>`.")
    return (f"{order_note}A fixed cipher transforms strings letter-by-letter. Here are examples:\n{ex}\n\n"
            f"Apply the SAME cipher to a new string: {t['query']}\n"
            f"Answer with the result string on the last line as `Answer: <string>`.")


if __name__ == "__main__":
    for kind in ("caesar", "atbash", "affine", "random"):
        t = gen_task(kind, 5)
        print(f"=== {kind} ({t['name']}) ===")
        print(render(t))
        print("answer:", t["answer"], "| query-seen-mask:", t["query_seen_mask"], "\n")
