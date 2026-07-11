"""caravan — state tracking over a natural-language event stream (atoms only).

A trade caravan's cargo and crew change through an ordered event stream; the
item asks one final-state question. All content is invented. The verifier is
an exact event simulator run at generation time.

Levels scale the number of events, goods, and (from L3) conditional and
exchange events that require carrying accurate intermediate state.
"""

from __future__ import annotations

from .. import base

FAMILY = "caravan"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = False

GOODS = (
    "saltglass",
    "emberwool",
    "tallowroot",
    "mirebeans",
    "quillnut",
    "duskwax",
    "pinnowseed",
    "hollowfern",
)
CREW = (
    "Odo",
    "Brint",
    "Sessa",
    "Marn",
    "Yara",
    "Tolvek",
    "Ferrin",
    "Ashka",
    "Dune",
    "Petch",
)
STOPS = (
    "the ford",
    "Gullwatch",
    "the tollgate",
    "Harrow Fen",
    "the salt road",
    "Cinderholt",
    "the low bridge",
    "Weirmarket",
)

_LEVEL_SHAPE = {
    # level: (n_events, n_goods, allow_crew, allow_conditional, allow_exchange)
    1: (6, 3, False, False, False),
    2: (10, 4, True, False, False),
    3: (14, 5, True, True, False),
    4: (18, 5, True, True, True),
}


def _simulate(events: list[dict], goods: list[str]) -> dict:
    cargo = {good: 0 for good in goods}
    crew: set[str] = set()
    for event in events:
        kind = event["kind"]
        if kind == "start_cargo":
            cargo[event["good"]] = event["count"]
        elif kind == "start_crew":
            crew.update(event["names"])
        elif kind == "gain":
            cargo[event["good"]] += event["count"]
        elif kind == "lose":
            cargo[event["good"]] = max(0, cargo[event["good"]] - event["count"])
        elif kind == "spoil_half":
            cargo[event["good"]] //= 2
        elif kind == "join":
            crew.add(event["name"])
        elif kind == "leave":
            crew.discard(event["name"])
        elif kind == "conditional":
            if cargo[event["good"]] > event["threshold"]:
                cargo[event["good"]] = max(0, cargo[event["good"]] - event["sell"])
            else:
                cargo[event["good"]] += event["buy"]
        elif kind == "exchange":
            moved = cargo[event["src"]]
            cargo[event["src"]] = 0
            cargo[event["dst"]] += moved
        else:  # pragma: no cover - generator bug
            raise ValueError(f"unknown event kind {kind!r}")
    return {"cargo": cargo, "crew": crew}


def _render_event(event: dict, rng) -> str:
    kind = event["kind"]
    if kind == "start_cargo":
        return f"At dawn the caravan carries {event['count']} crates of {event['good']}."
    if kind == "start_crew":
        return "Crew at dawn: " + ", ".join(event["names"]) + "."
    if kind == "gain":
        verb = rng.choice(["takes on", "buys", "is paid"])
        return f"At {event['stop']} the caravan {verb} {event['count']} crates of {event['good']}."
    if kind == "lose":
        verb = rng.choice(["sells", "loses", "trades away"])
        return f"At {event['stop']} the caravan {verb} {event['count']} crates of {event['good']}."
    if kind == "spoil_half":
        return (
            f"Damp rot claims half the {event['good']} crates, rounded down."
        )
    if kind == "join":
        return f"{event['name']} joins the crew at {event['stop']}."
    if kind == "leave":
        return f"{event['name']} leaves the crew at {event['stop']}."
    if kind == "conditional":
        return (
            f"If more than {event['threshold']} crates of {event['good']} "
            f"remain, {event['sell']} are sold; otherwise {event['buy']} more "
            f"are bought."
        )
    if kind == "exchange":
        return (
            f"All remaining {event['src']} crates are repacked and counted as "
            f"{event['dst']} from here on."
        )
    raise ValueError(f"unknown event kind {kind!r}")


def _gen_events(rng, level: int) -> tuple[list[dict], list[str]]:
    n_events, n_goods, allow_crew, allow_conditional, allow_exchange = _LEVEL_SHAPE[level]
    goods = list(rng.sample(GOODS, n_goods))
    events: list[dict] = []
    for good in goods[: max(2, n_goods - 1)]:
        events.append({"kind": "start_cargo", "good": good, "count": rng.randint(2, 9)})
    names: list[str] = []
    if allow_crew:
        names = list(rng.sample(CREW, rng.randint(3, 5)))
        events.append({"kind": "start_crew", "names": sorted(names)})

    kinds = ["gain", "lose"]
    if allow_crew:
        kinds += ["join", "leave"]
    if level >= 2:
        kinds.append("spoil_half")
    if allow_conditional:
        kinds += ["conditional", "conditional"]
    if allow_exchange:
        kinds.append("exchange")

    body_count = n_events - len(events)
    off_crew = [name for name in CREW if name not in names]
    for _ in range(body_count):
        kind = rng.choice(kinds)
        stop = rng.choice(STOPS)
        if kind in ("gain", "lose"):
            events.append(
                {
                    "kind": kind,
                    "good": rng.choice(goods),
                    "count": rng.randint(1, 6),
                    "stop": stop,
                }
            )
        elif kind == "spoil_half":
            events.append({"kind": kind, "good": rng.choice(goods)})
        elif kind == "join":
            if not off_crew:
                events.append(
                    {"kind": "gain", "good": rng.choice(goods), "count": rng.randint(1, 6), "stop": stop}
                )
                continue
            name = off_crew.pop(rng.randrange(len(off_crew)))
            names.append(name)
            events.append({"kind": "join", "name": name, "stop": stop})
        elif kind == "leave":
            if not names:
                events.append(
                    {"kind": "lose", "good": rng.choice(goods), "count": rng.randint(1, 6), "stop": stop}
                )
                continue
            name = names.pop(rng.randrange(len(names)))
            off_crew.append(name)
            events.append({"kind": "leave", "name": name, "stop": stop})
        elif kind == "conditional":
            events.append(
                {
                    "kind": "conditional",
                    "good": rng.choice(goods),
                    "threshold": rng.randint(2, 8),
                    "sell": rng.randint(1, 4),
                    "buy": rng.randint(1, 4),
                }
            )
        elif kind == "exchange":
            src, dst = rng.sample(goods, 2)
            events.append({"kind": "exchange", "src": src, "dst": dst})
    return events, goods


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        for attempt in range(20):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) <= base.ATOM_PROMPT_CHAR_LIMIT:
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    events, goods = _gen_events(rng, level)
    final = _simulate(events, goods)

    question_kind = rng.choice(["good", "good", "good", "crew"] if final["crew"] else ["good"])
    if question_kind == "good":
        # Prefer goods with a nonzero final count so a constant "0" reply
        # stays a floor, not a strategy.
        nonzero = [g for g in goods if final["cargo"][g] > 0]
        pool = nonzero if nonzero and rng.random() > 0.10 else goods
        good = rng.choice(pool)
        question = f"How many crates of {good} does the caravan carry at the end?"
        gold = final["cargo"][good]
    else:
        question = "How many people are in the crew at the end?"
        gold = len(final["crew"])

    lines = [
        "A trade caravan travels for one season. Track its state through the",
        "events below, in order.",
        "",
    ]
    lines += [f"{i + 1}. {_render_event(event, rng)}" for i, event in enumerate(events)]
    lines += ["", question, "", base.ATOM_ANSWER_INSTRUCTION]
    prompt = "\n".join(lines)

    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
    }


def score_atom(item: dict, reply_text: str) -> float:
    return base.score_exact_int(item["gold"], reply_text)


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


def selftest() -> dict:
    return base.selftest_atoms(__import__(__name__, fromlist=["x"]))
