from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .abi import Example, Task, candidate_programs, canonical_json, execute_program, prompt_for_task


NAMES = ["Ada", "Grace", "Linus", "Barbara", "Edsger", "Katherine", "Alan", "Joan"]
REGIONS = ["East", "West", "North", "South"]
PRODUCTS = ["pen", "pad", "clip", "tape", "box"]
STATUSES = ["active", "paused", "inactive", "open", "closed"]


def ex(inp: Any, out: Any, kind: str) -> Example:
    return Example(inp, out, kind)


def run(program: dict[str, Any], inp: Any) -> Any:
    return execute_program(program, inp)


def make_task(task_id: str, domain: str, description: str, program: dict[str, Any], visible: list[Any], hidden: list[Any], adversarial: list[Any]) -> Task:
    return Task(
        task_id=task_id,
        domain=domain,
        depth=len(program["steps"]),
        description=description,
        visible=tuple(ex(inp, run(program, inp), "visible") for inp in visible),
        hidden=tuple(ex(inp, run(program, inp), "hidden") for inp in hidden),
        adversarial=tuple(ex(inp, run(program, inp), "adversarial") for inp in adversarial),
        target_program=program,
    )


def rowset(rng: random.Random, n: int = 4) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        name = rng.choice(NAMES)
        status = rng.choice(STATUSES)
        region = rng.choice(REGIONS)
        score = rng.choice([5, 12, 49, 50, 67, 88, 101])
        amount = rng.choice(["$10", "12.50", "$1,200", "99", "101", "250"])
        product = rng.choice(PRODUCTS)
        rows.append(
            {
                "id": str(i + 1),
                "name": f"{name} {rng.choice(NAMES)}",
                "status": status,
                "region": region,
                "score": str(score),
                "amount": amount,
                "product": product,
                "email": f" {name.lower()}@Example.COM ",
            }
        )
    return rows


def scalar_inputs(kind: str, rng: random.Random) -> tuple[list[Any], list[Any], list[Any]]:
    if kind == "date_iso":
        return ["01/02/2024"], ["2024-03-04"], ["September 5, 2024"]
    if kind == "date_us":
        return ["2024-12-25"], ["25-Dec-2024"], ["Dec 5, 2024"]
    if kind == "phone":
        return ["(555) 123-4567"], ["1-212-555-7890"], ["+1 303.555.0100 ext 9"]
    if kind == "sku":
        return ["ab-123"], ["Xy_999"], [" q r 42 "]
    if kind == "slug":
        return ["Hello, World!"], [" A/B test "], ["Already---Slug"]
    if kind == "title":
        return ["ada lovelace"], ["GRACE HOPPER"], ["jean-luc picard"]
    if kind == "email":
        return ["contact a@example.com now"], ["Email: bob.smith+q@foo.co"], ["none"]
    if kind == "zip":
        return ["Ship to 94105 today"], ["zip: 02108-1234"], ["no postal code"]
    if kind == "ticket":
        return ["ticket #ABC-123 opened"], ["see #z9"], ["no hash"]
    return [str(rng.randint(1000, 9999))], [str(rng.randint(1000, 9999))], [str(rng.randint(1000, 9999))]


def program_for_template(template: str, rng: random.Random) -> tuple[str, dict[str, Any], str, list[Any], list[Any], list[Any]]:
    visible_rows = [rowset(rng, 4)]
    hidden_rows = [rowset(rng, 4)]
    adv_rows = [rowset(rng, 5)]
    if template == "filter_select":
        value = visible_rows[0][0]["status"]
        program = {"steps": [{"op": "filter_eq", "col": "status", "value": value}, {"op": "select_cols", "cols": ["name", "email"]}]}
        return "csv_etl", program, f"Keep rows whose status is {value}; return only name and email.", visible_rows, hidden_rows, adv_rows
    if template == "score_select":
        program = {"steps": [{"op": "filter_num_cmp", "col": "score", "cmp": "ge", "threshold": 50}, {"op": "select_cols", "cols": ["name", "score"]}]}
        return "csv_etl", program, "Keep rows with score at least 50 and return name and score.", visible_rows, hidden_rows, adv_rows
    if template == "sort_select":
        program = {"steps": [{"op": "sort_by", "col": "score", "numeric": True, "reverse": True}, {"op": "select_cols", "cols": ["name", "score"]}]}
        return "csv_etl", program, "Sort rows by numeric score descending and return name and score.", visible_rows, hidden_rows, adv_rows
    if template == "normalize_filter_select":
        for rows in [*visible_rows, *hidden_rows, *adv_rows]:
            for idx, row in enumerate(rows):
                row["status"] = " OPEN " if idx % 2 == 0 else " closed "
        program = {
            "steps": [
                {"op": "normalize_col", "col": "status", "mode": "strip_lower"},
                {"op": "filter_eq", "col": "status", "value": "open"},
                {"op": "select_cols", "cols": ["name", "status"]},
            ]
        }
        return "csv_etl", program, "Trim/lowercase status, keep open rows, and return name and status.", visible_rows, hidden_rows, adv_rows
    if template == "split_select":
        program = {"steps": [{"op": "split_full_name"}, {"op": "select_cols", "cols": ["first_name", "last_name"]}]}
        return "csv_etl", program, "Split full names and return only first_name and last_name.", visible_rows, hidden_rows, adv_rows
    if template == "parse_filter_select":
        program = {
            "steps": [
                {"op": "parse_money_col", "col": "amount", "out_col": "amount_num"},
                {"op": "filter_num_cmp", "col": "amount_num", "cmp": "gt", "threshold": 100},
                {"op": "select_cols", "cols": ["product", "amount_num"]},
            ]
        }
        return "csv_etl", program, "Parse amount to amount_num, keep amount_num greater than 100, and return product and amount_num.", visible_rows, hidden_rows, adv_rows
    if template == "group_sum":
        program = {"steps": [{"op": "group_sum", "key_col": "region", "value_col": "score"}]}
        return "csv_etl", program, "Sum score by region.", visible_rows, hidden_rows, adv_rows
    if template == "group_count":
        program = {"steps": [{"op": "group_count", "col": "product"}]}
        return "csv_etl", program, "Count rows by product.", visible_rows, hidden_rows, adv_rows
    if template == "normalize_email":
        program = {"steps": [{"op": "normalize_col", "col": "email", "mode": "strip_lower"}, {"op": "select_cols", "cols": ["name", "email"]}]}
        return "csv_etl", program, "Normalize email cells by trimming and lowercasing, then return name and email.", visible_rows, hidden_rows, adv_rows
    if template == "select_cols":
        program = {"steps": [{"op": "select_cols", "cols": ["name", "region", "score"]}]}
        return "csv_etl", program, "Return only name, region, and score columns.", visible_rows, hidden_rows, adv_rows

    scalar_map = {
        "date_iso": ("date_id_irregular", {"steps": [{"op": "normalize_date_iso"}]}, "Normalize dates to ISO YYYY-MM-DD."),
        "date_us": ("date_id_irregular", {"steps": [{"op": "normalize_date_us"}]}, "Normalize dates to US MM/DD/YYYY."),
        "phone": ("date_id_irregular", {"steps": [{"op": "normalize_phone"}]}, "Normalize US phone numbers to 555-123-4567."),
        "sku": ("date_id_irregular", {"steps": [{"op": "normalize_sku"}]}, "Normalize SKU strings by stripping punctuation and uppercasing."),
        "slug": ("date_id_irregular", {"steps": [{"op": "slugify"}]}, "Slugify title text."),
        "title": ("date_id_irregular", {"steps": [{"op": "title_case"}]}, "Title-case person names."),
        "email": ("date_id_irregular", {"steps": [{"op": "extract_regex", "pattern": "email"}]}, "Extract the email address from text."),
        "zip": ("date_id_irregular", {"steps": [{"op": "extract_regex", "pattern": "zip5"}]}, "Extract the first five-digit ZIP code."),
        "ticket": ("date_id_irregular", {"steps": [{"op": "extract_regex", "pattern": "ticket"}]}, "Extract the ticket id after a # marker."),
    }
    domain, program, description = scalar_map[template]
    visible, hidden, adv = scalar_inputs(template, rng)
    return domain, program, description, visible, hidden, adv


TEMPLATES = [
    "filter_select",
    "score_select",
    "sort_select",
    "normalize_filter_select",
    "split_select",
    "parse_filter_select",
    "group_sum",
    "group_count",
    "normalize_email",
    "select_cols",
    "date_iso",
    "date_us",
    "phone",
    "sku",
    "slug",
    "title",
    "email",
    "zip",
    "ticket",
]


def make_tasks(n: int, seed: int, prefix: str) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)]
        domain, program, description, visible, hidden, adversarial = program_for_template(template, rng)
        tasks.append(make_task(f"{prefix}_{i:04d}_{template}", domain, description, program, visible, hidden, adversarial))
    return tasks


def task_to_record(task: Task, split: str) -> dict[str, Any]:
    candidates = candidate_programs(task)
    target_text = canonical_json(task.target_program)
    return {
        "task_id": task.task_id,
        "split": split,
        "domain": task.domain,
        "depth": task.depth,
        "description": task.description,
        "visible": [{"input": ex.inp, "output": ex.out} for ex in task.visible],
        "hidden": [{"input": ex.inp, "output": ex.out} for ex in task.hidden],
        "adversarial": [{"input": ex.inp, "output": ex.out} for ex in task.adversarial],
        "prompt": prompt_for_task(task),
        "target_program": task.target_program,
        "target_text": target_text,
        "candidate_texts": [canonical_json(program) for program in candidates],
        "candidate_count": len(candidates),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def build_dataset(out_dir: Path, train_n: int = 180, val_n: int = 40, eval_n: int = 48) -> dict[str, Any]:
    train = [task_to_record(task, "train") for task in make_tasks(train_n, 101, "train")]
    val = [task_to_record(task, "validation") for task in make_tasks(val_n, 202, "val")]
    eval_rows = [task_to_record(task, "eval") for task in make_tasks(eval_n, 303, "eval")]
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "validation.jsonl", val)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    summary = {
        "train_n": len(train),
        "validation_n": len(val),
        "eval_n": len(eval_rows),
        "eval_depth_counts": {str(depth): sum(row["depth"] == depth for row in eval_rows) for depth in sorted({row["depth"] for row in eval_rows})},
        "eval_domain_counts": {domain: sum(row["domain"] == domain for row in eval_rows) for domain in sorted({row["domain"] for row in eval_rows})},
        "candidate_count_mean_eval": sum(row["candidate_count"] for row in eval_rows) / len(eval_rows),
    }
    write_json(out_dir / "dataset_summary.json", summary)
    return summary

