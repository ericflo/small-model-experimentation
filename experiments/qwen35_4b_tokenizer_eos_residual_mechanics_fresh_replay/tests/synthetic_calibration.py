"""Model-free synthetic calibration decision for pre-live mechanics tests."""

from __future__ import annotations

from typing import Any

from interface_analysis import BOUNDARY_ORDER, PAIR_CONDITIONS, choose_interface


def passing_decision(inputs: Any) -> dict[str, Any]:
    config = inputs.config["interface"]["calibration"]
    gate = {
        "rows": int(config["rows_per_cell"]),
        "exact_echo_successes_min": int(config["exact_echo_successes_min"]),
        "parse_successes_min": int(config["parse_successes_min"]),
        "answer_cap_contacts_max": int(config["answer_cap_contacts_max"]),
        "each_arity_rows": int(config["each_arity_rows"]),
        "each_arity_exact_successes_min": int(
            config["each_arity_exact_successes_min"]
        ),
        "each_arity_parse_successes_min": int(
            config["each_arity_parse_successes_min"]
        ),
        "each_arity_answer_cap_contacts_max": int(
            config["each_arity_answer_cap_contacts_max"]
        ),
    }

    def metrics(passes: bool) -> dict[str, Any]:
        total = gate["rows"] if passes else 0
        per_arity = gate["each_arity_rows"] if passes else 0
        return {
            "rows": gate["rows"],
            "exact_echo_successes": total,
            "parse_successes": total,
            "answer_cap_contacts": 0,
            "by_arity": {
                str(arity): {
                    "rows": gate["each_arity_rows"],
                    "exact_echo_successes": per_arity,
                    "parse_successes": per_arity,
                    "answer_cap_contacts": 0,
                }
                for arity in (2, 3)
            },
        }

    cells = {
        f"{boundary}_{condition}": metrics(
            boundary == "tokenizer_eos"
            and condition == "no_think_program_slot"
        )
        for boundary in BOUNDARY_ORDER
        for condition in PAIR_CONDITIONS
    }
    selection = choose_interface(cells, gate)
    return {
        "schema_version": 1,
        "stage": "authenticated_tokenizer_eos_calibration_decision",
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        **selection,
        "gate": gate,
        "shared_thought": {"synthetic_model_free_fixture": True},
        "boundary_pairs": 192,
        "answer_requests": 384,
        "all_pair_authentication": "PASS",
        "condition_receipts": {},
        "cells": cells,
        "transaction_chain": {},
        "calibration_input_receipt": inputs.read_receipt,
        "hidden_files_read": [],
        "mechanics_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "implementation_lock_sha256": "a" * 64,
        "live_preflight_sha256": "b" * 64,
        "runner_sha256": "c" * 64,
    }
