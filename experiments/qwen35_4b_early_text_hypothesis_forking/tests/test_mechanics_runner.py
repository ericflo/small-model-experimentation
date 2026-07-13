from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import yaml
import pytest


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts" / "run_mechanics.py"
sys.path.insert(0, str(EXP / "src"))
spec = importlib.util.spec_from_file_location("early_fork_mechanics", SCRIPT)
assert spec and spec.loader
mechanics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mechanics
spec.loader.exec_module(mechanics)

from protocol import program_source  # noqa: E402


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def config():
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


@pytest.fixture
def repo_tmp_path():
    with tempfile.TemporaryDirectory(prefix=".mechanics-test-", dir=mechanics.ROOT) as value:
        yield Path(value)


def test_user_prompt_has_menu_but_no_singled_out_candidate():
    prompt = mechanics.mechanics_user_prompt([3, -1, 2, 0, 5])
    assert "Public operation menu" in prompt
    assert "Concrete first operation:" not in prompt
    assert "provisional concrete first-operation hypothesis" in prompt


def test_prepared_rows_are_seed_id_base_prompt_and_length_matched():
    prepared = EXP / "runs" / "mechanics" / "prepared"
    rows = {
        arm: read_jsonl(prepared / f"{arm}_requests.jsonl")
        for arm in mechanics.ARMS
    }
    assert all(len(value) == 96 for value in rows.values())
    systematic_ids = [row["id"] for row in rows["systematic"]]
    assert len(set(systematic_ids)) == 96
    for arm in mechanics.ARMS[1:]:
        assert [row["id"] for row in rows[arm]] == systematic_ids

    for group in zip(*(rows[arm] for arm in mechanics.ARMS), strict=True):
        metas = [row["meta"] for row in group]
        assert len({meta["base_prompt_sha256"] for meta in metas}) == 1
        assert len({meta["base_prompt_token_count"] for meta in metas}) == 1
        assert len({meta["injection_token_count"] for meta in metas}) == 1
        base_count = metas[0]["base_prompt_token_count"]
        for row, meta in zip(group, metas, strict=True):
            injection = row["prompt_token_ids"][base_count:]
            assert len(injection) == meta["injection_token_count"]
            assert 248069 not in injection
            assert 248044 not in injection
            assert injection[-1] == group[0]["prompt_token_ids"][-1]
    receipt = json.loads((prepared / "preoutcome_receipt.json").read_text())
    assert receipt["decision"] == "TOKEN_STITCH_PREPARE_PASS"
    assert receipt["model_loaded"] is False
    assert receipt["outcomes_loaded"] is False
    assert receipt["unique_derangement_compositions"] == 4
    assert receipt["program_ceiling_rows"] == 8

    program = read_jsonl(prepared / "program_ceiling_requests.jsonl")
    assert [row["meta"]["registered_operation"] for row in program] == [
        mechanics.canonical_operation(operation)
        for operation in mechanics.MECHANICS_PROGRAM_FIRST_OPERATIONS
    ]
    assert {operation[0] for operation in mechanics.MECHANICS_PROGRAM_FIRST_OPERATIONS} >= {
        "add_k",
        "mul_k",
        "take_k",
        "rotate_k",
    }


def _synthetic_raw_rows(arm: str):
    prepared = read_jsonl(
        EXP / "runs" / "mechanics" / "prepared" / f"{arm}_requests.jsonl"
    )
    rows = []
    for request in prepared:
        meta = request["meta"]
        if arm == "placebo":
            result = [999_999]
        else:
            result = meta["expected_supplied"]
        rows.append(
            {
                "id": request["id"],
                "meta": meta,
                "outputs": [
                    {
                        "text": f"reason</think>\nRESULT: {result!r}",
                        "seed_stage1": 101,
                        "seed_stage2": 202,
                        "forced_close": True,
                        "n_answer_tokens": 8,
                        "n_sampled_tokens": 24,
                        "n_stage1_prompt_tokens": len(request["prompt_token_ids"]),
                        "n_stage2_prompt_tokens": len(request["prompt_token_ids"]) + 18,
                        "stage1_token_ids": list(range(16)),
                        "stage2_token_ids": list(range(8)),
                        "stage2_finish_reason": "stop",
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    return rows


def _authenticated_rows_and_metadata(arm: str):
    cfg = config()
    prepared = read_jsonl(
        EXP / "runs" / "mechanics" / "prepared" / f"{arm}_requests.jsonl"
    )
    sampling = mechanics._sampling(cfg)
    rows = []
    for request in prepared:
        parent = mechanics._runner_stable_seed(
            sampling.run_seed, request["id"], -1, "stage1"
        )
        rows.append(
            {
                "id": request["id"],
                "meta": request["meta"],
                "prompt_sha256": mechanics._sha256_bytes(b"decoded"),
                "prompt_token_ids_sha256": mechanics._token_ids_sha256(
                    request["prompt_token_ids"]
                ),
                "n_prompt_tokens": len(request["prompt_token_ids"]),
                "prompt_channel": "custom",
                "prompt_logprobs": None,
                "outputs": [
                    {
                        "sample_index": 0,
                        "stage1_parent_seed": parent,
                        "seed_stage1": parent,
                        "seed_stage2": mechanics._runner_stable_seed(
                            sampling.run_seed, request["id"], 0, "stage2"
                        ),
                        "text": "decoded",
                        "token_ids": [10, 11, 248069, 271, 12],
                        "stage1_token_ids": [10, 11],
                        "retained_thinking_token_ids": [10, 11],
                        "injected_token_ids": [248069, 271],
                        "stage2_token_ids": [12],
                        "n_thinking_tokens": 2,
                        "n_answer_tokens": 1,
                        "n_sampled_tokens": 3,
                        "n_injected_tokens": 2,
                        "n_completion_tokens": 5,
                        "n_terminal_tokens_trimmed": 0,
                        "n_stage1_prompt_tokens": len(request["prompt_token_ids"]),
                        "n_stage2_prompt_tokens": len(request["prompt_token_ids"]) + 4,
                        "thinking_closed": True,
                        "forced_close": True,
                        "finish_reason": "stop",
                        "stop_reason": 248044,
                        "stage1_finish_reason": "length",
                        "stage1_stop_reason": None,
                        "truncated": False,
                        "stage1_cumulative_logprob": None,
                        "stage2_cumulative_logprob": None,
                        "sampled_cumulative_logprob": None,
                        "stage1_logprobs": None,
                        "stage2_logprobs": None,
                    }
                ],
            }
        )
    engine = mechanics._engine_config(cfg)
    metadata = {
        "schema_version": mechanics.RUNNER_SCHEMA_VERSION,
        "model": mechanics.MODEL_ID,
        "model_revision": mechanics.MODEL_REVISION,
        "runner_sha256": mechanics._sha256_file(EXP / "src" / "vllm_runner.py"),
        "engine": asdict(engine),
        "engine_args": mechanics._expected_engine_args(engine),
        "resolved_cudagraph": {
            "cudagraph_capture_sizes": list(engine.cudagraph_capture_sizes),
            "max_cudagraph_capture_size": engine.max_num_seqs,
            "has_full_cudagraphs": True,
            "decode_mode": "FULL",
        },
        "sampling": asdict(sampling),
        "resolved_sampling": sampling.resolved_sampling(),
        "adapter": None,
        "think_token_ids": {
            "open": 248068,
            "close": 248069,
            "forced_close_sequence": [248069, 271],
        },
        "termination": {"hf_model_eos_token_id": 248044},
        "runtime": {
            "packages": {
                **mechanics._locked_environment_versions(),
                "vllm": cfg["model"]["vllm_version"],
            },
            "environment_lock": {
                "sha256": mechanics._sha256_file(mechanics.ROOT / "requirements-vllm.lock.txt")
            },
        },
    }
    metadata["counts"] = mechanics._metadata_expected_counts(rows)
    return prepared, rows, metadata


def test_raw_authentication_binds_prompt_seed_metadata_and_output_count(monkeypatch):
    monkeypatch.setattr(mechanics, "_decode_token_ids", lambda _ids: "decoded")
    prepared, rows, metadata = _authenticated_rows_and_metadata("systematic")
    mechanics._authenticate_generation(
        "systematic", rows, metadata, prepared, config()
    )

    natural_rows = deepcopy(rows)
    natural_metadata = deepcopy(metadata)
    natural = natural_rows[0]["outputs"][0]
    natural_ids = [10, 248069] + [12] * 129
    natural.pop("retained_thinking_token_ids")
    natural.update(
        {
            "seed_stage2": None,
            "token_ids": natural_ids,
            "stage1_token_ids": natural_ids,
            "injected_token_ids": [],
            "stage2_token_ids": [],
            "n_thinking_tokens": 1,
            "n_answer_tokens": 129,
            "n_sampled_tokens": len(natural_ids),
            "n_injected_tokens": 0,
            "n_completion_tokens": len(natural_ids),
            "n_stage2_prompt_tokens": 0,
            "thinking_closed": True,
            "forced_close": False,
            "finish_reason": "stop",
            "stage1_finish_reason": "stop",
        }
    )
    natural_metadata["counts"] = mechanics._metadata_expected_counts(natural_rows)
    mechanics._authenticate_generation(
        "systematic", natural_rows, natural_metadata, prepared, config()
    )

    mutations = []

    changed = (deepcopy(rows), deepcopy(metadata))
    changed[0][0]["meta"]["slot"] = 99
    mutations.append(changed)
    changed = (deepcopy(rows), deepcopy(metadata))
    changed[0][0]["prompt_token_ids_sha256"] = "0" * 64
    mutations.append(changed)
    changed = (deepcopy(rows), deepcopy(metadata))
    changed[0][0]["outputs"][0]["seed_stage1"] += 1
    mutations.append(changed)
    changed = (deepcopy(rows), deepcopy(metadata))
    changed[0][0]["outputs"].append(deepcopy(changed[0][0]["outputs"][0]))
    mutations.append(changed)
    changed = (deepcopy(rows), deepcopy(metadata))
    changed[1]["model_revision"] = "wrong"
    mutations.append(changed)
    changed = (deepcopy(rows), deepcopy(metadata))
    changed[1]["sampling"]["thinking_budget"] += 1
    mutations.append(changed)

    for raw_mutation, metadata_mutation in mutations:
        with pytest.raises(RuntimeError):
            mechanics._authenticate_generation(
                "systematic",
                raw_mutation,
                metadata_mutation,
                prepared,
                config(),
            )


def test_receipt_last_transaction_finalizes_without_resampling(monkeypatch, repo_tmp_path):
    tmp_path = repo_tmp_path
    monkeypatch.setattr(mechanics, "_decode_token_ids", lambda _ids: "decoded")
    monkeypatch.setattr(mechanics, "RAW", tmp_path / "raw")
    prepared, rows, metadata = _authenticated_rows_and_metadata("systematic")
    lock_path = tmp_path / "implementation_lock.json"
    live_path = mechanics.RAW / "live_preflight.json"
    mechanics._write_json(lock_path, {"locked": True})
    mechanics._write_json(live_path, {"pass": True})
    paths = mechanics._artifact_paths("systematic")
    mechanics._exclusive_json(
        paths["started"],
        mechanics._started_receipt(
            "systematic", prepared, config(), lock_path, live_path
        ),
    )
    mechanics._write_jsonl(paths["raw"], rows)
    mechanics._write_json(paths["metadata"], metadata)

    loaded = mechanics._load_completed_invocation(
        "systematic",
        prepared,
        config(),
        lock_path,
        live_path,
        allow_finalize=True,
    )
    assert loaded is not None
    assert paths["complete"].is_file()
    first_complete = paths["complete"].read_bytes()
    loaded_again = mechanics._load_completed_invocation(
        "systematic",
        prepared,
        config(),
        lock_path,
        live_path,
        allow_finalize=False,
    )
    assert loaded_again is not None
    assert paths["complete"].read_bytes() == first_complete

    corrupted = deepcopy(rows)
    corrupted[0]["outputs"][0]["text"] = "tampered"
    mechanics._write_jsonl(paths["raw"], corrupted)
    with pytest.raises(RuntimeError):
        mechanics._load_completed_invocation(
            "systematic",
            prepared,
            config(),
            lock_path,
            live_path,
            allow_finalize=False,
        )


def test_started_only_transaction_is_ambiguous_and_never_pending(monkeypatch, repo_tmp_path):
    tmp_path = repo_tmp_path
    monkeypatch.setattr(mechanics, "RAW", tmp_path / "raw")
    prepared = read_jsonl(
        EXP / "runs" / "mechanics" / "prepared" / "systematic_requests.jsonl"
    )
    lock_path = tmp_path / "implementation_lock.json"
    live_path = mechanics.RAW / "live_preflight.json"
    mechanics._write_json(lock_path, {"locked": True})
    mechanics._write_json(live_path, {"pass": True})
    paths = mechanics._artifact_paths("systematic")
    mechanics._exclusive_json(
        paths["started"],
        mechanics._started_receipt(
            "systematic", prepared, config(), lock_path, live_path
        ),
    )
    with pytest.raises(RuntimeError, match="ambiguous started model call"):
        mechanics._load_completed_invocation(
            "systematic",
            prepared,
            config(),
            lock_path,
            live_path,
            allow_finalize=True,
        )


def test_locked_package_mismatch_fails_before_runner_construction(
    monkeypatch, repo_tmp_path
):
    entered = {"runner": False}

    class ForbiddenRunner:
        def __init__(self, *_args, **_kwargs):
            entered["runner"] = True
            raise AssertionError("runner construction must remain unreachable")

    installed = {
        **mechanics._locked_environment_versions(),
        "vllm": config()["model"]["vllm_version"],
    }
    installed["torch"] = "wrong"
    monkeypatch.setattr(mechanics, "verify_implementation_lock", lambda _path: {})
    monkeypatch.setattr(
        mechanics, "_installed_environment_versions", lambda: installed
    )
    monkeypatch.setattr(mechanics, "VLLMRunner", ForbiddenRunner)
    with pytest.raises(RuntimeError, match="live environment differs"):
        mechanics.run_live(repo_tmp_path / "unused-lock.json")
    assert entered["runner"] is False


def test_gate_analyzer_passes_only_complete_specific_control(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    scored = tmp_path / "scored"
    summary = tmp_path / "summary.json"
    raw.mkdir()
    for arm in mechanics.ARMS:
        mechanics._write_jsonl(raw / f"{arm}.jsonl", _synthetic_raw_rows(arm))

    public = {
        "task_id": "synthetic",
        "depth": 2,
        "visible": [{"input": [1, 2], "output": [1, 2]}],
        "unlabeled_probe_inputs": [[3, 4]],
    }
    program_text = program_source([("reverse", None), ("reverse", None)])
    program_rows = []
    for index in range(8):
        program_rows.append(
            {
                "id": f"program-{index}",
                "meta": {
                    "public_task": public,
                    "injection_token_count": 26,
                    "registered_operation": mechanics.canonical_operation(
                        mechanics.MECHANICS_PROGRAM_FIRST_OPERATIONS[index]
                    ),
                },
                "outputs": [
                    {
                        "text": f"reason</think>\n{program_text}",
                        "n_answer_tokens": 40,
                        "n_sampled_tokens": 80,
                        "n_stage1_prompt_tokens": 200,
                        "n_stage2_prompt_tokens": 260,
                        "stage1_token_ids": list(range(40)),
                        "stage2_token_ids": list(range(40)),
                        "stage2_finish_reason": "stop",
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    mechanics._write_jsonl(raw / "program_ceiling.jsonl", program_rows)
    monkeypatch.setattr(mechanics, "RAW", raw)
    monkeypatch.setattr(mechanics, "SCORED", scored)
    monkeypatch.setattr(mechanics, "SUMMARY", summary)
    result = mechanics._score_authenticated(
        config(),
        {
            **{arm: _synthetic_raw_rows(arm) for arm in mechanics.ARMS},
            mechanics.PROGRAM_ARM: program_rows,
        },
    )
    assert result["decision"] == "EARLY_HYPOTHESIS_MECHANICS_PASS"
    assert result["metrics"]["systematic"]["supplied_execution_rate"] == 1.0
    assert result["metrics"]["deranged"]["registered_execution_rate"] == 0.0
    assert result["metrics"]["duplicate"]["registered_execution_rate"] < 0.20
    assert result["metrics"]["placebo"]["registered_execution_rate"] == 0.0
    assert result["program_ceiling"]["visible_pass_rate"] == 1.0
    assert result["program_ceiling"]["scope"] == "noncausal_reachability_only"
    assert result["gates"]["context_adherence_valid"] is True

    parameterized_fail = deepcopy(program_rows)
    wrong_source = program_source([("reverse", None), ("square", None)])
    for row in parameterized_fail[4:]:
        row["outputs"][0]["text"] = f"reason</think>\n{wrong_source}"
    stratified = mechanics._score_authenticated(
        config(),
        {
            **{arm: _synthetic_raw_rows(arm) for arm in mechanics.ARMS},
            mechanics.PROGRAM_ARM: parameterized_fail,
        },
    )
    assert stratified["program_ceiling"]["visible_pass_rate"] == 0.5
    assert stratified["program_ceiling"]["parameterized_visible_pass_rate"] == 0.0
    assert stratified["decision"] == "NO_CORRECT_HYPOTHESIS_CEILING"

    cap_fail = deepcopy(program_rows)
    cap_fail[0]["outputs"][0]["n_answer_tokens"] = 128
    cap_fail[0]["outputs"][0]["finish_reason"] = "length"
    capped = mechanics._score_authenticated(
        config(),
        {
            **{arm: _synthetic_raw_rows(arm) for arm in mechanics.ARMS},
            mechanics.PROGRAM_ARM: cap_fail,
        },
    )
    assert capped["gates"]["program_interface_valid"] is False
    assert capped["decision"] == "INVALID_INTERFACE_PARSE"


def test_contextwise_gate_rejects_one_weak_context(monkeypatch, tmp_path):
    scored = tmp_path / "scored"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(mechanics, "SCORED", scored)
    monkeypatch.setattr(mechanics, "SUMMARY", summary)
    rows = {arm: _synthetic_raw_rows(arm) for arm in mechanics.ARMS}
    for row in rows["systematic"]:
        if row["meta"]["context_id"] == "mechanics-00003":
            row["outputs"][0]["text"] = "reason</think>\nRESULT: [999999]"
    public = {
        "task_id": "synthetic",
        "depth": 2,
        "visible": [{"input": [1, 2], "output": [1, 2]}],
        "unlabeled_probe_inputs": [[3, 4]],
    }
    source = program_source([("reverse", None), ("reverse", None)])
    programs = []
    for index in range(8):
        programs.append(
            {
                "id": f"program-{index}",
                "meta": {
                    "public_task": public,
                    "injection_token_count": 26,
                    "registered_operation": mechanics.canonical_operation(
                        mechanics.MECHANICS_PROGRAM_FIRST_OPERATIONS[index]
                    ),
                },
                "outputs": [
                    {
                        "text": f"reason</think>\n{source}",
                        "n_answer_tokens": 40,
                        "n_sampled_tokens": 80,
                        "n_stage1_prompt_tokens": 200,
                        "n_stage2_prompt_tokens": 260,
                        "stage1_token_ids": list(range(40)),
                        "stage2_token_ids": list(range(40)),
                        "stage2_finish_reason": "stop",
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    result = mechanics._score_authenticated(
        config(), {**rows, mechanics.PROGRAM_ARM: programs}
    )
    assert result["decision"] == "NO_HYPOTHESIS_ADHERENCE"
    assert result["gates"]["context_adherence_valid"] is False
    assert result["context_adherence"]["mechanics-00003"]["pass"] is False
