#!/usr/bin/env python3
"""Resumable, preregistration-locked evidence-acquisition curriculum pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import bank  # noqa: E402
import harness  # noqa: E402
import repo_tasks  # noqa: E402
import retention_tasks_legacy  # noqa: E402
from analyze_menagerie import validate_event_for_resume  # noqa: E402

FROZEN_FILES = harness.REQUIRED_FROZEN_FILES


class TrainingComputeMismatch(RuntimeError):
    def __init__(self, receipt: Path):
        super().__init__(f"trained-arm serial compute mismatch: {receipt}")
        self.receipt = receipt


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def registered_model_tokenizers(cfg: dict) -> dict[str, dict]:
    """Re-hash both immutable parent tokenizers against the frozen config."""
    registrations = {}
    for role, path_key, manifest_key in (
        ("start", "start_checkpoint", "start_tokenizer_manifest_sha256"),
        ("anchor", "locality_anchor", "anchor_tokenizer_manifest_sha256"),
    ):
        model_path = resolve(cfg["model"][path_key])
        try:
            provenance = harness.tokenizer_provenance(model_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid {role} tokenizer provenance: {exc}") from exc
        if (
            provenance["tokenizer_manifest_sha256"]
            != cfg["model"][manifest_key]
            or provenance["tokenizer_compatibility_sha256"]
            != cfg["model"]["tokenizer_compatibility_sha256"]
        ):
            raise SystemExit(f"{role} tokenizer differs from frozen config identity")
        registrations[role] = {
            "model_path": str(model_path.resolve()),
            **provenance,
        }
    if (
        registrations["start"]["tokenizer_compatibility_sha256"]
        != registrations["anchor"]["tokenizer_compatibility_sha256"]
    ):
        raise SystemExit("start and anchor tokenizer compatibility differs")
    return registrations


def registered_parent_checkpoints(cfg: dict) -> dict[str, dict]:
    """Re-hash every model-control surface for the two frozen parent roles."""
    lock_path = EXP / "runs" / "preregistration_receipt.json"
    registrations = {}
    for role, key in (("start", "start_checkpoint"), ("anchor", "locality_anchor")):
        path = resolve(cfg["model"][key])
        try:
            registrations[role] = harness.validate_registered_checkpoint(
                EXP, path, cfg, lock_path, role
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid frozen {role} checkpoint: {exc}") from exc
    return registrations


def expected_tokenizer_manifest(cfg: dict, model_path: Path) -> str:
    anchor = resolve(cfg["model"]["locality_anchor"]).resolve()
    return cfg["model"][
        "anchor_tokenizer_manifest_sha256"
        if model_path.resolve() == anchor
        else "start_tokenizer_manifest_sha256"
    ]


def cpu_tokenizer_equivalence(cfg: dict) -> dict:
    """Prove exact locality prompts and token IDs match without loading weights."""
    from transformers import AutoTokenizer

    registrations = registered_model_tokenizers(cfg)
    contexts_path = resolve(cfg["locality"]["contexts"])
    contexts = json.loads(contexts_path.read_text())["contexts"]
    rendered: dict[str, list[str]] = {}
    token_ids: dict[str, list[list[int]]] = {}
    for role in ("start", "anchor"):
        tokenizer = AutoTokenizer.from_pretrained(
            registrations[role]["model_path"],
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        rendered[role] = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            for row in contexts
        ]
        token_ids[role] = [
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for prompt in rendered[role]
        ]
    if rendered["start"] != rendered["anchor"]:
        raise AssertionError("start/anchor locality chat templates render differently")
    if token_ids["start"] != token_ids["anchor"]:
        raise AssertionError("start/anchor locality token IDs differ")
    maximum = max(map(len, token_ids["start"]))
    if maximum > int(cfg["locality"]["max_context_tokens"]):
        raise AssertionError(f"locality tokenizer context exceeds bound: {maximum}")
    rows = [
        {
            "id": context["id"],
            "rendered_prompt": prompt,
            "token_ids": ids,
        }
        for context, prompt, ids in zip(
            contexts, rendered["start"], token_ids["start"], strict=True
        )
    ]
    return {
        "registrations": registrations,
        "contexts_sha256": sha256_file(contexts_path),
        "n_contexts": len(contexts),
        "max_context_tokens": maximum,
        "rendered_prompts_equal": True,
        "tokenized_context_ids_equal": True,
        "context_tokenization_sha256": hashlib.sha256(
            json.dumps(
                rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
    }


def command(argv: list[str], allowed: tuple[int, ...] = (0,)) -> int:
    print("[run] " + shlex.join(argv), flush=True)
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, argv)
    return completed.returncode


def run_if_missing(
    output: Path, argv: list[str], allowed: tuple[int, ...] = (0,)
) -> tuple[int, dict]:
    if output.is_file():
        print(f"[resume] {output}", flush=True)
        code = 0
    else:
        code = command(argv, allowed)
    if not output.is_file():
        raise SystemExit(f"command did not create registered receipt: {output}")
    return code, json.loads(output.read_text())


def gate_if_missing(output: Path, argv: list[str]) -> tuple[bool, dict]:
    _, payload = run_if_missing(output, argv, allowed=(0, 4))
    return bool(payload.get("gate", {}).get("passed")), payload


def run_gate_fresh(output: Path, argv: list[str]) -> tuple[bool, dict]:
    """Re-run cheap analyzers so resumed gates cannot belong to stale weights."""
    output.unlink(missing_ok=True)
    command(argv, allowed=(0, 4))
    if not output.is_file():
        raise SystemExit(f"gate analyzer did not write {output}")
    payload = json.loads(output.read_text())
    return bool(payload.get("gate", {}).get("passed")), payload


def write_terminal_disposition(
    *,
    stage: str,
    verdict: str,
    receipts: list[Path],
    menagerie_exposed: bool = False,
) -> Path:
    """Make every scientific stop explicit so lifecycle closeout cannot vanish."""
    lock = EXP / "runs" / "preregistration_receipt.json"
    registrations = {}
    for path in receipts:
        if path.is_file():
            registrations[path.stem] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
    payload = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "terminal": True,
        "stage": stage,
        "verdict": verdict,
        "design_lock_sha256": sha256_file(lock),
        "gate_receipts": registrations,
        "menagerie_exposed": menagerie_exposed,
        "lifecycle_closed": False,
        "required_closeout": [
            "README_and_report",
            "experiment_log",
            "program_evidence_and_backlog",
            "synthesis_and_claims_if_warranted",
            "brief_chart_catalog_and_manifest",
            "make_check_direct_main_push_and_CI",
        ],
    }
    output = EXP / "runs" / "terminal_disposition.json"
    if output.is_file():
        try:
            prior = json.loads(output.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit("existing terminal disposition is malformed") from exc
        if prior.get("terminal") is True:
            raise SystemExit(
                "experiment already has a terminal disposition; rescue requires a new intake"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def write_instrument_failure_receipt(
    *, stage: str, error: BaseException, receipts: list[Path]
) -> Path:
    """Persist a non-scientific execution failure before terminalizing the run."""
    registrations = {
        path.stem: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in receipts
        if path.is_file()
    }
    output = EXP / "analysis" / f"{stage}_instrument_failure.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "stage": stage,
        "status": "INSTRUMENT_FAIL",
        "error_type": type(error).__name__,
        "error": str(error),
        "receipts": registrations,
        "menagerie_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    return output


def assert_scientific_run_open() -> None:
    """Make a terminal gate irreversible inside this experiment directory."""
    terminal_path = EXP / "runs" / "terminal_disposition.json"
    if not terminal_path.is_file():
        return
    try:
        payload = json.loads(terminal_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit("existing terminal disposition is malformed") from exc
    if payload.get("terminal") is True:
        raise SystemExit(
            "experiment is terminal; any rescue or rerun requires a new experiment"
        )


def final_menagerie_verdict(passed: bool) -> str:
    return (
        "COUNTERFACTUAL_EVIDENCE_ACQUISITION_POSITIVE"
        if passed
        else "MENAGERIE_FAIL"
    )


def unlogged_menagerie_reservations(cfg: dict, existing: list[dict]) -> list[Path]:
    """Find burned/reserved seeds that cannot safely be called again."""
    logged = {
        (str(row.get("tier")), int(row.get("seed", -1))) for row in existing
    }
    expected_paths = {
        (
            EXP / "runs" / "menagerie_reservations"
            / f"{tier}_seed{int(seed)}.json"
        ).resolve(): (str(tier), int(seed))
        for tier, seed in cfg["menagerie"]["paired_seeds"].items()
    }
    reservation_dir = EXP / "runs" / "menagerie_reservations"
    observed = sorted(reservation_dir.glob("*.json")) if reservation_dir.is_dir() else []
    return [
        path
        for path in observed
        if path.resolve() not in expected_paths
        or expected_paths[path.resolve()] not in logged
    ]


def external_menagerie_seed_collisions(
    cfg: dict, existing: list[dict]
) -> dict[int, list[Path]]:
    """Read seed metadata only and detect frozen seeds consumed by other runs."""
    locally_logged = {
        int(row["seed"]) for row in existing if "seed" in row
    }
    required = {
        int(seed) for seed in cfg["menagerie"]["paired_seeds"].values()
    } - locally_logged
    collisions: dict[int, list[Path]] = defaultdict(list)
    for path in (ROOT / "experiments").glob("*/runs/menagerie_log.jsonl"):
        if EXP in path.parents:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            seed = json.loads(line).get("seed")
            if isinstance(seed, int) and seed in required:
                collisions[seed].append(path)
    for path in (ROOT / "experiments").glob(
        "*/runs/menagerie_reservations/*.json"
    ):
        if EXP in path.parents:
            continue
        seed = json.loads(path.read_text(encoding="utf-8")).get("seed")
        if isinstance(seed, int) and seed in required:
            collisions[seed].append(path)
    return {
        seed: sorted(set(paths)) for seed, paths in sorted(collisions.items())
    }


def terminalize_unavailable_menagerie_seed(
    collisions: dict[int, list[Path]], authorization: Path,
    local_evidence: list[Path] | None = None,
) -> int:
    local_evidence = local_evidence or []
    output = EXP / "analysis" / "menagerie_seed_availability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "stage": "menagerie_seed_availability",
        "collisions": {
            str(seed): [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in paths
            ]
            for seed, paths in collisions.items()
        },
        "gate": {"passed": False, "verdict": "MENAGERIE_SEED_UNAVAILABLE"},
        "local_exposure_evidence_present": bool(local_evidence),
        "public_call_may_have_started": bool(local_evidence),
    }, indent=2, sort_keys=True) + "\n")
    evidence = [path for paths in collisions.values() for path in paths]
    write_terminal_disposition(
        stage="menagerie_seed_availability",
        verdict="MENAGERIE_SEED_UNAVAILABLE",
        receipts=[authorization, output, *local_evidence, *evidence],
        menagerie_exposed=bool(local_evidence),
    )
    return 4


def terminalize_burned_menagerie_seed(
    *, error: BaseException, authorization: Path, evidence: list[Path]
) -> int:
    diagnostic = write_instrument_failure_receipt(
        stage="menagerie_seed",
        error=error,
        receipts=[authorization, *evidence],
    )
    write_terminal_disposition(
        stage="menagerie_instrument",
        verdict="MENAGERIE_INSTRUMENT_FAIL",
        receipts=[authorization, *evidence, diagnostic],
        menagerie_exposed=True,
    )
    return 4


def calibration_terminal_verdict(gate_path: Path) -> str:
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        payload.get("stage") == "trained_calibration_feasibility"
        and payload.get("gate", {}).get("passed") is False
    ):
        return "CALIBRATION_INFEASIBLE"
    return "CALIBRATION_FAIL"


def transfer_terminal_verdict(block: str, gate_path: Path) -> str:
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        payload.get("stage") == "transfer_feasibility"
        and payload.get("block") == block
        and payload.get("gate", {}).get("passed") is False
    ):
        expected = f"{block.upper()}_INFEASIBLE"
        if payload.get("gate", {}).get("verdict") != expected:
            raise SystemExit("transfer feasibility receipt has an invalid stop verdict")
        return expected
    return f"{block.upper()}_FAIL"


def _closeout_stable_paths() -> tuple[Path, ...]:
    """Human-curated surfaces whose exact result state must survive sealing."""
    return (
        EXP / "README.md",
        EXP / "experiment_log.md",
        EXP / "reports" / "report.md",
        EXP / "reports" / "artifact_manifest.yaml",
        ROOT / "research_programs" / "agentic_breadth_installation" / "evidence.md",
        ROOT / "research_programs" / "agentic_breadth_installation" / "backlog.md",
        ROOT / "knowledge" / "program_scorecards.md",
        ROOT / "knowledge" / "synthesis.md",
        ROOT / "knowledge" / "experiment_brief.json",
        ROOT / "knowledge" / "experiment_viz.json",
        ROOT / "knowledge" / "claims" / "claim_ledger.json",
        ROOT / "knowledge" / "experiment_status.json",
        ROOT / "knowledge" / "experiment_dates.json",
    )


def _csv_contains_experiment(path: Path) -> bool:
    return any(
        line.split(",", 1)[0] == EXP.name
        for line in path.read_text().splitlines()[1:]
    )


def _validate_closeout_semantics(payload: dict) -> None:
    """Reject a nominal closeout whose public/program surfaces are still active."""
    verdict = str(payload.get("verdict", ""))
    if not verdict:
        raise SystemExit("terminal disposition has no verdict")
    readme = (EXP / "README.md").read_text()
    report = (EXP / "reports" / "report.md").read_text()
    log = (EXP / "experiment_log.md").read_text()
    if "**Status:** finished" not in readme:
        raise SystemExit("README is not marked finished")
    if "Designed and preregistered; no scientific result" in report:
        raise SystemExit("report still contains the design-only status")
    if verdict not in report or verdict not in log:
        raise SystemExit("report and experiment log must record the terminal verdict")

    for path in (
        ROOT / "research_programs" / "agentic_breadth_installation" / "evidence.md",
        ROOT / "research_programs" / "agentic_breadth_installation" / "backlog.md",
        ROOT / "knowledge" / "program_scorecards.md",
    ):
        text = path.read_text()
        if EXP.name not in text:
            raise SystemExit(f"closeout surface does not name the experiment: {path}")
        if verdict not in text:
            raise SystemExit(
                f"closeout surface does not record the terminal verdict: {path}"
            )

    status = json.loads((ROOT / "knowledge" / "experiment_status.json").read_text())
    if EXP.name in status.get("experiments", {}):
        raise SystemExit(
            "finished experiment remains in knowledge/experiment_status.json"
        )

    briefs = json.loads((ROOT / "knowledge" / "experiment_brief.json").read_text())
    brief = briefs.get("experiments", {}).get(EXP.name)
    if not isinstance(brief, dict):
        raise SystemExit("finished experiment has no practitioner brief")
    stale_brief_phrases = (
        "no model has run",
        "designed and preregistered",
        "this is a design and preregistration",
        "model calls so far",
    )
    brief_text = json.dumps(brief, sort_keys=True).lower()
    if any(phrase in brief_text for phrase in stale_brief_phrases):
        raise SystemExit("practitioner brief still describes an active design")

    visualizations = json.loads(
        (ROOT / "knowledge" / "experiment_viz.json").read_text()
    )
    charts = visualizations.get("experiments", {}).get(EXP.name, {}).get("charts", [])
    headline_charts = [chart for chart in charts if chart.get("headline") is True]
    stale_chart_phrases = (
        "design only",
        "no model-bearing run",
        "planned training rows",
        "preregistered design",
    )
    if not headline_charts or all(
        any(
            phrase in json.dumps(chart, sort_keys=True).lower()
            for phrase in stale_chart_phrases
        )
        for chart in headline_charts
    ):
        raise SystemExit("headline chart still describes only the preregistered design")

    manifest = yaml.safe_load(
        (EXP / "reports" / "artifact_manifest.yaml").read_text()
    )
    if manifest.get("experiment_id") != EXP.name:
        raise SystemExit("artifact manifest is not bound to this experiment")
    manifest_text = json.dumps(manifest, sort_keys=True).lower()
    if verdict.lower() not in manifest_text or any(
        phrase in manifest_text
        for phrase in (
            "design-only manifest",
            "when generated",
            "if the stage is authorized",
        )
    ):
        raise SystemExit("artifact manifest still describes future design artifacts")
    metadata = yaml.safe_load((EXP / "metadata.yaml").read_text())
    if (
        metadata.get("id") != EXP.name
        or not metadata.get("file_counts")
        or int(metadata.get("total_files", 0)) <= 0
    ):
        raise SystemExit("generated experiment metadata is stale")
    dates = json.loads((ROOT / "knowledge" / "experiment_dates.json").read_text())
    if EXP.name not in dates.get("experiments", {}):
        raise SystemExit("finished experiment has no experiment-date entry")
    for path in (
        ROOT / "knowledge" / "experiment_catalog.csv",
        ROOT / "knowledge" / "experiment_readiness.csv",
    ):
        if not _csv_contains_experiment(path):
            raise SystemExit(f"generated closeout index is stale: {path}")


def _validate_closeout_seal_contract(payload: dict) -> None:
    expected_files = {
        str(path.relative_to(ROOT)) for path in _closeout_stable_paths()
    }
    closeout_files = payload.get("closeout_files")
    validation = payload.get("closeout_validation")
    documentation_commit = payload.get("closeout_documentation_commit")
    if (
        payload.get("schema_version") != 1
        or payload.get("terminal") is not True
        or payload.get("experiment_id") != EXP.name
        or payload.get("lifecycle_closed") is not True
        or payload.get("closeout_receipt_requires_push") is not True
        or not isinstance(documentation_commit, str)
        or len(documentation_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in documentation_commit
        )
        or not is_sha256(payload.get("open_terminal_disposition_sha256"))
        or not isinstance(closeout_files, dict)
        or set(closeout_files) != expected_files
        or any(not is_sha256(value) for value in closeout_files.values())
        or validation != {
            "command": "make check",
            "passed": True,
            "validated_commit": documentation_commit,
        }
    ):
        raise SystemExit("terminal disposition has an incomplete or forged closeout seal")


def close_terminal_lifecycle() -> Path:
    """Bind a pushed result-documentation commit and write its publishable seal."""
    verify_design_lock()
    terminal_path = EXP / "runs" / "terminal_disposition.json"
    if not terminal_path.is_file():
        raise SystemExit("no terminal scientific disposition exists to close")
    payload = json.loads(terminal_path.read_text())
    if payload.get("terminal") is not True or payload.get("experiment_id") != EXP.name:
        raise SystemExit("terminal disposition is malformed")
    if payload.get("lifecycle_closed") is True:
        raise SystemExit("terminal disposition is already sealed")
    closeout_paths = _closeout_stable_paths()
    missing = [str(path) for path in closeout_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"closeout surface is missing: {missing}")
    _validate_closeout_semantics(payload)
    terminal_relative = str(terminal_path.relative_to(ROOT))
    try:
        _git_output("ls-files", "--error-unmatch", terminal_relative)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "open terminal disposition must be committed before sealing"
        ) from exc
    if _git_output("status", "--short"):
        raise SystemExit("the repository must be clean before lifecycle sealing")
    head = _git_output("rev-parse", "HEAD")
    if head != _git_output("rev-parse", "origin/main"):
        raise SystemExit("closeout documentation must be pushed to origin/main")
    command(["make", "check"])
    open_terminal_sha256 = sha256_file(terminal_path)
    payload["lifecycle_closed"] = True
    payload["closeout_documentation_commit"] = head
    payload["open_terminal_disposition_sha256"] = open_terminal_sha256
    payload["closeout_files"] = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in closeout_paths
    }
    payload["closeout_validation"] = {
        "command": "make check",
        "passed": True,
        "validated_commit": head,
    }
    payload["closeout_receipt_requires_push"] = True
    terminal_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return terminal_path


def verify_terminal_lifecycle() -> Path:
    """Prove the closed receipt and regenerated indexes are clean on origin/main."""
    verify_design_lock()
    terminal_path = EXP / "runs" / "terminal_disposition.json"
    if not terminal_path.is_file():
        raise SystemExit("closed terminal disposition is missing")
    payload = json.loads(terminal_path.read_text())
    _validate_closeout_seal_contract(payload)
    terminal_relative = str(terminal_path.relative_to(ROOT))
    try:
        _git_output("ls-files", "--error-unmatch", terminal_relative)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("closed terminal disposition is not tracked") from exc
    if _git_output("status", "--short"):
        raise SystemExit("closed lifecycle must be verified from a clean repository")
    head = _git_output("rev-parse", "HEAD")
    if head != _git_output("rev-parse", "origin/main"):
        raise SystemExit("closed lifecycle receipt is not pushed to origin/main")
    documentation_commit = str(payload.get("closeout_documentation_commit", ""))
    if len(documentation_commit) != 40:
        raise SystemExit("closeout documentation commit is not immutable")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", documentation_commit, head],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit("closeout documentation commit is not an ancestor of HEAD")
    terminal_relative_at_commit = str(terminal_path.relative_to(ROOT))
    opened = subprocess.run(
        ["git", "show", f"{documentation_commit}:{terminal_relative_at_commit}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if (
        opened.returncode != 0
        or hashlib.sha256(opened.stdout).hexdigest()
        != payload["open_terminal_disposition_sha256"]
    ):
        raise SystemExit("open terminal disposition is not bound to its documentation commit")
    for relative, expected in payload.get("closeout_files", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"closeout surface changed after sealing: {relative}")
    _validate_closeout_semantics(payload)
    command(["make", "check"])
    print(
        f"[run] lifecycle verified closed on origin/main at {head}",
        flush=True,
    )
    return terminal_path


def _registered_bank_receipt(cfg: dict) -> tuple[Path, dict]:
    path = resolve(cfg["artifacts"]["root"]) / "bank" / "receipt.json"
    if not path.is_file():
        raise SystemExit("bank receipt is missing; run --smoke first")
    payload = json.loads(path.read_text())
    expected = {
        "config_sha256": sha256_file(EXP / "configs" / "default.yaml"),
        "builder_sha256": sha256_file(EXP / "scripts" / "build_bank.py"),
        "bank_library_sha256": sha256_file(EXP / "src" / "bank.py"),
        "task_generator_sha256": sha256_file(EXP / "src" / "repo_tasks.py"),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(f"stale bank receipt at {key}: rerun --smoke")
    start_tokenizer = registered_model_tokenizers(cfg)["start"]
    registered_tokenizer = {
        key: payload.get(key)
        for key in harness.TOKENIZER_PROVENANCE_KEYS
    }
    observed_tokenizer = {
        key: start_tokenizer[key]
        for key in harness.TOKENIZER_PROVENANCE_KEYS
    }
    if registered_tokenizer != observed_tokenizer:
        raise SystemExit("bank receipt tokenizer differs from frozen start tokenizer")
    for arm in cfg["training"]["arms"]:
        bank_path = path.parent / f"{arm}.jsonl"
        if payload.get("bank_sha256", {}).get(arm) != sha256_file(bank_path):
            raise SystemExit(f"bank file/receipt mismatch for {arm}")
    return path, payload


def _smoke_input_files() -> dict[str, str]:
    return {relative: sha256_file(EXP / relative) for relative in FROZEN_FILES}


def _validate_smoke_receipt(
    cfg: dict,
    smoke_path: Path,
    bank_path: Path,
    bank_receipt: dict,
) -> dict:
    """Bind PASS to the exact frozen implementation and current model-free outputs."""
    if not smoke_path.is_file():
        raise SystemExit("deterministic smoke receipt is missing")
    smoke = json.loads(smoke_path.read_text())
    required = {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": EXP.name,
        "model_output_generated": False,
        "firewall_clean": True,
        "smoke_input_files": _smoke_input_files(),
        "bank_receipt_sha256": sha256_file(bank_path),
        "bank_sha256": bank_receipt["bank_sha256"],
        "rows_per_arm": bank_receipt["rows_per_arm"],
        "transition_counts": bank_receipt["transition_counts"],
        "weighted_action_mass_by_transition": bank_receipt[
            "weighted_action_mass_by_transition"
        ],
        "shuffle_proof": bank_receipt["within_dyad_shuffle_proof"],
        "parent_checkpoints": registered_parent_checkpoints(cfg),
    }
    for key, expected in required.items():
        if smoke.get(key) != expected:
            raise SystemExit(f"stale deterministic smoke receipt at {key}")
    artifacts = resolve(cfg["artifacts"]["root"])
    expected_encoding = {
        arm: sha256_file(
            artifacts / "preflight" / "encode" / arm / "encoding_receipt.json"
        )
        for arm in cfg["training"]["arms"]
    }
    if smoke.get("encoding_receipt_sha256") != expected_encoding:
        raise SystemExit("stale deterministic smoke encoding receipts")
    geometry_path = EXP / "reports" / "context_geometry_receipt.json"
    if (
        not geometry_path.is_file()
        or smoke.get("context_geometry") != json.loads(geometry_path.read_text())
    ):
        raise SystemExit("stale deterministic smoke context geometry")
    if smoke.get("tokenizer_equivalence") != cpu_tokenizer_equivalence(cfg):
        raise SystemExit("stale deterministic smoke tokenizer equivalence")
    return smoke


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def write_design_lock(design_commit: str) -> None:
    cfg = config()
    resolved_commit = _git_output("rev-parse", design_commit)
    if _git_output("rev-parse", "HEAD") != resolved_commit:
        raise SystemExit("--lock-design must name the current committed design HEAD")
    if _git_output("rev-parse", "origin/main") != resolved_commit:
        raise SystemExit("the immutable design commit must already be on origin/main")
    if _git_output("status", "--short"):
        raise SystemExit("the repository must be clean before writing the design lock")
    missing = [relative for relative in FROZEN_FILES if not (EXP / relative).is_file()]
    if missing:
        raise SystemExit(f"frozen design files are missing: {missing}")
    paths = [str(EXP / relative) for relative in FROZEN_FILES]
    dirty = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"frozen design files are not committed:\n{dirty}")
    bank_path, bank_receipt = _registered_bank_receipt(cfg)
    smoke_path = EXP / "reports" / "smoke_receipt.json"
    _validate_smoke_receipt(cfg, smoke_path, bank_path, bank_receipt)
    model_tokenizers = registered_model_tokenizers(cfg)
    model_checkpoints = registered_parent_checkpoints(cfg)
    artifacts = resolve(cfg["artifacts"]["root"])
    forbidden = []
    for local_root in (EXP / "analysis", EXP / "runs"):
        if local_root.exists():
            forbidden.extend(
                str(path) for path in local_root.rglob("*") if path.is_file()
            )
    if artifacts.exists():
        for path in artifacts.rglob("*"):
            relative_parts = path.relative_to(artifacts).parts
            if path.is_file() and (
                path.name in {
                    "training_receipt.json", "adapter_model.safetensors",
                    "merge_receipt.json", "model.safetensors",
                }
                or bool(
                    set(relative_parts)
                    & {"adapters", "merged", "eval", "locality", "uncertainty", "menagerie"}
                )
            ):
                forbidden.append(str(path))
    if forbidden:
        raise SystemExit(
            "model output exists before design lock; quarantine it before locking: "
            + ", ".join(forbidden[:5])
        )
    receipt = {
        "schema_version": 1,
        "status": "locked",
        "experiment_id": EXP.name,
        "design_commit": resolved_commit,
        "frozen_file_order": list(FROZEN_FILES),
        "frozen_files": {
            relative: sha256_file(EXP / relative) for relative in FROZEN_FILES
        },
        "smoke_receipt_sha256": sha256_file(smoke_path),
        "bank_receipt": str(bank_path.resolve()),
        "bank_receipt_sha256": sha256_file(bank_path),
        "bank_sha256": bank_receipt["bank_sha256"],
        "model_tokenizers": model_tokenizers,
        "model_checkpoints": model_checkpoints,
        "model_output_precedes_lock": False,
        "note": "Only deterministic task execution, tokenizer geometry, bank construction, and encode-only audits preceded this lock.",
    }
    output = EXP / "runs" / "preregistration_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


def verify_design_lock(*, require_pushed: bool = True) -> dict:
    path = EXP / "runs" / "preregistration_receipt.json"
    if not path.is_file():
        raise SystemExit("preregistration receipt is missing; model execution is illegal")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "locked"
        or payload.get("experiment_id") != EXP.name
        or tuple(payload.get("frozen_file_order", ())) != FROZEN_FILES
        or set(payload.get("frozen_files", {})) != set(FROZEN_FILES)
        or payload.get("model_output_precedes_lock") is not False
    ):
        raise SystemExit("preregistration receipt is not the registered design lock")
    design_commit = payload.get("design_commit")
    if (
        not isinstance(design_commit, str)
        or len(design_commit) != 40
        or any(character not in "0123456789abcdef" for character in design_commit)
    ):
        raise SystemExit("design lock does not contain an immutable full commit SHA")
    for relative, expected in payload["frozen_files"].items():
        observed = sha256_file(EXP / relative)
        if observed != expected:
            raise SystemExit(f"frozen design changed: {relative} {observed} != {expected}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", design_commit, "HEAD"],
        cwd=ROOT, check=False,
    ).returncode:
        raise SystemExit("design commit is not an ancestor of HEAD")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    dirty = _git_output("status", "--short", "--", str(path))
    if tracked.returncode or dirty:
        raise SystemExit("design-lock receipt must be committed before model execution")
    if require_pushed:
        head = _git_output("rev-parse", "HEAD")
        origin = _git_output("rev-parse", "origin/main")
        if head != origin:
            raise SystemExit("design lock must be pushed to origin/main before model execution")
    cfg = config()
    bank_path, bank_receipt = _registered_bank_receipt(cfg)
    smoke_path = EXP / "reports" / "smoke_receipt.json"
    if (
        payload.get("bank_receipt_sha256") != sha256_file(bank_path)
        or payload.get("bank_sha256") != bank_receipt.get("bank_sha256")
        or payload.get("smoke_receipt_sha256") != sha256_file(smoke_path)
    ):
        raise SystemExit("model-free smoke or bank changed after design lock")
    _validate_smoke_receipt(cfg, smoke_path, bank_path, bank_receipt)
    if payload.get("model_tokenizers") != registered_model_tokenizers(cfg):
        raise SystemExit("registered model tokenizer changed after design lock")
    if payload.get("model_checkpoints") != registered_parent_checkpoints(cfg):
        raise SystemExit("registered parent checkpoint changed after design lock")
    return payload


def _cpu_task_invariants(cfg: dict) -> dict:
    if cfg["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise AssertionError("single-model constraint drifted")
    if tuple(cfg["families"]["acquisition_train"]) != repo_tasks.TRAIN_FAMILIES:
        raise AssertionError("train family registration drifted")
    if tuple(cfg["families"]["acquisition_transfer"]) != repo_tasks.TRANSFER_FAMILIES:
        raise AssertionError("transfer family registration drifted")
    tasks = repo_tasks.make_pairs(
        repo_tasks.ALL_FAMILIES, 1, 92801, "cpu_smoke"
    )
    by_pair: dict[str, list[repo_tasks.RepoTask]] = defaultdict(list)
    for task in tasks:
        by_pair[task.pair_id].append(task)
        for state, expected in (
            ("initial", (False, False)),
            ("partial", (False, False)),
            ("oracle", (True, True)),
        ):
            env = repo_tasks.RepoEnv(task)
            try:
                if state == "partial":
                    env.apply_partial()
                elif state == "oracle":
                    env.apply_oracle()
                if env.score_workspace() != expected:
                    raise AssertionError((task.task_id, state, env.score_workspace()))
            finally:
                env.close()
    for pair_id, members in by_pair.items():
        a, b = sorted(members, key=lambda row: row.branch)
        if len(members) != 2 or repo_tasks.pair_static_digest(a) != repo_tasks.pair_static_digest(b):
            raise AssertionError(f"counterfactual pair hygiene failed: {pair_id}")
        if {path for path in a.files if a.files[path] != b.files[path]} != {a.evidence_path}:
            raise AssertionError(f"pair has a non-evidence byte difference: {pair_id}")
        for source, counterpart in ((a, b), (b, a)):
            env = repo_tasks.RepoEnv(counterpart)
            try:
                patch = source.oracle_patches[0]
                if not env.patch(patch.path, patch.old, patch.new).startswith("PATCH_OK"):
                    raise AssertionError("cross-branch patch was not applicable")
                if all(env.score_workspace()):
                    raise AssertionError("cross-branch patch unexpectedly passes")
            finally:
                env.close()
    blocks = {}
    for name, block_cfg in cfg["evaluation"]["blocks"].items():
        if block_cfg.get("legacy_retention"):
            continue
        block_tasks = repo_tasks.make_pairs(
            tuple(cfg["families"][block_cfg["families"]]),
            int(block_cfg["tasks_per_family"]) // 2,
            int(block_cfg["seed"]),
            name,
        )
        hashes = {repo_tasks.content_digest(task) for task in block_tasks}
        if len(hashes) != len(block_tasks):
            raise AssertionError(f"duplicate repository content in {name}")
        blocks[name] = hashes
    for name, block_cfg in cfg["evaluation"]["blocks"].items():
        for prior in block_cfg.get("disjoint_from", []):
            if blocks[name] & blocks[prior]:
                raise AssertionError(f"registered blocks overlap: {name}/{prior}")
    bank_tasks = repo_tasks.make_pairs(
        repo_tasks.TRAIN_FAMILIES,
        int(cfg["bank"]["inferred_pairs_per_train_family"]),
        int(cfg["bank"]["seed"]),
        str(cfg["bank"]["split"]),
    )
    if any(
        {repo_tasks.content_digest(task) for task in bank_tasks} & hashes
        for hashes in blocks.values()
    ):
        raise AssertionError("training bank overlaps an evaluation block")
    legacy_count = 0
    for name in ("old_broad_retention", "old_transaction_retention"):
        block_cfg = cfg["evaluation"]["blocks"][name]
        legacy = retention_tasks_legacy.make_tasks(
            tuple(cfg["families"][block_cfg["families"]]), 1,
            int(block_cfg["seed"]), f"{name}_cpu_smoke",
        )
        for task in legacy:
            env = retention_tasks_legacy.RepoEnv(task)
            try:
                if (env.visible_pass(), env.hidden_pass()) != (False, False):
                    raise AssertionError(f"legacy initial state passes: {task.task_id}")
                env.apply_oracle()
                if (env.visible_pass(), env.hidden_pass()) != (True, True):
                    raise AssertionError(f"legacy oracle fails: {task.task_id}")
            finally:
                env.close()
        legacy_count += len(legacy)
    locality = json.loads(resolve(cfg["locality"]["contexts"]).read_text())
    current = {row["content_sha256"] for row in locality["contexts"]}
    if len(current) != int(cfg["locality"]["count"]):
        raise AssertionError("locality count/hash uniqueness failed")
    prior_hashes = set()
    for path in (ROOT / "experiments").glob("*/data/*.json"):
        if EXP in path.parents:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("contexts", []) if isinstance(payload, dict) else []:
            if isinstance(row, dict) and row.get("content_sha256"):
                prior_hashes.add(row["content_sha256"])
    if current & prior_hashes:
        raise AssertionError("locality block reuses prior content")
    return {
        "task_pairs": len(by_pair),
        "channels": sorted({task.evidence_channel for task in tasks}),
        "path_regimes": sorted({task.evidence_path_regime for task in tasks}),
        "cross_branch_patches_fail": True,
        "initial_partial_fail_oracle_pass": True,
        "evaluation_blocks": {key: len(value) for key, value in blocks.items()},
        "bank_eval_content_disjoint": True,
        "legacy_tasks_selftested": legacy_count,
        "fresh_locality_contexts": len(current),
    }


def cpu_smoke() -> dict:
    cfg = config()
    artifacts = resolve(cfg["artifacts"]["root"])
    py = sys.executable
    tokenizer_equivalence = cpu_tokenizer_equivalence(cfg)
    command([py, str(EXP / "scripts" / "build_bank.py")])
    bank_path, bank_receipt = _registered_bank_receipt(cfg)
    tcfg = cfg["training"]
    encoding = {}
    for arm in tcfg["arms"]:
        output = artifacts / "preflight" / "encode" / arm
        command([
            py, str(EXP / "scripts" / "train.py"), "--arm", arm,
            "--base-model", str(resolve(cfg["model"]["start_checkpoint"])),
            "--expected-base-weight-sha256", cfg["model"]["start_weight_sha256"],
            "--train", str(bank_path.parent / f"{arm}.jsonl"),
            "--expected-train-sha256", bank_receipt["bank_sha256"][arm],
            "--bank-receipt", str(bank_path), "--out", str(output),
            "--epochs", str(tcfg["epochs"]), "--lr", str(tcfg["learning_rate"]),
            "--rank", str(tcfg["rank"]), "--alpha", str(tcfg["alpha"]),
            "--batch-size", str(tcfg["batch_size"]),
            "--grad-accum", str(tcfg["gradient_accumulation_steps"]),
            "--loss-chunk-positions", str(tcfg["loss_chunk_positions"]),
            "--max-length", str(tcfg["max_length"]), "--seed", str(tcfg["seed"]),
            "--encode-only",
        ])
        encoding[arm] = json.loads((output / "encoding_receipt.json").read_text())
    serial_tokens = [row["serial_forward_tokens_per_epoch"] for row in encoding.values()]
    serial_ratio = max(serial_tokens) / min(serial_tokens)
    if serial_ratio > float(tcfg["serial_token_compute_ratio_max"]):
        raise AssertionError(f"training arms exceed serial compute match: {serial_ratio}")
    geometry_path = EXP / "reports" / "context_geometry_receipt.json"
    command([
        py, str(EXP / "scripts" / "audit_context_geometry.py"),
        "--out", str(geometry_path),
    ])
    geometry = json.loads(geometry_path.read_text())
    command([
        py, "-m", "unittest", "discover", "-s", str(EXP / "tests"),
        "-p", "test_*.py",
    ])
    invariants = _cpu_task_invariants(cfg)
    return {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": EXP.name,
        "model_output_generated": False,
        "firewall_clean": True,
        "smoke_input_files": _smoke_input_files(),
        "tokenizer_equivalence": tokenizer_equivalence,
        "parent_checkpoints": registered_parent_checkpoints(cfg),
        "bank_receipt_sha256": sha256_file(bank_path),
        "bank_sha256": bank_receipt["bank_sha256"],
        "rows_per_arm": bank_receipt["rows_per_arm"],
        "transition_counts": bank_receipt["transition_counts"],
        "weighted_action_mass_by_transition": bank_receipt[
            "weighted_action_mass_by_transition"
        ],
        "shuffle_proof": bank_receipt["within_dyad_shuffle_proof"],
        "encoding_receipt_sha256": {
            arm: sha256_file(
                artifacts / "preflight" / "encode" / arm / "encoding_receipt.json"
            )
            for arm in tcfg["arms"]
        },
        "serial_forward_tokens_per_epoch": dict(zip(tcfg["arms"], serial_tokens)),
        "serial_token_compute_ratio": serial_ratio,
        "context_geometry": geometry,
        **invariants,
    }


def _python_paths() -> tuple[str, str]:
    py = ROOT / ".venv" / "bin" / "python"
    vpy = ROOT / ".venv-vllm" / "bin" / "python"
    if not py.is_file() or not vpy.is_file():
        raise SystemExit("registered Python environments are missing")
    return str(py), str(vpy)


def evaluate_repo(
    cfg: dict,
    *,
    arm: str,
    model: Path,
    weight_sha256: str,
    block: str,
    contract: str,
    scenario: str,
    mode: str,
    answer_max_tokens: int,
    scaffold: bool = False,
) -> Path:
    _, vpy = _python_paths()
    suffix = "_scaffold" if scaffold else ""
    output = resolve(cfg["artifacts"]["root"]) / "eval" / block / (
        f"{arm}_{contract}_{scenario}_{mode}{suffix}_a{answer_max_tokens}.json"
    )
    argv = [
        vpy, str(EXP / "scripts" / "eval_repo_agent.py"),
        "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
        "--arm", arm, "--model", str(model),
        "--expected-weight-sha256", weight_sha256,
        "--block", block, "--contract", contract,
        "--scenario-set", scenario, "--mode", mode,
        "--answer-max-tokens", str(answer_max_tokens), "--output", str(output),
    ]
    if scaffold:
        argv.append("--scaffold")
    run_if_missing(output, argv)
    return output


def run_interface(cfg: dict) -> dict:
    verify_design_lock()
    py, _ = _python_paths()
    analysis = EXP / "analysis"
    selection = analysis / "interface_answer_band.json"
    start = resolve(cfg["model"]["start_checkpoint"])
    rung_receipts = []
    for answer_tokens in map(int, cfg["evaluation"]["interface_answer_rungs"]):
        runs = {
            "unassisted": evaluate_repo(
                cfg, arm="start", model=start,
                weight_sha256=cfg["model"]["start_weight_sha256"],
                block="interface_preflight", contract="inferred",
                scenario="acquisition", mode="deep", answer_max_tokens=answer_tokens,
            ),
            "injected": evaluate_repo(
                cfg, arm="start", model=start,
                weight_sha256=cfg["model"]["start_weight_sha256"],
                block="interface_preflight", contract="inferred",
                scenario="injected", mode="deep", answer_max_tokens=answer_tokens,
            ),
            "control_search": evaluate_repo(
                cfg, arm="start", model=start,
                weight_sha256=cfg["model"]["start_weight_sha256"],
                block="interface_preflight", contract="inferred",
                scenario="random", mode="deep", answer_max_tokens=answer_tokens,
            ),
            "explicit": evaluate_repo(
                cfg, arm="start", model=start,
                weight_sha256=cfg["model"]["start_weight_sha256"],
                block="interface_preflight", contract="explicit",
                scenario="acquisition", mode="deep", answer_max_tokens=answer_tokens,
            ),
        }
        rung = analysis / f"interface_rung_{answer_tokens}.json"
        command([
            py, str(EXP / "scripts" / "analyze_interface_preflight.py"),
            "--unassisted", str(runs["unassisted"]),
            "--injected", str(runs["injected"]),
            "--control-search", str(runs["control_search"]),
            "--explicit", str(runs["explicit"]),
            "--answer-max-tokens", str(answer_tokens), "--out", str(rung),
        ], allowed=(0, 4))
        rung_receipts.append(rung)
        code = command([
            py, str(EXP / "scripts" / "select_interface_band.py"),
            *sum((["--rung-receipt", str(path)] for path in rung_receipts), []),
            "--out", str(selection),
        ], allowed=(0, 3, 4))
        payload = json.loads(selection.read_text())
        if payload.get("gate", {}).get("passed"):
            return payload
        if code == 4:
            return payload
    return json.loads(selection.read_text())


def validate_locality_receipt(
    path: Path,
    cfg: dict,
    *,
    before_model: Path,
    before_weight_sha256: str,
    after_model: Path,
    after_weight_sha256: str,
) -> dict:
    payload = json.loads(path.read_text())
    try:
        before_tokenizer = harness.tokenizer_provenance(before_model)
        after_tokenizer = harness.tokenizer_provenance(after_model)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"locality tokenizer provenance is invalid: {exc}") from exc
    expected = {
        "schema_version": 1,
        "auditor_sha256": sha256_file(EXP / "scripts" / "audit_locality.py"),
        "before_model": str(before_model.resolve()),
        "after_model": str(after_model.resolve()),
        "before_model_weight_sha256": before_weight_sha256,
        "after_model_weight_sha256": after_weight_sha256,
        "before_model_config_sha256": sha256_file(before_model / "config.json"),
        "after_model_config_sha256": sha256_file(after_model / "config.json"),
        "before_model_generation_config_sha256": sha256_file(
            before_model / "generation_config.json"
        ),
        "after_model_generation_config_sha256": sha256_file(
            after_model / "generation_config.json"
        ),
        "before_merge_receipt_sha256": sha256_file(
            before_model / "merge_receipt.json"
        ),
        "after_merge_receipt_sha256": sha256_file(
            after_model / "merge_receipt.json"
        ),
        "before_tokenizer_files": before_tokenizer["tokenizer_files"],
        "before_tokenizer_manifest_sha256": before_tokenizer[
            "tokenizer_manifest_sha256"
        ],
        "before_tokenizer_compatibility_sha256": before_tokenizer[
            "tokenizer_compatibility_sha256"
        ],
        "after_tokenizer_files": after_tokenizer["tokenizer_files"],
        "after_tokenizer_manifest_sha256": after_tokenizer[
            "tokenizer_manifest_sha256"
        ],
        "after_tokenizer_compatibility_sha256": after_tokenizer[
            "tokenizer_compatibility_sha256"
        ],
        "contexts": str(resolve(cfg["locality"]["contexts"]).resolve()),
        "contexts_sha256": sha256_file(resolve(cfg["locality"]["contexts"])),
        "n_contexts": int(cfg["locality"]["count"]),
        "ceiling": float(cfg["locality"]["median_non_target_logit_drift_max"]),
        "entropy_delta_min": float(cfg["locality"]["mean_entropy_delta_min"]),
        "max_context_tokens": int(cfg["locality"]["max_context_tokens"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(f"locality receipt provenance drift at {key}: {path}")
    if (
        sha256_file(before_model / "model.safetensors") != before_weight_sha256
        or sha256_file(after_model / "model.safetensors") != after_weight_sha256
    ):
        raise SystemExit(f"locality checkpoint bytes changed: {path}")
    if (
        before_tokenizer["tokenizer_manifest_sha256"]
        != expected_tokenizer_manifest(cfg, before_model)
        or after_tokenizer["tokenizer_manifest_sha256"]
        != expected_tokenizer_manifest(cfg, after_model)
        or before_tokenizer["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
        or after_tokenizer["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
        or payload.get("rendered_prompts_equal") is not True
        or payload.get("tokenized_context_ids_equal") is not True
        or payload.get("before_rendered_prompts_sha256")
        != payload.get("after_rendered_prompts_sha256")
        or payload.get("before_tokenized_contexts_sha256")
        != payload.get("after_tokenized_contexts_sha256")
    ):
        raise SystemExit(f"locality tokenizer equivalence is not proven: {path}")
    return payload


def run_lineage_locality(cfg: dict) -> dict:
    verify_design_lock()
    py, _ = _python_paths()
    output = EXP / "analysis" / "locality_start_vs_anchor.json"
    _, payload = run_if_missing(output, [
        py, str(EXP / "scripts" / "audit_locality.py"),
        "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
        "--before-model", str(resolve(cfg["model"]["locality_anchor"])),
        "--after-model", str(resolve(cfg["model"]["start_checkpoint"])),
        "--contexts", str(resolve(cfg["locality"]["contexts"])),
        "--out", str(output),
        "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min", str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
    ], allowed=(0, 4))
    return validate_locality_receipt(
        output, cfg,
        before_model=resolve(cfg["model"]["locality_anchor"]),
        before_weight_sha256=cfg["model"]["anchor_weight_sha256"],
        after_model=resolve(cfg["model"]["start_checkpoint"]),
        after_weight_sha256=cfg["model"]["start_weight_sha256"],
    )


def run_qualification(
    cfg: dict, interface: dict, lineage_locality: dict
) -> dict:
    verify_design_lock()
    if not interface.get("gate", {}).get("passed"):
        return {"gate": {"passed": False, "verdict": "INTERFACE_FAIL"}}
    py, _ = _python_paths()
    answer_tokens = int(interface["selected_answer_max_tokens"])
    start = resolve(cfg["model"]["start_checkpoint"])
    receipts = {}
    for short, block in (("a", "qualification_a"), ("b", "qualification_b")):
        for label, contract, scenario in (
            ("unassisted", "inferred", "acquisition"),
            ("injected", "inferred", "injected"),
            ("control_search", "inferred", "random"),
            ("explicit", "explicit", "acquisition"),
        ):
            receipts[(short, label)] = evaluate_repo(
                cfg, arm="start", model=start,
                weight_sha256=cfg["model"]["start_weight_sha256"],
                block=block, contract=contract, scenario=scenario,
                mode="deep", answer_max_tokens=answer_tokens,
            )
    gate_path = EXP / "analysis" / "qualification_gate.json"
    argv = [
        py, str(EXP / "scripts" / "analyze_qualification.py"),
        "--interface-receipt", str(EXP / "analysis" / "interface_answer_band.json"),
    ]
    for short in ("a", "b"):
        argv.extend([
            f"--unassisted-{short}", str(receipts[(short, "unassisted")]),
            f"--injected-{short}", str(receipts[(short, "injected")]),
            f"--control-search-{short}", str(receipts[(short, "control_search")]),
            f"--explicit-{short}", str(receipts[(short, "explicit")]),
        ])
    argv.extend(["--out", str(gate_path)])
    qualification_passed, qualification = run_gate_fresh(gate_path, argv)
    if not qualification_passed:
        return qualification
    locality_path = EXP / "analysis" / "locality_start_vs_anchor.json"
    locality_passed = bool(lineage_locality.get("gate", {}).get("passed"))
    authorization = {
        "schema_version": 1,
        "stage": "training_authorization",
        "experiment_id": EXP.name,
        "issuer_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(EXP / "configs" / "default.yaml"),
        "design_lock_sha256": sha256_file(
            EXP / "runs" / "preregistration_receipt.json"
        ),
        "ancestor_receipts": {
            "qualification_gate": {
                "path": str(gate_path.resolve()),
                "sha256": sha256_file(gate_path),
            },
            "lineage_locality_gate": {
                "path": str(locality_path.resolve()),
                "sha256": sha256_file(locality_path),
            },
        },
        "selected_answer_max_tokens": answer_tokens,
        "checks": {
            "acquisition_qualified": qualification_passed,
            "lineage_locality_feasible": locality_passed,
        },
        "gate": {
            "passed": qualification_passed and locality_passed,
            "verdict": (
                "TRAINING_AUTHORIZED" if locality_passed
                else "LINEAGE_LOCALITY_INFEASIBLE"
            ),
        },
        "training_authorized": qualification_passed and locality_passed,
        "menagerie_authorized": False,
    }
    output = EXP / "analysis" / "training_authorization.json"
    output.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    return authorization


def _merged_weight_sha256(path: Path, cfg: dict, arm: str) -> str:
    try:
        checkpoint = harness.validate_registered_checkpoint(
            EXP,
            path,
            cfg,
            EXP / "runs" / "preregistration_receipt.json",
            arm,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"merged checkpoint receipt mismatch: {path}: {exc}") from exc
    return str(checkpoint["model_weight_sha256"])


def train_arms(cfg: dict) -> dict[str, tuple[Path, str]]:
    lock = verify_design_lock()
    py, _ = _python_paths()
    artifacts = resolve(cfg["artifacts"]["root"])
    bank_path, bank_receipt = _registered_bank_receipt(cfg)
    training_authorization = EXP / "analysis" / "training_authorization.json"
    if (
        not training_authorization.is_file()
        or not json.loads(training_authorization.read_text()).get("training_authorized")
    ):
        raise SystemExit("passed training authorization is missing")
    start = resolve(cfg["model"]["start_checkpoint"])
    tcfg = cfg["training"]
    models = {
        "start": (start, cfg["model"]["start_weight_sha256"]),
        "incumbent": (
            resolve(cfg["model"]["menagerie_incumbent"]),
            cfg["model"]["anchor_weight_sha256"],
        ),
    }
    serial_tokens = {}
    optimizer_steps = {}
    for arm in tcfg["arms"]:
        adapter = artifacts / "adapters" / arm
        train_receipt = adapter / "training_receipt.json"
        run_if_missing(train_receipt, [
            py, str(EXP / "scripts" / "train.py"), "--arm", arm,
            "--base-model", str(start),
            "--expected-base-weight-sha256", cfg["model"]["start_weight_sha256"],
            "--train", str(bank_path.parent / f"{arm}.jsonl"),
            "--expected-train-sha256", bank_receipt["bank_sha256"][arm],
            "--bank-receipt", str(bank_path),
            "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
            "--training-authorization", str(training_authorization),
            "--out", str(adapter), "--epochs", str(tcfg["epochs"]),
            "--lr", str(tcfg["learning_rate"]), "--rank", str(tcfg["rank"]),
            "--alpha", str(tcfg["alpha"]), "--batch-size", str(tcfg["batch_size"]),
            "--grad-accum", str(tcfg["gradient_accumulation_steps"]),
            "--loss-chunk-positions", str(tcfg["loss_chunk_positions"]),
            "--max-length", str(tcfg["max_length"]), "--seed", str(tcfg["seed"]),
        ])
        receipt = json.loads(train_receipt.read_text())
        if (
            receipt.get("training_file", {}).get("sha256")
            != bank_receipt["bank_sha256"][arm]
            or receipt.get("design_lock_sha256")
            != sha256_file(EXP / "runs" / "preregistration_receipt.json")
            or receipt.get("design_commit") != lock["design_commit"]
            or receipt.get("training_authorization_sha256")
            != sha256_file(training_authorization)
        ):
            raise SystemExit(f"stale training receipt for {arm}")
        preflight_path = (
            artifacts / "preflight" / "encode" / arm / "encoding_receipt.json"
        )
        preflight = json.loads(preflight_path.read_text())
        if (
            receipt.get("ordered_schedule_sha256")
            != preflight.get("ordered_schedule_sha256")
            or receipt.get("preflight_encoding_receipt")
            != {
                "path": str(preflight_path.resolve()),
                "sha256": sha256_file(preflight_path),
            }
        ):
            raise SystemExit(f"incomplete or preflight-divergent training receipt for {arm}")
        if (
            receipt.get("physical_forward_batch_size") != 1
            or receipt.get("logical_microbatch_size") != int(tcfg["batch_size"])
        ):
            raise SystemExit(f"unsafe padded physical training receipt for {arm}")
        serial_tokens[arm] = receipt["serial_forward_tokens_per_epoch"]
        optimizer_steps[arm] = {
            "planned": int(receipt["max_steps"]),
            "actual": int(receipt["optimizer_steps"]),
        }
        merged = artifacts / "merged" / arm
        run_if_missing(merged / "merge_receipt.json", [
            py, str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(start),
            "--expected-base-weight-sha256", cfg["model"]["start_weight_sha256"],
            "--adapter", str(adapter), "--arm", arm,
            "--expected-training-receipt-sha256", sha256_file(train_receipt),
            "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
            "--training-authorization", str(training_authorization),
            "--out", str(merged),
        ])
        merge_receipt = json.loads((merged / "merge_receipt.json").read_text())
        if (
            merge_receipt.get("arm") != arm
            or merge_receipt.get("training_receipt_sha256") != sha256_file(train_receipt)
            or merge_receipt.get("design_lock_sha256")
            != sha256_file(EXP / "runs" / "preregistration_receipt.json")
            or merge_receipt.get("base_weight_sha256")
            != cfg["model"]["start_weight_sha256"]
            or merge_receipt.get("training_authorization_sha256")
            != sha256_file(training_authorization)
        ):
            raise SystemExit(f"stale or unregistered merge receipt for {arm}")
        models[arm] = (merged, _merged_weight_sha256(merged, cfg, arm))
    ratio = max(serial_tokens.values()) / min(serial_tokens.values())
    compute_gate = EXP / "analysis" / "training_compute_gate.json"
    compute_passed = (
        ratio <= float(tcfg["serial_token_compute_ratio_max"])
        and all(
            row["planned"] == row["actual"] for row in optimizer_steps.values()
        )
    )
    compute_gate.write_text(json.dumps({
        "schema_version": 1,
        "stage": "training_compute",
        "issuer_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(EXP / "configs" / "default.yaml"),
        "design_lock_sha256": sha256_file(
            EXP / "runs" / "preregistration_receipt.json"
        ),
        "serial_forward_tokens_per_epoch": serial_tokens,
        "optimizer_steps": optimizer_steps,
        "merged_model_weight_sha256": {
            arm: models[arm][1] for arm in tcfg["arms"]
        },
        "candidate_model_weight_sha256": models["evidence_binding"][1],
        "max_to_min_ratio": ratio,
        "registered_ratio_max": float(tcfg["serial_token_compute_ratio_max"]),
        "training_receipts": {
            arm: {
                "path": str((artifacts / "adapters" / arm / "training_receipt.json").resolve()),
                "sha256": sha256_file(
                    artifacts / "adapters" / arm / "training_receipt.json"
                ),
            }
            for arm in tcfg["arms"]
        },
        "gate": {
            "passed": compute_passed,
            "verdict": (
                "TRAINING_COMPUTE_MATCHED"
                if compute_passed else "TRAINING_COMPUTE_MISMATCH"
            ),
        },
        "menagerie_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    if not compute_passed:
        raise TrainingComputeMismatch(compute_gate)
    return models


def evaluate_legacy_retention(
    cfg: dict,
    *,
    arm: str,
    model: Path,
    weight_sha256: str,
    block: str,
    scenario: str,
    answer_max_tokens: int,
) -> Path:
    _, vpy = _python_paths()
    output = resolve(cfg["artifacts"]["root"]) / "eval" / block / (
        f"{arm}_{scenario}_deep_a{answer_max_tokens}.json"
    )
    run_if_missing(output, [
        vpy, str(EXP / "scripts" / "eval_retention.py"),
        "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
        "--arm", arm, "--model", str(model),
        "--expected-weight-sha256", weight_sha256,
        "--block", block, "--scenario-set", scenario, "--mode", "deep",
        "--answer-max-tokens", str(answer_max_tokens), "--output", str(output),
    ])
    return output


def _candidate_locality(
    cfg: dict, models: dict[str, tuple[Path, str]]
) -> tuple[bool, Path, dict[str, Path]]:
    py, _ = _python_paths()
    candidate = models["evidence_binding"][0]
    anchor = models["incumbent"][0]
    start = models["start"][0]
    direct = EXP / "analysis" / "locality_candidate_vs_anchor.json"
    passed, _ = gate_if_missing(direct, [
        py, str(EXP / "scripts" / "audit_locality.py"),
        "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
        "--before-model", str(anchor), "--after-model", str(candidate),
        "--contexts", str(resolve(cfg["locality"]["contexts"])),
        "--out", str(direct),
        "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min", str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
    ])
    direct_payload = validate_locality_receipt(
        direct, cfg,
        before_model=anchor,
        before_weight_sha256=models["incumbent"][1],
        after_model=candidate,
        after_weight_sha256=models["evidence_binding"][1],
    )
    passed = bool(direct_payload.get("gate", {}).get("passed"))
    incremental = EXP / "analysis" / "locality_candidate_vs_start_diagnostic.json"
    run_if_missing(incremental, [
        py, str(EXP / "scripts" / "audit_locality.py"),
        "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
        "--before-model", str(start), "--after-model", str(candidate),
        "--contexts", str(resolve(cfg["locality"]["contexts"])),
        "--out", str(incremental),
        "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min", str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
    ], allowed=(0, 4))
    validate_locality_receipt(
        incremental, cfg,
        before_model=start,
        before_weight_sha256=models["start"][1],
        after_model=candidate,
        after_weight_sha256=models["evidence_binding"][1],
    )
    diagnostics = {"candidate_vs_start": incremental}
    for arm in ("explicit_redundant", "shuffled_binding"):
        output = EXP / "analysis" / f"locality_{arm}_vs_anchor_diagnostic.json"
        run_if_missing(output, [
            py, str(EXP / "scripts" / "audit_locality.py"),
            "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
            "--before-model", str(anchor),
            "--after-model", str(models[arm][0]),
            "--contexts", str(resolve(cfg["locality"]["contexts"])),
            "--out", str(output),
            "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
            "--entropy-delta-min", str(cfg["locality"]["mean_entropy_delta_min"]),
            "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
        ], allowed=(0, 4))
        validate_locality_receipt(
            output, cfg,
            before_model=anchor,
            before_weight_sha256=models["incumbent"][1],
            after_model=models[arm][0],
            after_weight_sha256=models[arm][1],
        )
        diagnostics[f"{arm}_vs_anchor"] = output
    return passed, direct, diagnostics


def _calibration_gate(
    cfg: dict,
    models: dict[str, tuple[Path, str]],
    answer_max_tokens: int,
    locality_path: Path,
) -> tuple[bool, Path]:
    py, _ = _python_paths()
    raw = {}
    for arm, model_key in (
        ("start", "start"),
        ("explicit_redundant", "explicit_redundant"),
        ("shuffled_binding", "shuffled_binding"),
    ):
        model, weight = models[model_key]
        raw[arm] = evaluate_repo(
            cfg, arm=arm, model=model, weight_sha256=weight,
            block="trained_calibration", contract="inferred",
            scenario="acquisition", mode="deep",
            answer_max_tokens=answer_max_tokens,
        )
    feasibility = EXP / "analysis" / "calibration_feasibility.json"
    feasible, _ = run_gate_fresh(feasibility, [
        py, str(EXP / "scripts" / "analyze_calibration.py"),
        "--start", str(raw["start"]),
        "--explicit-control", str(raw["explicit_redundant"]),
        "--shuffled-control", str(raw["shuffled_binding"]),
        "--expected-explicit-control-weight-sha256",
        models["explicit_redundant"][1],
        "--expected-shuffled-control-weight-sha256",
        models["shuffled_binding"][1],
        "--out", str(feasibility),
    ])
    if not feasible:
        return False, feasibility
    candidate_model, candidate_hash = models["evidence_binding"]
    candidate = evaluate_repo(
        cfg, arm="candidate", model=candidate_model,
        weight_sha256=candidate_hash, block="trained_calibration",
        contract="inferred", scenario="acquisition", mode="deep",
        answer_max_tokens=answer_max_tokens,
    )
    candidate_explicit = evaluate_repo(
        cfg, arm="candidate", model=candidate_model,
        weight_sha256=candidate_hash, block="explicit_retention",
        contract="explicit", scenario="acquisition", mode="deep",
        answer_max_tokens=answer_max_tokens,
    )
    gate = EXP / "analysis" / "calibration_gate.json"
    passed, _ = run_gate_fresh(gate, [
        py, str(EXP / "scripts" / "analyze_calibration.py"),
        "--start", str(raw["start"]),
        "--explicit-control", str(raw["explicit_redundant"]),
        "--shuffled-control", str(raw["shuffled_binding"]),
        "--expected-explicit-control-weight-sha256",
        models["explicit_redundant"][1],
        "--expected-shuffled-control-weight-sha256",
        models["shuffled_binding"][1],
        "--candidate", str(candidate),
        "--expected-candidate-weight-sha256", candidate_hash,
        "--candidate-explicit", str(candidate_explicit),
        "--locality", str(locality_path), "--out", str(gate),
    ])
    return passed, gate


def _transfer_gate(
    cfg: dict,
    models: dict[str, tuple[Path, str]],
    *,
    block: str,
    answer_max_tokens: int,
) -> tuple[bool, Path]:
    py, _ = _python_paths()
    candidate_model, candidate_hash = models["evidence_binding"]

    def eval_model(
        arm: str,
        model_key: str,
        scenario: str,
        *,
        contract: str = "inferred",
        mode: str = "deep",
        scaffold: bool = False,
        target_block: str = block,
    ) -> Path:
        model, weight = models[model_key]
        return evaluate_repo(
            cfg, arm=arm, model=model, weight_sha256=weight,
            block=target_block, contract=contract, scenario=scenario,
            mode=mode, answer_max_tokens=answer_max_tokens, scaffold=scaffold,
        )

    # Freeze all candidate-independent baselines, controls, and sample pools before
    # opening the primary candidate comparison. This lets an impossible margin stop
    # without spending or selectively exposing candidate outcomes.
    receipts = {
        "start": eval_model("start", "start", "acquisition"),
        "incumbent": eval_model("incumbent", "incumbent", "acquisition"),
        "explicit_control": eval_model(
            "explicit_redundant", "explicit_redundant", "acquisition"
        ),
        "shuffled_control": eval_model(
            "shuffled_binding", "shuffled_binding", "acquisition"
        ),
        "start_normal": eval_model("start", "start", "normal"),
        "pool_start": eval_model(
            "start", "start", "acquisition", mode="sample_pool"
        ),
        "pool_incumbent": eval_model(
            "incumbent", "incumbent", "acquisition", mode="sample_pool"
        ),
    }
    baseline_feasibility = EXP / "analysis" / f"{block}_baseline_feasibility.json"
    baseline_passed, _ = run_gate_fresh(baseline_feasibility, [
        py, str(EXP / "scripts" / "analyze_transfer_feasibility.py"),
        "--block", block, "--phase", "baseline",
        "--start", str(receipts["start"]),
        "--incumbent", str(receipts["incumbent"]),
        "--explicit-control", str(receipts["explicit_control"]),
        "--shuffled-control", str(receipts["shuffled_control"]),
        "--start-normal", str(receipts["start_normal"]),
        "--expected-explicit-control-weight-sha256",
        models["explicit_redundant"][1],
        "--expected-shuffled-control-weight-sha256",
        models["shuffled_binding"][1],
        "--out", str(baseline_feasibility),
    ])
    if not baseline_passed:
        return False, baseline_feasibility

    receipts["candidate"] = eval_model(
        "candidate", "evidence_binding", "acquisition"
    )
    matches = {}
    for pool_arm in ("start", "incumbent"):
        output = EXP / "analysis" / f"{block}_sample_match_{pool_arm}.json"
        command([
            py, str(EXP / "scripts" / "analyze_sample_pool.py"),
            "--target-deep", str(receipts["candidate"]),
            "--sample-pool", str(receipts[f"pool_{pool_arm}"]),
            "--expected-target-arm", "candidate",
            "--expected-target-weight-sha256", candidate_hash,
            "--expected-pool-arm", pool_arm,
            "--expected-pool-weight-sha256", models[pool_arm][1],
            "--out", str(output),
        ], allowed=(0, 4))
        matches[pool_arm] = output

    receipts["control_search"] = eval_model(
        "candidate", "evidence_binding", "random"
    )
    feasibility = EXP / "analysis" / f"{block}_feasibility.json"
    comparator_passed, _ = run_gate_fresh(feasibility, [
        py, str(EXP / "scripts" / "analyze_transfer_feasibility.py"),
        "--block", block, "--phase", "comparators",
        "--candidate", str(receipts["candidate"]),
        "--control-search", str(receipts["control_search"]),
        "--sample-match-start", str(matches["start"]),
        "--sample-match-incumbent", str(matches["incumbent"]),
        "--baseline-feasibility", str(baseline_feasibility),
        "--expected-candidate-weight-sha256", candidate_hash,
        "--expected-explicit-control-weight-sha256",
        models["explicit_redundant"][1],
        "--expected-shuffled-control-weight-sha256",
        models["shuffled_binding"][1],
        "--out", str(feasibility),
    ])
    if not comparator_passed:
        return False, feasibility

    receipts.update({
        "candidate_injected": eval_model(
            "candidate", "evidence_binding", "injected"
        ),
        "candidate_normal": eval_model("candidate", "evidence_binding", "normal"),
        "candidate_recovery": eval_model(
            "candidate", "evidence_binding", "recovery"
        ),
        "start_recovery": eval_model("start", "start", "recovery"),
        "candidate_recovery_scaffold": eval_model(
            "candidate", "evidence_binding", "recovery", scaffold=True
        ),
        "candidate_explicit": eval_model(
            "candidate", "evidence_binding", "acquisition",
            contract="explicit", target_block="explicit_retention",
        ),
    })
    gate = EXP / "analysis" / f"{block}_gate.json"
    passed, _ = run_gate_fresh(gate, [
        py, str(EXP / "scripts" / "analyze_transfer.py"),
        "--block", block,
        "--candidate", str(receipts["candidate"]),
        "--start", str(receipts["start"]),
        "--incumbent", str(receipts["incumbent"]),
        "--explicit-control", str(receipts["explicit_control"]),
        "--shuffled-control", str(receipts["shuffled_control"]),
        "--expected-candidate-weight-sha256", candidate_hash,
        "--expected-explicit-control-weight-sha256",
        models["explicit_redundant"][1],
        "--expected-shuffled-control-weight-sha256",
        models["shuffled_binding"][1],
        "--control-search", str(receipts["control_search"]),
        "--candidate-injected", str(receipts["candidate_injected"]),
        "--candidate-normal", str(receipts["candidate_normal"]),
        "--start-normal", str(receipts["start_normal"]),
        "--candidate-recovery", str(receipts["candidate_recovery"]),
        "--start-recovery", str(receipts["start_recovery"]),
        "--candidate-recovery-scaffold", str(
            receipts["candidate_recovery_scaffold"]
        ),
        "--candidate-explicit", str(receipts["candidate_explicit"]),
        "--sample-match-start", str(matches["start"]),
        "--sample-match-incumbent", str(matches["incumbent"]),
        "--feasibility", str(feasibility),
        "--out", str(gate),
    ])
    return passed, gate


def _retention_gate(
    cfg: dict,
    models: dict[str, tuple[Path, str]],
    answer_max_tokens: int,
) -> tuple[bool, Path]:
    py, _ = _python_paths()
    receipts = {}
    for substrate, block in (
        ("broad", "old_broad_retention"),
        ("transaction", "old_transaction_retention"),
    ):
        for arm, model_key in (("candidate", "evidence_binding"), ("start", "start")):
            model, weight = models[model_key]
            for scenario in ("normal", "recovery"):
                receipts[(substrate, arm, scenario)] = evaluate_legacy_retention(
                    cfg, arm=arm, model=model, weight_sha256=weight,
                    block=block, scenario=scenario,
                    answer_max_tokens=answer_max_tokens,
                )
    gate = EXP / "analysis" / "retention_gate.json"
    argv = [py, str(EXP / "scripts" / "analyze_retention.py")]
    for substrate in ("broad", "transaction"):
        for arm in ("candidate", "start"):
            for scenario in ("normal", "recovery"):
                argv.extend([
                    f"--{substrate}-{arm}-{scenario}",
                    str(receipts[(substrate, arm, scenario)]),
                ])
    argv.extend([
        "--expected-candidate-weight-sha256", models["evidence_binding"][1]
    ])
    argv.extend(["--out", str(gate)])
    passed, _ = run_gate_fresh(gate, argv)
    return passed, gate


def _run_menagerie(
    cfg: dict,
    models: dict[str, tuple[Path, str]],
    authorization: Path,
) -> int:
    py, _ = _python_paths()
    candidate = models["evidence_binding"][0]
    incumbent = models["incumbent"][0]
    log = EXP / "runs" / "menagerie_log.jsonl"
    reservation_dir = EXP / "runs" / "menagerie_reservations"

    def exposure_evidence() -> list[Path]:
        paths = [log] if log.is_file() and log.stat().st_size else []
        if reservation_dir.is_dir():
            paths.extend(sorted(reservation_dir.glob("*.json")))
        menagerie_dir = EXP / "runs" / "menagerie"
        if menagerie_dir.is_dir():
            paths.extend(sorted(menagerie_dir.glob("*.json")))
        return paths

    try:
        existing = [
            json.loads(line)
            for line in log.read_text().splitlines()
            if line.strip()
        ] if log.is_file() else []
    except (OSError, json.JSONDecodeError) as exc:
        if exposure_evidence():
            return terminalize_burned_menagerie_seed(
                error=exc, authorization=authorization,
                evidence=exposure_evidence(),
            )
        raise
    expected_pairs = {
        (str(tier), int(seed))
        for tier, seed in cfg["menagerie"]["paired_seeds"].items()
    }
    observed_pairs = [
        (str(row.get("tier")), int(row.get("seed", -1))) for row in existing
    ]
    if (
        len(observed_pairs) != len(set(observed_pairs))
        or not set(observed_pairs).issubset(expected_pairs)
    ):
        return terminalize_burned_menagerie_seed(
            error=RuntimeError(
                "Menagerie resume log contains an unexpected or duplicate event"
            ),
            authorization=authorization,
            evidence=exposure_evidence(),
        )
    stranded = unlogged_menagerie_reservations(cfg, existing)
    if stranded:
        return terminalize_burned_menagerie_seed(
            error=RuntimeError(
                "Menagerie reservation has no authenticated append-only log event"
            ),
            authorization=authorization,
            evidence=exposure_evidence(),
        )
    design_lock = EXP / "runs" / "preregistration_receipt.json"
    try:
        for index, row in enumerate(existing, 1):
            validate_event_for_resume(
                row,
                cfg,
                authorization,
                design_lock,
                source=f"{log}:line {index}",
            )
    except SystemExit as exc:
        return terminalize_burned_menagerie_seed(
            error=exc, authorization=authorization,
            evidence=exposure_evidence(),
        )
    collisions = external_menagerie_seed_collisions(cfg, existing)
    if collisions:
        return terminalize_unavailable_menagerie_seed(
            collisions, authorization, exposure_evidence()
        )
    for tier, seed in cfg["menagerie"]["paired_seeds"].items():
        prior = next((
            row for row in existing
            if row.get("tier") == tier and int(row.get("seed", -1)) == int(seed)
        ), None)
        if prior is not None:
            print(f"[resume] Menagerie {tier} seed {seed}", flush=True)
            continue
        try:
            command([
                py, str(EXP / "scripts" / "bench.py"),
                "--tier", str(tier), "--seed", str(seed),
                "--incumbent", str(incumbent), "--candidate", str(candidate),
                "--authorization", str(authorization),
                "--design-lock", str(EXP / "runs" / "preregistration_receipt.json"),
            ])
        except subprocess.CalledProcessError as exc:
            reservation = (
                reservation_dir / f"{tier}_seed{int(seed)}.json"
            )
            if reservation.is_file():
                return terminalize_burned_menagerie_seed(
                    error=exc, authorization=authorization,
                    evidence=exposure_evidence(),
                )
            collisions = external_menagerie_seed_collisions(cfg, existing)
            if collisions:
                return terminalize_unavailable_menagerie_seed(
                    collisions, authorization, exposure_evidence()
                )
            raise
    try:
        refreshed = [
            json.loads(line)
            for line in log.read_text().splitlines()
            if line.strip()
        ] if log.is_file() else []
    except (OSError, json.JSONDecodeError) as exc:
        return terminalize_burned_menagerie_seed(
            error=exc, authorization=authorization,
            evidence=exposure_evidence(),
        )
    stranded = unlogged_menagerie_reservations(cfg, refreshed)
    if stranded:
        return terminalize_burned_menagerie_seed(
            error=RuntimeError(
                "Menagerie command completed without an authenticated log event"
            ),
            authorization=authorization,
            evidence=exposure_evidence(),
        )
    gate = EXP / "analysis" / "menagerie_gate.json"
    passed, _ = run_gate_fresh(gate, [
        py, str(EXP / "scripts" / "analyze_menagerie.py"),
        "--log", str(log), "--authorization", str(authorization),
        "--out", str(gate),
    ])
    write_terminal_disposition(
        stage="menagerie",
        verdict=final_menagerie_verdict(passed),
        receipts=[authorization, gate],
        menagerie_exposed=True,
    )
    return 0 if passed else 4


def run_full_whitebox(
    cfg: dict,
    models: dict[str, tuple[Path, str]],
    answer_max_tokens: int,
) -> int:
    verify_design_lock()
    compute_path = EXP / "analysis" / "training_compute_gate.json"
    if (
        not compute_path.is_file()
        or json.loads(compute_path.read_text()).get("gate")
        != {"passed": True, "verdict": "TRAINING_COMPUTE_MATCHED"}
    ):
        raise SystemExit("passed training-compute gate is missing")
    locality_passed, locality_path, locality_diagnostics = _candidate_locality(
        cfg, models
    )
    if not locality_passed:
        write_terminal_disposition(
            stage="candidate_locality",
            verdict="LOCALITY_FAIL",
            receipts=[compute_path, locality_path],
        )
        print("[run] candidate failed direct apex locality; behavior remains sealed", flush=True)
        return 4
    calibration_passed, calibration_path = _calibration_gate(
        cfg, models, answer_max_tokens, locality_path
    )
    if not calibration_passed:
        write_terminal_disposition(
            stage="trained_calibration",
            verdict=calibration_terminal_verdict(calibration_path),
            receipts=[compute_path, locality_path, calibration_path],
        )
        print("[run] trained calibration failed; transfer and Menagerie remain sealed", flush=True)
        return 4
    transfer_paths = []
    for block in ("transfer_dev", "transfer_confirm"):
        passed, path = _transfer_gate(
            cfg, models, block=block, answer_max_tokens=answer_max_tokens
        )
        transfer_paths.append(path)
        if not passed:
            write_terminal_disposition(
                stage=block,
                verdict=transfer_terminal_verdict(block, path),
                receipts=[
                    compute_path, locality_path, calibration_path, *transfer_paths
                ],
            )
            print(f"[run] {block} failed; later exposure remains sealed", flush=True)
            return 4
    retention_passed, retention_path = _retention_gate(
        cfg, models, answer_max_tokens
    )
    if not retention_passed:
        write_terminal_disposition(
            stage="legacy_retention",
            verdict="RETENTION_FAIL",
            receipts=[
                compute_path, locality_path, calibration_path,
                *transfer_paths, retention_path,
            ],
        )
        print("[run] conditional-loop retention failed; Menagerie remains sealed", flush=True)
        return 4
    py, _ = _python_paths()
    bank_path, _ = _registered_bank_receipt(cfg)
    uncertainty = EXP / "analysis" / "transition_uncertainty_diagnostic.json"
    try:
        command([
            py, str(EXP / "scripts" / "audit_transition_uncertainty.py"),
            "--design-lock", str(
                EXP / "runs" / "preregistration_receipt.json"
            ),
            "--before-model", str(models["start"][0]),
            "--after-model", str(models["evidence_binding"][0]),
            "--bank", str(bank_path.parent / "evidence_binding.jsonl"),
            "--rows-per-transition", str(
                cfg["uncertainty"]["rows_per_transition"]
            ),
            "--strata", str(cfg["uncertainty"]["strata"]),
            "--out", str(uncertainty),
        ])
    except (subprocess.CalledProcessError, SystemExit) as exc:
        uncertainty.write_text(json.dumps({
            "schema_version": 1,
            "status": "diagnostic_unavailable_non_gating",
            "error": str(exc),
            "used_for_selection_or_loss": False,
        }, indent=2, sort_keys=True) + "\n")
    gates = [
        compute_path, locality_path, calibration_path,
        *transfer_paths, retention_path,
    ]
    if not all(json.loads(path.read_text()).get("gate", {}).get("passed") for path in gates):
        raise SystemExit("white-box authorization assembly found a failed gate")
    authorization_payload = {
        "schema_version": 1,
        "stage": "whitebox_authorization",
        "all_whitebox_gates_passed": True,
        "menagerie_authorized": True,
        "gate": {"passed": True, "verdict": "WHITEBOX_PASS"},
        "candidate_model_weight_sha256": models["evidence_binding"][1],
        "incumbent_model_weight_sha256": models["incumbent"][1],
        "selected_answer_max_tokens": answer_max_tokens,
        "gate_receipts": {
            path.stem: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in gates
        },
    }
    authorization = EXP / "analysis" / "whitebox_authorization.json"
    authorization.write_text(
        json.dumps(authorization_payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        "[run] WHITEBOX_PASS: locality, calibration, transfer-dev, "
        "content-disjoint confirmation, and legacy retention all pass.",
        flush=True,
    )
    return _run_menagerie(cfg, models, authorization)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--lock-design", metavar="COMMIT")
    parser.add_argument("--interface", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--closeout", action="store_true")
    parser.add_argument("--verify-closeout", action="store_true")
    args = parser.parse_args()
    if sum((
        args.smoke, bool(args.lock_design), args.interface,
        args.qualify, args.full, args.closeout, args.verify_closeout,
    )) != 1:
        parser.error("choose exactly one mode")
    if args.smoke:
        receipt = cpu_smoke()
        output = EXP / "reports" / "smoke_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.lock_design:
        write_design_lock(args.lock_design)
        return 0
    if args.closeout:
        close_terminal_lifecycle()
        return 0
    if args.verify_closeout:
        verify_terminal_lifecycle()
        return 0
    assert_scientific_run_open()
    cfg = config()
    lineage_locality = run_lineage_locality(cfg)
    if not lineage_locality.get("gate", {}).get("passed"):
        write_terminal_disposition(
            stage="lineage_locality",
            verdict="LINEAGE_LOCALITY_INFEASIBLE",
            receipts=[EXP / "analysis" / "locality_start_vs_anchor.json"],
        )
        print(
            "[run] LINEAGE_LOCALITY_INFEASIBLE: interface, qualification, and "
            "training remain sealed",
            flush=True,
        )
        return 4
    interface = run_interface(cfg)
    if not interface.get("gate", {}).get("passed"):
        write_terminal_disposition(
            stage="interface_preflight",
            verdict=str(interface.get("gate", {}).get("verdict", "INSTRUMENT_FAIL")),
            receipts=[EXP / "analysis" / "interface_answer_band.json"],
        )
        return 4
    if args.interface:
        return 0
    qualification = run_qualification(cfg, interface, lineage_locality)
    if args.qualify:
        if qualification.get("gate", {}).get("passed"):
            return 0
    if not qualification.get("gate", {}).get("passed"):
        write_terminal_disposition(
            stage="acquisition_qualification",
            verdict=str(qualification.get("gate", {}).get("verdict", "INSTRUMENT_FAIL")),
            receipts=[EXP / "analysis" / "qualification_gate.json"],
        )
        print("[run] training and all later stages remain sealed", flush=True)
        return 4
    # Full white-box evaluation and Menagerie sealing are appended below after
    # the frozen downstream analyzers; reaching this line without them is illegal.
    try:
        models = train_arms(cfg)
    except TrainingComputeMismatch as exc:
        write_terminal_disposition(
            stage="training_compute",
            verdict="TRAINING_COMPUTE_MISMATCH",
            receipts=[
                EXP / "analysis" / "training_authorization.json",
                exc.receipt,
            ],
        )
        return 4
    return run_full_whitebox(cfg, models, int(interface["selected_answer_max_tokens"]))


if __name__ == "__main__":
    raise SystemExit(main())
