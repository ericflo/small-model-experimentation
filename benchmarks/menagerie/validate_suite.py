"""CPU-only instrument validation for Menagerie families and tiers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from harness import SUITE_VERSION
from harness import backends, engine


SCRIPT_DIR = Path(__file__).resolve().parent


def load_run_module():
    spec = importlib.util.spec_from_file_location("menagerie_run", SCRIPT_DIR / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg = (cursor + 1 + end) / 2.0
        for pos in range(cursor, end):
            ranks[indexed[pos][0]] = avg
        cursor = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    # If rank variance is zero, both-constant vectors are perfectly stable and
    # one-constant vectors fail with rho 0.0.
    rx = average_ranks(x)
    ry = average_ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((value - mx) ** 2 for value in rx)
    vy = sum((value - my) ** 2 for value in ry)
    if vx == 0.0 or vy == 0.0:
        return 1.0 if vx == 0.0 and vy == 0.0 else 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (vx**0.5 * vy**0.5)


def selftest_family(families_dir: Path, name: str) -> dict:
    cmd = [sys.executable, "-m", f"families.{name}.selftest"]
    proc = subprocess.run(
        cmd,
        cwd=str(families_dir.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = proc.stdout.splitlines()
    return {"returncode": proc.returncode, "tail": "\n".join(lines[-20:])}


def run_validation(families_dir: Path, tiers_dir: Path, seed: int) -> dict:
    run_mod = load_run_module()
    families = engine.discover_families(families_dir)
    tier_paths = sorted(tiers_dir.glob("*.json"))
    eps_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    family_results: dict = {}
    for name in families:
        st = selftest_family(families_dir, name)
        family_results[name] = {
            "selftest_pass": st["returncode"] == 0,
            "selftest_tail": st["tail"],
        }

    tier_backend_results: dict[str, dict[str, dict]] = {}
    ladder: dict[str, dict[str, float]] = {}
    for tier_path in tier_paths:
        tier_cfg = run_mod.load_tier(str(tier_path))
        tier_name = tier_cfg["tier"]
        tier_backend_results[tier_name] = {}

        oracle_result = run_mod.run_tier(
            families,
            tier_cfg,
            backends.make_backend("oracle", seed=seed),
            seed,
            include_transcripts=False,
            backend_spec="oracle",
        )
        random_result = run_mod.run_tier(
            families,
            tier_cfg,
            backends.make_backend("random", seed=seed),
            seed,
            include_transcripts=False,
            backend_spec="random",
        )
        tier_backend_results[tier_name]["oracle"] = oracle_result
        tier_backend_results[tier_name]["random"] = random_result

        for name in families:
            family_results[name].setdefault(tier_name, {})
            family_results[name][tier_name]["oracle"] = oracle_result["per_family"][name]["score"]
            family_results[name][tier_name]["random"] = random_result["per_family"][name]["score"]

        ladder[tier_name] = {}
        for eps in eps_values:
            spec = f"noisy:{eps}"
            noisy_result = run_mod.run_tier(
                families,
                tier_cfg,
                backends.make_backend(spec, seed=seed),
                seed,
                include_transcripts=False,
                backend_spec=spec,
            )
            ladder[tier_name][str(eps)] = noisy_result["aggregate"]

    checks = {
        "selftests_pass": all(data["selftest_pass"] for data in family_results.values()),
        "oracle_perfect": True,
        "random_floor": True,
        "ladder_rank_stability": True,
        "monotone_eps": True,
    }

    for tier_name, result in tier_backend_results.items():
        for name in families:
            if abs(result["oracle"]["per_family"][name]["score"] - 1.0) > 1e-9:
                checks["oracle_perfect"] = False
            if result["random"]["per_family"][name]["score"] > 0.15:
                checks["random_floor"] = False

    spearman_results: dict[str, float] = {}
    tier_names = sorted(ladder)
    for i, left in enumerate(tier_names):
        for right in tier_names[i + 1 :]:
            left_vec = [ladder[left][str(eps)] for eps in eps_values]
            right_vec = [ladder[right][str(eps)] for eps in eps_values]
            rho = spearman(left_vec, right_vec)
            spearman_results[f"{left}|{right}"] = rho
            if rho < 0.9:
                checks["ladder_rank_stability"] = False

    for tier_name, values in ladder.items():
        vector = [values[str(eps)] for eps in eps_values]
        for prev, cur in zip(vector, vector[1:]):
            if cur > prev + 0.05:
                checks["monotone_eps"] = False

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "suite_version": SUITE_VERSION,
        "families": family_results,
        "ladder": ladder,
        "spearman": spearman_results,
        "checks": checks,
        "pass": all(checks.values()),
    }


def print_validation(result: dict) -> None:
    eps_keys = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    print(f"{'tier':<8} " + " ".join(f"eps={eps:>4}" for eps in eps_keys))
    for tier_name, values in sorted(result["ladder"].items()):
        print(f"{tier_name:<8} " + " ".join(f"{values[eps]:>8.3f}" for eps in eps_keys))
    print("spearman:")
    for pair, rho in sorted(result["spearman"].items()):
        print(f"  {pair}: {rho:.3f}")
    check_line = " ".join(f"{name}={str(value).lower()}" for name, value in result["checks"].items())
    print(f"checks: {check_line}")
    print(f"pass={str(result['pass']).lower()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Menagerie family instruments with CPU policies.")
    parser.add_argument("--families-dir", default=str(SCRIPT_DIR / "families"))
    parser.add_argument("--tiers-dir", default=str(SCRIPT_DIR / "tiers"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(SCRIPT_DIR / "results" / "instrument_validation.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_validation(Path(args.families_dir), Path(args.tiers_dir), args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print_validation(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
