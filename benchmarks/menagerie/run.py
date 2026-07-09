"""Command-line runner for Menagerie benchmark tiers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from harness import SUITE_VERSION
from harness import adapter_spec, backends, engine


SCRIPT_DIR = Path(__file__).resolve().parent
TIER_NAMES = ("quick", "medium", "slow", "deep")
ALLOWED_MODEL_ID_PREFIX = "Qwen/Qwen3.5-4B"
SCORE_ONLY_PER_ITEM_KEYS = ("id", "family", "level", "mode", "score", "turns", "wall_s")

# measured on RTX 4090 (WSL2), vLLM 0.24.0 at gpu_memory_utilization 0.85,
# model-resident Qwen3.5-4B, two-phase thinking, seed 31337: quick = 80 gens
# think@1024 (79/80 budget-bound), ~85.2k gen tokens / 43.83 s = ~1944 tok/s;
# medium = 286 gens over 4 lockstep rounds, ~302.7k gen tokens / 147.98 s =
# ~2046 tok/s. Sustained aggregate ~1950 tok/s at >=60 concurrent sequences;
# 1500 remains a conservative floor for late-round concurrency decay as
# episodes terminate.
VLLM_SUSTAINED_TOKS_PER_S = 1950.0
VLLM_WORST_TOKS_PER_S = 1500.0
EXPECTED_THINK_FRACTION = 0.95  # Measured bind fraction is 0.97 at budget 1024; larger budgets close earlier, so this overestimates -> safe.
# Measured horizon use was 0.69 (medium, seed 31337: 166 of 240 turn-gens);
# 0.85 stays as the conservative blend for harder tiers.
EXPECTED_HORIZON_FRACTION = 0.85
# Context-bound check constants. Measured maxima were 392 prompt tokens for
# quick atoms and 637 for medium 4-turn episodes (~61/turn): a 3-4x margin.
EST_BASE_PROMPT_TOKENS = 800
EST_TOKENS_PER_TURN = 250
VLLM_MAX_MODEL_LEN = 16384  # Must match the harness/vllm_runner.py default.


def validate_model_id(model_id: str) -> None:
    """Enforce the repository's one-model rule for Menagerie runs."""

    if not model_id.startswith(ALLOWED_MODEL_ID_PREFIX):
        raise ValueError(
            "--model-id violates the repo one-model rule in AGENTS.md Non-Negotiables: "
            "only Qwen/Qwen3.5-4B or revision variants starting with Qwen/Qwen3.5-4B are allowed"
        )


def score_only_per_item(item: dict) -> dict:
    """Return the public, benchmark-content-free per-item record."""

    return {key: item[key] for key in SCORE_ONLY_PER_ITEM_KEYS if key in item}


def warn_debug_artifacts(out_path: Path) -> None:
    print("", file=sys.stderr)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
    print("WARNING: --debug-artifacts output contains held-out benchmark content.", file=sys.stderr)
    print("It includes transcripts and score details and must NEVER enter this repo", file=sys.stderr)
    print("or any training data.", file=sys.stderr)
    print(f"Writing debug artifact to: {out_path}", file=sys.stderr)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
    print("", file=sys.stderr)


def load_tier(name_or_path: str) -> dict:
    """Load a tier JSON by canonical name or path."""

    if "/" not in name_or_path and not name_or_path.endswith(".json"):
        path = SCRIPT_DIR / "tiers" / f"{name_or_path}.json"
    else:
        path = Path(name_or_path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_tier_requests(tier: str | None, estimate: bool) -> list[str]:
    """Resolve CLI tier selection into names or paths."""

    if estimate and (tier is None or tier == "all"):
        return list(TIER_NAMES)
    if tier is None:
        raise ValueError("--tier is required unless --estimate")
    if tier == "all":
        raise ValueError("--tier all is only valid with --estimate")
    return [tier]


def collect_items(families: dict, tier_cfg: dict, seed: int) -> dict[str, list[dict]]:
    """Generate tier items for each family and mode."""

    by_family: dict[str, list[dict]] = {}
    for name, module in families.items():
        items: list[dict] = []
        atom_cfg = tier_cfg["atoms"]
        for level in atom_cfg["levels"]:
            items.extend(module.generate(seed, level, atom_cfg["n_per_level"], "atom"))
        episode_cfg = tier_cfg.get("episodes")
        if episode_cfg is not None:
            for level in episode_cfg["levels"]:
                items.extend(module.generate(seed, level, episode_cfg["n_per_level"], "episode"))
        by_family[name] = items
    return by_family


def run_tier(
    families: dict,
    tier_cfg: dict,
    backend_obj,
    seed: int,
    include_transcripts: bool = False,
    backend_spec: str = "",
    think: bool = True,
    think_budget: int | dict[str, int] | None = None,
) -> dict:
    """Run a tier over already-discovered families with a constructed backend."""

    # Tier think budgets escalate 1024/1024/2048/4096, with a 1024 floor because
    # 256/512 are in the measured truncation-harm zone (~100% forced-close).
    # Above that floor only the tail of long-chain items binds. Deep episodes use
    # a separate 2048 per-turn cap purely to bound the 14-round worst-case wall
    # clock; --think-budget overrides both for compute-response studies.
    if think_budget is None:
        atom_tb = int(tier_cfg["think_budget"])
        episode_tb = int(tier_cfg.get("episode_think_budget", tier_cfg["think_budget"]))
    elif isinstance(think_budget, dict):
        atom_tb = int(think_budget["atom"])
        episode_tb = int(think_budget["episode"])
    else:
        atom_tb = int(think_budget)
        episode_tb = int(think_budget)
    think_budget = {"atom": atom_tb, "episode": episode_tb}

    all_episodes = []
    items_by_family = collect_items(families, tier_cfg, seed)
    episode_cfg = tier_cfg.get("episodes")
    episode_max_turns = episode_cfg.get("max_turns") if episode_cfg else None
    for name, items in items_by_family.items():
        all_episodes.extend(engine.build_episodes(name, families[name], items, episode_max_turns))

    t0 = perf_counter()
    engine_result = engine.run_lockstep(all_episodes, backend_obj.batch_act)
    wall_total_s = perf_counter() - t0

    per_family_scores: dict[str, list[float]] = defaultdict(list)
    for item in engine_result["per_item"]:
        per_family_scores[item["family"]].append(float(item["score"]))

    per_family = {}
    for name in families:
        scores = per_family_scores.get(name, [])
        mean_score = sum(scores) / len(scores) if scores else 0.0
        per_family[name] = {
            "score": mean_score,
            "n": len(scores),
            "wall_s": round(engine_result["per_family_wall_s"].get(name, 0.0), 3),
        }
    aggregate = sum(data["score"] for data in per_family.values()) / len(per_family) if per_family else 0.0

    per_item = engine_result["per_item"]
    if not include_transcripts:
        per_item = [score_only_per_item(item) for item in per_item]

    return {
        "suite_version": SUITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "tier": tier_cfg["tier"],
        "tier_config": tier_cfg,
        "backend": backend_spec,
        "think": think,
        "think_budget": think_budget if think else None,
        "per_family": per_family,
        "aggregate": aggregate,
        "per_item": per_item,
        "rounds": engine_result["rounds"],
        "wall_total_s": wall_total_s,
        "budget_s": tier_cfg["budget_s"],
        "within_budget": wall_total_s <= tier_cfg["budget_s"],
        "backend_stats": dict(getattr(backend_obj, "stats", {})),
    }


def default_out_path(tier_cfg: dict, backend_spec: str, seed: int, adapter_info: dict | None = None) -> Path:
    safe_backend = backend_spec.replace(":", "-")
    if adapter_info is not None:
        return SCRIPT_DIR / "results" / f"{tier_cfg['tier']}_{safe_backend}_seed{seed}_adapter-{adapter_info['weights_sha256'][:8]}.json"
    return SCRIPT_DIR / "results" / f"{tier_cfg['tier']}_{safe_backend}_seed{seed}.json"


def print_result_table(result: dict) -> None:
    """Print a compact per-family score table."""

    print(f"{'family':<18} {'n':>5} {'score':>8} {'wall_s':>8}")
    for name, data in result["per_family"].items():
        print(f"{name:<18} {data['n']:>5} {data['score']:>8.3f} {data['wall_s']:>8.3f}")
    print(f"{'AGGREGATE':<18} {'':>5} {result['aggregate']:>8.3f} {result['wall_total_s']:>8.3f}")
    status = "yes" if result["within_budget"] else "no"
    print(f"wall_total_s={result['wall_total_s']:.3f} budget_s={result['budget_s']} within_budget={status}")


def estimate_tier(
    tier_cfg: dict,
    family_count: int,
    think: bool,
    think_budget: int | None = None,
) -> tuple[float, float, int]:
    if think:
        if think_budget is None:
            atom_tb = int(tier_cfg["think_budget"])
            ep_tb = int(tier_cfg.get("episode_think_budget", tier_cfg["think_budget"]))
        else:
            atom_tb = int(think_budget)
            ep_tb = int(think_budget)
    else:
        atom_tb = 0
        ep_tb = 0

    atom_answer_tokens = int(tier_cfg["max_new_tokens"]["atom"])
    episode_answer_tokens = int(tier_cfg["max_new_tokens"]["episode"])
    atom_cfg = tier_cfg["atoms"]
    n_atom_gens = family_count * len(atom_cfg["levels"]) * atom_cfg["n_per_level"]

    episode_cfg = tier_cfg.get("episodes")
    if episode_cfg is None:
        n_episodes = 0
        max_turns = 0
    else:
        n_episodes = family_count * len(episode_cfg["levels"]) * episode_cfg["n_per_level"]
        max_turns = int(episode_cfg["max_turns"])
    worst_turn_gens = n_episodes * max_turns
    expected_turn_gens = worst_turn_gens * EXPECTED_HORIZON_FRACTION

    worst_s = (
        n_atom_gens * (atom_tb + atom_answer_tokens)
        + worst_turn_gens * (ep_tb + episode_answer_tokens)
    ) / VLLM_WORST_TOKS_PER_S
    expected_s = (
        n_atom_gens * (atom_tb * EXPECTED_THINK_FRACTION + atom_answer_tokens)
        + expected_turn_gens * (ep_tb * EXPECTED_THINK_FRACTION + episode_answer_tokens)
    ) / VLLM_SUSTAINED_TOKS_PER_S

    ctx_worst = EST_BASE_PROMPT_TOKENS + atom_tb + atom_answer_tokens
    if episode_cfg is not None:
        episode_ctx_worst = (
            EST_BASE_PROMPT_TOKENS
            + max_turns * EST_TOKENS_PER_TURN
            + ep_tb
            + episode_answer_tokens
        )
        ctx_worst = max(ctx_worst, episode_ctx_worst)
    return worst_s, expected_s, ctx_worst


def print_estimates(
    tier_names: list[str],
    families_dir: Path,
    assume_families: int,
    think_budget_override: int | None = None,
) -> None:
    # Estimation uses the explicit family-count assumption and never imports
    # held-out benchmark family modules.
    family_count = assume_families
    print(f"families: {family_count} (assumed)")
    print("model load time excluded (~35 s once, vLLM)")
    print(
        f"{'tier':<8} {'atom_think':>10} {'ep_think':>8} {'worst_s':>10} "
        f"{'expected_s':>11} {'no_think_worst_s':>16} {'no_think_expected_s':>19} "
        f"{'ctx_worst':>9} {'budget_s':>9} {'flag':>8}"
    )
    for tier_request in tier_names:
        tier_cfg = load_tier(tier_request)
        if think_budget_override is None:
            atom_think = int(tier_cfg["think_budget"])
            ep_think = int(tier_cfg.get("episode_think_budget", tier_cfg["think_budget"]))
        else:
            atom_think = int(think_budget_override)
            ep_think = int(think_budget_override)
        worst, expected, ctx_worst = estimate_tier(
            tier_cfg, family_count, True, think_budget_override
        )
        no_think_worst, no_think_expected, _ = estimate_tier(tier_cfg, family_count, False)
        # quick is pinned at 8 atoms/family by validate_suite's random-floor gate
        # (>=8 items keeps one chance hit at <=0.125), which puts its expected
        # wall at ~73% of the 60 s budget; 0.75 encodes 'fits with all the
        # headroom the instrument gates allow'. Other tiers sit at 50-65%.
        expected_cap = 0.75 * tier_cfg["budget_s"]
        if ctx_worst > VLLM_MAX_MODEL_LEN:
            flag = "CTX-OVER"
        elif worst <= tier_cfg["budget_s"] and expected <= expected_cap:
            flag = "WITHIN"
        else:
            flag = "OVER"
        print(
            f"{tier_cfg['tier']:<8} {atom_think:>10} {ep_think:>8} {worst:>10.1f} "
            f"{expected:>11.1f} {no_think_worst:>16.1f} {no_think_expected:>19.1f} "
            f"{ctx_worst:>9} {tier_cfg['budget_s']:>9} {flag:>8}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Menagerie benchmark tiers.",
        epilog=(
            "The qwen backend must run under "
            "/home/ericflo/Development/small-model-experimentation/.venv/bin/python "
            "(torch cu130 + transformers live there). CPU backends run under any python3."
        ),
    )
    parser.add_argument("--tier", help="quick|medium|slow|deep, a JSON path, or all with --estimate", default=None)
    parser.add_argument("--backend", default="qwen_vllm", help="qwen_vllm (default; fast vLLM path) | qwen (HF; deterministic parity oracle) | oracle | random | noisy:EPS | const:TEXT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="output JSON file")
    parser.set_defaults(think=True)
    parser.add_argument("--no-think", action="store_false", dest="think", help="disable qwen thinking mode")
    parser.add_argument(
        "--think-budget",
        type=int,
        default=None,
        help="override BOTH the tier atom and episode thinking budgets with one value (for explicit compute-response studies)",
    )
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--families-dir", default=str(SCRIPT_DIR / "families"))
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="write full transcripts and score details; requires an output filename containing DO_NOT_TRAIN",
    )
    parser.add_argument("--no-transcripts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to a PEFT LoRA adapter dir trained on Qwen/Qwen3.5-4B; validated at startup, applied by the qwen and qwen_vllm backends",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--estimate", action="store_true", help="print token-math wall-time estimates")
    parser.add_argument("--assume-families", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_model_id(args.model_id)
    except ValueError as exc:
        parser.error(str(exc))
    adapter_info = None
    if args.adapter is not None:
        try:
            adapter_info = adapter_spec.validate_adapter(args.adapter)
        except ValueError as exc:
            parser.error(str(exc))
    if args.no_transcripts and args.debug_artifacts:
        parser.error("--debug-artifacts restores transcripts and details; do not combine it with --no-transcripts")

    max_batch = args.max_batch
    if max_batch is None:
        max_batch = 48 if args.think else 96

    try:
        tier_requests = resolve_tier_requests(args.tier, args.estimate)
    except ValueError as exc:
        parser.error(str(exc))

    families_dir = Path(args.families_dir)
    if args.estimate:
        print_estimates(tier_requests, families_dir, args.assume_families, args.think_budget)
        return 0

    if adapter_info is not None and args.backend not in ("qwen", "qwen_vllm"):
        parser.error("--adapter requires --backend qwen or qwen_vllm")
    tier_cfg = load_tier(tier_requests[0])
    # Tier think budgets escalate 1024/1024/2048/4096, with a 1024 floor because
    # 256/512 are in the measured truncation-harm zone (~100% forced-close).
    # Above that floor only the tail of long-chain items binds. Deep episodes use
    # a separate 2048 per-turn cap purely to bound the 14-round worst-case wall
    # clock; --think-budget overrides both for compute-response studies.
    atom_tb = int(args.think_budget) if args.think_budget is not None else int(tier_cfg["think_budget"])
    episode_tb = (
        int(args.think_budget)
        if args.think_budget is not None
        else int(tier_cfg.get("episode_think_budget", tier_cfg["think_budget"]))
    )
    effective_think_budget = {"atom": atom_tb, "episode": episode_tb}
    out_path = Path(args.out) if args.out else default_out_path(tier_cfg, args.backend, args.seed, adapter_info)
    if args.debug_artifacts:
        if "DO_NOT_TRAIN" not in out_path.name:
            parser.error("--debug-artifacts requires an output filename containing DO_NOT_TRAIN")
        warn_debug_artifacts(out_path)

    families = engine.discover_families(families_dir)
    if not families:
        raise SystemExit(f"no runnable families found in {families_dir}")

    qwen_opts = {
        "model_id": args.model_id,
        "device": args.device,
        "think": args.think,
        "think_budget": effective_think_budget,
        "max_batch": max_batch,
        "max_new_tokens": tier_cfg["max_new_tokens"],
        "adapter": adapter_info["path"] if adapter_info else None,
    }
    backend_obj = backends.make_backend(args.backend, seed=args.seed, qwen_opts=qwen_opts)
    result = run_tier(
        families,
        tier_cfg,
        backend_obj,
        args.seed,
        include_transcripts=args.debug_artifacts,
        backend_spec=args.backend,
        think=args.think,
        think_budget=effective_think_budget,
    )
    result["adapter"] = adapter_info

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print_result_table(result)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
