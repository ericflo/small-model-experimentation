"""Gauntlet harness: batched atom generation and lockstep episode driving.

Runs under .venv-vllm (imports the experiment-local VLLMRunner). All scoring
and environment logic is CPU/gym-side; the model only ever sees prompts and
returns text.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_runner import EngineConfig, SamplingConfig, VLLMRunner  # noqa: E402
from gym import base  # noqa: E402
from gym.families import load as load_family  # noqa: E402


def make_runner(
    engine_cfg: dict,
    adapter: str | None = None,
    model_override: str | None = None,
) -> VLLMRunner:
    return VLLMRunner(
        EngineConfig(
            max_model_len=int(engine_cfg.get("max_model_len", 16384)),
            gpu_memory_utilization=float(engine_cfg.get("gpu_memory_utilization", 0.85)),
            max_num_seqs=int(engine_cfg.get("max_num_seqs", 64)),
            max_num_batched_tokens=int(engine_cfg.get("max_num_batched_tokens", 16384)),
            adapter=Path(adapter) if adapter else None,
            model_override=Path(model_override) if model_override else None,
        )
    )


def _sampling(
    *,
    think_budget: int,
    answer_max_tokens: int,
    n: int,
    run_seed: int,
    greedy: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> SamplingConfig:
    return SamplingConfig(
        thinking="budget",
        thinking_budget=think_budget,
        n=1 if greedy else n,
        answer_max_tokens=answer_max_tokens,
        greedy=greedy,
        temperature=None if greedy else temperature,
        top_p=None if greedy else top_p,
        top_k=None if greedy else top_k,
        run_seed=run_seed,
    )


def _slim(output: dict) -> dict:
    """Keep the fields downstream stages need; drop token-id arrays."""
    return {
        "sample_index": output["sample_index"],
        "text": output["text"],
        "n_thinking_tokens": output["n_thinking_tokens"],
        "n_answer_tokens": output["n_answer_tokens"],
        "n_sampled_tokens": output["n_sampled_tokens"],
        "thinking_closed": output["thinking_closed"],
        "forced_close": output["forced_close"],
        "finish_reason": output["finish_reason"],
        "truncated": output["truncated"],
    }


def run_atoms(
    runner: VLLMRunner,
    items: list[dict],
    *,
    k: int,
    think_budget: int,
    answer_max_tokens: int,
    run_seed: int,
    greedy: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Generate k samples per atom item and score them. Returns one row per item."""
    records = [
        {"id": item["id"], "messages": [{"role": "user", "content": item["prompt"]}]}
        for item in items
    ]
    sampling = _sampling(
        think_budget=think_budget,
        answer_max_tokens=answer_max_tokens,
        n=k,
        run_seed=run_seed,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    rows, summary = runner.generate(records, sampling)
    results = []
    for item, row in zip(items, rows):
        family = load_family(item["family"])
        outputs = []
        for output in row["outputs"]:
            slim = _slim(output)
            slim["score"] = float(family.score_atom(item, output["text"]))
            slim["answer_value"] = base.extract_answer(output["text"])
            outputs.append(slim)
        results.append(
            {
                "id": item["id"],
                "family": item["family"],
                "level": item["level"],
                "prompt": item["prompt"],
                "gold": item["gold"],
                "answer_domain": item.get("answer_domain"),
                "outputs": outputs,
            }
        )
    return results


def run_episodes(
    runner: VLLMRunner,
    specs: list[tuple[str, int, int]],
    *,
    k: int,
    think_budget: int,
    answer_max_tokens: int,
    run_seed: int,
    greedy: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run k rollouts per (family, level, ep_seed) spec, lockstep-batched
    across ALL rollouts of all families. Returns one row per rollout."""
    rollouts: list[dict[str, Any]] = []
    for family_name, level, ep_seed in specs:
        family = load_family(family_name)
        for rollout_index in range(1 if greedy else k):
            episode = family.Episode(ep_seed, level)
            rollouts.append(
                {
                    "rid": f"{family_name}-L{level}-e{ep_seed}-r{rollout_index}",
                    "family": family_name,
                    "level": level,
                    "ep_seed": ep_seed,
                    "rollout": rollout_index,
                    "episode": episode,
                    "messages": [
                        {"role": "system", "content": episode.system_prompt()},
                        {"role": "user", "content": episode.initial_observation()},
                    ],
                    "turns": [],
                    "done": False,
                }
            )

    sampling = _sampling(
        think_budget=think_budget,
        answer_max_tokens=answer_max_tokens,
        n=1,
        run_seed=run_seed,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    turn = 0
    max_horizon = max(r["episode"].max_turns for r in rollouts) if rollouts else 0
    while turn < max_horizon:
        active = [
            r for r in rollouts if not r["done"] and turn < r["episode"].max_turns
        ]
        if not active:
            break
        records = [
            {"id": f"{r['rid']}-t{turn}", "messages": r["messages"]} for r in active
        ]
        rows, _ = runner.generate(records, sampling)
        for rollout, row in zip(active, rows):
            output = row["outputs"][0]
            action = base.extract_action(output["text"])
            observation, done = rollout["episode"].step(action)
            slim = _slim(output)
            rollout["turns"].append(
                {
                    "turn": turn,
                    "action": action,
                    "action_ok": bool(getattr(rollout["episode"], "last_action_ok", True)),
                    "observation": observation,
                    "context_messages": len(rollout["messages"]),
                    **slim,
                }
            )
            rollout["messages"].append({"role": "assistant", "content": action})
            rollout["messages"].append({"role": "user", "content": observation})
            rollout["done"] = bool(done)
        if progress:
            live = sum(1 for r in rollouts if not r["done"])
            print(f"[episodes] turn {turn}: {len(active)} active, {live} still live", flush=True)
        turn += 1

    results = []
    for rollout in rollouts:
        results.append(
            {
                "rid": rollout["rid"],
                "family": rollout["family"],
                "level": rollout["level"],
                "ep_seed": rollout["ep_seed"],
                "rollout": rollout["rollout"],
                "spec": rollout["episode"].spec,
                "system_prompt": rollout["messages"][0]["content"],
                "initial_observation": rollout["messages"][1]["content"],
                "turns": rollout["turns"],
                "done": rollout["done"],
                "score": float(rollout["episode"].score()),
                "n_turns": len(rollout["turns"]),
                "max_turns": rollout["episode"].max_turns,
            }
        )
    return results
