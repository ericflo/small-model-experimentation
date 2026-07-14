# Gauntlet frontier: difficulty escalation past the breadth-install plateau

**Status:** finished

Follow-up to `qwen35_4b_gauntlet_breadth_round1` (claims C49/C50). Executes
every escalation lever against the +0.22..+0.33 plateau: attribution
ablations, L5/L6 difficulty + two new weak-axis families, a frontier harvest
with the round-3 model, and a deploy-budget-matched variant.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: after the emission-policy install, what buys the next
  capability increment on the blackbox instrument?
- Prior anchors: C50 (the install + plateau), C49 (deployment hazards),
  C11/C48 (saturation and locality precedents).

## Question

Is the plateau a difficulty-coverage artifact of the gym (fixable by harder
strata, new axes, richer recovery supervision, or budget-matched training),
or a second wall intrinsic to training on the model's own verified outputs?
And within the original recipe: which ingredient caused the round-2 step
change — emission repair, one substrate's content, or breadth?

## Hypothesis

Escalated difficulty + a working recovery arm + budget matching lift quick
and medium decisively beyond +0.32; breadth is causal for held-out-family
transfer.

## Setup

- Model: Qwen/Qwen3.5-4B (pinned), think-mode, QLoRA r32/α64 think-channel
  SFT, merged-composite deployment (C49).
- Gym: the round-1 gym extended to L1–L6 (L1–L4 byte-identical, verified;
  horizons to 22) + new families `patchwheel` (rewrite-rule repair) and
  `packhouse` (assignment optimization). 14 families, all selftests green.
- Data: ablation slices of rounds 2–3 (recovery-only 553; ferrier-only 665;
  breadth-matched 665); frontier harvest with the round-3 merged model
  (2,514 examples, mass L3–L6); sharp1024 subset (think ≤900 + recovery,
  3,756). Recovery contexts trimmed to fit the encode window (the C50
  correction: rounds 2–3 trained on zero recovery examples).
- Eval: paired base-vs-merged menagerie events, fresh seed each (53001–53009,
  vLLM backend both arms); gym-internal greedy@1024 on held-out seeds.
- Hidden-label boundary: unchanged from round 1 (verifiers never in prompts;
  benchmark firewall absolute).

## Run

```bash
python3 scripts/run.py --smoke                      # config + 14 family selftests
# ablations: data/ablation_*.jsonl -> train_think -> merge_adapter -> bench
# frontier:  harvest --config configs/frontier.yaml --merged <round3 merged>
#            build_sft -> train_think (union) -> merge -> bench + eval_gym
```

## Results

All menagerie numbers are paired same-seed vLLM events (merged deployment):

| treatment | quick | medium |
|---|---|---|
| recovery-only (553) | +0.122 | — |
| ferrier-only (665) | +0.314 | — |
| breadth-matched (665) | +0.330 | — |
| frontier union (~5,700; L3–L6 mass) | +0.272 / +0.314 | +0.320 |
| sharp1024 (deploy-budget-matched) | +0.293 / +0.238 | +0.342 |

- Treated absolute lands at 0.375–0.447 in every event (base 0.070–0.161).
- Attribution: one rich family ≈ full breadth on aggregate AND on held-out
  gym transfer (spindle 0.748 vs 0.752; runeward 0.944 vs 0.933); breadth's
  edge is axis-aligned (chronicle +0.750 vs +0.375). Pure emission repair
  is worth ~+0.12 with weak transfer.
- Difficulty escalation installs IN-GYM (L5–L6 mean ~0 → 0.466; L1–L4
  0.184 → 0.681) without moving the blackbox band.
- Deployable evidence throughout; oracle policies only validate instruments.

## Interpretation

The recipe family installs a substrate-agnostic behavioral policy (conclude
within budget, commit tersely, one-line actions) worth ~+0.30 aggregate as a
one-time step; no train-on-own-verified-outputs variant moves the band
further (claim C53 — the second wall). The residual deficit is a capability
core (menders/lockpick/stockade/rites/warren at L2+), reachable — if at all —
by different mechanisms: scaffold-distillation of tool-found solutions,
on-policy RL at the residual failures, or axis curricula from failure
forensics.

## Knowledgebase Update

- Program evidence updated: `research_programs/agentic_breadth_installation/evidence.md`
- Program backlog updated: `research_programs/agentic_breadth_installation/backlog.md`
- Claim ledger updated: C51 (new), C50 (attribution corrections)

## Artifacts

- `src/gym/` — 14-family gym, L1–L6 (L1–L4 byte-identical to round 1)
- `configs/frontier.yaml`, `data/` (ablation + frontier + sharp1024 sets)
- `runs/` — harvest yields, gym evals, menagerie event log
- adapters/merged checkpoints external under `large_artifacts/` (manifest)
