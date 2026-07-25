#!/usr/bin/env bash
# Run a command inside a memory-limited cgroup scope, so that a runaway descendant is OOM-killed
# INSIDE the scope instead of triggering a global OOM whose victim cgroup is /init.scope -- which
# under WSL2 contains all of WSL, so the whole VM dies. See docs/wsl_stability.md.
#
# Verified on this box: `systemd-run --user --scope -p MemoryMax=1G` + a deliberate 3 GB allocation
# is killed with exit 137 and the VM stays up.
#
# Why this and not just `ulimit -v`: ulimit only binds processes we launch ourselves through a shell
# string we control. A cgroup scope binds EVERY descendant, including build backends, test-spawned
# subprocesses, and anything a third-party suite forks -- which is what actually killed the box
# (a single uncapped python reached ~31 GB of anonymous memory: 15.3 GB resident + 16.1 GB swapped).
#
# Usage:  scripts/guard.sh <command> [args...]
#         GUARD_MEM=12G GUARD_SWAP=2G scripts/guard.sh python train.py
#
# NOTE: MemoryMax bounds RESIDENT memory, so unlike `ulimit -v` it is safe for torch/vLLM processes
# (CUDA reserves huge virtual address space but not resident pages).
set -euo pipefail

MEM="${GUARD_MEM:-10G}"          # hard ceiling; VM total is 16G, leave room for the agent + system
SWAP="${GUARD_SWAP:-2G}"         # swap saturation is what turned pressure into a kill; keep it small
HIGH="${GUARD_HIGH:-8G}"         # throttle before killing, so a near-miss slows down instead of dying

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

exec systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM" -p MemoryHigh="$HIGH" -p MemorySwapMax="$SWAP" \
  -- "$@"
