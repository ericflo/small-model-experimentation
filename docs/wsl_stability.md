# WSL stability under sustained vLLM load

Five full-VM crashes in ~two days of long agentic-eval runs (not process kills — the whole WSL2 VM
dies and has to be restarted from Windows, which is slow and manual). What was ruled out, what is
left, and the settings that reduce it.

## Ruled out by measurement

- **Linux OOM.** The kernel OOM killer never fired: `swapon --show` shows 0B of the 4G swap used and
  `free -g` reports 10–14 GB available even at peak load. `dmesg` shows no `Out of memory` line for
  the crashing runs (and does not survive the reboot, but the pre-crash samples are clean).
- **Our process footprint.** The trainer was already fixed to ~2.2 GB RSS (`low_cpu_mem_usage=True`);
  evals run 2 workers; the vLLM server's host-side RSS is ~2–3 GB. Total stays well under the 15 GB
  the VM is given.

So the VM is being killed from outside the Linux memory system.

## The likely trigger

Sustained CUDA load through WSL2's GPU paravirtualization layer (`/dev/dxg`). The signature fits:
crashes occur only during multi-hour runs with vLLM resident at `--gpu-memory-utilization 0.90`,
never during CPU-only work (the 8-agent audit workflow, analysis, commits), and never immediately —
always deep into a run. WSLg/dxgkrnl instability under long-lived high-VRAM allocations is a known
class of failure; the host GPU driver resetting or the VM exceeding its memory ceiling while the
GPU partition is pinned takes the whole VM down rather than a single process.

## Fixes

### 1. Windows-side `.wslconfig` (apply once, biggest lever)

Create/edit `C:\Users\<you>\.wslconfig` (from WSL: `/mnt/c/Users/<you>/.wslconfig`), then
`wsl --shutdown` from PowerShell to apply:

```ini
[wsl2]
memory=24GB              # raise the VM ceiling (default is ~50% of host RAM)
swap=16GB                # real headroom instead of the 4G default
autoMemoryReclaim=gradual # return freed memory to Windows instead of holding it
pageReporting=true
guiApplications=false    # no WSLg surface needed for headless work
```

`memory=` is the important one: a VM that cannot grow when the GPU partition is pinned is exactly
the condition that ends in a hard VM kill.

### 2. Lower-pressure serve profile (already adopted here)

```
--gpu-memory-utilization 0.80   # not 0.90; leaves the driver headroom
--max-num-seqs 8                # fewer concurrent sequences => less bookkeeping
--enforce-eager                 # required anyway for Qwen3.5's hybrid arch
```

Also: **restart the server between long runs** rather than leaving one resident for many hours, and
prefer 2 eval workers over 4.

### 3. Crash discipline (assume it will still happen)

- `pi_episode.py` / `pi_repo_episode.py` checkpoint every 5 episodes and resume from their own
  partial file. This has now recovered five crashes with zero completed episodes lost, including the
  35-episode run that produced C64.
- Commit before every long run.
- On restart, use the gated sequence (kill by ANCHORED `pgrep -f "^/home.*python -u .*script"`, gate
  on `ss -tln` LISTENING, gate on a real `curl`, gate on proxy traffic growing). An unanchored
  `pgrep -f`/`pkill -f` matches the agent's own shell wrapper and has repeatedly killed the wrong
  thing; a proxy that lost a port-bind race once silently fast-failed 38 episodes.
