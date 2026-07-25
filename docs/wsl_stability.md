# WSL VM death under vLLM load — what is measured, and what is NOT explained

Seven full-VM deaths in ~two days of GPU work (the WSL2 VM disappears; recovery is a manual
Windows-side restart). Two wrong calls were made along the way and are corrected here: first,
recommending that the VM memory ceiling be *raised* to 24 GB (it should be capped); second, claiming
the resulting memory measurements identified the cause (they do not).

**STATUS (after crash #7): the cause is NOT established.** An earlier version of this document
claimed the host-memory/WDDM mechanism below *was* the cause. That claim is FALSIFIED: after capping
the VM at 16 GB and serving at `--gpu-memory-utilization 0.60`, host free memory stayed at **7.3 GB
under load** — no exhaustion — and the VM died anyway. The memory measurements below are real and
worth keeping (they describe a genuine resource constraint, and 2.9 GB free was genuinely dangerous),
but they do not explain the crashes.

Also ruled out since: no `nvlddmkm`/display-driver faults in the Windows System log *ever*; no Xid
errors; GPU healthy at idle (48 °C, 18 W of a 450 W limit); no Linux OOM. Seven VM deaths, all during
sustained GPU work, none during CPU-only work, none immediately after start.

What remains unexplained is why the VM dies at all. Candidates not yet distinguished: a dxgkrnl/WSL
GPU-paravirtualization fault under long-lived CUDA contexts, a WSL kernel panic (the guest log does
not survive the reboot), or a host-side watchdog. Diagnosing further requires Windows-side tracing
that cannot be done reliably from inside the VM that keeps dying.

## The memory measurements (real, but not the cause)

On this box (host 31.9 GB RAM, RTX 4090 24 GB, HAGS on), with `.wslconfig memory=24GB`:

| state | Windows host free | `vmmemWSL` working set | VRAM |
|---|---|---|---|
| VM up, GPU idle | 18.3 GB | 3.94 GB | ~1.8 GB |
| vLLM up at `--gpu-memory-utilization 0.80` | **2.9 GB** | **19.39 GB** | 20.3 GB |
| after killing vLLM | 4.3 GB | 17.1 GB | 1.9 GB |
| + `sync; drop_caches`, ~2 min later | 6.3 GB | 15.6 GB | 1.9 GB |

Three facts follow directly (all still true, none of them the crash cause):

1. **In WSL2, VRAM is mirrored into host memory ~1:1.** Reserving 20.3 GB of VRAM grew `vmmem` by
   15.5 GB and drove host free memory from 18.3 GB to **2.9 GB**. GPU allocations are backed through
   the Windows WDDM layer, so GPU memory is *not* free from the host's perspective.
2. **Windows itself needs ~13.6 GB here** (31.9 − 18.3 at idle). Budget: `vmmem` must stay under
   roughly `31.9 − 13.6 − margin`.
3. **`vmmem` does not give memory back promptly.** After vLLM exited and VRAM dropped to 1.9 GB,
   `vmmem` still held 17.1 GB, decaying only gradually (15.6 GB two minutes later) even though Linux
   inside the VM reported just 1 GB used of 23. Every serve/restart cycle therefore *ratchets* host
   memory usage upward — which is why crashes came deep into long sessions and why repeatedly
   restarting the server made things worse.

`memory=24GB` therefore left Windows only ~2.9 GB, which is a genuinely bad operating point and worth
avoiding on its own merits. It is simply not what kills the VM: at 16 GB / util 0.60 the same workload
had 7.3 GB free and still died.

## Settings worth keeping anyway

These do not stop the crashes, but they keep the box out of a genuinely bad memory regime and make
failures recoverable (a too-small KV cache fails cleanly inside Linux instead of thrashing).

### 1. `.wslconfig` — CAP the VM, do not raise it

`C:\Users\<you>\.wslconfig`, then `wsl --shutdown` from PowerShell:

```ini
[wsl2]
memory=12GB               # CAP. Not 24GB. vmmem must not be allowed to starve Windows.
swap=16GB                 # disk-backed, harmless, keeps Linux from needing to grow
processors=8

[experimental]
autoMemoryReclaim=gradual # helps, but does NOT return memory promptly — see the table
pageReporting=true
```

With a 12 GB cap: 13.6 (Windows) + ~12 (vmmem) ≈ 25.6 GB, leaving **~6 GB of headroom** instead of
2.9 GB. The cap is the real safety belt: it converts "host runs out and kills the VM" into "a CUDA
allocation fails inside Linux", which is a recoverable error.

### 2. vLLM serve profile — the model is 8.5 GB, so keep VRAM near that

```
--gpu-memory-utilization 0.45   # ≈10.8 GB VRAM: 8.5 GB weights + ~2.3 GB KV
--max-model-len 16384           # not 40960; pi's model entry caps output at 8192 anyway
--max-num-seqs 4
--enforce-eager                 # required for Qwen3.5's hybrid GDN arch regardless
```

`vmmem` then lands ~11–12 GB, inside the cap, with host headroom intact. (Do not go below ~0.40: the
bf16 weights alone are 8.5 GB and the KV cache would compute negative.)

### 3. Operating rules

- **Prefer SHORT GPU bursts to multi-hour runs.** Every crash landed deep into a long run; none
  happened at startup. Combined with checkpoint+resume this is the one mitigation with an evidence
  base — a 10-minute burst that completes is worth more than an hour-long run that dies at minute 50.
- **`wsl --shutdown` before a long GPU run.** Because `vmmem` never fully deflates, starting a
  multi-hour run on a VM that already holds 15 GB is starting in the danger zone.
- **Close host memory hogs first.** 13.6 GB of Windows usage at idle is a lot; freeing 4–5 GB
  (browser, game launchers — an Epic Online Services crash appears in this box's event log) buys
  more safety than any WSL setting.
- **One server, no restart churn.** Each serve/kill cycle ratchets `vmmem` up.
- **Watch the real number, not the Linux one.** `free -h` inside WSL is useless here (it showed 22 GB
  free while the host had 2.9 GB). Check the host:
  ```
  powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB"
  ```
  Abort a run if host free drops under ~4 GB.

## Crash discipline (assume it can still happen)

- `pi_episode.py` / `pi_repo_episode.py` checkpoint every 5 episodes and resume from their own
  partial file. This recovered all seven crashes with **zero completed episodes lost**, including the
  35-episode run behind claim C64.
- Commit before every long run.
- On restart use the gated sequence: kill stragglers by ANCHORED cmdline
  (`pgrep -f "^/home.*python -u .*script"` — an unanchored `pgrep -f`/`pkill -f` matches the agent's
  own shell wrapper and has repeatedly killed the wrong process), gate on `ss -tln` showing the port
  LISTENING, gate on a real `curl` answering, and gate on proxy traffic actually growing. A proxy
  that lost a port-bind race once silently fast-failed 38 episodes into empty results.
