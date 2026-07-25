# WSL VM death — CAUSE ESTABLISHED: guest OOM kills `init.scope`, which is the whole VM

Eight full-VM deaths (the WSL2 VM disappears; recovery is a Windows-side restart). **As of 2026-07-24
the mechanism is measured, replicated, and mitigable.** Three earlier calls in this document were
wrong and are corrected below: recommending the VM ceiling be *raised*; claiming host-memory
exhaustion was the cause; and — the expensive one — asserting "no Linux OOM" and "the guest log does
not survive the reboot". The guest log **does** survive (`journalctl -b -1`), and it contains an
explicit OOM kill. The evidence was retrievable through seven crashes and was never read.

## The mechanism

```
kernel: python invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0
kernel: oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,task_memcg=/init.scope,task=python,pid=6687
kernel: Out of memory: Killed process 6687 (python) total-vm:32658112kB, anon-rss:15096448kB
systemd[1]: init.scope: Failed with result 'oom-kill'.
systemd[1]: init.scope: Consumed 8min 36.070s CPU time, 15.1G memory peak, 15.8G memory swap peak.
```

Read it in order:

1. **One process grows to ~15.1 GB anonymous RSS** (`total-vm` 32 GB, so it had also mapped far more
   address space than it resident-touched).
2. That saturates the VM's `.wslconfig` ceiling — `memory=16GB` — **and its swap**: `15.8G memory swap
   peak` against `swap=16GB`. With both exhausted there is nothing left to reclaim.
3. The kernel fires a **global** OOM (`constraint=CONSTRAINT_NONE`, `global_oom`). Under WSL2 every
   process lives in the `/init.scope` cgroup, so the accounting scope charged for the kill is
   `init.scope`.
4. systemd marks **`init.scope`** — the scope containing PID 1 and the entire session — as
   `Failed with result 'oom-kill'`, tears down user sessions, and the VM goes away.

The fatal step is (4): the OOM does not merely kill the offending process, it condemns the scope that
*is* WSL. So **any single process that reaches ~15 GB of anonymous memory destroys the box**, and
nothing about the workload's nature matters.

### Replication and scope of the claim

| crash | when | workload | signature |
|---|---|---|---|
| #8 | 2026-07-24 22:31 | **CPU-only**: 4 concurrent third-party pytest suites, zero GPU work | `anon-rss 15,096,448 kB`, `init.scope … 'oom-kill'`, 15.1G/15.8G peaks |
| — | 2026-07-24 18:14 | earlier session (pre-dating this investigation) | `anon-rss 15,339,980 kB`, same `global_oom` → `init.scope … 'oom-kill'`, 15.1G/15.8G peaks |

Two independent deaths, byte-similar signatures, ~15 GB anon RSS both times. Honest scope: journald
retains only the last few boots, so the **first seven** crashes' logs are gone and cannot be
confirmed post hoc. Two prior claims are nonetheless falsified outright by crash #8 —
"none during CPU-only work" (it was entirely CPU-bound) and "no Linux OOM" (there is one, in the
guest journal). Given that the earlier crashes ran workloads with well-known ~8–16 GB
host-RAM appetites (an accidental fp32 4B load is ~16 GB; TRL's colocate sleep `level=2` offloads
8.5 GB of weights to host RAM), guest OOM is now the leading explanation for those too, and there is
no longer any need to invoke an unobservable dxgkrnl/WDDM fault.

### Who the ~15 GB process actually is

The kernel's OOM report carries a full task table. Parse it and the picture is unambiguous — one
process, everything else negligible:

```
python           pid=3020   totalvm_mb=32191  rss_mb=15305  swap_mb=16132
claude           pid=1423   totalvm_mb=6147   rss_mb=172    swap_mb=97
python3.10       pid=621    totalvm_mb=658    rss_mb=12     swap_mb=44
```

`rss 15.3 GB + swap 16.1 GB ≈ 31 GB of anonymous memory in a single python process.` Two distinct
mechanisms can produce this, and both were live in this repo:

1. **An unbounded child allocation** — we run third-party OSS test suites. `toolz`'s own
   `test_interpose` (`first(rest(interpose("a", range(1000000000))))`) raises `MemoryError` under a
   4 GB address-space cap, i.e. it wants more than 4 GB; uncapped it can run to tens of GB. It was
   in-flight during crash #8.
2. **Driver-side output buffering** — `subprocess.run(..., stdout=PIPE)` accumulates a child's ENTIRE
   output in the PARENT's memory. A capped child still cannot be stopped from PRINTING without bound,
   and every byte lands in the driver's RSS. This is why deaths arrived deep into long runs under both
   GPU and CPU workloads: every driver in this repo captured output that way.

## Mitigations (in order of effectiveness)

### 0. Run every heavy pipeline inside a memory-limited cgroup scope — the general fix

`experiments/qwen35_4b_realrepo_agentic_instrument/scripts/guard.sh`:

```bash
systemd-run --user --scope --quiet \
  -p MemoryMax=10G -p MemoryHigh=8G -p MemorySwapMax=2G -- "$@"
```

A cgroup scope binds **every descendant** — build backends, test-spawned subprocesses, anything a
third-party suite forks — so it does not depend on remembering to cap each call site. When the ceiling
is hit the kernel kills inside the scope and the VM survives.

Verified on this box: `-p MemoryMax=1G` plus a deliberate 3 GB allocation → killed with exit 137, VM
up. cgroup v2 with the `memory` controller delegated to the user session is present
(`/sys/fs/cgroup` is `cgroup2fs`; `user@1000.service` lists `cpu memory pids`).

`MemoryMax` bounds **resident** memory, so unlike `ulimit -v` it is safe for torch/vLLM (CUDA reserves
huge virtual address space but not resident pages). Give GPU pipelines a deliberately larger ceiling
(e.g. `GUARD_MEM=13G`) rather than none.

### 1. Cap untrusted children too, and never buffer their output in RAM

Convert "the VM dies and the run is lost" into "one subprocess fails with `MemoryError`". Use
`ulimit -v` **inside the shell command**, not `subprocess(preexec_fn=…)`, which is documented-unsafe
in multithreaded drivers and can deadlock:

```python
cmd = f"ulimit -v {6 * 1024 * 1024} 2>/dev/null; ulimit -c 0 2>/dev/null; {cmd}"   # 6 GiB
```

Confirmed to work under `/bin/sh` → **dash** on this box (dash honours `ulimit -v`; do not assume it,
the `2>/dev/null` would hide an unsupported-option error and silently leave the child uncapped —
test it after any shell change).

And **never capture unbounded child output in memory**: spool it to a file, stream the file to extract
what you need, keep only a tail, and kill any child that prints past a ceiling. Reference
implementation: `experiments/qwen35_4b_realrepo_agentic_instrument/scripts/env_util.py`
(`run_spooled()`, `capped()`, `run_tests()`), which runs third-party OSS suites — code we do not
control and cannot audit for allocation or output behaviour. Kill the process **group**
(`start_new_session=True` + `os.killpg`), or a killed pytest's children keep running and keep printing.

**Do not apply `ulimit -v` to torch/vLLM processes.** CUDA reserves enormous virtual address space,
so an address-space cap breaks them. Bound those with the cgroup scope above, and keep the load-time
invariants that avoid the danger zone entirely: `dtype=bfloat16` always (an accidental fp32 4B load is
~16 GB = exactly this crash), and prefer not to sleep the colocate engine to host RAM.

### 2. Gate new work on `MemAvailable`

Before spawning a worker, refuse to start below a floor (`env_util.wait_for_memory()`, default
3 GB). Cheap, and it stops a slow ratchet from becoming a kill.

### 3. Keep concurrency low for memory-hungry children

Crash #8 ran four third-party suites at once. Per-process caps are the fix; low concurrency
(`--jobs 2`) is the belt-and-braces, since peak demand multiplies with workers.

### 4. Swap is not protection — it is delay

`swap=16GB` let the runaway reach 15.8 GB of swap before dying, thrashing the disk on the way. It
does not prevent the OOM; it postpones it and degrades everything meanwhile. Keep it (a legitimate
transient spike survives) but never treat it as headroom.

### 5. `.wslconfig`

Current box:

```ini
[wsl2]
memory=16GB               # CAP. Protects Windows (which needs ~13.6 GB here); 24GB left only 2.9 GB.
swap=16GB
processors=8

[experimental]
autoMemoryReclaim=gradual
```

Raising `memory=` does **not** fix this class of failure — a runaway allocation reaches any ceiling.
It only helps if a *legitimate* workload genuinely needs more than ~14 GB of guest RAM, and the right
response to that is to fix the workload's memory profile. Changing this file requires
`wsl --shutdown`, which kills every running session.

## Diagnosing the next one — read the journal FIRST

```bash
journalctl --list-boots                      # -1 is the boot that died
journalctl -b -1 --no-pager | grep -E "oom-kill|memory peak|Out of memory|oom_reaper"
journalctl -b -1 -n 60 --no-pager            # the teardown sequence

# WHO it was: parse the OOM task table (pages -> MB), sorted by RSS. This is the step that ends the
# guesswork -- it names the ballooning process and shows that nothing else was close.
journalctl -b -1 --no-pager \
  | sed -n '/Tasks state/,/Out of memory/p' | sed 's/.*kernel: //' \
  | awk '/^\[/ {printf "%-16s pid=%-8s totalvm_mb=%-8.0f rss_mb=%-7.0f swap_mb=%-6.0f\n", \
                       $NF, $1$2, $5*4/1024, $6*4/1024, $11*4/1024}' \
  | sort -t= -k4 -rn | head
```

Also look ~2 minutes BEFORE the kill for early pressure: `p9_client_rpc` / `p9_fcall_init` page
allocation failures (the 9p mount to Windows) appeared first in crash #8 and are a useful early
warning that the VM is running out, not that the filesystem is broken.

If `Out of memory: Killed process … anon-rss:1[45],…` appears, it is this mechanism: find the
process, cap it, done. Only if the journal is *clean* through the death is a host-side or
paravirtualization fault worth considering.

## The host-memory measurements (real, and still worth respecting)

Not the crash cause, but a genuine constraint. Host 31.9 GB RAM, RTX 4090 24 GB, HAGS on, measured
with `.wslconfig memory=24GB`:

| state | Windows host free | `vmmemWSL` working set | VRAM |
|---|---|---|---|
| VM up, GPU idle | 18.3 GB | 3.94 GB | ~1.8 GB |
| vLLM up at `--gpu-memory-utilization 0.80` | **2.9 GB** | **19.39 GB** | 20.3 GB |
| after killing vLLM | 4.3 GB | 17.1 GB | 1.9 GB |
| + `sync; drop_caches`, ~2 min later | 6.3 GB | 15.6 GB | 1.9 GB |

1. **In WSL2, VRAM is mirrored into host memory ~1:1.** 20.3 GB of VRAM grew `vmmem` by 15.5 GB.
2. **Windows needs ~13.6 GB here**, so `vmmem` must stay under roughly `31.9 − 13.6 − margin`.
3. **`vmmem` does not give memory back promptly**, so serve/restart cycles ratchet host usage upward.

### vLLM serve profile — the model is 8.5 GB, so keep VRAM near that

```
--gpu-memory-utilization 0.45   # ≈10.8 GB VRAM: 8.5 GB weights + ~2.3 GB KV
--max-model-len 16384           # not 40960; pi's model entry caps output at 8192 anyway
--max-num-seqs 4
--enforce-eager                 # required for Qwen3.5's hybrid GDN arch regardless
```

Do not go below ~0.40: bf16 weights alone are 8.5 GB and the KV cache would compute negative.

## Crash discipline (still assume it can happen)

- Checkpoint every few episodes and resume from the partial file. This recovered all eight crashes
  with **zero completed episodes lost**, including the 35-episode run behind claim C64 and the
  12-repo fetch behind crash #8.
- Make long CPU pipelines resumable too, not just GPU ones — crash #8 was CPU-only.
- Prefer short bursts over multi-hour runs; commit before every long run.
- On restart use the gated sequence: kill stragglers by ANCHORED cmdline
  (`pgrep -f "^/home.*python -u .*script"` — an unanchored `pgrep -f`/`pkill -f` matches the agent's
  own shell wrapper and has repeatedly killed the wrong process), gate on `ss -tln` showing the port
  LISTENING, gate on a real `curl` answering, and gate on proxy traffic actually growing. A proxy
  that lost a port-bind race once silently fast-failed 38 episodes into empty results.
