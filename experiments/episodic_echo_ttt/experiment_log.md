# Episodic ECHO-TTT Experiment Log

## Run `smoke_episodic_echo_ttt_v1`

- Started: 2026-06-26 03:48:57 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `no_prefix,global_no_ttt,echo_ttt,shuffle_ttt`

Completed `smoke_episodic_echo_ttt_v1` in 82.9s.

- Metric rows: 12
- Detail rows: 48
- Report: `reports/episodic_echo_ttt_report.md`

## Run `pilot_episodic_echo_ttt_lr_v2`

- Started: 2026-06-26 03:51:01 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `no_prefix,global_no_ttt,echo_ttt,shuffle_ttt,generic_ttt`

Stopped during evaluation because candidate continuations were scored one
forward pass at a time. The runner was patched to score all candidates for an
episode in a single batch before the next pilot.

## Run `pilot_episodic_echo_ttt_batched_v3`

- Started: 2026-06-26 03:54:19 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `no_prefix,global_no_ttt,echo_ttt,shuffle_ttt,generic_ttt`

Completed `pilot_episodic_echo_ttt_batched_v3` in 116.7s.

- Metric rows: 22
- Detail rows: 176
- Report: `reports/episodic_echo_ttt_report.md`

## Run `smoke_episodic_echo_ttt_mc_v4`

- Started: 2026-06-26 03:57:34 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `no_prefix,global_no_ttt,echo_ttt,shuffle_ttt`

Completed `smoke_episodic_echo_ttt_mc_v4` in 24.6s.

- Metric rows: 12
- Detail rows: 48
- Report: `reports/episodic_echo_ttt_report.md`

## Run `main_episodic_echo_ttt_v1`

- Started: 2026-06-26 03:58:38 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202,303`
- Arms: `no_prefix,global_no_ttt,echo_ttt,shuffle_ttt,generic_ttt`

Completed `main_episodic_echo_ttt_v1` in 336.0s.

- Metric rows: 66
- Detail rows: 792
- Report: `reports/episodic_echo_ttt_report.md`
