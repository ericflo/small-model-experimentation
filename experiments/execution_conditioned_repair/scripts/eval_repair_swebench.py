#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def docker_preflight() -> dict:
    if not shutil.which("docker"):
        return {"ok": False, "reason": "docker binary not found"}
    version = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    if version.returncode != 0:
        return {"ok": False, "reason": "docker daemon unavailable", "output": version.stdout}
    probe = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "hello-world"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    return {
        "ok": probe.returncode == 0,
        "server_version": version.stdout.strip(),
        "probe_returncode": probe.returncode,
        "probe_output": probe.stdout[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/swebench_slice_results.json"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--predictions-path", type=Path)
    parser.add_argument("--instance-ids", nargs="*")
    parser.add_argument("--run-id", default="repair-lora-slice")
    args = parser.parse_args()

    preflight = docker_preflight()
    payload = {"docker_preflight": preflight, "status": "preflight_only"}
    if not args.preflight_only and preflight["ok"]:
        if not args.predictions_path:
            raise SystemExit("--predictions-path is required when running SWE-bench evaluation")
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--predictions_path",
            str(args.predictions_path),
            "--max_workers",
            "1",
            "--run_id",
            args.run_id,
        ]
        if args.instance_ids:
            cmd.extend(["--instance_ids", *args.instance_ids])
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        payload = {
            "docker_preflight": preflight,
            "status": "completed" if proc.returncode == 0 else "failed",
            "command": cmd,
            "returncode": proc.returncode,
            "output": proc.stdout[-12000:],
        }
    elif not args.preflight_only:
        payload["status"] = "blocked"
        payload["reason"] = "Docker preflight failed; official SWE-bench evaluation is not valid in this runner."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not preflight["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
