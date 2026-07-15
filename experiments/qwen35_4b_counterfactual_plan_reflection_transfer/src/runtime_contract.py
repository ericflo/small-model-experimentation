"""Immutable detached-worktree and runtime provenance contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import base64
import fcntl
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNTIME_PIN = Path(__file__).resolve().parents[1] / "configs" / "pinned_runtime_environments.json"
_BOOTSTRAP_RECEIPT: dict[str, Any] | None = None
_RUNTIME_IMPORT_GUARD: Any | None = None
_RUNTIME_IMPORT_CONTENT: dict[str, Any] | None = None
_RUNTIME_PREFLIGHT_GUARD: Any | None = None
_RUNTIME_PREFLIGHT_CONTENT: dict[str, Any] | None = None
_PREAUTH_GIT_FD = 192
_PREAUTH_LOADER_FD = 193
_PREAUTH_MANIFEST_FD = 194
_PREAUTH_RUNTIME_CONTRACT_FD = 195
_PREAUTH_LOAD_GUARD_FD = 196
_PREAUTH_STAGE_FD = 197
_STATIC_LAUNCHER_PROOF_FD = 198
_PREAUTH_INTERPRETER_FD = 199
_SNAPSHOT_ROOT = Path("/workspace/sme-reflection-runtime")
_SNAPSHOT_LIBRARY_PATH = (
    "/workspace/sme-reflection-runtime/runtime-libs:/usr/local/cuda/lib64"
)


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def require_detached_execution_worktree(repo_root: Path) -> dict[str, str]:
    """Require commands to run from one clean, detached, exact-SHA worktree."""
    repo_root = repo_root.resolve()
    executable = Path(sys.executable).resolve()
    if Path.cwd().resolve() != repo_root:
        raise ValueError("execution must be invoked with the detached worktree root as cwd")
    if repo_root == executable or repo_root in executable.parents:
        raise ValueError("execution interpreter must be provisioned outside the worktree")
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode or sys.flags.no_site != 1:
        raise ValueError("execution requires an isolated -I -B -S Python interpreter")
    top = _run_preauthenticated_git(["rev-parse", "--show-toplevel"], cwd=repo_root)
    head = _run_preauthenticated_git(["rev-parse", "HEAD"], cwd=repo_root)
    status = _run_preauthenticated_git(
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        cwd=repo_root,
    )
    branch = _run_preauthenticated_git(["symbolic-ref", "-q", "HEAD"], cwd=repo_root)
    if (
        top.returncode != 0
        or Path(top.stdout.strip()).resolve() != repo_root
        or head.returncode != 0
        or len(head.stdout.strip()) != 40
        or status.returncode != 0
        or status.stdout
        or branch.returncode == 0
    ):
        raise ValueError(
            "execution requires a clean detached exact-SHA Git worktree with no ignored state"
        )
    return {
        "repo_root": str(repo_root),
        "git_commit": head.stdout.strip(),
        "head_mode": "detached",
        "cwd": str(Path.cwd().resolve()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_pin(backend: str) -> dict[str, Any]:
    value = json.loads(RUNTIME_PIN.read_text())
    if (
        set(value) != {"schema_version", "system", "training", "vllm"}
        or value["schema_version"] != 4
    ):
        raise ValueError("runtime environment pin schema changed")
    if backend not in {"training", "vllm"}:
        raise ValueError(f"unknown runtime backend: {backend}")
    selected = value[backend]
    if (
        not isinstance(selected, dict)
        or set(selected)
        != {
            "environment_root",
            "bin_surface_sha256",
            "lock_file",
            "path_extensions",
            "record_surface_sha256",
            "site_surface_sha256",
            "startup_files",
        }
        or not isinstance(selected["startup_files"], dict)
        or not isinstance(selected["path_extensions"], list)
        or any(
            not isinstance(item, str) or not item or Path(item).is_absolute()
            for item in selected["path_extensions"]
        )
    ):
        raise ValueError("runtime environment backend pin changed")
    return selected


def _system_pin() -> dict[str, Any]:
    value = json.loads(RUNTIME_PIN.read_text()).get("system")
    required = {
        "resolved_python",
        "resolved_python_sha256",
        "stdlib_root",
        "stdlib_surface_sha256",
        "ld_library_path",
        "native_library_roots",
        "native_mappings",
        "executables",
        "pre_python_manifest",
        "stage_entrypoints",
        "launchers",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or not isinstance(value.get("native_mappings"), dict)
        or not value["native_mappings"]
        or not isinstance(value.get("native_library_roots"), dict)
        or not value["native_library_roots"]
        or not isinstance(value.get("executables"), dict)
        or set(value["executables"]) != {"git", "nvidia_smi", "nvcc", "uv"}
        or any(
            not isinstance(item, dict) or set(item) != {"path", "sha256"}
            for item in value["executables"].values()
        )
        or not isinstance(value.get("pre_python_manifest"), dict)
        or set(value["pre_python_manifest"]) != {"path", "sha256"}
        or not isinstance(value.get("stage_entrypoints"), dict)
        or set(value["stage_entrypoints"]) != {"training", "vllm"}
        or any(
            not isinstance(stages, dict)
            or not stages
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(item, dict)
                or set(item) != {"path", "sha256"}
                for name, item in stages.items()
            )
            for stages in value["stage_entrypoints"].values()
        )
        or not isinstance(value.get("launchers"), dict)
        or set(value["launchers"]) != {"training", "vllm"}
        or any(
            not isinstance(item, dict) or set(item) != {"path", "sha256"}
            for item in value["launchers"].values()
        )
    ):
        raise ValueError("runtime system pin changed")
    return value


def _launcher_environment(backend: str, stage: str) -> dict[str, str]:
    environment_root = Path(
        "/workspace/small-model-experimentation/.venv"
        if backend == "training"
        else "/workspace/small-model-experimentation/.venv-vllm"
    )
    return {
        "HOME": "/root",
        "PATH": (
            f"{environment_root}/bin:/usr/local/cuda-12.8/bin:"
            "/workspace/sme-reflection-runtime/tools"
        ),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "Etc/UTC",
        "PYTHONNOUSERSITE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_EXEC_PATH": "/workspace/sme-reflection-runtime/lib/git-core",
        "LD_LIBRARY_PATH": _SNAPSHOT_LIBRARY_PATH,
        "LOCPATH": "/workspace/sme-reflection-runtime/lib/locale",
        "GCONV_PATH": "/workspace/sme-reflection-runtime/runtime-libs/gconv",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SME_RUNTIME_BACKEND": backend,
        "SME_RUNTIME_STAGE": stage,
    }


def _descriptor_authentication(descriptor: int) -> dict[str, Any]:
    if not os.get_inheritable(descriptor):
        raise RuntimeError("pre-Python proof descriptor is not inheritable")
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("pre-Python proof descriptor is not a regular file")
    path = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve(strict=True)
    path_stat = os.stat(path, follow_symlinks=False)
    digest = _sha256_fd(descriptor)
    after = os.fstat(descriptor)
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if (
        path.is_symlink()
        or (before.st_dev, before.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        or any(getattr(before, field) != getattr(after, field) for field in stable)
    ):
        raise RuntimeError("pre-Python proof inode changed or differs from its path")
    return {
        "fd": descriptor,
        "path": str(path),
        "sha256": digest,
        "device": before.st_dev,
        "inode": before.st_ino,
        "size": before.st_size,
    }


def _manifest_rows(descriptor: int) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    payload = os.pread(descriptor, os.fstat(descriptor).st_size, 0).decode()
    lines = payload.splitlines()
    if not lines or lines[0] != "schema\t1":
        raise RuntimeError("pre-Python manifest schema changed")
    roles: dict[str, dict[str, str]] = {}
    stages: dict[str, dict[str, dict[str, str]]] = {"training": {}, "vllm": {}}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) == 4 and fields[0] == "file":
            _kind, role, path, digest = fields
            if role != "preauth":
                if role in roles:
                    raise RuntimeError("pre-Python manifest duplicated a proof role")
                roles[role] = {"path": path, "sha256": digest}
        elif len(fields) == 5 and fields[0] == "stage":
            _kind, backend, name, path, digest = fields
            if backend not in stages or name in stages[backend]:
                raise RuntimeError("pre-Python manifest stage table changed")
            stages[backend][name] = {"path": path, "sha256": digest}
        else:
            raise RuntimeError("pre-Python manifest row changed")
    return roles, stages


def authenticate_static_launcher(repo_root: Path, backend: str) -> dict[str, Any]:
    """Authenticate the live static parent and every inherited pre-Python inode."""
    repo_root = Path(repo_root).resolve()
    if backend not in {"training", "vllm"}:
        raise RuntimeError("runtime launcher backend is invalid")
    stage = os.environ.get("SME_RUNTIME_STAGE", "")
    if os.environ.get("SME_RUNTIME_BACKEND") != backend or not stage:
        raise RuntimeError("runtime launcher backend/stage environment is absent")
    expected_environment = _launcher_environment(backend, stage)
    observed_environment = dict(os.environ)
    selector = observed_environment.pop("CUDA_VISIBLE_DEVICES", None)
    if selector is not None and re.fullmatch(r"GPU-[A-Za-z0-9-]+", selector) is None:
        raise RuntimeError("runtime launcher supplied an invalid physical GPU UUID")
    if observed_environment != expected_environment:
        raise RuntimeError("runtime launcher replacement environment differs from its pin")
    if sys.argv[0] != f"/proc/self/fd/{_PREAUTH_STAGE_FD}":
        raise RuntimeError("runtime stage was not entered through its proof descriptor")
    launcher = (
        repo_root
        / "experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts"
        / f"{backend}_launcher"
    )
    parent_descriptor: int | None = None
    try:
        parent_pid = os.getppid()
        if parent_pid <= 1:
            raise RuntimeError("static runtime-launcher parent is absent")
        parent_descriptor = os.open(f"/proc/{parent_pid}/exe", os.O_RDONLY)
        parent_before = os.fstat(parent_descriptor)
        proof = _descriptor_authentication(_STATIC_LAUNCHER_PROOF_FD)
        path_stat = os.stat(launcher, follow_symlinks=False)
        if (
            launcher.is_symlink()
            or not stat.S_ISREG(parent_before.st_mode)
            or (parent_before.st_dev, parent_before.st_ino)
            != (proof["device"], proof["inode"])
            or (path_stat.st_dev, path_stat.st_ino)
            != (proof["device"], proof["inode"])
        ):
            raise RuntimeError("runtime-launcher proof names different executable bytes")
        parent_after = os.fstat(parent_descriptor)
    except OSError as error:
        raise RuntimeError(
            "runtime stage requires a live pinned static-launcher parent"
        ) from error
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    if any(
        getattr(parent_before, field) != getattr(parent_after, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    ):
        raise RuntimeError("runtime-launcher parent changed during authentication")
    descriptors = {
        "git": _descriptor_authentication(_PREAUTH_GIT_FD),
        "loader": _descriptor_authentication(_PREAUTH_LOADER_FD),
        "manifest": _descriptor_authentication(_PREAUTH_MANIFEST_FD),
        "runtime_contract": _descriptor_authentication(_PREAUTH_RUNTIME_CONTRACT_FD),
        "load_window_guard": _descriptor_authentication(_PREAUTH_LOAD_GUARD_FD),
        "stage": _descriptor_authentication(_PREAUTH_STAGE_FD),
        "interpreter": _descriptor_authentication(_PREAUTH_INTERPRETER_FD),
    }
    roles, stages = _manifest_rows(_PREAUTH_MANIFEST_FD)
    selected = stages.get(backend, {}).get(stage)
    required_roles = {"git", "loader", "runtime_contract", "load_window_guard", "interpreter"}
    if (
        set(roles) != required_roles
        or selected is None
        or any(
            descriptors[name]["path"] != roles[name]["path"]
            or descriptors[name]["sha256"] != roles[name]["sha256"]
            for name in required_roles
        )
        or descriptors["stage"]["path"] != selected["path"]
        or descriptors["stage"]["sha256"] != selected["sha256"]
        or descriptors["runtime_contract"]["path"] != str(Path(__file__).resolve())
        or descriptors["load_window_guard"]["path"]
        != str(Path(__file__).resolve().parent / "load_window_guard.py")
        or Path(sys.executable) != _SNAPSHOT_ROOT / "bin/python3.12"
        or Path(sys.executable).resolve() != Path(descriptors["interpreter"]["path"])
    ):
        raise RuntimeError("pre-Python proof descriptors differ from their manifest")
    return {
        "backend": backend,
        "stage": stage,
        "path": str(launcher),
        "sha256": proof["sha256"],
        "parent_pid": parent_pid,
        "proof_fd": _STATIC_LAUNCHER_PROOF_FD,
        "cuda_visible_devices": selector,
        "environment_sha256": hashlib.sha256(
            json.dumps(dict(os.environ), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "pre_python_descriptors": descriptors,
    }


def _run_preauthenticated_git(
    arguments: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run the launcher-authenticated Git inode through its authenticated loader."""
    fixed = [
        "-c", "core.fsmonitor=false",
        "-c", "core.untrackedCache=false",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.pager=cat",
    ]
    return subprocess.run(
        [
            f"/proc/self/fd/{_PREAUTH_LOADER_FD}",
            "--library-path", _SNAPSHOT_LIBRARY_PATH,
            "--argv0", "/workspace/sme-reflection-runtime/tools/git",
            f"/proc/self/fd/{_PREAUTH_GIT_FD}",
            *fixed,
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
        pass_fds=(_PREAUTH_GIT_FD, _PREAUTH_LOADER_FD),
    )


def _pinned_executable(name: str) -> tuple[Path, str]:
    pin = _system_pin()["executables"].get(name)
    if not isinstance(pin, dict):
        raise ValueError(f"unknown pinned executable: {name}")
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"pinned executable is absent or not regular: {name}")
    return path, pin["sha256"]


def run_pinned_executable(
    name: str,
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Hash, lease, and execute one exact inode through an inherited descriptor."""
    path, expected_sha256 = _pinned_executable(name)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    previous_handler = signal.getsignal(signal.SIGIO)
    break_signals = 0

    def saw_sigio(signum: int, frame: Any) -> None:
        nonlocal break_signals
        break_signals += 1
        if callable(previous_handler):
            previous_handler(signum, frame)

    leased = False
    try:
        before_stat = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
        before_sha256 = _sha256_fd(descriptor)
        if (
            not stat.S_ISREG(before_stat.st_mode)
            or (before_stat.st_dev, before_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or before_sha256 != expected_sha256
        ):
            raise ValueError(f"pinned executable differs before invocation: {name}")
        signal.signal(signal.SIGIO, saw_sigio)
        try:
            fcntl.fcntl(descriptor, fcntl.F_SETLEASE, fcntl.F_RDLCK)
            leased = True
        except OSError as error:
            raise RuntimeError(
                f"mandatory read lease denied for pinned executable: {name}"
            ) from error
        command = [f"/proc/self/fd/{descriptor}", *arguments]
        pass_descriptors = (descriptor,)
        if name != "uv":
            command = [
                f"/proc/self/fd/{_PREAUTH_LOADER_FD}",
                "--library-path",
                _SNAPSHOT_LIBRARY_PATH,
                "--argv0",
                str(path),
                f"/proc/self/fd/{descriptor}",
                *arguments,
            ]
            pass_descriptors = (descriptor, _PREAUTH_LOADER_FD)
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            pass_fds=pass_descriptors,
        )
        after_stat = os.fstat(descriptor)
        after_path_stat = os.stat(path, follow_symlinks=False)
        if (
            break_signals
            or any(
                getattr(before_stat, field) != getattr(after_stat, field)
                for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
            )
            or (after_stat.st_dev, after_stat.st_ino)
            != (after_path_stat.st_dev, after_path_stat.st_ino)
            or _sha256_fd(descriptor) != expected_sha256
        ):
            raise RuntimeError(f"pinned executable changed during invocation: {name}")
        return result
    finally:
        if leased:
            try:
                fcntl.fcntl(descriptor, fcntl.F_SETLEASE, fcntl.F_UNLCK)
            except OSError:
                pass
        signal.signal(signal.SIGIO, previous_handler)
        os.close(descriptor)


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _bootstrap_locked_versions(lock_path: Path, backend: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_path.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match:
            versions[_normalized_distribution_name(match.group(1))] = match.group(2)
    if backend == "vllm":
        versions["vllm"] = "0.24.0+cu129"
    elif backend == "training":
        versions.update(
            {
                "causal-conv1d": "1.6.2.post1",
                "iniconfig": "2.3.0",
                "pluggy": "1.6.0",
                "pytest": "9.1.1",
            }
        )
    return dict(sorted(versions.items()))


def authenticate_site_packages(
    site_packages: Path,
    expected_versions: dict[str, str],
    expected_record_surface_sha256: str,
    expected_site_surface_sha256: str,
) -> dict[str, Any]:
    """Verify RECORD contents and the complete importable site-packages file surface."""
    site_packages = site_packages.resolve()
    distributions = list(importlib.metadata.distributions(path=[str(site_packages)]))
    observed_versions: dict[str, str] = {}
    record_rows: list[tuple[str, str, str]] = []
    record_claims: dict[Path, list[tuple[str, str, str]]] = {}
    claim_count = 0
    verified_bytes = 0
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ValueError("installed distribution lacks a canonical name")
        name = _normalized_distribution_name(raw_name)
        if name in observed_versions:
            raise ValueError(f"duplicate installed distribution: {name}")
        observed_versions[name] = distribution.version
        files = list(distribution.files or ())
        record_files = [
            item
            for item in files
            if str(item).endswith(".dist-info/RECORD")
            and len(Path(str(item)).parts) == 2
        ]
        if len(record_files) != 1:
            raise ValueError(f"distribution lacks one exact RECORD file: {name}")
        record_path = Path(distribution.locate_file(record_files[0]))
        record_rows.append((name, distribution.version, _sha256_file(record_path)))
        for item in files:
            path = Path(distribution.locate_file(item))
            if item == record_files[0]:
                continue
            if item.hash is None or not path.is_file() or path.is_symlink():
                raise ValueError(f"distribution has unauthenticated installed content: {name}")
            record_claims.setdefault(path.resolve(), []).append(
                (name, item.hash.mode, item.hash.value)
            )
            claim_count += 1
    superseded_claims = 0
    for path, claims in record_claims.items():
        modes = {mode for _name, mode, _value in claims}
        digests = {mode: hashlib.new(mode) for mode in modes}
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                for digest in digests.values():
                    digest.update(block)
                verified_bytes += len(block)
        encoded = {
            mode: base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
            for mode, digest in digests.items()
        }
        matched = sum(encoded[mode] == value for _name, mode, value in claims)
        if not matched:
            raise ValueError(
                "installed distribution file differs from every RECORD claim: "
                f"{path}"
            )
        superseded_claims += len(claims) - matched
    if observed_versions != expected_versions:
        raise ValueError("installed distribution surface differs from the stage lock")
    aggregate = hashlib.sha256()
    for row in sorted(record_rows):
        aggregate.update(("\0".join(row) + "\n").encode())
    if aggregate.hexdigest() != expected_record_surface_sha256:
        raise ValueError("installed RECORD surface differs from the pinned environment")
    complete_site_files: dict[str, tuple[str, int]] = {}
    for path in site_packages.rglob("*"):
        if path.is_symlink():
            raise ValueError("site-packages contains an unauthenticated symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(site_packages).as_posix()
        complete_site_files[relative] = (_sha256_file(path), path.stat().st_size)
    site_surface = hashlib.sha256()
    for relative, (digest, size) in sorted(complete_site_files.items()):
        site_surface.update(f"{relative}\0{digest}\0{size}\n".encode())
    if site_surface.hexdigest() != expected_site_surface_sha256:
        raise ValueError(
            "complete site-packages file surface differs from its pin: "
            f"{site_surface.hexdigest()}"
        )
    return {
        "distribution_count": len(distributions),
        "record_claims": claim_count,
        "verified_files": len(record_claims),
        "verified_bytes": verified_bytes,
        "superseded_record_claims": superseded_claims,
        "record_surface_sha256": aggregate.hexdigest(),
        "site_files": len(complete_site_files),
        "site_surface_sha256": site_surface.hexdigest(),
        "packages_sha256": hashlib.sha256(
            json.dumps(
                observed_versions, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def authenticate_tree(root: Path, expected_surface_sha256: str) -> dict[str, Any]:
    """Authenticate regular files plus exact symlink targets under one runtime tree."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("authenticated runtime tree is absent")
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = path.resolve(strict=True)
            if target.is_file():
                size = target.stat().st_size
                rows.append(
                    {
                        "path": relative,
                        "kind": "file_symlink",
                        "target": str(target),
                        "target_sha256": _sha256_file(target),
                        "target_size": size,
                    }
                )
                total_bytes += size
            elif target.is_dir():
                rows.append(
                    {
                        "path": relative,
                        "kind": "directory_symlink",
                        "target": str(target),
                    }
                )
            else:
                raise ValueError("runtime tree symlink has an unsupported target")
        elif path.is_file():
            size = path.stat().st_size
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": _sha256_file(path),
                    "size": size,
                }
            )
            total_bytes += size
        elif not path.is_dir():
            raise ValueError("runtime tree contains an unsupported filesystem entry")
    surface = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if surface != expected_surface_sha256:
        raise ValueError(f"runtime tree differs from its pin: {surface}")
    return {"entries": len(rows), "bytes": total_bytes, "surface_sha256": surface}


def _native_mapping_authentication() -> dict[str, str]:
    mappings = {
        fields[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if (fields := line.split()) and fields[-1].startswith("/")
    }
    return {path: _sha256_file(Path(path)) for path in sorted(mappings)}


def _authenticate_system_runtime() -> dict[str, Any]:
    pin = _system_pin()
    resolved_python = Path(sys.executable).resolve()
    if (
        str(resolved_python) != pin["resolved_python"]
        or _sha256_file(resolved_python) != pin["resolved_python_sha256"]
        or os.environ.get("LD_LIBRARY_PATH", "") != pin["ld_library_path"]
        or any(os.environ.get(name) for name in ("LD_PRELOAD", "LD_AUDIT"))
    ):
        raise ValueError("interpreter or pre-Python native environment differs from its pin")
    mappings = _native_mapping_authentication()
    if mappings != pin["native_mappings"]:
        expected_mappings = pin["native_mappings"]
        missing = sorted(set(expected_mappings) - set(mappings))
        extra = sorted(set(mappings) - set(expected_mappings))
        changed = sorted(
            path
            for path in set(mappings) & set(expected_mappings)
            if mappings[path] != expected_mappings[path]
        )
        raise ValueError(
            "initial native mapping closure differs from its pin: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "resolved_python": str(resolved_python),
        "resolved_python_sha256": _sha256_file(resolved_python),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "native_mappings": mappings,
        "native_mappings_sha256": hashlib.sha256(
            json.dumps(mappings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _authenticate_loaded_native_closure(
    site_packages: Path, stdlib_root: Path, native_roots: list[Path]
) -> dict[str, Any]:
    mappings = _native_mapping_authentication()
    initial_mappings = _system_pin()["native_mappings"]
    allowed_roots = [
        site_packages.resolve(),
        stdlib_root.resolve(),
        *(root.resolve() for root in native_roots),
    ]
    interpreter = Path(_system_pin()["resolved_python"]).resolve()
    unexpected = [
        path
        for path in mappings
        if Path(path).resolve() != interpreter
        and not any(
            root == Path(path).resolve() or root in Path(path).resolve().parents
            for root in allowed_roots
        )
    ]
    if unexpected:
        raise ValueError(
            "loaded native mapping escaped authenticated roots: "
            + ", ".join(sorted(unexpected)[:5])
        )
    if any(mappings.get(path) != digest for path, digest in initial_mappings.items()):
        raise ValueError("loaded native mapping closure omits the pinned initial closure")
    return {
        "mappings": mappings,
        "mappings_sha256": hashlib.sha256(
            json.dumps(mappings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _initial_import_path_allowed(value: str) -> bool:
    source_root = str(Path(__file__).resolve().parent)
    snapshot_stdlib = str(_SNAPSHOT_ROOT / f"lib/python{sys.version_info.major}.{sys.version_info.minor}")
    return (
        value == source_root
        or value == str(_SNAPSHOT_ROOT / f"lib/python{sys.version_info.major}{sys.version_info.minor}.zip")
        or value == snapshot_stdlib
        or value.startswith(f"{snapshot_stdlib}/")
    )


def _validate_pinned_launcher_authentication(
    repo_root: Path, backend: str, authentication: dict[str, Any]
) -> None:
    system_pin = _system_pin()
    launcher_pin = system_pin["launchers"][backend]
    manifest_pin = system_pin["pre_python_manifest"]
    stage = authentication["stage"]
    stage_pin = system_pin["stage_entrypoints"][backend].get(stage)
    descriptors = authentication["pre_python_descriptors"]
    if (
        authentication["path"] != str(repo_root / launcher_pin["path"])
        or authentication["sha256"] != launcher_pin["sha256"]
        or descriptors["manifest"]["path"] != str(repo_root / manifest_pin["path"])
        or descriptors["manifest"]["sha256"] != manifest_pin["sha256"]
        or stage_pin is None
        or descriptors["stage"]["path"] != str(repo_root / stage_pin["path"])
        or descriptors["stage"]["sha256"] != stage_pin["sha256"]
    ):
        raise ValueError("live pre-Python launcher proof differs from the committed pin")


def bootstrap_runtime_environment(repo_root: Path, backend: str) -> dict[str, Any]:
    """Authenticate and guard stdlib/site bytes before enabling third-party imports."""
    global _BOOTSTRAP_RECEIPT, _RUNTIME_IMPORT_CONTENT, _RUNTIME_IMPORT_GUARD
    global _RUNTIME_PREFLIGHT_CONTENT, _RUNTIME_PREFLIGHT_GUARD
    if _BOOTSTRAP_RECEIPT is not None:
        if _BOOTSTRAP_RECEIPT.get("backend") != backend:
            raise ValueError("process already bootstrapped for a different runtime backend")
        return dict(_BOOTSTRAP_RECEIPT)
    repo_root = repo_root.resolve()
    launcher_authentication = authenticate_static_launcher(repo_root, backend)
    from load_window_guard import LoadWindowGuard, root_commitments

    experiment = repo_root / "experiments/qwen35_4b_counterfactual_plan_reflection_transfer"
    preflight_roots = [experiment / "src", experiment / "scripts", experiment / "configs"]
    preflight_content = {"worktree_code_surface": root_commitments(preflight_roots)}
    preflight_guard = LoadWindowGuard(
        preflight_roots,
        expected_content=preflight_content,
    )
    preflight_guard.__enter__()
    try:
        worktree = require_detached_execution_worktree(repo_root)
        preflight_after = {"worktree_code_surface": root_commitments(preflight_roots)}
        if preflight_after != preflight_content:
            raise RuntimeError("worktree code surface changed during Git preflight")
        pin = _runtime_pin(backend)
        system_pin = _system_pin()
        _validate_pinned_launcher_authentication(
            repo_root, backend, launcher_authentication
        )
        system_authentication = _authenticate_system_runtime()
    except BaseException as error:
        preflight_guard.__exit__(type(error), error, error.__traceback__)
        raise
    environment_root = Path(pin["environment_root"])
    bin_root = environment_root / "bin"
    invoked_python = _SNAPSHOT_ROOT / "bin/python3.12"
    if Path(sys.executable) != invoked_python or not invoked_python.is_file():
        raise ValueError("stage used the wrong pinned runtime interpreter path")
    site_packages = (
        environment_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if (
        not bin_root.is_dir()
        or not site_packages.is_dir()
        or repo_root.resolve() in site_packages.resolve().parents
    ):
        raise ValueError("pinned runtime import root is absent or inside the worktree")
    stdlib_root = Path(system_pin["stdlib_root"])
    if not stdlib_root.is_dir():
        raise ValueError("pinned standard-library root is absent")
    if not sys.path or any(not _initial_import_path_allowed(item) for item in sys.path):
        raise ValueError("unapproved import roots were active before runtime authentication")
    startup_paths = {
        path.name: path
        for pattern in ("*.pth", "sitecustomize.py", "usercustomize.py")
        for path in site_packages.glob(pattern)
    }
    observed_startup = {
        name: _sha256_file(path) for name, path in sorted(startup_paths.items())
    }
    if observed_startup != pin["startup_files"]:
        raise ValueError("pinned runtime startup-file surface changed")
    lock_path = repo_root / pin["lock_file"]
    if not lock_path.is_file():
        raise ValueError("pinned stage-specific runtime lock is absent")
    expected_versions = _bootstrap_locked_versions(lock_path, backend)
    native_roots = [Path(path) for path in system_pin["native_library_roots"]]
    expected_content = {
        "record_surface_sha256": pin["record_surface_sha256"],
        "site_surface_sha256": pin["site_surface_sha256"],
        "packages_sha256": hashlib.sha256(
            json.dumps(
                expected_versions, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "stdlib_surface_sha256": system_pin["stdlib_surface_sha256"],
        "bin_surface_sha256": pin["bin_surface_sha256"],
        "native_library_surfaces": system_pin["native_library_roots"],
    }
    import_guard = LoadWindowGuard(
        [bin_root, site_packages, stdlib_root, *native_roots],
        expected_content={"runtime_environment": expected_content},
    )
    try:
        import_guard.__enter__()
    except BaseException as error:
        preflight_guard.__exit__(type(error), error, error.__traceback__)
        raise
    before = list(sys.path)
    try:
        environment_authentication = authenticate_site_packages(
            site_packages,
            expected_versions,
            pin["record_surface_sha256"],
            pin["site_surface_sha256"],
        )
        bin_authentication = authenticate_tree(bin_root, pin["bin_surface_sha256"])
        stdlib_authentication = authenticate_tree(
            stdlib_root, system_pin["stdlib_surface_sha256"]
        )
        native_library_authentication = {
            str(root): authenticate_tree(
                root, system_pin["native_library_roots"][str(root)]
            )
            for root in native_roots
        }
        content = {
            "record_surface_sha256": environment_authentication[
                "record_surface_sha256"
            ],
            "site_surface_sha256": environment_authentication[
                "site_surface_sha256"
            ],
            "packages_sha256": environment_authentication["packages_sha256"],
            "bin_surface_sha256": bin_authentication["surface_sha256"],
            "stdlib_surface_sha256": stdlib_authentication["surface_sha256"],
            "native_library_surfaces": {
                path: authentication["surface_sha256"]
                for path, authentication in native_library_authentication.items()
            },
        }
        if content != expected_content:
            raise ValueError("runtime environment content differs from its committed pin")
        activated_extensions: list[str] = []
        sys.path.append(str(site_packages.resolve()))
        for relative in pin["path_extensions"]:
            extension = (site_packages / relative).resolve()
            if (
                not extension.is_dir()
                or extension.is_symlink()
                or site_packages.resolve() not in extension.parents
            ):
                raise ValueError("pinned site-packages path extension is invalid")
            sys.path.append(str(extension))
            activated_extensions.append(str(extension))
    except BaseException as error:
        import_guard.__exit__(type(error), error, error.__traceback__)
        preflight_guard.__exit__(type(error), error, error.__traceback__)
        raise
    _RUNTIME_IMPORT_GUARD = import_guard
    _RUNTIME_IMPORT_CONTENT = {"runtime_environment": content}
    _RUNTIME_PREFLIGHT_GUARD = preflight_guard
    _RUNTIME_PREFLIGHT_CONTENT = preflight_content
    _BOOTSTRAP_RECEIPT = {
        "schema_version": 4,
        "backend": backend,
        "launcher_authentication": launcher_authentication,
        "worktree": worktree,
        "environment_root": str(environment_root),
        "bin_root": str(bin_root.resolve()),
        "bin_authentication": bin_authentication,
        "invoked_python": str(invoked_python),
        "resolved_python": str(invoked_python.resolve()),
        "resolved_python_sha256": _sha256_file(invoked_python.resolve()),
        "system_authentication": system_authentication,
        "stdlib_root": str(stdlib_root.resolve()),
        "stdlib_authentication": stdlib_authentication,
        "native_library_authentication": native_library_authentication,
        "site_packages": str(site_packages.resolve()),
        "initial_sys_path": before,
        "activated_path_extensions": activated_extensions,
        "startup_files": observed_startup,
        "environment_authentication": environment_authentication,
        "lock_file": pin["lock_file"],
        "lock_sha256": _sha256_file(lock_path),
        "python_isolated": True,
        "python_dont_write_bytecode": True,
        "python_no_site": True,
    }
    return dict(_BOOTSTRAP_RECEIPT)


def seal_runtime_environment(repo_root: Path, expected_backend: str) -> dict[str, Any]:
    """Reauthenticate and close the immutable environment import window."""
    global _RUNTIME_IMPORT_CONTENT, _RUNTIME_IMPORT_GUARD
    global _RUNTIME_PREFLIGHT_CONTENT, _RUNTIME_PREFLIGHT_GUARD
    if _BOOTSTRAP_RECEIPT is None or _BOOTSTRAP_RECEIPT.get("backend") != expected_backend:
        raise ValueError("runtime bootstrap is absent or belongs to another backend")
    if _RUNTIME_IMPORT_GUARD is None:
        if "import_window_guard" not in _BOOTSTRAP_RECEIPT:
            raise ValueError("runtime import guard disappeared before sealing")
        return dict(_BOOTSTRAP_RECEIPT)
    if _RUNTIME_IMPORT_CONTENT is None:
        raise ValueError("runtime import content commitment is absent")
    if _RUNTIME_PREFLIGHT_GUARD is None or _RUNTIME_PREFLIGHT_CONTENT is None:
        raise ValueError("runtime preflight guard disappeared before sealing")
    pin = _runtime_pin(expected_backend)
    system_pin = _system_pin()
    environment_root = Path(pin["environment_root"])
    bin_root = environment_root / "bin"
    site_packages = (
        environment_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    stdlib_root = Path(system_pin["stdlib_root"])
    native_roots = [Path(path) for path in system_pin["native_library_roots"]]
    lock_path = repo_root.resolve() / pin["lock_file"]
    try:
        environment_authentication = authenticate_site_packages(
            site_packages,
            _bootstrap_locked_versions(lock_path, expected_backend),
            pin["record_surface_sha256"],
            pin["site_surface_sha256"],
        )
        bin_authentication = authenticate_tree(bin_root, pin["bin_surface_sha256"])
        stdlib_authentication = authenticate_tree(
            stdlib_root, system_pin["stdlib_surface_sha256"]
        )
        native_library_authentication = {
            str(root): authenticate_tree(
                root, system_pin["native_library_roots"][str(root)]
            )
            for root in native_roots
        }
        loaded_native_closure = _authenticate_loaded_native_closure(
            site_packages, stdlib_root, native_roots
        )
        after_content = {
            "runtime_environment": {
                "record_surface_sha256": environment_authentication[
                    "record_surface_sha256"
                ],
                "site_surface_sha256": environment_authentication[
                    "site_surface_sha256"
                ],
                "packages_sha256": environment_authentication["packages_sha256"],
                "bin_surface_sha256": bin_authentication["surface_sha256"],
                "stdlib_surface_sha256": stdlib_authentication["surface_sha256"],
                "native_library_surfaces": {
                    path: authentication["surface_sha256"]
                    for path, authentication in native_library_authentication.items()
                },
            }
        }
        _RUNTIME_IMPORT_GUARD.bind_authenticated_content(
            _RUNTIME_IMPORT_CONTENT, after_content
        )
        guard_receipt = _RUNTIME_IMPORT_GUARD.verify()
        experiment = repo_root.resolve() / "experiments/qwen35_4b_counterfactual_plan_reflection_transfer"
        from load_window_guard import root_commitments

        preflight_roots = [
            experiment / "src", experiment / "scripts", experiment / "configs"
        ]
        preflight_after = {"worktree_code_surface": root_commitments(preflight_roots)}
        _RUNTIME_PREFLIGHT_GUARD.bind_authenticated_content(
            _RUNTIME_PREFLIGHT_CONTENT, preflight_after
        )
        preflight_guard_receipt = _RUNTIME_PREFLIGHT_GUARD.verify()
    except BaseException as error:
        _RUNTIME_IMPORT_GUARD.__exit__(type(error), error, error.__traceback__)
        if _RUNTIME_PREFLIGHT_GUARD is not None:
            _RUNTIME_PREFLIGHT_GUARD.__exit__(type(error), error, error.__traceback__)
        raise
    _BOOTSTRAP_RECEIPT["post_import_environment_authentication"] = (
        environment_authentication
    )
    _BOOTSTRAP_RECEIPT["post_import_bin_authentication"] = bin_authentication
    _BOOTSTRAP_RECEIPT["post_import_stdlib_authentication"] = stdlib_authentication
    _BOOTSTRAP_RECEIPT["post_import_native_library_authentication"] = (
        native_library_authentication
    )
    _BOOTSTRAP_RECEIPT["post_import_loaded_native_closure"] = loaded_native_closure
    _BOOTSTRAP_RECEIPT["import_window_guard"] = guard_receipt
    _BOOTSTRAP_RECEIPT["preflight_window_guard"] = preflight_guard_receipt
    _RUNTIME_IMPORT_GUARD = None
    _RUNTIME_IMPORT_CONTENT = None
    _RUNTIME_PREFLIGHT_GUARD = None
    _RUNTIME_PREFLIGHT_CONTENT = None
    return dict(_BOOTSTRAP_RECEIPT)


def runtime_bootstrap_receipt(expected_backend: str) -> dict[str, Any]:
    if (
        _BOOTSTRAP_RECEIPT is None
        or _BOOTSTRAP_RECEIPT.get("backend") != expected_backend
        or _RUNTIME_IMPORT_GUARD is not None
        or _RUNTIME_PREFLIGHT_GUARD is not None
        or "import_window_guard" not in _BOOTSTRAP_RECEIPT
        or "preflight_window_guard" not in _BOOTSTRAP_RECEIPT
    ):
        raise ValueError("runtime import window was not sealed")
    return dict(_BOOTSTRAP_RECEIPT)


def installed_packages() -> dict[str, str]:
    values: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            normalized = name.lower().replace("_", "-")
            values[normalized] = importlib.metadata.version(name)
    return dict(sorted(values.items()))


def selected_gpu_identity(repo_root: Path) -> dict[str, Any]:
    """Bind one UUID selector to the exact physical GPU row used by the process."""
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selector or "," in selector or not selector.startswith("GPU-"):
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU UUID")
    query = run_pinned_executable(
        "nvidia_smi",
        [
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/workspace/sme-reflection-runtime/tools",
            "LD_LIBRARY_PATH": _system_pin()["ld_library_path"],
        },
    )
    if query.returncode != 0:
        raise ValueError("selected GPU inventory query failed")
    matches: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError("selected GPU inventory schema changed")
        index, name, uuid, driver, memory = fields
        if uuid == selector:
            try:
                matches.append(
                    {
                        "cuda_visible_devices": selector,
                        "physical_index": int(index),
                        "name": name,
                        "uuid": uuid,
                        "driver_version": driver,
                        "memory_total_mib": int(memory),
                    }
                )
            except ValueError as error:
                raise ValueError("selected GPU inventory has invalid numeric fields") from error
    if len(matches) != 1:
        raise ValueError("CUDA_VISIBLE_DEVICES does not resolve to one exact GPU row")
    return matches[0]


def _normalized_cuda_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        if len(value) != 16:
            raise ValueError("active CUDA UUID byte width changed")
        encoded = value.hex()
        value = (
            f"{encoded[:8]}-{encoded[8:12]}-{encoded[12:16]}-"
            f"{encoded[16:20]}-{encoded[20:]}"
        )
    text = str(value)
    if not text.startswith("GPU-"):
        text = f"GPU-{text}"
    if re.fullmatch(r"GPU-[A-Za-z0-9-]+", text) is None:
        raise ValueError("active CUDA UUID schema changed")
    return text


def bind_active_cuda_identity(
    repo_root: Path, torch_module: Any
) -> dict[str, Any]:
    """Bind the selected physical UUID to CUDA's sole active logical device."""
    selected = selected_gpu_identity(repo_root)
    cuda = torch_module.cuda
    if not cuda.is_initialized():
        raise ValueError("CUDA runtime was not initialized before device authentication")
    visible_count = int(cuda.device_count())
    logical_index = int(cuda.current_device())
    if visible_count != 1 or logical_index != 0:
        raise ValueError("runtime does not expose exactly one active logical CUDA device")
    properties = cuda.get_device_properties(logical_index)
    active_name = str(properties.name)
    active_memory_mib = int(properties.total_memory) // (1024 * 1024)
    active_uuid = _normalized_cuda_uuid(getattr(properties, "uuid", None))
    if (
        active_name != selected["name"]
        or active_memory_mib != selected["memory_total_mib"]
        or active_uuid.lower() != str(selected["uuid"]).lower()
    ):
        raise ValueError("active CUDA device differs from the selected physical GPU row")
    return {
        **selected,
        "active_logical_index": logical_index,
        "active_visible_device_count": visible_count,
        "active_name": active_name,
        "active_memory_total_mib": active_memory_mib,
        "active_uuid": active_uuid,
    }


def runtime_metadata(
    repo_root: Path, lock_path: Path, gpu_identity: dict[str, Any]
) -> dict[str, Any]:
    """Record the complete installed/runtime/hardware identity for training."""
    worktree = require_detached_execution_worktree(repo_root)
    nvcc = run_pinned_executable("nvcc", ["--version"], cwd=repo_root)
    packages = installed_packages()
    python_version = platform.python_version()
    platform_value = platform.platform()
    bootstrap = seal_runtime_environment(repo_root, "training")
    value = {
        "schema_version": 4,
        "bootstrap": bootstrap,
        "worktree": worktree,
        "python": python_version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
        "python_isolated": sys.flags.isolated == 1,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "python_no_site": sys.flags.no_site == 1,
        "platform": platform_value,
        "packages": packages,
        "packages_sha256": hashlib.sha256(
            json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "environment_lock": {
            "path": str(lock_path.resolve()),
            "sha256": _sha256_file(lock_path),
        },
        "gpu": dict(gpu_identity),
        "cuda_toolkit": nvcc.stdout.strip() if nvcc.returncode == 0 else "",
    }
    return value
