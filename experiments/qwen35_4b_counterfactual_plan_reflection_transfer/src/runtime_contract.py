"""Immutable detached-worktree and runtime provenance contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import base64
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNTIME_PIN = Path(__file__).resolve().parents[1] / "configs" / "pinned_runtime_environments.json"
_BOOTSTRAP_RECEIPT: dict[str, Any] | None = None
_RUNTIME_IMPORT_GUARD: Any | None = None
_RUNTIME_IMPORT_CONTENT: dict[str, Any] | None = None


def _run(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False, env=env
    )


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
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=repo_root)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        cwd=repo_root,
    )
    branch = _run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo_root)
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
        or value["schema_version"] != 2
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
        "nvidia_smi",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or not isinstance(value.get("native_mappings"), dict)
        or not value["native_mappings"]
        or not isinstance(value.get("native_library_roots"), dict)
        or not value["native_library_roots"]
        or not isinstance(value.get("nvidia_smi"), dict)
        or set(value["nvidia_smi"]) != {"path", "sha256"}
    ):
        raise ValueError("runtime system pin changed")
    return value


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
        raise ValueError("initial native mapping closure differs from its pin")
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
    return {
        "mappings": mappings,
        "mappings_sha256": hashlib.sha256(
            json.dumps(mappings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _initial_import_path_allowed(value: str) -> bool:
    source_root = str(Path(__file__).resolve().parent)
    return (
        value == source_root
        or value == f"/usr/lib/python{sys.version_info.major}{sys.version_info.minor}.zip"
        or value == f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}"
        or value.startswith(
            f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/"
        )
    )


def bootstrap_runtime_environment(repo_root: Path, backend: str) -> dict[str, Any]:
    """Authenticate and guard stdlib/site bytes before enabling third-party imports."""
    global _BOOTSTRAP_RECEIPT, _RUNTIME_IMPORT_CONTENT, _RUNTIME_IMPORT_GUARD
    if _BOOTSTRAP_RECEIPT is not None:
        if _BOOTSTRAP_RECEIPT.get("backend") != backend:
            raise ValueError("process already bootstrapped for a different runtime backend")
        return dict(_BOOTSTRAP_RECEIPT)
    worktree = require_detached_execution_worktree(repo_root)
    pin = _runtime_pin(backend)
    system_pin = _system_pin()
    system_authentication = _authenticate_system_runtime()
    environment_root = Path(pin["environment_root"])
    invoked_python = environment_root / "bin" / "python"
    if Path(sys.executable) != invoked_python or not invoked_python.is_file():
        raise ValueError("stage used the wrong pinned runtime interpreter path")
    site_packages = (
        environment_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not site_packages.is_dir() or repo_root.resolve() in site_packages.resolve().parents:
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
        "native_library_surfaces": system_pin["native_library_roots"],
    }
    from load_window_guard import LoadWindowGuard

    import_guard = LoadWindowGuard(
        [site_packages, stdlib_root, *native_roots],
        expected_content={"runtime_environment": expected_content},
        unleased_roots=[stdlib_root, *native_roots],
    )
    import_guard.__enter__()
    before = list(sys.path)
    try:
        environment_authentication = authenticate_site_packages(
            site_packages,
            expected_versions,
            pin["record_surface_sha256"],
            pin["site_surface_sha256"],
        )
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
        raise
    _RUNTIME_IMPORT_GUARD = import_guard
    _RUNTIME_IMPORT_CONTENT = {"runtime_environment": content}
    _BOOTSTRAP_RECEIPT = {
        "schema_version": 2,
        "backend": backend,
        "worktree": worktree,
        "environment_root": str(environment_root),
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
    if _BOOTSTRAP_RECEIPT is None or _BOOTSTRAP_RECEIPT.get("backend") != expected_backend:
        raise ValueError("runtime bootstrap is absent or belongs to another backend")
    if _RUNTIME_IMPORT_GUARD is None:
        if "import_window_guard" not in _BOOTSTRAP_RECEIPT:
            raise ValueError("runtime import guard disappeared before sealing")
        return dict(_BOOTSTRAP_RECEIPT)
    if _RUNTIME_IMPORT_CONTENT is None:
        raise ValueError("runtime import content commitment is absent")
    pin = _runtime_pin(expected_backend)
    system_pin = _system_pin()
    environment_root = Path(pin["environment_root"])
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
    except BaseException as error:
        _RUNTIME_IMPORT_GUARD.__exit__(type(error), error, error.__traceback__)
        raise
    _BOOTSTRAP_RECEIPT["post_import_environment_authentication"] = (
        environment_authentication
    )
    _BOOTSTRAP_RECEIPT["post_import_stdlib_authentication"] = stdlib_authentication
    _BOOTSTRAP_RECEIPT["post_import_native_library_authentication"] = (
        native_library_authentication
    )
    _BOOTSTRAP_RECEIPT["post_import_loaded_native_closure"] = loaded_native_closure
    _BOOTSTRAP_RECEIPT["import_window_guard"] = guard_receipt
    _RUNTIME_IMPORT_GUARD = None
    _RUNTIME_IMPORT_CONTENT = None
    return dict(_BOOTSTRAP_RECEIPT)


def runtime_bootstrap_receipt(expected_backend: str) -> dict[str, Any]:
    if (
        _BOOTSTRAP_RECEIPT is None
        or _BOOTSTRAP_RECEIPT.get("backend") != expected_backend
        or _RUNTIME_IMPORT_GUARD is not None
        or "import_window_guard" not in _BOOTSTRAP_RECEIPT
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
    executable_pin = _system_pin()["nvidia_smi"]
    executable = Path(executable_pin["path"])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or executable.is_symlink()
        or _sha256_file(executable) != executable_pin["sha256"]
    ):
        raise ValueError("selected GPU inventory executable differs from its pin")
    query = _run(
        [
            str(executable),
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
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
    if (
        active_name != selected["name"]
        or active_memory_mib != selected["memory_total_mib"]
    ):
        raise ValueError("active CUDA device differs from the selected physical GPU row")
    return {
        **selected,
        "active_logical_index": logical_index,
        "active_visible_device_count": visible_count,
        "active_name": active_name,
        "active_memory_total_mib": active_memory_mib,
    }


def runtime_metadata(
    repo_root: Path, lock_path: Path, gpu_identity: dict[str, Any]
) -> dict[str, Any]:
    """Record the complete installed/runtime/hardware identity for training."""
    worktree = require_detached_execution_worktree(repo_root)
    nvcc = _run(["nvcc", "--version"], cwd=repo_root)
    packages = installed_packages()
    value = {
        "schema_version": 4,
        "bootstrap": runtime_bootstrap_receipt("training"),
        "worktree": worktree,
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
        "python_isolated": sys.flags.isolated == 1,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "python_no_site": sys.flags.no_site == 1,
        "platform": platform.platform(),
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
