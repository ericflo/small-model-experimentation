from __future__ import annotations

import json
import os
import base64
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import runtime_contract as R  # noqa: E402
import tokenizer_lineage as T  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require(root: Path, *, executable: Path | None = None) -> dict[str, str]:
    original_flags = sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1

        def __getattr__(self, name: str):
            return getattr(original_flags, name)

    with mock.patch.object(
        R.sys, "flags", IsolatedFlags()
    ), mock.patch.object(
        R.sys, "dont_write_bytecode", True
    ), mock.patch.object(
        R.sys, "executable", str(executable or Path(sys.executable).resolve())
    ):
        return R.require_detached_execution_worktree(root)


class RuntimeContractTests(unittest.TestCase):
    def test_static_launchers_rebuild_exactly_and_direct_entry_is_rejected(self) -> None:
        pin = json.loads(R.RUNTIME_PIN.read_text())
        source = EXP / "scripts" / "runtime_launcher.S"
        with tempfile.TemporaryDirectory() as temporary:
            for backend, define in (("training", "TRAINING"), ("vllm", "VLLM")):
                output = Path(temporary) / f"{backend}_launcher"
                subprocess.run(
                    [
                        "/usr/bin/gcc", "-nostdlib", "-static", "-no-pie", "-s",
                        "-Wl,--build-id=none", f"-D{define}=1", "-o", str(output),
                        str(source),
                    ],
                    check=True,
                )
                self.assertEqual(
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                    pin["system"]["launchers"][backend]["sha256"],
                )
        environment = R._launcher_environment("training")
        with mock.patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
            RuntimeError, "live pinned static-launcher parent"
        ):
            R.authenticate_static_launcher(EXP.parents[1], "training")

    def test_pinned_git_execution_ignores_path_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "shadow-ran"
            shadow = root / "git"
            shadow.write_text(f"#!/bin/sh\ntouch {marker}\nexit 99\n")
            shadow.chmod(0o755)
            result = R.run_pinned_executable(
                "git", ["--version"], cwd=EXP.parents[1], env={"PATH": str(root)}
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("git version", result.stdout)
            self.assertFalse(marker.exists())

    def test_loaded_native_closure_requires_the_initial_mapping_set(self) -> None:
        system_pin = {
            "resolved_python": "/synthetic/python",
            "native_mappings": {"/synthetic/python": "f" * 64},
        }
        with mock.patch.object(R, "_native_mapping_authentication", return_value={}), mock.patch.object(
            R, "_system_pin", return_value=system_pin
        ), self.assertRaisesRegex(ValueError, "omits the pinned initial closure"):
            R._authenticate_loaded_native_closure(
                Path("/synthetic/site"),
                Path("/synthetic/stdlib"),
                [Path("/synthetic/native")],
            )

    def test_site_package_record_bytes_are_authenticated_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            module = site / "demo.py"
            module.write_text("VALUE = 1\n")
            info = site / "demo-1.0.dist-info"
            info.mkdir()
            metadata = info / "METADATA"
            metadata.write_text("Name: demo\nVersion: 1.0\n")

            def record_line(path: Path) -> str:
                digest = hashlib.sha256(path.read_bytes()).digest()
                encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                return f"{path.relative_to(site).as_posix()},sha256={encoded},{path.stat().st_size}"

            record = info / "RECORD"
            record.write_text(
                "\n".join(
                    [
                        record_line(module),
                        record_line(metadata),
                        "demo-1.0.dist-info/RECORD,,",
                    ]
                )
                + "\n"
            )
            surface = hashlib.sha256()
            surface.update(
                (
                    "demo\0"
                    "1.0\0"
                    f"{hashlib.sha256(record.read_bytes()).hexdigest()}\n"
                ).encode()
            )
            site_surface = hashlib.sha256()
            for path in sorted((module, metadata, record)):
                relative = path.relative_to(site).as_posix()
                site_surface.update(
                    f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\0"
                    f"{path.stat().st_size}\n".encode()
                )
            receipt = R.authenticate_site_packages(
                site, {"demo": "1.0"}, surface.hexdigest(), site_surface.hexdigest()
            )
            self.assertEqual(receipt["record_claims"], 2)
            self.assertEqual(receipt["verified_files"], 2)
            self.assertEqual(receipt["superseded_record_claims"], 0)
            self.assertEqual(receipt["site_files"], 3)
            module.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "differs from every RECORD"):
                R.authenticate_site_packages(
                    site,
                    {"demo": "1.0"},
                    surface.hexdigest(),
                    site_surface.hexdigest(),
                )
            module.write_text("VALUE = 1\n")
            (site / "shadow_module.py").write_text("VALUE = 'injected'\n")
            with self.assertRaisesRegex(ValueError, "complete site-packages"):
                R.authenticate_site_packages(
                    site,
                    {"demo": "1.0"},
                    surface.hexdigest(),
                    site_surface.hexdigest(),
                )

    def test_no_site_mode_keeps_pth_startup_code_inert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "executed"
            (root / "injected.pth").write_text(
                f"import pathlib; pathlib.Path({str(marker)!r}).write_text('bad')\n"
            )
            code = (
                "import sys; "
                f"sys.path.append({str(root)!r}); "
                f"print(int(__import__('pathlib').Path({str(marker)!r}).exists()))"
            )
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", code],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "0")
            self.assertFalse(marker.exists())

    def test_selected_gpu_requires_one_exact_uuid_not_host_inventory(self) -> None:
        inventory = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "0, GPU A, GPU-AAAA, 999.0, 80000\n"
                "1, GPU B, GPU-BBBB, 999.0, 40000\n"
            ),
            stderr="",
        )
        with mock.patch.object(
            R, "run_pinned_executable", return_value=inventory
        ) as run, mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-BBBB"}, clear=False
        ):
            identity = R.selected_gpu_identity(Path("/synthetic"))
        self.assertEqual(identity["uuid"], "GPU-BBBB")
        self.assertEqual(identity["physical_index"], 1)
        self.assertEqual(run.call_args.args[0], "nvidia_smi")
        self.assertEqual(run.call_args.args[1][0], "--query-gpu=index,name,uuid,driver_version,memory.total")
        self.assertEqual(run.call_args.kwargs["env"]["PATH"], "/usr/bin:/bin")
        with mock.patch.object(
            R, "run_pinned_executable", return_value=inventory
        ), mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False
        ), self.assertRaisesRegex(ValueError, "physical GPU UUID"):
            R.selected_gpu_identity(Path("/synthetic"))

    def test_active_cuda_device_is_bound_to_selected_uuid_row(self) -> None:
        selected = {
            "cuda_visible_devices": "GPU-BBBB",
            "physical_index": 1,
            "name": "Synthetic GPU",
            "uuid": "GPU-BBBB",
            "driver_version": "999.0",
            "memory_total_mib": 80000,
        }

        class FakeCuda:
            @staticmethod
            def is_initialized() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def current_device() -> int:
                return 0

            @staticmethod
            def get_device_properties(_index: int):
                return type(
                    "Properties",
                    (),
                    {
                        "name": "Synthetic GPU",
                        "total_memory": 80000 * 1024 * 1024,
                        "uuid": "GPU-BBBB",
                    },
                )()

        torch_module = type("Torch", (), {"cuda": FakeCuda})()
        with mock.patch.object(R, "selected_gpu_identity", return_value=selected):
            identity = R.bind_active_cuda_identity(Path("/synthetic"), torch_module)
        self.assertEqual(identity["active_logical_index"], 0)
        self.assertEqual(identity["active_visible_device_count"], 1)
        self.assertEqual(identity["active_name"], identity["name"])
        self.assertEqual(
            identity["active_memory_total_mib"], identity["memory_total_mib"]
        )
        self.assertEqual(identity["active_uuid"], identity["uuid"])

        class WrongUuidCuda(FakeCuda):
            @staticmethod
            def get_device_properties(_index: int):
                return type(
                    "Properties",
                    (),
                    {
                        "name": "Synthetic GPU",
                        "total_memory": 80000 * 1024 * 1024,
                        "uuid": "GPU-AAAA",
                    },
                )()

        wrong_torch = type("Torch", (), {"cuda": WrongUuidCuda})()
        with mock.patch.object(R, "selected_gpu_identity", return_value=selected), self.assertRaisesRegex(
            ValueError, "active CUDA device differs"
        ):
            R.bind_active_cuda_identity(Path("/synthetic"), wrong_torch)

    def test_execution_requires_clean_detached_root_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            _git(root, "init", "-q")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "tracked.txt").write_text("v1\n")
            _git(root, "add", "tracked.txt")
            _git(root, "commit", "-qm", "initial")
            commit = _git(root, "rev-parse", "HEAD")

            with mock.patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "clean detached"):
                    _require(root)

            _git(root, "checkout", "--detach", "-q", commit)
            with mock.patch("pathlib.Path.cwd", return_value=root):
                self.assertEqual(
                    _require(root),
                    {
                        "repo_root": str(root.resolve()),
                        "git_commit": commit,
                        "head_mode": "detached",
                        "cwd": str(root.resolve()),
                    },
                )

            (root / ".gitignore").write_text("__pycache__/\n")
            _git(root, "add", ".gitignore")
            _git(root, "commit", "-qm", "ignore bytecode")
            ignored = root / "__pycache__"
            ignored.mkdir()
            (ignored / "injected.pyc").write_bytes(b"injected")
            with mock.patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "no ignored state"):
                    _require(root)
            (ignored / "injected.pyc").unlink()
            ignored.rmdir()

            fake_interpreter = root / "python"
            fake_interpreter.write_bytes(b"synthetic")
            with mock.patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "outside the worktree"):
                    _require(root, executable=fake_interpreter)
            fake_interpreter.unlink()

            (root / "tracked.txt").write_text("dirty\n")
            with mock.patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "clean detached"):
                    _require(root)
            with mock.patch("pathlib.Path.cwd", return_value=root.parent):
                with self.assertRaisesRegex(ValueError, "worktree root as cwd"):
                    _require(root)

    def test_tokenizer_authentication_is_exact_and_mutation_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "chat_template.jinja": b"template\n",
                "merges.txt": b"a b\n",
                "tokenizer.json": b"{}\n",
                "tokenizer_config.json": b"{}\n",
                "vocab.json": b"{}\n",
            }
            for name, payload in files.items():
                (root / name).write_bytes(payload)
            pin = {
                "schema_version": 2,
                "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "absent_files": ["added_tokens.json", "special_tokens_map.json"],
                "files": {
                    name: {
                        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                    for name, payload in files.items()
                },
            }
            with mock.patch.object(T, "load_pinned_tokenizer", return_value=pin):
                receipt = T.authenticate_tokenizer_snapshot(root)
                self.assertEqual(receipt["files"], pin["files"])
                self.assertEqual(len(receipt["files_sha256"]), 64)
                (root / "added_tokens.json").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "must be absent"):
                    T.authenticate_tokenizer_snapshot(root)
                (root / "added_tokens.json").unlink()
                closed = root / "closed"
                closed.mkdir()
                for name, payload in files.items():
                    (closed / name).write_bytes(payload)
                self.assertEqual(
                    T.authenticate_closed_tokenizer_view(closed), receipt
                )
                (closed / "special_tokens_map.json").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "missing or extra"):
                    T.authenticate_closed_tokenizer_view(closed)
                (root / "vocab.json").write_bytes(b"tampered\n")
                with self.assertRaisesRegex(ValueError, "differs from exact revision"):
                    T.authenticate_tokenizer_snapshot(root)

    def test_pinned_tokenizer_file_set_is_exact(self) -> None:
        pin = json.loads((EXP / "configs" / "pinned_tokenizer_structure.json").read_text())
        self.assertEqual(
            set(pin["files"]),
            {
                "chat_template.jinja",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            },
        )
        self.assertEqual(
            pin["absent_files"],
            ["added_tokens.json", "special_tokens_map.json"],
        )
        self.assertEqual(T.load_pinned_tokenizer(), pin)


if __name__ == "__main__":
    unittest.main()
