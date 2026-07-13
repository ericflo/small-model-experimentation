from __future__ import annotations

import contextlib
import errno
import gzip
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.safe_io as safe_io  # noqa: E402
from src.safe_io import (  # noqa: E402
    StableArtifactError,
    ensure_canonical_directory,
    fsync_canonical_directory,
    move_new_entry,
    open_stable_directory_for_update,
    open_stable_regular,
    open_stable_regular_for_update,
    publish_new_bytes,
    publish_new_file,
    read_verified_bytes,
    read_verified_json_object,
    read_verified_jsonl_gzip,
    rename_new_entry,
)


class StableArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def test_verified_bytes_are_read_from_the_hashed_open_inode(self) -> None:
        payload = self.root / "payload.bin"
        payload.write_bytes(b"registered")
        self.assertEqual(
            read_verified_bytes(self.root, payload, self.digest(b"registered")),
            b"registered",
        )
        with self.assertRaisesRegex(StableArtifactError, "bound SHA-256"):
            read_verified_bytes(self.root, payload, self.digest(b"different"))

    def test_symlink_leaf_ancestor_and_hardlink_alias_fail_closed(self) -> None:
        real = self.root / "real.bin"
        real.write_bytes(b"bytes")
        (self.root / "leaf-link").symlink_to(real)
        with self.assertRaises(StableArtifactError):
            read_verified_bytes(self.root, self.root / "leaf-link", self.digest(b"bytes"))

        directory = self.root / "directory"
        directory.mkdir()
        (directory / "nested.bin").write_bytes(b"nested")
        (self.root / "directory-link").symlink_to(directory, target_is_directory=True)
        with self.assertRaises(StableArtifactError):
            read_verified_bytes(
                self.root,
                self.root / "directory-link" / "nested.bin",
                self.digest(b"nested"),
            )

        alias = self.root / "alias.bin"
        os.link(real, alias)
        with self.assertRaisesRegex(StableArtifactError, "hardlink"):
            read_verified_bytes(self.root, real, self.digest(b"bytes"))

    def test_atomic_replace_cannot_redirect_an_open_consumer(self) -> None:
        payload = self.root / "payload.bin"
        payload.write_bytes(b"registered")
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(b"substitute")
        with self.assertRaisesRegex(StableArtifactError, "changed while"):
            with open_stable_regular(
                self.root,
                payload,
                expected_sha256=self.digest(b"registered"),
            ) as handle:
                os.replace(replacement, payload)
                self.assertEqual(handle.read(), b"registered")

    def test_rename_away_and_replacement_cannot_unbind_the_canonical_path(self) -> None:
        payload = self.root / "payload.bin"
        moved = self.root / "moved.bin"
        payload.write_bytes(b"registered")
        with self.assertRaisesRegex(StableArtifactError, "canonical path changed"):
            with open_stable_regular(
                self.root,
                payload,
                expected_sha256=self.digest(b"registered"),
            ) as handle:
                payload.rename(moved)
                payload.write_bytes(b"substitute")
                self.assertEqual(handle.read(), b"registered")

    def test_in_place_mutation_is_detected_after_consumption(self) -> None:
        payload = self.root / "payload.bin"
        payload.write_bytes(b"registered")
        with self.assertRaisesRegex(StableArtifactError, "changed while"):
            with open_stable_regular(
                self.root,
                payload,
                expected_sha256=self.digest(b"registered"),
            ) as handle:
                payload.write_bytes(b"substitute")
                handle.read()

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        for name, encoded in (
            ("duplicate.json", b'{"x":1,"x":2}\n'),
            ("nan.json", b'{"x":NaN}\n'),
        ):
            path = self.root / name
            path.write_bytes(encoded)
            with self.assertRaises(StableArtifactError):
                read_verified_json_object(self.root, path, self.digest(encoded))

    def test_gzip_jsonl_is_parsed_from_the_verified_compressed_snapshot(self) -> None:
        path = self.root / "rows.jsonl.gz"
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as archive:
            archive.write(b'{"id":"a","value":1}\n{"id":"b","value":2}\n')
        encoded = path.read_bytes()
        self.assertEqual(
            read_verified_jsonl_gzip(self.root, path, self.digest(encoded)),
            [{"id": "a", "value": 1}, {"id": "b", "value": 2}],
        )

    def test_noncanonical_and_escaping_paths_are_rejected(self) -> None:
        payload = self.root / "payload.bin"
        payload.write_bytes(b"registered")
        outside = self.root.parent / "outside.bin"
        outside.write_bytes(b"outside")
        try:
            with self.assertRaises(StableArtifactError):
                read_verified_bytes(self.root, "../outside.bin", self.digest(b"outside"))
            with self.assertRaises(StableArtifactError):
                read_verified_bytes(self.root, "./payload.bin", self.digest(b"registered"))
            for alias in (
                f"{self.root}/nested/../payload.bin",
                f"{self.root}//payload.bin",
                f"{self.root}/payload.bin/",
            ):
                with self.subTest(alias=alias), self.assertRaisesRegex(
                    StableArtifactError, "not canonical"
                ):
                    read_verified_bytes(self.root, alias, self.digest(b"registered"))
        finally:
            outside.unlink()

    def test_noncanonical_relative_or_double_slash_roots_are_rejected(self) -> None:
        payload = self.root / "payload.bin"
        payload.write_bytes(b"registered")
        relative = os.path.relpath(self.root, Path.cwd())
        leaf = Path(relative).name
        parent = Path(relative).parent.as_posix()
        aliased_relative = f"{parent}/{leaf}/../{leaf}" if parent != "." else f"{leaf}/../{leaf}"
        for root_alias in (
            aliased_relative,
            f"{relative}//",
            f"//{self.root.as_posix().lstrip('/')}",
        ):
            with self.subTest(root_alias=root_alias), self.assertRaisesRegex(
                StableArtifactError, "root is not"
            ):
                read_verified_bytes(root_alias, payload, self.digest(b"registered"))

    def test_publish_new_bytes_is_exact_and_refuses_overwrite(self) -> None:
        directory = self.root / "directory"
        directory.mkdir()
        destination = directory / "artifact.bin"
        publish_new_bytes(self.root, destination, b"registered", mode=0o640)
        self.assertEqual(destination.read_bytes(), b"registered")
        self.assertEqual(destination.stat().st_mode & 0o777, 0o640)
        self.assertEqual(destination.stat().st_nlink, 1)
        with self.assertRaisesRegex(StableArtifactError, "refusing to overwrite"):
            publish_new_bytes(self.root, destination, b"substitute")
        self.assertEqual(destination.read_bytes(), b"registered")
        self.assertEqual(list(directory.glob(".publish-*.tmp")), [])

    def test_publish_rejects_symlink_ancestors_and_existing_alias_leaves(self) -> None:
        real_directory = self.root / "real-directory"
        real_directory.mkdir()
        linked_directory = self.root / "linked-directory"
        linked_directory.symlink_to(real_directory, target_is_directory=True)
        with self.assertRaises(StableArtifactError):
            publish_new_bytes(self.root, linked_directory / "artifact.bin", b"payload")
        self.assertFalse((real_directory / "artifact.bin").exists())

        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside")
        symlink_leaf = self.root / "symlink-leaf"
        symlink_leaf.symlink_to(outside)
        with self.assertRaisesRegex(StableArtifactError, "refusing to overwrite"):
            publish_new_bytes(self.root, symlink_leaf, b"payload")
        self.assertEqual(outside.read_bytes(), b"outside")

        hardlink_leaf = self.root / "hardlink-leaf"
        os.link(outside, hardlink_leaf)
        with self.assertRaisesRegex(StableArtifactError, "refusing to overwrite"):
            publish_new_bytes(self.root, hardlink_leaf, b"payload")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_failed_publication_removes_private_staging_file(self) -> None:
        destination = self.root / "artifact.bin"
        destination.write_bytes(b"existing")
        with self.assertRaises(StableArtifactError):
            publish_new_bytes(self.root, destination, b"new")
        self.assertEqual(destination.read_bytes(), b"existing")
        self.assertEqual(list(self.root.glob(".publish-*.tmp")), [])

    def test_streaming_writer_publishes_exact_chunks_mode_and_digest(self) -> None:
        destination = self.root / "stream.bin"

        def write_chunks(handle: object) -> None:
            handle.write(b"first-")  # type: ignore[attr-defined]
            handle.write(b"second")  # type: ignore[attr-defined]

        digest = publish_new_file(self.root, destination, write_chunks, mode=0o751)
        self.assertEqual(destination.read_bytes(), b"first-second")
        self.assertEqual(digest, self.digest(b"first-second"))
        self.assertEqual(destination.stat().st_mode & 0o777, 0o751)
        self.assertEqual(destination.stat().st_nlink, 1)

    def test_dead_stage_debris_does_not_alias_or_block_a_fresh_stage(self) -> None:
        debris = self.root / ".publish-collision.tmp"
        debris.write_bytes(b"dead-process")
        with mock.patch.object(
            safe_io.secrets,
            "token_hex",
            side_effect=["collision", "fresh"],
        ):
            publish_new_bytes(self.root, self.root / "artifact.bin", b"payload")
        destination = self.root / "artifact.bin"
        self.assertEqual(destination.read_bytes(), b"payload")
        self.assertEqual(destination.stat().st_nlink, 1)
        self.assertEqual(debris.read_bytes(), b"dead-process")
        self.assertNotEqual(destination.stat().st_ino, debris.stat().st_ino)

    def test_writer_failure_removes_private_stage_without_a_final_alias(self) -> None:
        destination = self.root / "artifact.bin"

        def fail_after_write(handle: object) -> None:
            handle.write(b"partial")  # type: ignore[attr-defined]
            raise RuntimeError("injected writer failure")

        with self.assertRaisesRegex(RuntimeError, "injected writer failure"):
            publish_new_file(self.root, destination, fail_after_write)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".publish-*.tmp")), [])

    def test_pre_rename_file_fsync_failure_cleans_stage_and_does_not_commit(self) -> None:
        destination = self.root / "artifact.bin"
        with mock.patch.object(
            safe_io.os,
            "fsync",
            side_effect=OSError("injected pre-rename fsync failure"),
        ):
            with self.assertRaisesRegex(StableArtifactError, "durably published"):
                publish_new_bytes(self.root, destination, b"payload")
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".publish-*.tmp")), [])

    def test_post_rename_parent_fsync_failure_reports_committed_destination(self) -> None:
        destination = self.root / "artifact.bin"
        with mock.patch.object(
            safe_io.os,
            "fsync",
            side_effect=[None, OSError("injected post-rename fsync failure")],
        ):
            with self.assertRaisesRegex(StableArtifactError, "rename committed.*fsync"):
                publish_new_bytes(self.root, destination, b"payload")
        self.assertEqual(destination.read_bytes(), b"payload")
        self.assertEqual(destination.stat().st_nlink, 1)
        self.assertEqual(list(self.root.glob(".publish-*.tmp")), [])

    def test_stage_mode_failure_cleans_the_new_named_inode(self) -> None:
        destination = self.root / "artifact.bin"
        with mock.patch.object(
            safe_io.os,
            "fchmod",
            side_effect=OSError("injected mode failure"),
        ):
            with self.assertRaisesRegex(StableArtifactError, "durably published"):
                publish_new_bytes(self.root, destination, b"payload")
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".publish-*.tmp")), [])

    def test_embedded_nul_cannot_truncate_a_c_rename_destination(self) -> None:
        truncated = self.root / "artifact"
        with self.assertRaisesRegex(StableArtifactError, "not canonical"):
            publish_new_bytes(self.root, f"{truncated}\x00.bin", b"payload")
        self.assertFalse(truncated.exists())

    def test_unsupported_atomic_rename_fails_closed_and_cleans_stage(self) -> None:
        for unsupported in (None, "ENOSYS", "EINVAL"):
            with self.subTest(unsupported=unsupported):
                destination = self.root / f"artifact-{unsupported}.bin"
                if unsupported is None:
                    patches = (
                        mock.patch.object(safe_io, "_RENAMEAT2", None),
                        contextlib.nullcontext(),
                    )
                else:
                    error_number = getattr(errno, unsupported)
                    patches = (
                        mock.patch.object(safe_io, "_RENAMEAT2", return_value=-1),
                        mock.patch.object(safe_io.ctypes, "get_errno", return_value=error_number),
                    )
                with patches[0], patches[1]:
                    with self.assertRaisesRegex(StableArtifactError, "unavailable|unsupported"):
                        publish_new_bytes(self.root, destination, b"payload")
                self.assertFalse(destination.exists())
                self.assertEqual(list(self.root.glob(".publish-*.tmp")), [])

    def test_same_parent_directory_commit_is_no_replace_and_preserves_tree(self) -> None:
        staging = self.root / ".tree-stage"
        staging.mkdir()
        (staging / "payload.bin").write_bytes(b"tree")
        destination = self.root / "tree"
        rename_new_entry(self.root, staging, destination)
        self.assertFalse(staging.exists())
        self.assertEqual((destination / "payload.bin").read_bytes(), b"tree")

        second_staging = self.root / ".tree-stage-2"
        second_staging.mkdir()
        (second_staging / "payload.bin").write_bytes(b"second")
        with self.assertRaisesRegex(StableArtifactError, "refusing to overwrite"):
            rename_new_entry(self.root, second_staging, destination)
        self.assertEqual((destination / "payload.bin").read_bytes(), b"tree")
        self.assertEqual((second_staging / "payload.bin").read_bytes(), b"second")

    def test_cross_parent_move_is_no_replace_and_fsyncs_both_parents(self) -> None:
        source_parent = self.root / "source"
        destination_parent = self.root / "destination"
        source_parent.mkdir()
        destination_parent.mkdir()
        source = source_parent / "payload.bin"
        destination = destination_parent / "payload.bin"
        source.write_bytes(b"registered")
        expected_parent_inodes = {
            source_parent.stat().st_ino,
            destination_parent.stat().st_ino,
        }
        fsynced_parent_inodes: set[int] = set()
        real_fsync = os.fsync

        def record_fsync(descriptor: int) -> None:
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode):
                fsynced_parent_inodes.add(info.st_ino)
            real_fsync(descriptor)

        with mock.patch.object(safe_io.os, "fsync", side_effect=record_fsync):
            move_new_entry(self.root, source, destination)

        self.assertFalse(source.exists())
        self.assertEqual(destination.read_bytes(), b"registered")
        self.assertTrue(expected_parent_inodes.issubset(fsynced_parent_inodes))

        competing_source = source_parent / "second.bin"
        competing_source.write_bytes(b"second")
        with self.assertRaisesRegex(StableArtifactError, "refusing to overwrite"):
            move_new_entry(self.root, competing_source, destination)
        self.assertEqual(competing_source.read_bytes(), b"second")
        self.assertEqual(destination.read_bytes(), b"registered")

    def test_cross_parent_move_attempts_both_fsyncs_after_commit_error(self) -> None:
        source_parent = self.root / "fsync-source"
        destination_parent = self.root / "fsync-destination"
        source_parent.mkdir()
        destination_parent.mkdir()
        source = source_parent / "payload.bin"
        destination = destination_parent / "payload.bin"
        source.write_bytes(b"registered")
        source_inode = source_parent.stat().st_ino
        destination_inode = destination_parent.stat().st_ino
        directory_attempts: list[int] = []
        failed = False
        real_fsync = os.fsync

        def fail_source_parent_once(descriptor: int) -> None:
            nonlocal failed
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode):
                directory_attempts.append(info.st_ino)
                if info.st_ino == source_inode and not failed:
                    failed = True
                    raise OSError(errno.EIO, "synthetic source-parent fsync failure")
            real_fsync(descriptor)

        with mock.patch.object(
            safe_io.os,
            "fsync",
            side_effect=fail_source_parent_once,
        ):
            with self.assertRaisesRegex(
                StableArtifactError, "committed but durability"
            ):
                move_new_entry(self.root, source, destination)

        self.assertFalse(source.exists())
        self.assertEqual(destination.read_bytes(), b"registered")
        self.assertIn(source_inode, directory_attempts)
        self.assertIn(destination_inode, directory_attempts)

    def test_cross_parent_move_rejects_hardlink_and_parent_replacement(self) -> None:
        source_parent = self.root / "source"
        destination_parent = self.root / "destination"
        source_parent.mkdir()
        destination_parent.mkdir()
        real = source_parent / "real.bin"
        real.write_bytes(b"registered")
        alias = source_parent / "alias.bin"
        os.link(real, alias)
        with self.assertRaisesRegex(StableArtifactError, "hardlink"):
            move_new_entry(
                self.root,
                alias,
                destination_parent / "hardlink-final.bin",
            )

        alias.unlink()
        moved_parent = self.root / "destination-moved"
        destination = destination_parent / "final.bin"
        real_rename = safe_io._rename_noreplace_at

        def rename_then_replace_parent(*args: object, **kwargs: object) -> None:
            real_rename(*args, **kwargs)  # type: ignore[arg-type]
            destination_parent.rename(moved_parent)
            destination_parent.mkdir()

        with mock.patch.object(
            safe_io,
            "_rename_noreplace_at",
            side_effect=rename_then_replace_parent,
        ):
            with self.assertRaisesRegex(
                StableArtifactError, "durability or binding"
            ):
                move_new_entry(self.root, real, destination)
        self.assertFalse(destination.exists())
        self.assertEqual((moved_parent / "final.bin").read_bytes(), b"registered")

    def test_update_contexts_hold_ancestors_and_reject_path_replacement(self) -> None:
        ancestor = self.root / "ancestor"
        directory = ancestor / "tree"
        directory.mkdir(parents=True)
        payload = directory / "payload.bin"
        payload.write_bytes(b"registered")
        moved = self.root / "ancestor-moved"
        attacker = self.root / "attacker"
        (attacker / "tree").mkdir(parents=True)
        (attacker / "tree" / "payload.bin").write_bytes(b"replacement")

        with self.assertRaises(StableArtifactError):
            with open_stable_regular_for_update(self.root, payload) as descriptor:
                ancestor.rename(moved)
                ancestor.symlink_to(attacker, target_is_directory=True)
                os.ftruncate(descriptor, 0)

        self.assertEqual((moved / "tree" / "payload.bin").stat().st_size, 0)
        self.assertEqual(
            (attacker / "tree" / "payload.bin").read_bytes(), b"replacement"
        )

        ancestor.unlink()
        moved.rename(ancestor)
        moved_tree = self.root / "tree-moved"
        with self.assertRaises(StableArtifactError):
            with open_stable_directory_for_update(
                self.root, ancestor / "tree"
            ) as descriptor:
                (ancestor / "tree").rename(moved_tree)
                (ancestor / "tree").mkdir()
                os.fsync(descriptor)
        self.assertTrue((moved_tree / "payload.bin").is_file())

    def test_update_contexts_validate_bindings_on_exceptional_exit(self) -> None:
        ancestor = self.root / "exception-ancestor"
        directory = ancestor / "tree"
        directory.mkdir(parents=True)
        payload = directory / "payload.bin"
        payload.write_bytes(b"registered")
        moved = self.root / "exception-ancestor-moved"
        replacement = self.root / "exception-replacement"
        (replacement / "tree").mkdir(parents=True)
        (replacement / "tree" / "payload.bin").write_bytes(b"replacement")

        with self.assertRaisesRegex(
            StableArtifactError, "changed while it was held"
        ) as raised:
            with open_stable_regular_for_update(self.root, payload) as descriptor:
                ancestor.rename(moved)
                ancestor.symlink_to(replacement, target_is_directory=True)
                os.ftruncate(descriptor, 0)
                raise RuntimeError("synthetic body failure")
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(
            (replacement / "tree" / "payload.bin").read_bytes(), b"replacement"
        )

        ancestor.unlink()
        moved.rename(ancestor)
        with self.assertRaisesRegex(RuntimeError, "unchanged body failure"):
            with open_stable_regular_for_update(self.root, payload):
                raise RuntimeError("unchanged body failure")

    def test_canonical_directory_creation_is_recursive_durable_and_no_follow(self) -> None:
        target = self.root / "one" / "two" / "three"
        fsynced: set[int] = set()
        real_fsync = os.fsync

        def record(descriptor: int) -> None:
            fsynced.add(os.fstat(descriptor).st_ino)
            real_fsync(descriptor)

        with mock.patch.object(safe_io.os, "fsync", side_effect=record):
            ensure_canonical_directory(self.root, target)
            fsync_canonical_directory(self.root, target)
        self.assertTrue(target.is_dir())
        self.assertTrue(
            {
                self.root.stat().st_ino,
                (self.root / "one").stat().st_ino,
                (self.root / "one" / "two").stat().st_ino,
                target.stat().st_ino,
            }.issubset(fsynced)
        )

        real = self.root / "real-directory"
        real.mkdir()
        linked = self.root / "linked-directory"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaises(StableArtifactError):
            ensure_canonical_directory(self.root, linked / "child")
        self.assertFalse((real / "child").exists())

    def test_canonical_directory_creation_rebinds_every_ancestor(self) -> None:
        target = self.root / "stable-one" / "stable-two"
        moved = self.root / "stable-one-moved"
        real_rebind = safe_io._require_canonical_directory_chain
        injected = False

        def replace_ancestor(
            root_descriptor: int,
            parts: tuple[str, ...],
            expected: tuple[tuple[int, int, int], ...],
        ) -> None:
            nonlocal injected
            (self.root / "stable-one").rename(moved)
            (self.root / "stable-one").mkdir()
            injected = True
            real_rebind(root_descriptor, parts, expected)

        with mock.patch.object(
            safe_io,
            "_require_canonical_directory_chain",
            side_effect=replace_ancestor,
        ):
            with self.assertRaisesRegex(
                StableArtifactError, "canonical directory chain changed"
            ):
                ensure_canonical_directory(self.root, target)
        self.assertTrue(injected)
        self.assertFalse(target.exists())
        self.assertTrue((moved / "stable-two").is_dir())

    def test_same_parent_commit_rejects_symlink_and_hardlinked_sources(self) -> None:
        real = self.root / "real.bin"
        real.write_bytes(b"real")
        symlink = self.root / "source-link"
        symlink.symlink_to(real)
        with self.assertRaisesRegex(StableArtifactError, "symlink"):
            rename_new_entry(self.root, symlink, self.root / "symlink-final")
        self.assertFalse((self.root / "symlink-final").exists())

        hardlink = self.root / "source-hardlink"
        os.link(real, hardlink)
        with self.assertRaisesRegex(StableArtifactError, "hardlink"):
            rename_new_entry(self.root, hardlink, self.root / "hardlink-final")
        self.assertFalse((self.root / "hardlink-final").exists())

    def test_directory_commit_rebinds_the_full_canonical_parent_path(self) -> None:
        parent = self.root / "parent"
        parent.mkdir()
        staging = parent / ".stage"
        staging.mkdir()
        destination = parent / "final"
        moved_parent = self.root / "moved-parent"
        real_commit = safe_io._commit_new_entry_at

        def commit_then_move_parent(*args: object, **kwargs: object) -> None:
            real_commit(*args, **kwargs)  # type: ignore[arg-type]
            parent.rename(moved_parent)
            parent.mkdir()

        with mock.patch.object(
            safe_io,
            "_commit_new_entry_at",
            side_effect=commit_then_move_parent,
        ):
            with self.assertRaisesRegex(StableArtifactError, "changed during commit"):
                rename_new_entry(self.root, staging, destination)
        self.assertFalse(destination.exists())
        self.assertTrue((moved_parent / "final").is_dir())

    def test_trusted_root_replacement_cannot_redirect_publication(self) -> None:
        trusted = self.root / "trusted"
        trusted.mkdir()
        moved = self.root / "trusted-moved"
        destination = trusted / "artifact.bin"
        real_open_parent = safe_io._open_parent
        injected = False

        def replace_root(root_descriptor: int, parts: tuple[str, ...]):
            nonlocal injected
            if not injected:
                trusted.rename(moved)
                trusted.mkdir()
                injected = True
            return real_open_parent(root_descriptor, parts)

        with mock.patch.object(
            safe_io, "_open_parent", side_effect=replace_root
        ):
            with self.assertRaisesRegex(StableArtifactError, "root changed"):
                publish_new_bytes(trusted, destination, b"registered")
        self.assertTrue(injected)
        self.assertFalse(destination.exists())
        self.assertEqual((moved / "artifact.bin").read_bytes(), b"registered")


if __name__ == "__main__":
    unittest.main()
