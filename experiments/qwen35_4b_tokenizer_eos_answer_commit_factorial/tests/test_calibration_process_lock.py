from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts/run_calibration.py"
HELPER = r"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_calibration_lock_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.RUN_LOCK = Path(sys.argv[2])
mode = sys.argv[3]
try:
    with module.calibration_process_lock():
        if mode == "holder":
            Path(sys.argv[4]).write_text("ready")
            sys.stdin.readline()
        else:
            raise SystemExit(23)
except RuntimeError as error:
    if "another calibration process holds" not in str(error):
        raise
    raise SystemExit(0)
"""


class CalibrationProcessLockTests(unittest.TestCase):
    def test_failed_contender_cannot_replace_locked_inode_for_third_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "run.lock"
            ready = root / "ready"
            holder = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    HELPER,
                    str(SCRIPT),
                    str(lock_path),
                    "holder",
                    str(ready),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(100):
                    if ready.is_file():
                        break
                    if holder.poll() is not None:
                        self.fail(holder.stderr.read())
                    time.sleep(0.05)
                else:
                    self.fail("lock holder did not become ready")
                inode = lock_path.stat().st_ino
                command = [
                    sys.executable,
                    "-B",
                    "-c",
                    HELPER,
                    str(SCRIPT),
                    str(lock_path),
                    "contender",
                ]
                first = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertEqual(lock_path.stat().st_ino, inode)
                third = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(third.returncode, 0, third.stderr)
                self.assertEqual(lock_path.stat().st_ino, inode)
            finally:
                if holder.stdin is not None:
                    holder.stdin.write("release\n")
                    holder.stdin.flush()
                    holder.stdin.close()
                holder.wait(timeout=10)
                error_output = holder.stderr.read() if holder.stderr is not None else ""
                if holder.stdout is not None:
                    holder.stdout.close()
                if holder.stderr is not None:
                    holder.stderr.close()
                if holder.returncode != 0:
                    self.fail(error_output)
            after_release = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(after_release.returncode, 23, after_release.stderr)
            self.assertEqual(lock_path.stat().st_ino, inode)


if __name__ == "__main__":
    unittest.main()
