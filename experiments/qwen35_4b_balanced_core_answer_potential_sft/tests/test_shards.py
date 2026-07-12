from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from shards import read_jsonl_gz, valid_receipt, write_jsonl_gz


class ShardTests(unittest.TestCase):
    def test_atomic_gzip_round_trip_and_receipt(self) -> None:
        rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            receipt = write_jsonl_gz(path, rows)
            self.assertTrue(valid_receipt(receipt))
            self.assertEqual(read_jsonl_gz(path), rows)


if __name__ == "__main__":
    unittest.main()
