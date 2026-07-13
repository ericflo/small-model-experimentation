from __future__ import annotations

import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_think.py"


class TrainSeedContractTests(unittest.TestCase):
    def test_global_seed_precedes_model_creation(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        main = source[source.index("def main()") :]
        self.assertLess(main.index("set_seed(args.seed)"), main.index("model = load_text_model"))

    def test_lora_seed_is_reset_immediately_before_adapter_creation(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        loader = source[source.index("def load_text_model") : source.index("def main()")]
        self.assertIn("*, seed: int", loader)
        self.assertLess(loader.index("set_seed(seed)"), loader.index("get_peft_model("))

    def test_receipt_exposes_seed_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"global_seed_before_model": True', source)
        self.assertIn('"lora_seed_reset_before_adapter_init": True', source)

    def test_training_receipt_is_restart_safe_and_not_self_hashed(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('receipt_path.unlink(missing_ok=True)', source)
        self.assertIn('path.name != "training_receipt.json"', source)
        self.assertIn("write_json_atomic(receipt_path, receipt)", source)


if __name__ == "__main__":
    unittest.main()
