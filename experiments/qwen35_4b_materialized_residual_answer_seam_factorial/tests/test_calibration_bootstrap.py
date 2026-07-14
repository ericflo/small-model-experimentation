from __future__ import annotations

import ast
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_calibration_verifies_and_installs_audit_before_local_imports(self) -> None:
        source = (EXP / "scripts/run_calibration.py").read_text()
        ast.parse(source)
        bootstrap = source.index("_bootstrap_verify_before_local_imports()")
        path_insert = source.index("sys.path.insert(0, str(SRC))")
        local_import = source.index("from calibration_lock import")
        self.assertLess(bootstrap, path_insert)
        self.assertLess(bootstrap, local_import)
        self.assertIn("sys.addaudithook(audit)", source[:local_import])
        self.assertIn("forbids benchmark access", source[:local_import])
        self.assertIn("forbids unregistered repository access", source[:local_import])
        self.assertIn("pre-import calibration refuses local Python caches", source)

    def test_mechanics_lock_stage_authenticates_calibration_frozen_sources(self) -> None:
        source = (EXP / "scripts/run_mechanics.py").read_text()
        tree = ast.parse(source)
        self.assertIn(
            'stage not in {"lock", "run", "analyze-visible", "score-hidden"}',
            source,
        )
        self.assertIn("_MECHANICS_FROZEN_IMPORT_FILES", source)
        self.assertIn("frozen = calibration.get", source)
        self.assertIn('if stage != "lock"', source)
        self.assertIn("pre-import frozen mechanics blob changed", source)
        self.assertIsInstance(tree, ast.Module)

    def test_mechanics_gold_hook_opens_only_after_publication_authorization(self) -> None:
        source = (EXP / "scripts/run_mechanics.py").read_text()
        ast.parse(source)
        bootstrap = source.index("_bootstrap_verify_before_local_imports()")
        local_import = source.index("from calibration_lock import")
        self.assertLess(bootstrap, local_import)
        self.assertIn("sys.addaudithook(audit)", source[:local_import])
        self.assertIn("mechanics gold remains sealed", source[:local_import])
        authorize = source.index("authorization = authorize_hidden_read()")
        open_hook = source.index("_bootstrap_authorize_gold_path()", authorize)
        hidden_score = source.index("score_hidden(visible=visible", open_hook)
        self.assertLess(authorize, open_hook)
        self.assertLess(open_hook, hidden_score)


if __name__ == "__main__":
    unittest.main()
