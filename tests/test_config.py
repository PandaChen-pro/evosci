import os
import tempfile
import unittest
from pathlib import Path

from evosci.config import EvoSciConfig


class ConfigTests(unittest.TestCase):
    def test_toml_load_and_validation(self) -> None:
        text = """
[llm]
provider = "heuristic"
[run]
rounds = 2
team_size = 3
ideas_per_round = 2
reviewer_count = 2
[evolution]
selection_ratio = 0.6
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(text)
            config = EvoSciConfig.from_toml(path)
        config.validate()
        self.assertEqual(config.run.rounds, 2)
        self.assertEqual(config.run.team_size, 3)
        self.assertAlmostEqual(config.evolution.selection_ratio, 0.6)

    def test_openai_provider_requires_key(self) -> None:
        config = EvoSciConfig.from_dict({
            "llm": {"provider": "openai-compatible", "api_key_env": "EVOSCI_TEST_KEY"}
        })
        old_value = os.environ.pop("EVOSCI_TEST_KEY", None)
        try:
            with self.assertRaisesRegex(ValueError, "Missing API key"):
                config.validate()
        finally:
            if old_value is not None:
                os.environ["EVOSCI_TEST_KEY"] = old_value


if __name__ == "__main__":
    unittest.main()
