"""Tests for the web layer's secret handling.

These exist because a real API key was once pasted into ``llm.api_key_env`` — a legitimate
dataclass field — and reached config.json, which the API serves.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web.backend.artifacts import REDACTED, redact_config
from web.backend.configspec import apply_overrides, field_spec, is_secret_path
from web.backend.keys import name_error

# Shaped like a real key — long, mixed case, leading `sk-` — but synthetic. A genuine key
# used as a fixture would be committed to history, which is the exact failure these tests exist to prevent.
REAL_KEY_SHAPE = "sk-" + "0f3a" * 8 + "TESTFIXTURENOTAREALKEY"


class NameValidationTest(unittest.TestCase):
    def test_rejects_a_pasted_secret(self) -> None:
        self.assertIsNotNone(name_error(REAL_KEY_SHAPE))

    def test_rejects_long_mixed_case_without_a_dash(self) -> None:
        self.assertIsNotNone(name_error("a" * 40))

    def test_accepts_ordinary_variable_names(self) -> None:
        for name in ("PRISMLLM_API_KEY", "OPENAI_API_KEY", "_x", "K1"):
            self.assertIsNone(name_error(name), name)

    def test_rejects_empty_and_malformed(self) -> None:
        for name in ("", "1BAD", "has space", "has-dash", "a" * 65):
            self.assertIsNotNone(name_error(name), name)


class OverrideValidationTest(unittest.TestCase):
    def test_secret_in_api_key_env_is_rejected(self) -> None:
        data, invalid, rejected = apply_overrides({"llm.api_key_env": REAL_KEY_SHAPE})
        self.assertIn("llm.api_key_env", rejected)
        self.assertNotIn("llm", data)
        self.assertEqual(invalid, [])

    def test_variable_name_passes_through(self) -> None:
        data, _, rejected = apply_overrides({"llm.api_key_env": "PRISMLLM_API_KEY"})
        self.assertEqual(rejected, {})
        self.assertEqual(data["llm"]["api_key_env"], "PRISMLLM_API_KEY")

    def test_unknown_provider_is_rejected(self) -> None:
        _, _, rejected = apply_overrides({"llm.provider": "gpt-5.6-sol"})
        self.assertIn("llm.provider", rejected)

    def test_known_providers_pass(self) -> None:
        for provider in ("heuristic", "openai-compatible"):
            _, _, rejected = apply_overrides({"llm.provider": provider})
            self.assertEqual(rejected, {}, provider)

    def test_provider_is_offered_as_a_closed_choice(self) -> None:
        spec = {item["path"]: item for item in field_spec()}
        self.assertEqual(
            spec["llm.provider"].get("choices"), ["openai-compatible", "heuristic"]
        )


class SecretPathTest(unittest.TestCase):
    def test_token_hint_does_not_swallow_max_completion_tokens(self) -> None:
        # A substring match refused this ordinary integer knob as if it were a credential.
        self.assertFalse(is_secret_path("llm.max_completion_tokens"))
        _, invalid, _ = apply_overrides({"llm.max_completion_tokens": 2000})
        self.assertEqual(invalid, [])

    def test_real_secret_leaves_are_still_caught(self) -> None:
        for path in ("llm.api_key", "x.auth_token", "x.token", "x.client_secret", "x.password"):
            self.assertTrue(is_secret_path(path), path)

    def test_api_key_env_is_exempt(self) -> None:
        self.assertFalse(is_secret_path("llm.api_key_env"))


class RedactionTest(unittest.TestCase):
    def test_legacy_config_with_a_pasted_key_is_redacted(self) -> None:
        out = redact_config({"llm": {"api_key_env": REAL_KEY_SHAPE, "model": "m"}})
        self.assertEqual(out["llm"]["api_key_env"], REDACTED)
        self.assertEqual(out["llm"]["model"], "m")

    def test_ordinary_config_is_untouched(self) -> None:
        config = {"llm": {"api_key_env": "PRISMLLM_API_KEY"}}
        self.assertEqual(redact_config(config), config)

    def test_tolerates_missing_and_malformed_sections(self) -> None:
        self.assertEqual(redact_config({}), {})
        self.assertEqual(redact_config({"llm": None}), {"llm": None})
        self.assertIsNone(redact_config(None))


if __name__ == "__main__":
    unittest.main()
