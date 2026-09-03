"""API keys entered through the UI.

A key set here is held for the service's lifetime and handed to a run the same way an
exported variable is: through the spawned process's environment, filtered to the one name
that run's config asks for. It is written to a 0600 file so a service restart does not
silently turn a configured run into a failing one.

The value is write-only from the API's point of view. Nothing here has a read path that
returns it, which is what keeps it out of responses, run directories, and event logs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

STORE_FILE = Path(__file__).resolve().parents[1] / ".keys.json"

# POSIX-ish environment variable name. The dash in every `sk-…` key fails this, which is
# the check that would have caught a real key being pasted into the api_key_env field.
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_NAME_LENGTH = 64
MAX_VALUE_LENGTH = 4096

# Beyond this length a lowercase-bearing string is a credential, not a variable name:
# real names are short and SCREAMING_SNAKE_CASE, real keys are long and mixed-case.
SECRET_SHAPE_LENGTH = 32


def name_error(name: str) -> str | None:
    """Why this string cannot be an environment variable name, or None if it can.

    The secret-shape test runs before the generic length and charset ones: a pasted key
    fails all three, and "这看起来是密钥" is the message that tells the person what they
    actually did wrong. "不能超过 64 个字符" would send them off to shorten it.
    """
    if not name:
        return "环境变量名不能为空"
    if len(name) >= SECRET_SHAPE_LENGTH and any(char.islower() for char in name):
        return "这看起来是密钥本身，不是环境变量名 —— 这一栏只填变量名，密钥用下面的输入框保存"
    if len(name) > MAX_NAME_LENGTH:
        return f"环境变量名不能超过 {MAX_NAME_LENGTH} 个字符"
    if not NAME_PATTERN.match(name):
        return "环境变量名只能包含字母、数字和下划线，且不能以数字开头"
    return None


class KeyStore:
    """Keys set through the UI, applied into this process's own environment.

    Applying them to ``os.environ`` rather than keeping a parallel lookup means
    ``LLMConfig.api_key`` and ``EvoSciConfig.validate`` see a UI-set key exactly as they
    see an exported one, and ``JobStore._spawn``'s existing allowlist forwards it to the
    run with no change. The key was already in this process either way.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or STORE_FILE
        self._names: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for name, value in data.items():
            if isinstance(name, str) and isinstance(value, str) and not name_error(name):
                os.environ[name] = value
                self._names.add(name)

    def _save(self) -> None:
        payload = {name: os.environ.get(name, "") for name in sorted(self._names)}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.path.chmod(0o600)

    def set(self, name: str, value: str) -> None:
        problem = name_error(name)
        if problem:
            raise ValueError(problem)
        value = value.strip()
        if not value:
            raise ValueError("密钥不能为空")
        if len(value) > MAX_VALUE_LENGTH:
            raise ValueError(f"密钥不能超过 {MAX_VALUE_LENGTH} 个字符")
        os.environ[name] = value
        self._names.add(name)
        self._save()

    def delete(self, name: str) -> bool:
        if name not in self._names:
            return False
        self._names.discard(name)
        os.environ.pop(name, None)
        self._save()
        return True

    def has(self, name: str) -> bool:
        return bool(name and os.environ.get(name))

    def describe(self, name: str) -> dict[str, Any]:
        """Presence and origin only — never the value."""
        return {
            "name": name,
            "present": self.has(name),
            "source": "ui" if name in self._names else ("env" if os.environ.get(name) else None),
        }

    def names(self) -> list[str]:
        return sorted(self._names)
