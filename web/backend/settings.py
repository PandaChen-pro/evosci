"""Runtime settings for the EvoSci web service.

Everything is environment-driven so the service can be started with nothing but
``uvicorn`` and the one API key its runs are configured to use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"


def _paths(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item.strip()]


@dataclass(slots=True)
class WebSettings:
    runs_root: Path = DEFAULT_RUNS_ROOT
    scan_roots: list[Path] = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8000
    max_concurrent: int = 1
    max_rounds: int = 10
    token: str = ""

    @classmethod
    def from_env(cls) -> "WebSettings":
        runs_root = Path(
            os.environ.get("EVOSCI_RUNS_ROOT", str(DEFAULT_RUNS_ROOT))
        ).expanduser().resolve()
        scan_roots = _paths(os.environ.get("EVOSCI_SCAN_ROOTS"))
        if runs_root not in scan_roots:
            scan_roots.insert(0, runs_root)
        return cls(
            runs_root=runs_root,
            scan_roots=scan_roots,
            host=os.environ.get("EVOSCI_WEB_HOST", "127.0.0.1"),
            port=int(os.environ.get("EVOSCI_WEB_PORT", "8000")),
            max_concurrent=int(os.environ.get("EVOSCI_MAX_CONCURRENT", "1")),
            max_rounds=int(os.environ.get("EVOSCI_MAX_ROUNDS", "10")),
            token=os.environ.get("EVOSCI_WEB_TOKEN", ""),
        )
