"""Start the EvoSci web service.

    PYTHONPATH=src web/.venv/bin/python -m web.serve

Exists because a generated token is useless unless it is shown once, and because binding
to the LAN is the point at which the operator has to be told what that exposes.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import uvicorn

from .backend.auth import MIN_TOKEN_LENGTH
from .backend.main import create_app
from .backend.settings import WebSettings

TOKEN_FILE = Path(__file__).resolve().parent / ".token"


def _lan_address() -> str:
    """Best-effort local address for the printed URL; no packet is actually sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def main() -> int:
    settings = WebSettings.from_env()
    exposed = settings.host not in {"127.0.0.1", "localhost", "::1"}
    if exposed and settings.token and len(settings.token) < MIN_TOKEN_LENGTH:
        print(
            f"拒绝绑定 {settings.host}：token 短于 {MIN_TOKEN_LENGTH} 个字符",
            file=sys.stderr,
        )
        return 2

    app = create_app(settings)
    token = app.state.token
    TOKEN_FILE.write_text(token, encoding="utf-8")
    TOKEN_FILE.chmod(0o600)

    host = _lan_address() if exposed else settings.host
    lines = [
        f"\nEvoSci web  →  http://{host}:{settings.port}",
        f"token       →  {token}",
        f"            （同时写入 {TOKEN_FILE}）",
        f"runs 根目录 →  {settings.runs_root}",
        f"扫描目录    →  {', '.join(str(root) for root in settings.scan_roots)}",
        "",
    ]
    # Flushed explicitly: uvicorn logs to stderr, so a buffered stdout would strand the
    # token below the server output — or lose it entirely when redirected to a file.
    print("\n".join(lines), flush=True)
    if exposed:
        print(
            f"警告：已绑定 {settings.host}，且使用明文 HTTP。同一网络内拿到 token 的人可以读取\n"
            "每个任务的想法与评审，也可以提交任务从而消耗 API 额度。他们读不到 API key。\n"
            "请仅在你信任的网络中使用。\n",
            file=sys.stderr,
            flush=True,
        )

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
