"""Bearer-token auth for a LAN-exposed service.

The token is compared with ``compare_digest`` and never appears in a URL, so it stays out
of access logs and ``Referer`` headers. Repeated failures from one address are throttled
so a token cannot be brute-forced over the LAN.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from .settings import WebSettings

MIN_TOKEN_LENGTH = 24
FAILURE_WINDOW_SECONDS = 60
MAX_FAILURES_PER_WINDOW = 10

_failures: dict[str, deque[float]] = defaultdict(deque)


def resolve_token(settings: WebSettings) -> str:
    if settings.token:
        if len(settings.token) < MIN_TOKEN_LENGTH:
            raise SystemExit(
                f"EVOSCI_WEB_TOKEN must be at least {MIN_TOKEN_LENGTH} characters"
            )
        return settings.token
    return secrets.token_urlsafe(32)


def _record_failure(client: str) -> None:
    now = time.monotonic()
    recent = _failures[client]
    recent.append(now)
    while recent and now - recent[0] > FAILURE_WINDOW_SECONDS:
        recent.popleft()


def _is_throttled(client: str) -> bool:
    now = time.monotonic()
    recent = _failures[client]
    while recent and now - recent[0] > FAILURE_WINDOW_SECONDS:
        recent.popleft()
    return len(recent) >= MAX_FAILURES_PER_WINDOW


def require_token(request: Request, authorization: str = Header(default="")) -> None:
    client = request.client.host if request.client else "unknown"
    if _is_throttled(client):
        raise HTTPException(status_code=429, detail="Too many failed attempts")
    expected = request.app.state.token
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented, expected):
        _record_failure(client)
        raise HTTPException(status_code=401, detail="Unauthorized")
