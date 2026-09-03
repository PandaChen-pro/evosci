"""Durable event log: one JSON object per line, appended by the run subprocess.

Writing to a file rather than an in-memory queue is what makes a reload survivable. A
browser that reconnects at minute nine replays from its last sequence number and misses
nothing, and a client that was never connected still gets the whole history.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

EVENTS_FILE = "events.jsonl"
POLL_SECONDS = 0.25
TERMINAL = {"job.finished", "job.failed", "job.cancelled"}


def events_path(run_dir: Path) -> Path:
    return run_dir / EVENTS_FILE


def last_seq(run_dir: Path) -> int:
    """High-water mark of the log. A resumed run continues numbering from here, so a
    browser cursor stays monotonic across attempts and old events are never re-sent."""
    highest = 0
    for record in read_events(run_dir):
        highest = max(highest, int(record.get("seq", 0)))
    return highest


class EventWriter:
    """Append-only writer used inside the run subprocess."""

    def __init__(self, run_dir: Path) -> None:
        self._path = events_path(run_dir)
        self._seq = last_seq(run_dir)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": event_type,
            "data": data,
        }
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def read_events(run_dir: Path, since: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    path = events_path(run_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a torn final line; earlier ones are intact.
                continue
            if record.get("seq", 0) > since:
                out.append(record)
                if limit is not None and len(out) >= limit:
                    break
    return out


async def stream_events(
    run_dir: Path, since: int = 0, is_active=lambda: True, terminal_from: int = 0
) -> AsyncIterator[str]:
    """Replay past events, then tail the file until the run reaches a terminal event.

    ``terminal_from`` is the current attempt's watermark: a resumed run appends to the
    same log, so a replayed ``job.finished`` from the previous attempt must not close a
    stream that is following the new one.
    """
    last = since
    idle = 0.0
    while True:
        batch = read_events(run_dir, since=last)
        for record in batch:
            last = record["seq"]
            yield f"data: {json.dumps(record, ensure_ascii=False)}\n\n"
            if record.get("type") in TERMINAL and record["seq"] > terminal_from:
                return
        if batch:
            idle = 0.0
        else:
            idle += POLL_SECONDS
            if not is_active() and idle > 2.0:
                return
            if idle >= 15.0:
                idle = 0.0
                yield ": keepalive\n\n"
        await asyncio.sleep(POLL_SECONDS)
