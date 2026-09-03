"""Job lifecycle: spawn, queue, cancel, and re-discover runs after a service restart.

Each run is its own process group, so cancelling is a real kill rather than a cooperative
flag the engine has no seam to check — ``EvoSciEngine.run`` blocks inside ``urlopen`` and
``time.sleep`` with nothing to poll. The cost is that all cross-process state travels
through files, which is also what lets a run outlive the service that started it.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import events_path, last_seq, read_events
from .settings import WebSettings

RUNNER = Path(__file__).resolve().parent / "run_job.py"
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE")
ACTIVE = {"queued", "running"}
TERMINAL_EVENTS = {
    "job.finished": "finished",
    "job.failed": "failed",
    "job.cancelled": "cancelled",
}


def _slug(value: str) -> str:
    """Keep CJK and other letters, so a Chinese topic yields a recognisable directory
    name instead of collapsing to the "run" fallback."""
    cleaned = re.sub(r"[^\w]+", "-", value.lower(), flags=re.UNICODE)
    return cleaned.strip("-")[:40] or "run"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings
        self.settings.runs_root.mkdir(parents=True, exist_ok=True)
        self._children: dict[int, subprocess.Popen[bytes]] = {}

    def _reap(self) -> None:
        """An exited child stays a zombie until waited on, and ``os.kill(pid, 0)``
        reports a zombie as alive — so cancel would burn its full grace period."""
        for pid, child in list(self._children.items()):
            if child.poll() is not None:
                self._children.pop(pid, None)

    def _alive(self, pid: int) -> bool:
        child = self._children.get(pid)
        if child is not None:
            if child.poll() is not None:
                self._children.pop(pid, None)
                return False
            return True
        # Not ours: spawned before a service restart, so only the signal probe is left.
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    # ---- discovery -------------------------------------------------------

    def _run_dirs(self) -> list[Path]:
        self._reap()
        seen: dict[Path, None] = {}
        for root in self.settings.scan_roots:
            if not root.is_dir():
                continue
            for candidate in sorted(root.iterdir()):
                # Runs are identified by their contents; sibling venvs and scratch
                # directories share the same parents and must not be listed as runs.
                if candidate.is_dir() and (candidate / "state.json").exists():
                    seen.setdefault(candidate.resolve(), None)
                elif candidate.is_dir() and (candidate / "job.json").exists():
                    seen.setdefault(candidate.resolve(), None)
        return list(seen)

    def resolve_dir(self, job_id: str) -> Path | None:
        if "/" in job_id or "\\" in job_id or job_id in {"", ".", ".."}:
            return None
        for candidate in self._run_dirs():
            if candidate.name == job_id:
                return candidate
        return None

    # ---- metadata --------------------------------------------------------

    def _read_job(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "job.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    def _write_job(self, run_dir: Path, job: dict[str, Any]) -> None:
        (run_dir / "job.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _reconcile(self, run_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
        """Trust the event log over the stored status; a killed process cannot update it.

        Only events after ``events_from`` count: a resumed run appends to the same log, so
        the previous attempt's ``job.finished`` would otherwise resolve the new attempt.
        """
        status = job.get("status", "unknown")
        if status not in ACTIVE:
            return job
        floor = int(job.get("events_from", 0))
        for record in reversed(read_events(run_dir, since=floor)):
            resolved = TERMINAL_EVENTS.get(record.get("type", ""))
            if resolved:
                job["status"] = resolved
                job["finished_at"] = record.get("ts")
                self._write_job(run_dir, job)
                return job
        pid = job.get("pid")
        if status == "running" and (not pid or not self._alive(int(pid))):
            job["status"] = "interrupted"
            job["finished_at"] = _now()
            self._write_job(run_dir, job)
        return job

    def summary(self, run_dir: Path) -> dict[str, Any]:
        job = self._reconcile(run_dir, self._read_job(run_dir))
        topic = job.get("topic")
        disciplines = job.get("disciplines", [])
        rounds_done = 0
        state_file = run_dir / "state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                topic = topic or state.get("topic")
                disciplines = disciplines or state.get("disciplines", [])
                rounds_done = len(state.get("rounds", []))
            except json.JSONDecodeError:
                pass
        return {
            "job_id": run_dir.name,
            "run_dir": str(run_dir),
            "scan_root": str(run_dir.parent),
            "status": job.get("status", "archived" if state_file.exists() else "unknown"),
            "topic": topic,
            "disciplines": disciplines,
            "label": job.get("label"),
            "created_at": job.get("created_at"),
            "finished_at": job.get("finished_at"),
            "rounds_done": rounds_done,
            "rounds_target": job.get("rounds_target"),
            "model": job.get("model"),
            "provider": job.get("provider"),
            "managed": (run_dir / "job.json").exists(),
            "has_events": events_path(run_dir).exists(),
        }

    def list_jobs(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        items = [self.summary(path) for path in self._run_dirs()]
        if status:
            items = [item for item in items if item["status"] == status]
        items.sort(key=lambda item: (item["created_at"] or "", item["job_id"]), reverse=True)
        return items[:limit]

    def active_count(self) -> int:
        return sum(
            1 for path in self._run_dirs()
            if self._reconcile(path, self._read_job(path)).get("status") == "running"
        )

    # ---- lifecycle -------------------------------------------------------

    def create(
        self,
        *,
        topic: str,
        disciplines: list[str],
        config: dict[str, Any],
        label: str | None,
    ) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{stamp}-{_slug(topic)}-{secrets.token_hex(3)}"
        run_dir = self.settings.runs_root / job_id
        run_dir.mkdir(parents=True)
        (run_dir / "request-config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        job = {
            "job_id": job_id,
            "topic": topic,
            "disciplines": disciplines,
            "label": label,
            "mode": "run",
            "status": "queued",
            "events_from": 0,
            "created_at": _now(),
            "rounds_target": config.get("run", {}).get("rounds"),
            "model": config.get("llm", {}).get("model"),
            "provider": config.get("llm", {}).get("provider"),
            "api_key_env": config.get("llm", {}).get("api_key_env"),
        }
        self._write_job(run_dir, job)
        self.maybe_start()
        return self.summary(run_dir)

    def maybe_start(self) -> None:
        if self.active_count() >= self.settings.max_concurrent:
            return
        queued = [
            path for path in self._run_dirs()
            if self._read_job(path).get("status") == "queued"
        ]
        queued.sort(key=lambda path: self._read_job(path).get("created_at") or "")
        for run_dir in queued[: self.settings.max_concurrent - self.active_count()]:
            self._spawn(run_dir)

    def _spawn(self, run_dir: Path) -> None:
        job = self._read_job(run_dir)
        env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
        env["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), str(RUNNER.parent)])
        env["PYTHONUNBUFFERED"] = "1"
        # A run sees only the one API key its own config names.
        key_env = job.get("api_key_env")
        if key_env and key_env in os.environ:
            env[key_env] = os.environ[key_env]
        with (run_dir / "runner.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(RUNNER), str(run_dir)],
                cwd=str(RUNNER.parents[2]),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
            )
        job.update(status="running", pid=process.pid, started_at=_now())
        self._children[process.pid] = process
        self._write_job(run_dir, job)

    def cancel(self, run_dir: Path) -> dict[str, Any]:
        job = self._reconcile(run_dir, self._read_job(run_dir))
        if job.get("status") == "queued":
            job.update(status="cancelled", finished_at=_now())
            self._write_job(run_dir, job)
            return self.summary(run_dir)
        pid = job.get("pid")
        if job.get("status") == "running" and pid:
            try:
                pgid = os.getpgid(int(pid))
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(50):
                    if not self._alive(int(pid)):
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            job.update(status="cancelled", finished_at=_now())
            self._write_job(run_dir, job)
        self.maybe_start()
        return self.summary(run_dir)

    def resume(self, run_dir: Path) -> dict[str, Any]:
        job = self._reconcile(run_dir, self._read_job(run_dir))
        if job.get("status") in ACTIVE:
            return self.summary(run_dir)
        if not (run_dir / "job.json").exists():
            # A run found under a scan root is somebody else's finished artifact, often a
            # committed one. Resuming would rewrite its state.json and litter the directory.
            raise ValueError(
                "该任务不是通过本界面启动的，在这里只能只读查看；"
                "若要从它继续运行，请先把它的目录复制到 runs 根目录下"
            )
        if not (run_dir / "config.json").exists():
            raise ValueError("该任务没有 config.json，无法继续运行")
        if not (run_dir / "request-config.json").exists():
            (run_dir / "request-config.json").write_text(
                (run_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        job.update({
            "job_id": run_dir.name,
            "topic": state.get("topic"),
            "disciplines": state.get("disciplines", []),
            "mode": "resume",
            "status": "queued",
            "events_from": last_seq(run_dir),
            "created_at": job.get("created_at") or _now(),
            "finished_at": None,
        })
        # The run resumes from config.json, so the env allowlist must follow that file
        # rather than whatever key the first attempt happened to name.
        llm = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        ).get("llm", {})
        job["api_key_env"] = llm.get("api_key_env")
        job["model"] = llm.get("model")
        job["provider"] = llm.get("provider")
        self._write_job(run_dir, job)
        self.maybe_start()
        return self.summary(run_dir)

    def is_active(self, run_dir: Path) -> bool:
        return self._reconcile(run_dir, self._read_job(run_dir)).get("status") in ACTIVE

    def events_from(self, run_dir: Path) -> int:
        """Watermark of the current attempt; events at or below it belong to earlier ones."""
        return int(self._read_job(run_dir).get("events_from", 0))
