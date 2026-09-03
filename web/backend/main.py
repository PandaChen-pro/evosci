"""EvoSci web API.

A tool for a trusted LAN, not an internet service. See web/README.md for what a token
holder can and cannot do.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from evosci.config import EvoSciConfig

from .artifacts import (
    artifact_list,
    find_idea,
    graph_payload,
    load_json,
    read_artifact,
    summarize_state,
)
from .auth import require_token, resolve_token
from .configspec import apply_overrides, default_config, field_spec
from .events import read_events, stream_events
from .jobs import JobStore
from .ledger import build_ledger
from .settings import WebSettings

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


class RunRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    disciplines: list[str] = Field(min_length=1, max_length=12)
    overrides: dict[str, Any] = Field(default_factory=dict)
    label: str | None = Field(default=None, max_length=120)


def create_app(settings: WebSettings | None = None) -> FastAPI:
    settings = settings or WebSettings.from_env()
    app = FastAPI(title="EvoSci Web", version="1.0")
    app.state.settings = settings
    app.state.token = resolve_token(settings)
    app.state.jobs = JobStore(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    guard = [Depends(require_token)]

    def store(request: Request) -> JobStore:
        return request.app.state.jobs

    def run_dir_or_404(request: Request, job_id: str) -> Path:
        path = store(request).resolve_dir(job_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        return path

    # ---- meta ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": app.version, "runs_root": str(settings.runs_root)}

    @app.get("/api/config/defaults", dependencies=guard)
    def config_defaults() -> dict[str, Any]:
        return {
            "config": default_config(),
            "spec": field_spec(),
            "limits": {"max_rounds": settings.max_rounds},
            "presets": _presets(),
        }

    @app.get("/api/env/check", dependencies=guard)
    def env_check(names: str = Query(default="")) -> dict[str, bool]:
        # Reports presence only. The value of an API key never leaves this process.
        wanted = [item.strip() for item in names.split(",") if item.strip()][:10]
        return {name: bool(os.environ.get(name)) for name in wanted}

    # ---- runs ------------------------------------------------------------

    @app.post("/api/runs", status_code=201, dependencies=guard)
    def create_run(request: Request, body: RunRequest) -> dict[str, Any]:
        data, invalid = apply_overrides(body.overrides)
        if invalid:
            raise HTTPException(
                status_code=422,
                detail={"message": "Unknown or forbidden config keys", "invalid_keys": invalid},
            )
        disciplines = [item.strip().lower() for item in body.disciplines if item.strip()]
        if not disciplines:
            raise HTTPException(status_code=422, detail="At least one discipline is required")
        try:
            config = EvoSciConfig.from_dict(data)
            if config.run.rounds > settings.max_rounds:
                raise ValueError(
                    f"run.rounds exceeds the server limit of {settings.max_rounds}"
                )
            config.validate()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return store(request).create(
            topic=body.topic.strip(),
            disciplines=disciplines,
            config=asdict(config),
            label=body.label,
        )

    @app.get("/api/runs", dependencies=guard)
    def list_runs(
        request: Request,
        status: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        jobs = store(request)
        jobs.maybe_start()
        return {"runs": jobs.list_jobs(status=status, limit=limit)}

    @app.get("/api/runs/{job_id}", dependencies=guard)
    def get_run(request: Request, job_id: str) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        state = load_json(path, "state.json")
        return {
            "job": store(request).summary(path),
            "state": summarize_state(state) if state else None,
            "config": load_json(path, "config.json"),
            "artifacts": artifact_list(path),
        }

    @app.get("/api/runs/{job_id}/ideas/{idea_id}", dependencies=guard)
    def get_idea(request: Request, job_id: str, idea_id: str) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        state = load_json(path, "state.json") or {}
        item = find_idea(state, idea_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Unknown idea")
        return item

    @app.get("/api/runs/{job_id}/graph", dependencies=guard)
    def get_graph(request: Request, job_id: str) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        graph = load_json(path, "graph.json")
        if graph is None:
            return {"entities": [], "edges": {}, "clusters": [], "has_clusters": False}
        return graph_payload(graph)

    @app.get("/api/runs/{job_id}/diagnostics", dependencies=guard)
    def get_diagnostics(request: Request, job_id: str) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        data = load_json(path, "diagnostics.json")
        return {"available": data is not None, "data": data}

    @app.get("/api/runs/{job_id}/feedback-ledger", dependencies=guard)
    def get_ledger(request: Request, job_id: str) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        state = load_json(path, "state.json")
        if state is None:
            return {"entries": [], "available": False, "note": "该任务没有 state.json。"}
        return build_ledger(state)

    @app.get("/api/runs/{job_id}/artifacts/{name}", dependencies=guard)
    def get_artifact(request: Request, job_id: str, name: str) -> Response:
        path = run_dir_or_404(request, job_id)
        found = read_artifact(path, name)
        if found is None:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        body, media = found
        return Response(content=body, media_type=media)

    # ---- events ----------------------------------------------------------

    @app.get("/api/runs/{job_id}/events", dependencies=guard)
    def get_events(request: Request, job_id: str, since: int = 0) -> StreamingResponse:
        path = run_dir_or_404(request, job_id)
        jobs = store(request)
        return StreamingResponse(
            stream_events(
                path,
                since=since,
                is_active=lambda: jobs.is_active(path),
                terminal_from=jobs.events_from(path),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{job_id}/events/page", dependencies=guard)
    def get_events_page(
        request: Request,
        job_id: str,
        since: int = 0,
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        events = read_events(path, since=since, limit=limit)
        return {
            "events": events,
            "last_seq": events[-1]["seq"] if events else since,
            "active": store(request).is_active(path),
        }

    # ---- control ---------------------------------------------------------

    @app.post("/api/runs/{job_id}/cancel", dependencies=guard)
    def cancel_run(request: Request, job_id: str) -> dict[str, Any]:
        return store(request).cancel(run_dir_or_404(request, job_id))

    @app.post("/api/runs/{job_id}/resume", dependencies=guard)
    def resume_run(request: Request, job_id: str) -> dict[str, Any]:
        path = run_dir_or_404(request, job_id)
        try:
            return store(request).resume(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if FRONTEND_DIST.is_dir():
        # Client-side routes have no file behind them, so a refresh on /runs/<id> — the
        # mid-run reload the event cursor exists for — must still get the app shell.
        class SpaFiles(StaticFiles):
            async def get_response(self, path: str, scope: Any) -> Response:
                try:
                    return await super().get_response(path, scope)
                except StarletteHTTPException as exc:
                    if exc.status_code != 404:
                        raise
                    # An unknown /api path must stay a 404; only app routes get the shell.
                    # Dot segments are checked too: StaticFiles normalises
                    # ``api/runs/../..`` down past the api/ prefix before we see it.
                    if path == "api" or path.startswith("api/") or ".." in path.split("/"):
                        raise
                    return await super().get_response("index.html", scope)

        app.mount("/", SpaFiles(directory=str(FRONTEND_DIST), html=True), name="ui")

    return app


def _presets() -> list[dict[str, Any]]:
    """Offer the repository's example TOMLs as starting points, minus any secret."""
    examples = Path(__file__).resolve().parents[2] / "examples"
    out = []
    for path in sorted(examples.glob("*.toml")):
        try:
            config = EvoSciConfig.from_toml(path)
        except (OSError, ValueError, TypeError):
            continue
        out.append({"name": path.stem, "config": asdict(config)})
    return out


def __getattr__(name: str) -> Any:
    """Build the app only when something asks for it.

    ``uvicorn web.backend.main:app`` needs a module attribute, but creating it eagerly
    would read the environment and build a JobStore just from importing this module —
    which is how ``web/serve.py`` ends up with two of them and loses its own startup checks.
    """
    if name == "app":
        return create_app()
    raise AttributeError(name)
