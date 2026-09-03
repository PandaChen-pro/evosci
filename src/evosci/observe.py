"""Per-call telemetry around any LLM backend.

Every model call in this package goes through ``LLMBackend.generate_json``, so wrapping
that one method observes the whole run without touching agents, evolution, or either
backend implementation. Retries are the exception: they happen inside
``OpenAICompatibleBackend.generate_json``, so that class reports them through its own
``on_attempt`` hook.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .llm import LLMBackend, OpenAICompatibleBackend

Emit = Callable[[str, dict[str, Any]], None]


def _size(payload: dict[str, Any]) -> int | None:
    """Response size as a proxy for cost. The API's own token `usage` is discarded by
    OpenAICompatibleBackend, so this is the only measure available."""
    try:
        return len(json.dumps(payload, default=str))
    except (TypeError, ValueError):
        return None


class ObservedBackend(LLMBackend):
    def __init__(self, inner: LLMBackend, emit: Emit) -> None:
        self._inner = inner
        self._emit = emit
        self._counter = 0
        self._context: dict[str, Any] = {}
        if isinstance(inner, OpenAICompatibleBackend):
            inner.on_attempt = self._on_attempt

    def set_context(self, **fields: Any) -> None:
        """Tag subsequent calls with the round and phase the engine is in."""
        self._context = {key: value for key, value in fields.items() if value is not None}

    def _publish(self, event: str, data: dict[str, Any]) -> None:
        try:
            self._emit(event, {**self._context, **data})
        except Exception:
            pass

    def _on_attempt(self, payload: dict[str, Any]) -> None:
        self._publish("llm.attempt", {"call_id": self._call_id, **payload})

    @property
    def _call_id(self) -> str:
        return f"c-{self._counter:04d}"

    def generate_json(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._counter += 1
        call_id = self._call_id
        model = getattr(getattr(self._inner, "config", None), "model", "offline-heuristic")
        prompt_chars = len(system) + len(user)
        self._publish(
            "llm.start",
            {"call_id": call_id, "task": task, "model": model, "prompt_chars": prompt_chars},
        )
        started = time.perf_counter()
        try:
            result = self._inner.generate_json(
                task=task, system=system, user=user, schema=schema, context=context
            )
        except Exception as exc:
            self._publish(
                "llm.end",
                {
                    "call_id": call_id,
                    "task": task,
                    "model": model,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        self._publish(
            "llm.end",
            {
                "call_id": call_id,
                "task": task,
                "model": model,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "ok": True,
                "error": None,
                "prompt_chars": prompt_chars,
                "response_chars": _size(result),
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define. Guarding the private
        # names keeps an unpickled or partially built instance from recursing forever.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)
