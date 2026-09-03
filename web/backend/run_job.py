"""Subprocess entry point for one EvoSci run.

Started by the web service with ``start_new_session=True`` so the run survives a service
restart and can be cancelled as a whole process group. Everything it reports flows out
through ``events.jsonl`` in its own run directory — there is no IPC channel to keep alive.

The service never imports this module; it only spawns it.
"""

from __future__ import annotations

import json
import re
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evosci.config import EvoSciConfig  # noqa: E402
from evosci.engine import EvoSciEngine  # noqa: E402
from evosci.llm import build_backend  # noqa: E402
from evosci.observe import ObservedBackend  # noqa: E402

from events import EventWriter  # noqa: E402

PHASE_PATTERNS = [
    (re.compile(r"knowledge graph", re.I), "init"),
    (re.compile(r"problem space", re.I), "problems"),
    (re.compile(r"research team", re.I), "team"),
    (re.compile(r"reviewing", re.I), "review"),
    (re.compile(r"evolving", re.I), "evolve"),
    (re.compile(r"tournament", re.I), "tournament"),
]
ROUND_PATTERN = re.compile(r"Round (\d+)")


def classify(message: str) -> tuple[str, int | None]:
    phase = next((name for pattern, name in PHASE_PATTERNS if pattern.search(message)), "other")
    match = ROUND_PATTERN.search(message)
    return phase, int(match.group(1)) if match else None


class Cancelled(Exception):
    pass


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    writer = EventWriter(run_dir)

    # Default SIGTERM handling exits without unwinding, so a cancelled run would never
    # write a terminal event and every attached SSE client would hang until timeout.
    def on_terminate(signum, frame):
        raise Cancelled()

    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGINT, on_terminate)

    try:
        # A resume must read the same file EvoSciEngine.resume does, or the observed
        # backend would be built from a config the engine never loads.
        source = "config.json" if job.get("mode") == "resume" else "request-config.json"
        config = EvoSciConfig.from_dict(
            json.loads((run_dir / source).read_text(encoding="utf-8"))
        )
        config.run.output_dir = str(run_dir.parent)
        config.validate()

        backend = ObservedBackend(
            build_backend(config.llm, seed=config.run.random_seed),
            lambda event, data: writer.emit(event, data),
        )

        def progress(message: str) -> None:
            phase, round_index = classify(message)
            backend.set_context(phase=phase, round=round_index)
            writer.emit("step", {"message": message, "phase": phase, "round": round_index})

        writer.emit("job.started", {
            "topic": job["topic"],
            "disciplines": job["disciplines"],
            "model": config.llm.model,
            "provider": config.llm.provider,
            "rounds": config.run.rounds,
        })

        if job.get("mode") == "resume":
            state = EvoSciEngine.load_state(run_dir)
            engine = EvoSciEngine.resume(run_dir, backend=backend, progress=progress)
            state, _ = engine.run(
                state.topic, state.disciplines, run_dir=run_dir, state=state
            )
        else:
            engine = EvoSciEngine(config, backend=backend, progress=progress)
            state, _ = engine.run(job["topic"], job["disciplines"], run_dir=run_dir)

        writer.emit("job.finished", {
            "rounds": len(state.rounds),
            "ideas": sum(len(item.evaluated_ideas) for item in state.rounds),
        })
        return 0
    except Cancelled:
        writer.emit("job.cancelled", {"reason": "terminated by request"})
        return 130
    except BaseException as exc:
        writer.emit("job.failed", {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
