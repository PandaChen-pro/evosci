from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import EvoSciConfig
from .engine import EvoSciEngine


def _progress(message: str) -> None:
    print(f"[evosci] {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evosci", description="Run the EvoSci reproduction workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="start a new evolutionary research run")
    run.add_argument("--topic", required=True)
    run.add_argument("--disciplines", required=True, help="comma-separated target disciplines")
    run.add_argument("--config", type=Path, help="TOML configuration file")
    run.add_argument("--output-dir", type=Path, help="override run.output_dir")

    resume = subparsers.add_parser("resume", help="continue an interrupted run")
    resume.add_argument("run_dir", type=Path)

    inspect = subparsers.add_parser("inspect", help="print a compact run summary")
    inspect.add_argument("run_dir", type=Path)
    inspect.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            config = EvoSciConfig.from_toml(args.config) if args.config else EvoSciConfig()
            if args.output_dir:
                config.run.output_dir = str(args.output_dir)
            engine = EvoSciEngine(config, progress=_progress)
            disciplines = [item.strip() for item in args.disciplines.split(",")]
            state, destination = engine.run(args.topic, disciplines)
            print(f"Run completed: {destination}")
            print(f"Rounds: {len(state.rounds)}")
            print(f"Report: {destination / 'report.md'}")
            return 0

        if args.command == "resume":
            state = EvoSciEngine.load_state(args.run_dir)
            engine = EvoSciEngine.resume(args.run_dir, progress=_progress)
            state, destination = engine.run(
                state.topic, state.disciplines, run_dir=args.run_dir, state=state
            )
            print(f"Run completed: {destination}")
            print(f"Rounds: {len(state.rounds)}")
            return 0

        if args.command == "inspect":
            state = EvoSciEngine.load_state(args.run_dir)
            evaluated = [item for result in state.rounds for item in result.evaluated_ideas]
            summary = {
                "topic": state.topic,
                "disciplines": state.disciplines,
                "rounds": len(state.rounds),
                "ideas": len(evaluated),
                "best_idea": max(evaluated, key=lambda item: item.fitness).idea.title if evaluated else None,
            }
            if args.as_json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                for key, value in summary.items():
                    print(f"{key}: {value}")
            return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"evosci: error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
