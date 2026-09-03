"""Read run artifacts, trimmed for the wire.

A real ``state.json`` reaches 125KB, and a single ``idea.experiment`` field has measured
6785 characters — so the list view returns scores and titles only, and the long prose is
fetched per idea on demand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT_FILES = {
    "config.json": "application/json",
    "state.json": "application/json",
    "graph.json": "application/json",
    "diagnostics.json": "application/json",
    "diagnostics.md": "text/markdown; charset=utf-8",
    "report.md": "text/markdown; charset=utf-8",
    "runner.log": "text/plain; charset=utf-8",
}

IDEA_LIST_FIELDS = ("id", "title", "hypothesis", "round_index", "entity_ids", "authors")
SCORE_FIELDS = ("novelty", "feasibility", "validity", "excitement", "overall", "confidence")


def load_json(run_dir: Path, name: str) -> Any | None:
    path = run_dir / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def artifact_list(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for name, media in ARTIFACT_FILES.items():
        path = run_dir / name
        if path.exists():
            out.append({"name": name, "media_type": media, "bytes": path.stat().st_size})
    return out


def read_artifact(run_dir: Path, name: str) -> tuple[str, str] | None:
    """Only the fixed set above is readable; the requested name is never path-joined."""
    media = ARTIFACT_FILES.get(name)
    if media is None:
        return None
    path = run_dir / name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace"), media


def _scores(review: dict[str, Any]) -> dict[str, Any]:
    return {field: review.get(field) for field in SCORE_FIELDS}


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    rounds = []
    for entry in state.get("rounds", []):
        ideas = []
        for item in entry.get("evaluated_ideas", []):
            idea = item.get("idea", {})
            meta = item.get("meta_review", {})
            ideas.append({
                **{field: idea.get(field) for field in IDEA_LIST_FIELDS},
                "fitness": item.get("fitness"),
                "review_count": len(item.get("reviews", [])),
                "meta_scores": _scores(meta),
                "suggestions": meta.get("suggestions", []),
            })
        rounds.append({
            "round_index": entry.get("round_index"),
            "problems": entry.get("problems", []),
            "ideas": ideas,
            "evolution_summary": normalize_evolution(entry.get("evolution_summary", {})),
        })
    return {
        "topic": state.get("topic"),
        "disciplines": state.get("disciplines", []),
        "rounds": rounds,
    }


def normalize_evolution(summary: dict[str, Any]) -> dict[str, Any]:
    """Crossovers were bare entity ids before they became {entity_id, from, to} records.

    Both shapes exist in real run directories, so the endpoint reports which one it found
    rather than making the frontend guess from the payload.
    """
    raw = summary.get("crossovers", [])
    detailed = bool(raw) and isinstance(raw[0], dict)
    crossovers = [
        item if isinstance(item, dict) else {"entity_id": item, "from": None, "to": None}
        for item in raw
    ]
    return {**summary, "crossovers": crossovers, "crossovers_detailed": detailed}


def find_idea(state: dict[str, Any], idea_id: str) -> dict[str, Any] | None:
    for entry in state.get("rounds", []):
        for item in entry.get("evaluated_ideas", []):
            if item.get("idea", {}).get("id") == idea_id:
                return {**item, "round_index": entry.get("round_index")}
    return None


def graph_payload(graph: dict[str, Any]) -> dict[str, Any]:
    clusters = graph.get("clusters")
    return {
        "entities": graph.get("entities", []),
        "edges": graph.get("edges", {}),
        "clusters": clusters or [],
        "has_clusters": clusters is not None,
    }
