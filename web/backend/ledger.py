"""Reconstruct which review suggestions were carried into the next round's prompt.

``EvoSciEngine._prior_feedback`` ranks the previous round's ideas by fitness, takes the
top three, flattens their meta-review suggestions, dedupes, and truncates to eight. That
last step is recorded in no artifact: a suggestion that fell off the end looks identical
to one that was never made. This rebuilds the boundary from ``state.json`` alone.

The selection below must stay a literal mirror of ``engine.py:_prior_feedback`` — if that
function changes, this one is wrong and the ledger silently misattributes.
"""

from __future__ import annotations

from typing import Any

TOP_IDEAS = 3
MAX_SUGGESTIONS = 8


def _candidates(round_entry: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = sorted(
        round_entry.get("evaluated_ideas", []),
        key=lambda item: item.get("fitness", 0.0),
        reverse=True,
    )
    seen: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(ranked[:TOP_IDEAS], 1):
        idea = item.get("idea", {})
        for text in item.get("meta_review", {}).get("suggestions", []):
            if text in seen:
                continue
            seen[text] = {
                "text": text,
                "source_idea_id": idea.get("id"),
                "source_idea_title": idea.get("title"),
                "source_idea_rank": rank,
                "source_idea_fitness": item.get("fitness"),
            }
    return list(seen.values())


def build_ledger(state: dict[str, Any]) -> dict[str, Any]:
    rounds = state.get("rounds", [])
    entries = []
    for index in range(1, len(rounds)):
        previous = rounds[index - 1]
        candidates = _candidates(previous)
        carried = candidates[:MAX_SUGGESTIONS]
        dropped = candidates[MAX_SUGGESTIONS:]
        contributing = len(previous.get("evaluated_ideas", []))
        entries.append({
            "into_round": rounds[index].get("round_index"),
            "from_round": previous.get("round_index"),
            "candidate_count": len(candidates),
            "carried": carried,
            "dropped": dropped,
            "limit": MAX_SUGGESTIONS,
            "considered_ideas": min(TOP_IDEAS, contributing),
            "total_ideas_in_source_round": contributing,
        })
    return {
        "entries": entries,
        "available": bool(entries),
        "note": (
            "本账本由 state.json 重建，逐字镜像 EvoSciEngine._prior_feedback 的选取逻辑。"
            "被带入的建议原文进入了下一轮的 prompt —— 这是一份溯源记录，"
            "不是「模型采纳了这些建议」的论断。"
        ),
    }
