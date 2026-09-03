"""Structural health metrics for an evolutionary run.

The paper claims three properties in Appendix D.4 — heritable variation, fitness-guided
selection, and diversity that contracts without collapsing — plus intra-/inter-round
similarity in D.3. Nothing here judges idea *quality*: Appendix D.2 states quality gains
are non-monotonic, and the offline heuristic reviewer scores by length and keyword
presence, so it cannot evidence quality. These metrics answer whether the population is
structurally alive.

Read-only: this module never mutates the graph or the run state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .knowledge import KnowledgeGraph, text_similarity
from .models import EvaluatedIdea, RunState


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def gini(values: list[float]) -> float:
    """Inequality of a non-negative distribution: 0.0 is uniform, 1.0 maximally unequal."""
    positive = [value for value in values if value > 0.0]
    if len(positive) < 2:
        return 0.0
    ordered = sorted(positive)
    total = sum(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered)


def edge_density(graph: KnowledgeGraph) -> float:
    count = len(graph.entities)
    if count < 2:
        return 0.0
    edges = sum(len(neighbors) for neighbors in graph.edges.values()) / 2
    return edges / (count * (count - 1) / 2)


def _idea_text(idea: Any) -> str:
    return " ".join([idea.title, idea.hypothesis, idea.rationale, idea.method])


def intra_round_convergence(ideas: list[EvaluatedIdea]) -> float:
    """Mean pairwise similarity within one round (paper D.3)."""
    texts = [_idea_text(item.idea) for item in ideas]
    pairs = [
        text_similarity(texts[left], texts[right])
        for left in range(len(texts))
        for right in range(left + 1, len(texts))
    ]
    return mean(pairs)


def inter_round_continuity(
    earlier: list[EvaluatedIdea], later: list[EvaluatedIdea]
) -> float:
    """Similarity between the aggregated text of two adjacent rounds (paper D.3)."""
    if not earlier or not later:
        return 0.0
    return text_similarity(
        " ".join(_idea_text(item.idea) for item in earlier),
        " ".join(_idea_text(item.idea) for item in later),
    )


@dataclass(slots=True)
class ClusterHealth:
    discipline: str
    count: int
    sizes: list[int]
    largest_share: float
    singletons: int
    fitness_spread: float


@dataclass(slots=True)
class RoundDiagnostics:
    round_index: int
    idea_count: int
    idea_fitness_mean: float
    idea_fitness_spread: float
    clusters_credited: int
    intra_round_convergence: float
    inter_round_continuity: float


@dataclass(slots=True)
class RunDiagnostics:
    entity_count: int
    edge_density: float
    entity_fitness_mean: float
    entity_fitness_spread: float
    entity_fitness_gini: float
    longest_entity_name: int
    concatenated_names: int
    clusters: list[ClusterHealth] = field(default_factory=list)
    rounds: list[RoundDiagnostics] = field(default_factory=list)
    distinct_clusters_credited: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "edge_density": self.edge_density,
            "entity_fitness_mean": self.entity_fitness_mean,
            "entity_fitness_spread": self.entity_fitness_spread,
            "entity_fitness_gini": self.entity_fitness_gini,
            "longest_entity_name": self.longest_entity_name,
            "concatenated_names": self.concatenated_names,
            "distinct_clusters_credited": self.distinct_clusters_credited,
            "clusters": [
                {
                    "discipline": item.discipline,
                    "count": item.count,
                    "sizes": item.sizes,
                    "largest_share": item.largest_share,
                    "singletons": item.singletons,
                    "fitness_spread": item.fitness_spread,
                }
                for item in self.clusters
            ],
            "rounds": [
                {
                    "round_index": item.round_index,
                    "idea_count": item.idea_count,
                    "idea_fitness_mean": item.idea_fitness_mean,
                    "idea_fitness_spread": item.idea_fitness_spread,
                    "clusters_credited": item.clusters_credited,
                    "intra_round_convergence": item.intra_round_convergence,
                    "inter_round_continuity": item.inter_round_continuity,
                }
                for item in self.rounds
            ],
        }

    def render(self) -> str:
        lines = [
            "# Structural diagnostics",
            "",
            f"- entities: {self.entity_count}",
            f"- edge density: {self.edge_density:.3f}",
            f"- entity fitness: mean {self.entity_fitness_mean:.3f}, "
            f"sd {self.entity_fitness_spread:.3f}, gini {self.entity_fitness_gini:.3f}",
            f"- longest entity name: {self.longest_entity_name} chars "
            f"({self.concatenated_names} concatenated)",
            f"- distinct clusters credited: {self.distinct_clusters_credited}",
            "",
            "## Clusters",
            "",
            "| discipline | clusters | sizes | largest share | singletons | fitness sd |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.clusters:
            sizes = ", ".join(str(size) for size in item.sizes)
            lines.append(
                f"| {item.discipline} | {item.count} | {sizes} | "
                f"{item.largest_share:.2f} | {item.singletons} | {item.fitness_spread:.3f} |"
            )
        lines += [
            "",
            "## Rounds",
            "",
            "| round | ideas | fitness mean | fitness sd | clusters credited "
            "| intra-round | inter-round |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.rounds:
            lines.append(
                f"| {item.round_index} | {item.idea_count} | {item.idea_fitness_mean:.3f} "
                f"| {item.idea_fitness_spread:.3f} | {item.clusters_credited} "
                f"| {item.intra_round_convergence:.3f} | {item.inter_round_continuity:.3f} |"
            )
        return "\n".join(lines) + "\n"


def _credited_clusters(graph: KnowledgeGraph, ideas: list[EvaluatedIdea]) -> set[str]:
    referenced = {
        entity_id for item in ideas for entity_id in item.idea.entity_ids
        if entity_id in graph.entities
    }
    credited = set()
    for cluster in graph.all_clusters():
        if referenced & set(cluster.entity_ids):
            credited.add(cluster.id)
    return credited


def diagnose(state: RunState, graph: KnowledgeGraph) -> RunDiagnostics:
    entities = list(graph.entities.values())
    fitness = [entity.fitness for entity in entities]
    names = [entity.name for entity in entities]
    report = RunDiagnostics(
        entity_count=len(entities),
        edge_density=edge_density(graph),
        entity_fitness_mean=mean(fitness),
        entity_fitness_spread=stdev(fitness),
        entity_fitness_gini=gini(fitness),
        longest_entity_name=max((len(name) for name in names), default=0),
        concatenated_names=sum(1 for name in names if " + " in name),
    )
    for discipline in state.disciplines:
        clusters = graph.clusters(discipline, state.topic)
        if not clusters:
            continue
        sizes = [len(cluster.entity_ids) for cluster in clusters]
        report.clusters.append(ClusterHealth(
            discipline=discipline,
            count=len(clusters),
            sizes=sizes,
            largest_share=max(sizes) / sum(sizes),
            singletons=sum(1 for size in sizes if size < 2),
            fitness_spread=stdev([cluster.fitness for cluster in clusters]),
        ))
    credited_overall: set[str] = set()
    for index, round_result in enumerate(state.rounds):
        ideas = round_result.evaluated_ideas
        credited = _credited_clusters(graph, ideas)
        credited_overall |= credited
        previous = state.rounds[index - 1].evaluated_ideas if index else []
        report.rounds.append(RoundDiagnostics(
            round_index=round_result.round_index,
            idea_count=len(ideas),
            idea_fitness_mean=mean([item.fitness for item in ideas]),
            idea_fitness_spread=stdev([item.fitness for item in ideas]),
            clusters_credited=len(credited),
            intra_round_convergence=intra_round_convergence(ideas),
            inter_round_continuity=inter_round_continuity(previous, ideas),
        ))
    report.distinct_clusters_credited = len(credited_overall)
    return report
