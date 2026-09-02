from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

from .config import GraphConfig
from .models import Entity, EntityCluster


DISCIPLINE_SEEDS: dict[str, list[str]] = {
    "computer science": [
        "representation learning", "optimization", "generalization", "attention",
        "causal inference", "graph neural networks", "reinforcement learning", "uncertainty",
    ],
    "mathematics": [
        "dynamical systems", "topology", "information geometry", "spectral analysis",
        "stochastic processes", "optimization theory", "symmetry", "fixed points",
    ],
    "physics": [
        "phase transition", "entropy", "statistical mechanics", "energy landscape",
        "critical phenomena", "renormalization", "nonequilibrium dynamics", "symmetry breaking",
    ],
    "cognitive science": [
        "working memory", "concept formation", "skill acquisition", "attention control",
        "metacognition", "cognitive load", "memory consolidation", "transfer learning",
    ],
    "biology": [
        "evolution", "adaptation", "regulatory networks", "homeostasis",
        "population dynamics", "selection pressure", "phenotypic plasticity", "robustness",
    ],
    "medicine": [
        "clinical validation", "biomarkers", "treatment response", "confounding",
        "heterogeneity", "risk stratification", "longitudinal study", "external validity",
    ],
    "chemistry": [
        "reaction kinetics", "catalysis", "molecular dynamics", "chemical equilibrium",
        "structure property relation", "reaction pathway", "free energy", "selectivity",
    ],
    "economics": [
        "causal identification", "incentives", "equilibrium", "market dynamics",
        "mechanism design", "bounded rationality", "counterfactual policy", "heterogeneity",
    ],
    "earth science": [
        "seismic dynamics", "spatial correlation", "extreme events", "fault systems",
        "early warning", "geophysical inversion", "temporal clustering", "uncertainty",
    ],
    "materials science": [
        "crystal symmetry", "defect dynamics", "multiscale modeling", "phase stability",
        "structure property relation", "equivariant representation", "microstructure", "transport",
    ],
    "software vulnerability detection": [
        "CodeBERT token semantics",
        "GraphCodeBERT data-flow representation",
        "TACC-S/DFA program analysis",
        "interprocedural data-flow graph",
        "call graph and control-flow graph",
        "code property graph",
        "repository dependency graph",
        "commit and patch history",
        "API usage and configuration",
        "natural-language security context",
        "cross-function taint propagation",
        "weak supervision and vulnerability localization",
        "multimodal contrastive alignment",
        "missing-modality robustness",
        "long-context hierarchical encoding",
        "cross-repository generalization",
    ],
}


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    trigrams = [word[index : index + 3] for word in words for index in range(max(0, len(word) - 2))]
    return words + trigrams


def text_similarity(left: str, right: str) -> float:
    a = Counter(_tokens(left))
    b = Counter(_tokens(right))
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(token, 0) for token, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b)


class KnowledgeGraph:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, dict[str, float]] = defaultdict(dict)

    def add_entity(self, entity: Entity) -> Entity:
        existing = self.find_by_name(entity.name, entity.discipline)
        if existing:
            existing.fitness = max(existing.fitness, entity.fitness)
            if entity.description and not existing.description:
                existing.description = entity.description
            return existing
        self.entities[entity.id] = entity
        return entity

    def find_by_name(self, name: str, discipline: str | None = None) -> Entity | None:
        normalized = name.strip().lower()
        for entity in self.entities.values():
            if entity.name.strip().lower() != normalized:
                continue
            if discipline is None or entity.discipline.lower() == discipline.lower():
                return entity
        return None

    def entities_for(self, discipline: str) -> list[Entity]:
        normalized = discipline.lower()
        return [
            entity for entity in self.entities.values()
            if entity.discipline.lower() == normalized
        ]

    def initialize(self, disciplines: Iterable[str]) -> None:
        for discipline in disciplines:
            normalized = discipline.strip().lower()
            seeds = DISCIPLINE_SEEDS.get(normalized, [
                f"{normalized} mechanism", f"{normalized} modeling",
                f"{normalized} measurement", f"{normalized} validation",
            ])
            for name in seeds:
                self.add_entity(Entity(
                    id=stable_id("ent", normalized, name),
                    name=name,
                    discipline=normalized,
                    source="built-in",
                ))
            if self.config.use_wikipedia:
                for name, description in self._wikipedia_entities(normalized):
                    if len(self.entities_for(normalized)) >= self.config.max_entities_per_discipline:
                        break
                    self.add_entity(Entity(
                        id=stable_id("ent", normalized, name),
                        name=name,
                        discipline=normalized,
                        description=description,
                        source="wikipedia",
                    ))
        self.rebuild_edges()

    def _wikipedia_entities(self, discipline: str) -> list[tuple[str, str]]:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
            discipline.replace(" ", "_")
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "EvoSciReproduction/0.1"})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            extract = payload.get("extract", "")
            phrases = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+|$)){1,3}", extract)
            unique = list(dict.fromkeys(phrase.strip() for phrase in phrases if len(phrase.strip()) > 4))
            return [(phrase, extract[:300]) for phrase in unique[:12]]
        except (OSError, ValueError, json.JSONDecodeError):
            return []

    def rebuild_edges(self) -> None:
        self.edges = defaultdict(dict)
        entities = list(self.entities.values())
        for index, left in enumerate(entities):
            for right in entities[index + 1 :]:
                score = text_similarity(
                    f"{left.name} {left.description}",
                    f"{right.name} {right.description}",
                )
                if score >= self.config.similarity_threshold:
                    self.edges[left.id][right.id] = score
                    self.edges[right.id][left.id] = score

    def clusters(self, discipline: str, topic: str) -> list[EntityCluster]:
        members = self.entities_for(discipline)
        member_ids = {entity.id for entity in members}
        unseen = set(member_ids)
        components: list[list[str]] = []
        while unseen:
            start = unseen.pop()
            queue = deque([start])
            component = [start]
            while queue:
                current = queue.popleft()
                for neighbor in self.edges.get(current, {}):
                    if neighbor in unseen and neighbor in member_ids:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
            components.append(component)
        components.sort(
            key=lambda ids: max(
                text_similarity(topic, self.entities[entity_id].name) for entity_id in ids
            ),
            reverse=True,
        )
        clusters = []
        for index, ids in enumerate(components[: self.config.cluster_limit]):
            relevance = sum(
                0.65 * text_similarity(topic, self.entities[entity_id].name)
                + 0.35 * self.entities[entity_id].fitness
                for entity_id in ids
            ) / len(ids)
            clusters.append(EntityCluster(
                id=stable_id("cluster", discipline, str(index), *sorted(ids)),
                discipline=discipline,
                entity_ids=ids,
                score=relevance,
            ))
        return clusters

    def top_entities(self, discipline: str, topic: str, limit: int = 8) -> list[Entity]:
        return sorted(
            self.entities_for(discipline),
            key=lambda entity: 0.55 * text_similarity(topic, entity.name) + 0.45 * entity.fitness,
            reverse=True,
        )[:limit]

    def to_dict(self) -> dict[str, object]:
        return {
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "edges": {key: value for key, value in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, config: GraphConfig, data: dict[str, object]) -> "KnowledgeGraph":
        graph = cls(config)
        for item in data.get("entities", []):
            graph.add_entity(Entity.from_dict(item))
        graph.edges = defaultdict(dict, {
            str(key): {str(other): float(score) for other, score in value.items()}
            for key, value in data.get("edges", {}).items()
        })
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
