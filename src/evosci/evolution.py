from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .config import EvolutionConfig
from .knowledge import KnowledgeGraph, stable_id
from .llm import LLMBackend
from .models import Entity, EvaluatedIdea
from .parsing import as_string_list


class EntityEvolution:
    def __init__(
        self,
        graph: KnowledgeGraph,
        backend: LLMBackend,
        config: EvolutionConfig,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.backend = backend
        self.config = config
        self.rng = random.Random(seed)

    def evolve(
        self,
        topic: str,
        evaluated_ideas: list[EvaluatedIdea],
        disciplines: list[str],
        generation: int,
    ) -> dict[str, Any]:
        self._apply_fitness(evaluated_ideas)
        summary: dict[str, Any] = {
            "generation": generation,
            "selected": [],
            "crossovers": [],
            "variations": [],
            "pruned": [],
        }
        for discipline in disciplines:
            population = self.graph.entities_for(discipline)
            if not population:
                continue
            population.sort(key=lambda entity: entity.fitness, reverse=True)
            keep_count = max(2, round(len(population) * self.config.selection_ratio))
            selected = population[:keep_count]
            summary["selected"].extend(entity.id for entity in selected)
            summary["crossovers"].extend(
                self._crossover(discipline, selected, generation)
            )
            summary["variations"].extend(
                self._variation(topic, discipline, selected, generation)
            )
            summary["pruned"].extend(self._prune(discipline))
        self.graph.rebuild_edges()
        return summary

    def _apply_fitness(self, evaluated_ideas: list[EvaluatedIdea]) -> None:
        evidence: dict[str, list[float]] = defaultdict(list)
        for item in evaluated_ideas:
            for entity_id in item.idea.entity_ids:
                if entity_id in self.graph.entities:
                    evidence[entity_id].append(item.fitness)
        for entity in self.graph.entities.values():
            entity.fitness *= self.config.fitness_decay
            if entity.id in evidence:
                signal = sum(evidence[entity.id]) / len(evidence[entity.id])
                entity.fitness = 0.45 * entity.fitness + 0.55 * signal
            entity.fitness = max(0.0, min(1.0, entity.fitness))

    def _crossover(
        self, discipline: str, selected: list[Entity], generation: int
    ) -> list[str]:
        if len(selected) < 2:
            return []
        created = []
        pairs = []
        for offset in range(self.config.crossover_count):
            left = selected[offset % len(selected)]
            right = selected[-((offset % len(selected)) + 1)]
            if left.id != right.id:
                pairs.append((left, right))
        for left, right in pairs:
            name = f"{left.name} + {right.name}"
            entity = self.graph.add_entity(Entity(
                id=stable_id("ent", discipline, name),
                name=name,
                discipline=discipline,
                kind="CrossoverConcept",
                description=(
                    f"Recombines {left.name} and {right.name} for interdisciplinary exploration."
                ),
                source="crossover",
                fitness=(left.fitness + right.fitness) / 2,
                generation=generation,
                parents=[left.id, right.id],
            ))
            created.append(entity.id)
        return list(dict.fromkeys(created))

    def _variation(
        self,
        topic: str,
        discipline: str,
        selected: list[Entity],
        generation: int,
    ) -> list[str]:
        anchors = [entity.name for entity in selected[:4]]
        if not anchors:
            return []
        response = self.backend.generate_json(
            task="variation",
            system=(
                "Generate scientifically meaningful low-frequency concepts that vary the "
                "selected entity population without merely renaming existing entities."
            ),
            user=(
                f"Topic: {topic}\nDiscipline: {discipline}\nAnchors: {anchors}\n"
                f"Generate {self.config.variation_count} variations."
            ),
            schema={
                "entities": [{"name": "string", "kind": "string", "description": "string"}]
            },
            context={
                "topic": topic,
                "discipline": discipline,
                "anchors": anchors,
                "count": self.config.variation_count,
                "generation": generation,
            },
        )
        created = []
        for item in response.get("entities", [])[: self.config.variation_count]:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            parent = selected[len(created) % len(selected)]
            entity = self.graph.add_entity(Entity(
                id=stable_id("ent", discipline, name),
                name=name,
                discipline=discipline,
                kind=str(item.get("kind") or "EvolvedConcept"),
                description=str(item.get("description") or ""),
                source="variation",
                fitness=max(0.25, parent.fitness * 0.8),
                generation=generation,
                parents=[parent.id],
            ))
            created.append(entity.id)
        return list(dict.fromkeys(created))

    def _prune(self, discipline: str) -> list[str]:
        population = self.graph.entities_for(discipline)
        ranked = sorted(population, key=lambda entity: entity.fitness, reverse=True)
        protected = {entity.id for entity in ranked[:2]}
        remove = [
            entity for entity in ranked
            if entity.id not in protected and entity.fitness < self.config.min_fitness
        ]
        remaining = len(ranked) - len(remove)
        if remaining > self.config.max_population_per_discipline:
            overflow = remaining - self.config.max_population_per_discipline
            candidates = [
                entity for entity in reversed(ranked)
                if entity.id not in protected and entity not in remove
            ]
            remove.extend(candidates[:overflow])
        removed_ids = []
        for entity in remove:
            removed_ids.append(entity.id)
            self.graph.entities.pop(entity.id, None)
            self.graph.edges.pop(entity.id, None)
            for neighbors in self.graph.edges.values():
                neighbors.pop(entity.id, None)
        return removed_ids
