import unittest

from evosci.config import EvolutionConfig, GraphConfig
from evosci.evolution import EntityEvolution
from evosci.knowledge import KnowledgeGraph, stable_id
from evosci.llm import HeuristicBackend
from evosci.models import EvaluatedIdea, ResearchIdea, Review


def review(score: float) -> Review:
    return Review(
        reviewer_id="reviewer",
        novelty=score,
        feasibility=score,
        validity=score,
        excitement=score,
        overall=score,
        confidence=0.8,
        strengths=["testable"],
        weaknesses=["limited"],
        suggestions=["add controls"],
    )


class EvolutionTests(unittest.TestCase):
    def test_evolution_updates_fitness_and_population(self) -> None:
        graph = KnowledgeGraph(GraphConfig(similarity_threshold=0.3))
        graph.initialize(["physics"])
        entity_ids = [entity.id for entity in graph.entities_for("physics")[:2]]
        idea = ResearchIdea(
            id=stable_id("idea", "test"),
            title="Test idea",
            hypothesis="A changes B",
            rationale="Mechanistic bridge",
            method="controlled experiment",
            experiment="Matched intervention and control across five seeds",
            expected_outcome="Dose response",
            risks=["confounding"],
            entity_ids=entity_ids,
            round_index=1,
        )
        item = EvaluatedIdea(idea, [review(8.0)], review(8.0))
        before = len(graph.entities)
        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(crossover_count=1, variation_count=1),
            seed=1,
        )
        summary = evolution.evolve("grokking", [item], ["physics"], generation=1)
        self.assertGreater(len(graph.entities), before)
        self.assertTrue(summary["crossovers"])
        self.assertTrue(summary["variations"])
        for entity_id in entity_ids:
            self.assertGreater(graph.entities[entity_id].fitness, 0.5)


if __name__ == "__main__":
    unittest.main()
