import unittest

from evosci.config import GraphConfig
from evosci.diagnostics import (
    diagnose,
    edge_density,
    gini,
    inter_round_continuity,
    intra_round_convergence,
    stdev,
)
from evosci.knowledge import KnowledgeGraph, stable_id
from evosci.models import EvaluatedIdea, ResearchIdea, Review, RoundResult, RunState


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


def idea(suffix: str, entity_ids: list[str], round_index: int) -> ResearchIdea:
    return ResearchIdea(
        id=stable_id("idea", suffix),
        title=f"Idea {suffix}",
        hypothesis=f"{suffix} drives the transition",
        rationale="Mechanistic bridge",
        method="controlled experiment",
        experiment="Matched intervention and control",
        expected_outcome="Dose response",
        risks=["confounding"],
        entity_ids=entity_ids,
        round_index=round_index,
    )


class DiagnosticsTests(unittest.TestCase):
    def test_stdev_and_gini_on_known_distributions(self) -> None:
        self.assertEqual(stdev([0.5]), 0.0)
        self.assertEqual(stdev([0.5, 0.5, 0.5]), 0.0)
        self.assertGreater(stdev([0.1, 0.9]), stdev([0.4, 0.6]))
        self.assertEqual(gini([0.5, 0.5, 0.5]), 0.0)
        self.assertGreater(gini([0.05, 0.05, 0.9]), gini([0.3, 0.3, 0.4]))

    def test_edge_density_is_zero_without_edges(self) -> None:
        graph = KnowledgeGraph(GraphConfig(similarity_threshold=1.0))
        graph.initialize(["physics"])
        self.assertEqual(edge_density(graph), 0.0)

    def test_intra_and_inter_round_similarity(self) -> None:
        first = [
            EvaluatedIdea(idea("a", [], 1), [review(8.0)], review(8.0)),
            EvaluatedIdea(idea("b", [], 1), [review(7.0)], review(7.0)),
        ]
        second = [EvaluatedIdea(idea("c", [], 2), [review(8.0)], review(8.0))]
        self.assertGreater(intra_round_convergence(first), 0.0)
        self.assertEqual(intra_round_convergence(first[:1]), 0.0)
        self.assertGreater(inter_round_continuity(first, second), 0.0)
        self.assertEqual(inter_round_continuity([], second), 0.0)

    def test_diagnose_reports_cluster_and_round_structure(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        first_ids = clusters[0].entity_ids[:1]
        second_ids = clusters[1].entity_ids[:1]
        state = RunState(topic="grokking", disciplines=["physics"], rounds=[
            RoundResult(
                round_index=1,
                problems=[],
                evaluated_ideas=[
                    EvaluatedIdea(idea("a", first_ids, 1), [review(8.0)], review(8.0))
                ],
                evolution_summary={},
            ),
            RoundResult(
                round_index=2,
                problems=[],
                evaluated_ideas=[
                    EvaluatedIdea(idea("b", second_ids, 2), [review(6.0)], review(6.0))
                ],
                evolution_summary={},
            ),
        ])

        report = diagnose(state, graph)
        self.assertEqual(report.entity_count, len(graph.entities))
        self.assertEqual(report.concatenated_names, 0)
        self.assertEqual(len(report.clusters), 1)
        self.assertEqual(report.clusters[0].singletons, 0)
        self.assertEqual(len(report.rounds), 2)
        self.assertEqual(report.rounds[0].clusters_credited, 1)
        self.assertEqual(report.rounds[1].clusters_credited, 1)
        self.assertEqual(report.distinct_clusters_credited, 2)
        self.assertEqual(report.rounds[0].inter_round_continuity, 0.0)
        self.assertGreater(report.rounds[1].inter_round_continuity, 0.0)

    def test_diagnose_is_read_only(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        graph.clusters("physics", "grokking")
        before = graph.to_dict()
        state = RunState(topic="grokking", disciplines=["physics"])
        diagnose(state, graph)
        after = graph.to_dict()
        self.assertEqual(
            [entity["fitness"] for entity in before["entities"]],
            [entity["fitness"] for entity in after["entities"]],
        )
        self.assertEqual(
            [(c["id"], c["entity_ids"], c["fitness"]) for c in before["clusters"]],
            [(c["id"], c["entity_ids"], c["fitness"]) for c in after["clusters"]],
        )

    def test_render_emits_markdown_tables(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        state = RunState(topic="grokking", disciplines=["physics"])
        text = diagnose(state, graph).render()
        self.assertIn("# Structural diagnostics", text)
        self.assertIn("| discipline | clusters |", text)
        self.assertTrue(text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
