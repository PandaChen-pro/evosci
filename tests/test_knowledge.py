import unittest

from evosci.config import GraphConfig
from evosci.knowledge import KnowledgeGraph, text_similarity


class KnowledgeGraphTests(unittest.TestCase):
    def test_similarity_is_semantically_stable_for_shared_terms(self) -> None:
        related = text_similarity("phase transition dynamics", "phase transition modeling")
        unrelated = text_similarity("phase transition dynamics", "clinical biomarkers")
        self.assertGreater(related, unrelated)

    def test_initialize_cluster_and_round_trip(self) -> None:
        graph = KnowledgeGraph(GraphConfig(similarity_threshold=0.25))
        graph.initialize(["physics", "computer science"])
        self.assertGreaterEqual(len(graph.entities_for("physics")), 8)
        clusters = graph.clusters("physics", "grokking phase transition")
        self.assertTrue(clusters)
        restored = KnowledgeGraph.from_dict(graph.config, graph.to_dict())
        self.assertEqual(set(graph.entities), set(restored.entities))
        self.assertEqual(dict(graph.edges), dict(restored.edges))


if __name__ == "__main__":
    unittest.main()
