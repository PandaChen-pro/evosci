import unittest

from evosci.config import GraphConfig
from evosci.knowledge import KnowledgeGraph, containment, stable_id, text_similarity
from evosci.models import Entity


class KnowledgeGraphTests(unittest.TestCase):
    def test_similarity_is_semantically_stable_for_shared_terms(self) -> None:
        related = text_similarity("phase transition dynamics", "phase transition modeling")
        unrelated = text_similarity("phase transition dynamics", "clinical biomarkers")
        self.assertGreater(related, unrelated)

    def test_containment_separates_concatenation_from_relatedness(self) -> None:
        concatenations = [
            ("symmetry breaking", "symmetry breaking + phase transition"),
            ("entropy", "energy landscape + entropy"),
            (
                "multimodal contrastive alignment",
                "multimodal contrastive alignment + cross-function taint propagation",
            ),
        ]
        for parent, child in concatenations:
            self.assertEqual(containment(parent, child), 1.0, msg=child)
        related = [
            ("working memory", "memory consolidation"),
            ("phase transition dynamics", "phase transition modeling"),
            ("optimization", "generalization"),
        ]
        for left, right in related:
            self.assertLess(containment(left, right), 0.90, msg=f"{left}|{right}")

    def test_containment_veto_rejects_subsuming_names(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        name = "symmetry breaking + phase transition"
        vetoed = graph.add_entity(Entity(
            id=stable_id("ent", "physics", name), name=name, discipline="physics"
        ))
        self.assertIsNone(vetoed)
        self.assertNotIn(name, {entity.name for entity in graph.entities_for("physics")})

        admitted = graph.add_entity(Entity(
            id=stable_id("ent", "physics", "kinetic arrest"),
            name="kinetic arrest",
            discipline="physics",
        ))
        self.assertIsNotNone(admitted)

    def test_containment_veto_is_skipped_for_small_populations(self) -> None:
        """Only the recombination rule waits for the floor, so the name must fit the budget.

        ``entropy + phase transition`` is 4 words against physics' 3-word budget, so the
        drift rule — which has no floor — would reject it regardless and the floor logic
        would go untested.
        """
        graph = KnowledgeGraph(GraphConfig(cluster_limit=4, min_cluster_size=2))
        graph.add_entity(Entity(id="a", name="entropy", discipline="physics"))
        graph.add_entity(Entity(id="b", name="phase transition", discipline="physics"))
        name = "entropy phase transition"
        admitted = graph.add_entity(Entity(id="c", name=name, discipline="physics"))
        self.assertIsNotNone(admitted)

        populated = KnowledgeGraph(GraphConfig(cluster_limit=4, min_cluster_size=2))
        populated.initialize(["physics"])
        self.assertIsNone(
            populated.add_entity(Entity(id="d", name=name, discipline="physics"))
        )

    def test_containment_veto_rejects_redundant_and_runaway_names(self) -> None:
        """A name adding no new word, or exceeding the discipline's word budget.

        The redundant case must fire even below the population floor and even when the
        shared parent is gone: ``working memory`` gets pruned, leaving only the one-suffix
        name, so the recombination rule finds a single subsumed name and would pass.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.add_entity(Entity(id="a", name="working memory Uncertainty",
                                discipline="cognitive science"))
        for redundant in (
            "working memory Uncertainty Uncertainty",
            "Uncertainty working memory Uncertainty",
        ):
            self.assertIsNone(
                graph.add_entity(Entity(id=stable_id("ent", "cognitive science", redundant),
                                        name=redundant, discipline="cognitive science")),
                msg=redundant,
            )

        budget = graph._name_word_budget("cognitive science")
        runaway = " ".join(f"word{index}" for index in range(budget + 1))
        self.assertIsNone(graph.add_entity(Entity(
            id=stable_id("ent", "cognitive science", runaway),
            name=runaway, discipline="cognitive science",
        )))

    def test_name_word_budget_follows_seed_complexity(self) -> None:
        """Budget is per-discipline: one global constant either drifts or over-blocks.

        Cognitive science seeds top out at two words while computer science reaches three
        (``graph neural networks``), so a constant tight enough for the first rejects
        legitimate concepts in the second.
        """
        graph = KnowledgeGraph(GraphConfig(max_name_word_growth=1))
        self.assertEqual(graph._name_word_budget("cognitive science"), 3)
        self.assertEqual(graph._name_word_budget("computer science"), 4)
        self.assertGreaterEqual(graph._name_word_budget("unknown discipline"), 2)

    def test_containment_veto_admits_single_word_extensions(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        admitted = graph.add_entity(Entity(
            id=stable_id("ent", "physics", "entropy production"),
            name="entropy production",
            discipline="physics",
        ))
        self.assertIsNotNone(admitted)

    def test_clusters_have_no_singletons_and_cover_the_population(self) -> None:
        config = GraphConfig()
        graph = KnowledgeGraph(config)
        disciplines = ["physics", "computer science", "cognitive science"]
        graph.initialize(disciplines)
        for discipline in disciplines:
            clusters = graph.clusters(discipline, "grokking phase transition")
            sizes = [len(cluster.entity_ids) for cluster in clusters]
            self.assertLessEqual(len(clusters), config.cluster_limit, msg=discipline)
            self.assertTrue(
                all(size >= config.min_cluster_size for size in sizes),
                msg=f"{discipline} has singletons: {sizes}",
            )
            self.assertEqual(
                sum(sizes), len(graph.entities_for(discipline)), msg=discipline
            )
            covered = [entity_id for cluster in clusters for entity_id in cluster.entity_ids]
            self.assertEqual(len(covered), len(set(covered)), msg=discipline)

    def test_initialize_cluster_and_round_trip(self) -> None:
        graph = KnowledgeGraph(GraphConfig(similarity_threshold=0.25))
        graph.initialize(["physics", "computer science"])
        self.assertGreaterEqual(len(graph.entities_for("physics")), 8)
        clusters = graph.clusters("physics", "grokking phase transition")
        self.assertTrue(clusters)
        restored = KnowledgeGraph.from_dict(graph.config, graph.to_dict())
        self.assertEqual(set(graph.entities), set(restored.entities))
        self.assertEqual(dict(graph.edges), dict(restored.edges))

    def test_clusters_persist_across_round_trip(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        before = graph.clusters("physics", "grokking")
        before[0].fitness = 0.77
        before[0].times_selected = 3
        before[0].stale = 1

        restored = KnowledgeGraph.from_dict(graph.config, graph.to_dict())
        after = restored.clusters("physics", "grokking")
        self.assertEqual(
            [(c.id, c.entity_ids) for c in before], [(c.id, c.entity_ids) for c in after]
        )
        self.assertEqual(after[0].fitness, 0.77)
        self.assertEqual(after[0].times_selected, 3)
        self.assertEqual(after[0].stale, 1)

    def test_legacy_graph_without_clusters_bootstraps(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        legacy = graph.to_dict()
        del legacy["clusters"]
        restored = KnowledgeGraph.from_dict(graph.config, legacy)
        self.assertTrue(restored.clusters("physics", "grokking"))

    def test_cluster_membership_drops_dangling_ids(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        graph.clusters("physics", "grokking")
        data = graph.to_dict()
        data["clusters"][0]["entity_ids"].append("ent_does_not_exist")
        restored = KnowledgeGraph.from_dict(graph.config, data)
        for cluster in restored.clusters("physics", "grokking"):
            self.assertTrue(cluster.entity_ids)
            self.assertTrue(set(cluster.entity_ids) <= set(restored.entities))

    def test_cluster_ids_are_stable_when_members_change(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        before = {cluster.id for cluster in graph.clusters("physics", "grokking")}
        graph.add_entity(Entity(
            id=stable_id("ent", "physics", "kinetic arrest"),
            name="kinetic arrest",
            discipline="physics",
        ))
        after = {cluster.id for cluster in graph.clusters("physics", "grokking")}
        self.assertEqual(before, after)
        self.assertEqual(
            sum(len(c.entity_ids) for c in graph.clusters("physics", "grokking")),
            len(graph.entities_for("physics")),
        )


if __name__ == "__main__":
    unittest.main()
