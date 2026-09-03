import unittest

from evosci.config import EvolutionConfig, GraphConfig
from evosci.diagnostics import stdev
from evosci.evolution import EntityEvolution
from evosci.knowledge import KnowledgeGraph, stable_id
from evosci.llm import HeuristicBackend
from evosci.models import Entity, EvaluatedIdea, ResearchIdea, Review


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
        hypothesis="A changes B",
        rationale="Mechanistic bridge",
        method="controlled experiment",
        experiment="Matched intervention and control across five seeds",
        expected_outcome="Dose response",
        risks=["confounding"],
        entity_ids=list(entity_ids),
        round_index=round_index,
    )


class EvolutionTests(unittest.TestCase):
    def test_evolution_updates_fitness_and_population(self) -> None:
        graph = KnowledgeGraph(GraphConfig(similarity_threshold=0.3))
        graph.initialize(["physics"])
        entity_ids = [entity.id for entity in graph.entities_for("physics")[:2]]
        item = EvaluatedIdea(idea("test", entity_ids, 1), [review(8.0)], review(8.0))
        membership_before = {
            cluster.id: set(cluster.entity_ids)
            for cluster in graph.clusters("physics", "grokking")
        }
        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(crossover_count=1, variation_count=1),
            seed=1,
        )
        summary = evolution.evolve("grokking", [item], ["physics"], generation=1)
        membership_after = {
            cluster.id: set(cluster.entity_ids)
            for cluster in graph.clusters("physics", "grokking")
        }
        self.assertTrue(summary["variations"])
        self.assertNotEqual(membership_after, membership_before)
        for entity_id in entity_ids:
            self.assertGreater(graph.entities[entity_id].fitness, 0.5)

    def test_crossover_migrates_between_clusters_without_minting(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        donor, receiver = clusters[0], clusters[1]
        donor.fitness, receiver.fitness = 0.9, 0.2
        for other in clusters[2:]:
            other.fitness = 0.5
        donor.entity_ids.append(receiver.entity_ids.pop())
        entities_before = set(graph.entities)
        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(crossover_count=1, variation_count=0),
            seed=1,
        )
        summary = evolution.evolve("grokking", [], ["physics"], generation=1)

        self.assertEqual(len(summary["crossovers"]), 1)
        migration = summary["crossovers"][0]
        self.assertEqual(migration["from"], donor.id)
        self.assertEqual(migration["to"], receiver.id)
        self.assertIn(migration["entity_id"], receiver.entity_ids)
        self.assertNotIn(migration["entity_id"], donor.entity_ids)
        self.assertEqual(set(graph.entities), entities_before)

    def test_crossover_never_drains_a_cluster_below_the_minimum(self) -> None:
        graph = KnowledgeGraph(GraphConfig(min_cluster_size=3))
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        for index, cluster in enumerate(clusters):
            cluster.fitness = 0.9 if index == 0 else 0.2
        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(crossover_count=8, variation_count=0),
            seed=1,
        )
        for generation in range(1, 6):
            evolution.evolve("grokking", [], ["physics"], generation=generation)
            for cluster in graph.clusters("physics", "grokking"):
                self.assertGreaterEqual(len(cluster.entity_ids), 3, msg=cluster.id)

    def test_pruning_fires_within_a_short_run_and_spares_min_size_clusters(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        pruned = 0
        for generation in range(1, 7):
            clusters = graph.clusters("physics", "grokking")
            items = [
                EvaluatedIdea(
                    idea(f"good{generation}", clusters[0].entity_ids, generation),
                    [review(9.5)],
                    review(9.5),
                ),
                EvaluatedIdea(
                    idea(f"bad{generation}", clusters[-1].entity_ids, generation),
                    [review(1.5)],
                    review(1.5),
                ),
            ]
            summary = evolution.evolve(
                "grokking", items, ["physics"], generation=generation
            )
            pruned += len(summary["pruned"])
            for cluster in graph.clusters("physics", "grokking"):
                self.assertGreaterEqual(len(cluster.entity_ids), 2, msg=cluster.id)
        self.assertGreater(pruned, 0)

    def test_pruning_never_removes_the_fittest_entities(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        for generation in range(1, 7):
            clusters = graph.clusters("physics", "grokking")
            best = max(
                graph.entities_for("physics"),
                key=lambda entity: (entity.fitness, entity.name),
            )
            item = EvaluatedIdea(
                idea(f"r{generation}", clusters[0].entity_ids, generation),
                [review(9.0)],
                review(9.0),
            )
            summary = evolution.evolve(
                "grokking", [item], ["physics"], generation=generation
            )
            self.assertNotIn(best.id, summary["pruned"])

    def test_variation_never_places_one_entity_in_two_clusters(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics", "computer science"])
        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(variation_count=3),
            seed=1,
        )
        for generation in range(1, 9):
            evolution.evolve(
                "grokking", [], ["physics", "computer science"], generation=generation
            )
            members = [
                entity_id
                for cluster in graph.all_clusters()
                for entity_id in cluster.entity_ids
            ]
            self.assertEqual(len(members), len(set(members)))

    def test_uncredited_clusters_are_unchanged_not_decayed(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        credited, ignored = clusters[0], clusters[1]
        ignored_before = {i: graph.entities[i].fitness for i in ignored.entity_ids}
        item = EvaluatedIdea(
            idea("credited", credited.entity_ids, 1), [review(8.0)], review(8.0)
        )
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        summary = evolution.evolve("grokking", [item], ["physics"], generation=1)

        self.assertIn(credited.id, summary["credited_clusters"])
        self.assertNotIn(ignored.id, summary["credited_clusters"])
        self.assertGreater(credited.fitness, 0.5)
        self.assertEqual(ignored.fitness, 0.5)
        self.assertEqual(ignored.stale, 1)
        self.assertEqual(
            {i: graph.entities[i].fitness for i in ignored_before}, ignored_before
        )

    def test_credit_differentiates_the_population(self) -> None:
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        spreads = []
        for generation in range(1, 7):
            target = clusters[generation % 2]
            item = EvaluatedIdea(
                idea(f"r{generation}", target.entity_ids, generation),
                [review(9.0 if generation % 2 else 3.0)],
                review(9.0 if generation % 2 else 3.0),
            )
            evolution.evolve("grokking", [item], ["physics"], generation=generation)
            values = [entity.fitness for entity in graph.entities_for("physics")]
            spreads.append(stdev(values))

        self.assertGreater(spreads[-1], spreads[0])
        self.assertGreater(stdev([c.fitness for c in graph.clusters("physics", "grokking")]), 0.0)

    def test_evolution_never_admits_concatenated_names(self) -> None:
        graph = KnowledgeGraph(GraphConfig(similarity_threshold=0.3))
        graph.initialize(["physics"])
        seed_longest = max(len(entity.name) for entity in graph.entities_for("physics"))
        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(crossover_count=2, variation_count=2),
            seed=1,
        )
        for generation in range(1, 9):
            evolution.evolve("grokking", [], ["physics"], generation=generation)
        names = [entity.name for entity in graph.entities.values()]
        self.assertFalse([name for name in names if " + " in name])
        self.assertLessEqual(max(len(name) for name in names), int(seed_longest * 1.2) + 12)


    def test_variation_reaches_every_cluster_not_just_the_fittest(self) -> None:
        """Deposits must spread across clusters, tracked by membership rather than by id.

        Pinning the id set would now assert the opposite of what dissolution and fission
        exist to do — a retired cluster's id is gone and a fission child's is new — so the
        property is checked over entities: every seed entity still in the population sits in
        some cluster, and the population grew in more than one place.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        seeds = {entity.id for entity in graph.entities_for("physics")}
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        for generation in range(1, 9):
            evolution.evolve("grokking", [], ["physics"], generation=generation)
        clusters = graph.clusters("physics", "grokking")
        members = {
            entity_id for cluster in clusters for entity_id in cluster.entity_ids
        }
        self.assertEqual(
            members, {entity.id for entity in graph.entities_for("physics")}
        )
        grown = [
            cluster for cluster in clusters
            if any(entity_id not in seeds for entity_id in cluster.entity_ids)
        ]
        self.assertGreaterEqual(len(grown), 2, msg=f"only {len(grown)} cluster(s) grew")

    def test_clusters_do_not_collapse_over_many_rounds(self) -> None:
        disciplines = ["physics", "computer science"]
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(disciplines)
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        for generation in range(1, 9):
            evolution.evolve("grokking", [], disciplines, generation=generation)
            for discipline in disciplines:
                sizes = [len(c.entity_ids) for c in graph.clusters(discipline, "grokking")]
                self.assertGreaterEqual(len(sizes), 3, msg=f"{discipline} r{generation}")
                self.assertLess(
                    max(sizes) / sum(sizes), 0.5, msg=f"{discipline} r{generation}: {sizes}"
                )

    def test_cluster_ids_stay_unique_across_a_long_run(self) -> None:
        """Two clusters sharing an id made ``_crossover``'s donor filter empty.

        A fission child keyed only on its membership can reproduce an id some earlier
        cluster already held, and the collision surfaced far away as
        ``ValueError: min() iterable argument is empty``.
        """
        disciplines = ["physics", "computer science"]
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(disciplines)
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        for generation in range(1, 16):
            evolution.evolve("grokking", [], disciplines, generation=generation)
            ids = [cluster.id for cluster in graph.all_clusters()]
            self.assertEqual(len(ids), len(set(ids)), msg=f"r{generation}: {ids}")

    def test_dissolution_retires_one_cluster_at_a_time_and_holds_a_floor(self) -> None:
        """Extinction must not become the collapse it is paired with.

        With no evaluated ideas every cluster's ``stale`` rises in lockstep, so retiring
        everything over the threshold took physics from 4 clusters to 1 in a single round.
        Each ``sync_clusters`` call needs its own generation: structural change is gated to
        once per generation, so a repeated generation is deliberately a no-op.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        self.assertGreaterEqual(len(clusters), 3)
        for cluster in clusters:
            cluster.stale = graph.config.cluster_max_stale + 5

        after = graph.sync_clusters("physics", generation=1)
        self.assertEqual(len(after), len(clusters) - 1)

        for generation in range(2, 8):
            for cluster in graph.sync_clusters("physics", generation=generation):
                cluster.stale = graph.config.cluster_max_stale + 5
        self.assertGreaterEqual(
            len(graph.sync_clusters("physics", generation=99)),
            min(graph.config.cluster_limit, KnowledgeGraph.CLUSTER_FLOOR),
        )

    def test_structural_change_happens_once_per_generation(self) -> None:
        """Reads must not mutate structure — ``clusters()`` is called many times per round.

        Ungated, fission fired 377 times over a 30-round run, oscillating a discipline
        between ``[2, 8, 2]`` and ``[2, 8, 2, 2]``, while repeated dissolution dumped orphans
        into whichever cluster was already largest and ratcheted it from 7 to 11 members.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        for cluster in graph.clusters("physics", "grokking"):
            cluster.stale = graph.config.cluster_max_stale + 5
        first = [c.id for c in graph.sync_clusters("physics", generation=3)]
        for _ in range(5):
            repeated = [c.id for c in graph.sync_clusters("physics", generation=3)]
            self.assertEqual(repeated, first)
        for _ in range(5):
            # clusters() orders by topic relevance, so compare membership not sequence
            self.assertEqual(
                {c.id for c in graph.clusters("physics", "grokking")}, set(first)
            )

    def test_cluster_membership_turns_over_across_a_run(self) -> None:
        """Fission and dissolution together must move the *set* of clusters, not just sizes.

        Fission alone fills a discipline to ``cluster_limit`` and then stops, which would
        make "clusters don't collapse" true by arithmetic rather than by dynamics.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        before = {cluster.id for cluster in graph.clusters("physics", "grokking")}
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
        )
        for generation in range(1, 9):
            evolution.evolve("grokking", [], ["physics"], generation=generation)
        after = {cluster.id for cluster in graph.clusters("physics", "grokking")}
        self.assertTrue(after - before, msg="no cluster was born")
        self.assertTrue(before - after, msg="no cluster was retired")

    def test_crossover_does_not_turn_the_least_fit_cluster_into_a_sink(self) -> None:
        """A cluster that only ever receives is a dumping ground, not a line of inquiry.

        Receiving members never changes a cluster's fitness — only crediting does — so a
        receiver picked on raw fitness stays the receiver forever. Under rotating feedback
        one physics cluster absorbed every migration for 20 rounds, growing from 2 members
        to 24 (an 0.80 share) while never once being selected.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        evolution = EntityEvolution(
            graph, HeuristicBackend(seed=42), EvolutionConfig(), seed=42
        )
        received: dict[str, int] = {}
        for generation in range(1, 21):
            clusters = sorted(graph.clusters("physics", "grokking"), key=lambda c: c.id)
            target = clusters[generation % len(clusters)]
            item = EvaluatedIdea(
                idea(f"r{generation}", list(target.entity_ids), generation),
                [review(8.0)],
                review(8.0),
            )
            summary = evolution.evolve(
                "grokking", [item], ["physics"], generation=generation
            )
            for migration in summary["crossovers"]:
                received[migration["to"]] = received.get(migration["to"], 0) + 1
            sizes = [len(c.entity_ids) for c in graph.clusters("physics", "grokking")]
            self.assertLess(
                max(sizes) / sum(sizes), 0.5, msg=f"r{generation}: {sizes}"
            )
        self.assertGreater(len(received), 1, msg=f"one receiver only: {received}")

    def test_a_crowded_cluster_can_donate_even_when_it_is_the_least_fit(self) -> None:
        """Surplus size alone must justify a migration, or a lopsided layout freezes.

        Ranking both ends on one blended fitness+crowding score deadlocks here: the giant
        is the only cluster big enough to donate, yet scores below every min-size cluster,
        so the guard rejects the one migration that would rebalance the discipline. Measured
        as a cognitive-science layout stuck at an 0.75 share for 14 consecutive rounds.
        """
        graph = KnowledgeGraph(GraphConfig())
        graph.initialize(["physics"])
        clusters = graph.clusters("physics", "grokking")
        crowded, lean = clusters[0], clusters[1]
        for index in range(8):
            name = f"absorbed concept {index}"
            entity = graph.add_entity(Entity(
                id=stable_id("ent", "physics", name),
                name=name,
                discipline="physics",
            ))
            self.assertIsNotNone(entity, msg=name)
            crowded.entity_ids.append(entity.id)
        for cluster in clusters[1:]:
            cluster.fitness = 0.75
        crowded.fitness = 0.5
        self.assertGreater(len(crowded.entity_ids), len(lean.entity_ids))

        evolution = EntityEvolution(
            graph,
            HeuristicBackend(seed=1),
            EvolutionConfig(crossover_count=1, variation_count=0),
            seed=1,
        )
        summary = evolution.evolve("grokking", [], ["physics"], generation=1)
        self.assertEqual(len(summary["crossovers"]), 1, msg="crowded cluster could not donate")
        self.assertEqual(summary["crossovers"][0]["from"], crowded.id)

    def test_evolution_is_deterministic_for_a_fixed_seed(self) -> None:
        def run() -> dict:
            graph = KnowledgeGraph(GraphConfig())
            graph.initialize(["physics"])
            evolution = EntityEvolution(
                graph, HeuristicBackend(seed=1), EvolutionConfig(), seed=1
            )
            for generation in range(1, 5):
                evolution.evolve("grokking", [], ["physics"], generation=generation)
            graph.clusters("physics", "grokking")
            return graph.to_dict()

        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
