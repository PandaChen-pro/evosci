from __future__ import annotations

import random
from typing import Any

from .config import EvolutionConfig
from .knowledge import KnowledgeGraph, stable_id
from .llm import LLMBackend
from .models import Entity, EntityCluster, EvaluatedIdea
from .parsing import as_string_list


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class EntityEvolution:
    TABU_ROUNDS = 2
    STALENESS_BONUS = 0.05
    STALENESS_CAP = 6
    CROWDING_PENALTY = 0.20

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
        self._migrated: dict[str, int] = {}

    def evolve(
        self,
        topic: str,
        evaluated_ideas: list[EvaluatedIdea],
        disciplines: list[str],
        generation: int,
    ) -> dict[str, Any]:
        credited = self._apply_fitness(evaluated_ideas, disciplines)
        summary: dict[str, Any] = {
            "generation": generation,
            "credited_clusters": credited,
            "selected": [],
            "crossovers": [],
            "variations": [],
            "pruned": [],
        }
        for discipline in disciplines:
            clusters = self.graph.sync_clusters(discipline, generation)
            if not clusters:
                continue
            selected = self._select(clusters)
            summary["selected"].extend(cluster.id for cluster in selected)
            summary["crossovers"].extend(self._crossover(clusters, generation))
            summary["variations"].extend(
                self._variation(topic, discipline, selected, generation)
            )
            summary["pruned"].extend(self._prune(discipline))
        self.graph.rebuild_edges()
        for discipline in disciplines:
            self.graph.sync_clusters(discipline)
        self._assert_cluster_invariants()
        return summary

    def _select(self, clusters: list[EntityCluster]) -> list[EntityCluster]:
        """Filter clusters by evaluation feedback (paper §3.4 Selection).

        Selection marks which clusters carry forward; it never deletes one. A cluster is a
        line of inquiry, and a round with no ideas touching it is absence of evidence, not
        evidence of failure — ``stale`` records that so pruning can act on a real trend.

        The two extra terms both exist to stop selection locking on. Ranked on fitness
        alone the same clusters win every round, absorb every variation, and the rest stay
        frozen at seed size — never producing anything a reviewer could judge, so their
        fitness can never move and they can never be selected.

        The staleness bonus lets an unjudged cluster overtake a fit one after a few idle
        rounds. Crowding is a graded penalty rather than a tie-break because within one
        discipline the fitness spread is small but rarely exactly zero — an exact-equality
        tie-break almost never fires, and a cluster only 0.01 fitter still wins every round
        and runs away with the population.
        """
        share = {
            cluster.id: len(cluster.entity_ids)
            / max(1, sum(len(other.entity_ids) for other in clusters))
            for cluster in clusters
        }
        ranked = sorted(
            clusters,
            key=lambda cluster: (
                -(
                    cluster.fitness
                    + self.STALENESS_BONUS * min(cluster.stale, self.STALENESS_CAP)
                    - self.CROWDING_PENALTY * share[cluster.id]
                ),
                cluster.id,
            ),
        )
        keep_count = max(1, round(len(ranked) * self.config.selection_ratio))
        return ranked[:keep_count]

    def _crossover(
        self, clusters: list[EntityCluster], generation: int
    ) -> list[dict[str, str]]:
        """Exchange entities between semantic clusters of one discipline (paper §3.4).

        One-way migration, not a swap: an entity moves from the fittest donor to the least
        fit receiver. Minting a recombined *entity* — the obvious reading of "crossover" —
        is what collapsed the graph, since a name spanning two clusters links to both and
        fuses them.

        The donor keeps its best member (its identity) and gives up its second-best, so the
        receiver gains something proven. A tabu stamp stops an entity ping-ponging between
        two clusters on consecutive rounds.

        Migration does two jobs, and they need separate justifications. Propagating proven
        material means moving from a *fitter* cluster; relieving crowding means moving from a
        *larger* one. Ranking on fitness alone made the least-fit cluster a permanent sink:
        receiving members does not change a cluster's fitness, only crediting does, so
        nothing it absorbed could ever disqualify it. Under rotating feedback one physics
        cluster took every migration for 20 rounds and grew from 2 members to 24 — an 0.80
        share — while never once being selected, since ``_select``'s own crowding penalty
        kept it off the host list. It was a dumping ground, not a line of inquiry.

        Blending the two into a single score does not fix it either, because a crowded
        cluster can be crowded *and* unfit. Ranked on the blend, a cognitive-science cluster
        holding 12 of 16 members at ``f=0.50`` was the only cluster large enough to donate,
        yet scored below every min-size cluster at ``f=0.75`` — so the guard rejected the one
        migration that would have rebalanced the discipline, and the layout froze at an 0.75
        share for 14 rounds. Hence the disjunction: either surplus is reason enough to move.
        """
        eligible = [
            cluster for cluster in clusters
            if len(cluster.entity_ids) > self.graph.config.min_cluster_size
        ]
        if not eligible or len(clusters) < 2:
            return []

        def share(cluster: EntityCluster) -> float:
            return len(cluster.entity_ids) / max(
                1, sum(len(other.entity_ids) for other in clusters)
            )

        def surplus(cluster: EntityCluster) -> float:
            return cluster.fitness + self.CROWDING_PENALTY * share(cluster)

        migrations: list[dict[str, str]] = []
        for _ in range(self.config.crossover_count):
            donor = max(eligible, key=lambda cluster: (surplus(cluster), cluster.id))
            receiver = min(
                (cluster for cluster in clusters if cluster.id != donor.id),
                key=lambda cluster: (surplus(cluster), cluster.id),
            )
            if donor.fitness <= receiver.fitness and share(donor) <= share(receiver):
                break
            ranked = sorted(
                (self.graph.entities[entity_id] for entity_id in donor.entity_ids),
                key=lambda entity: (-entity.fitness, entity.name),
            )
            traveller = next(
                (
                    entity for entity in ranked[1:]
                    if generation - self._migrated.get(entity.id, -self.TABU_ROUNDS)
                    >= self.TABU_ROUNDS
                ),
                None,
            )
            if traveller is None:
                break
            donor.entity_ids.remove(traveller.id)
            receiver.entity_ids.append(traveller.id)
            self._migrated[traveller.id] = generation
            migrations.append({
                "entity_id": traveller.id,
                "from": donor.id,
                "to": receiver.id,
            })
            if len(donor.entity_ids) <= self.graph.config.min_cluster_size:
                eligible = [
                    cluster for cluster in eligible if cluster.id != donor.id
                ]
                if not eligible:
                    break
        return migrations

    def _assert_cluster_invariants(self) -> None:
        """Guard the assumptions every cluster consumer makes.

        An empty cluster reaches the mentor prompt as an empty entity list, and the
        heuristic backend indexes it modulo its length — so a violation surfaces far from
        its cause as a ZeroDivisionError.
        """
        for cluster in self.graph.all_clusters():
            if not cluster.entity_ids:
                raise AssertionError(f"cluster {cluster.id} is empty")
            dangling = set(cluster.entity_ids) - set(self.graph.entities)
            if dangling:
                raise AssertionError(f"cluster {cluster.id} references {sorted(dangling)}")
            if len(cluster.entity_ids) != len(set(cluster.entity_ids)):
                raise AssertionError(f"cluster {cluster.id} has duplicate members")

    def _apply_fitness(
        self, evaluated_ideas: list[EvaluatedIdea], disciplines: list[str]
    ) -> list[str]:
        """Route review feedback to clusters, then from each cluster to its members.

        Credit is attributed by membership overlap rather than by a single cluster id: an
        idea may draw on entities from several clusters, and there is no idea-to-problem
        edge to follow.

        Decay lives strictly *inside* the credited branch. A global decay sweep punishes
        every cluster that happened not to be examined this round as if it had been judged
        bad, which drags the whole population down together — that uniform drift is what
        made the fitness curve look flat.
        """
        for discipline in disciplines:
            self.graph.sync_clusters(discipline)
        credited: list[str] = []
        for cluster in self.graph.all_clusters():
            members = set(cluster.entity_ids)
            if not members:
                continue
            weighted = [
                (len(members & set(item.idea.entity_ids)) / len(members), item.fitness)
                for item in evaluated_ideas
            ]
            weighted = [(weight, fitness) for weight, fitness in weighted if weight > 0.0]
            if not weighted:
                cluster.stale += 1
                continue
            total = sum(weight for weight, _ in weighted)
            credit = sum(weight * fitness for weight, fitness in weighted) / total
            cluster.fitness = _clamp(
                0.45 * cluster.fitness * self.config.fitness_decay + 0.55 * credit
            )
            cluster.stale = 0
            credited.append(cluster.id)
            self._inherit(cluster)
        return credited

    def _inherit(self, cluster: EntityCluster) -> None:
        """Propagate a credited cluster's fitness to its members (paper §3.4 Inheritance).

        The rank multiplier spans [0.85, 1.15] around the cluster's own fitness: it keeps
        members distinguishable — without it every member converges on one value and
        selection inside a cluster becomes arbitrary — while staying mean-preserving, so
        good review feedback cannot leave a cited entity worse off than it started.
        """
        members = [self.graph.entities[entity_id] for entity_id in cluster.entity_ids]
        ranked = sorted(members, key=lambda entity: (-entity.fitness, entity.name))
        span = max(1, len(ranked) - 1)
        for position, entity in enumerate(ranked):
            rank_share = 1.0 - position / span
            entity.fitness = _clamp(
                0.65 * entity.fitness + 0.35 * cluster.fitness * (0.85 + 0.30 * rank_share)
            )

    def _variation(
        self,
        topic: str,
        discipline: str,
        selected: list[EntityCluster],
        generation: int,
    ) -> list[str]:
        """Introduce new or low-frequency entities into existing clusters (paper §3.4).

        The host is the *thinnest* selected cluster, recomputed per entity. Depositing into
        a fixed slice of the selection instead lets one cluster absorb every round's
        variation — with flat fitness the same slice wins every time — which is the
        collapse this rewrite exists to prevent. Anchors come from the same clusters that
        receive, so a new entity stays semantically close to its host.
        """
        if not selected:
            return []
        hosts = sorted(selected, key=lambda cluster: (len(cluster.entity_ids), cluster.id))
        anchor_clusters = hosts[:2]
        anchors = [
            self.graph.entities[entity_id].name
            for cluster in anchor_clusters
            for entity_id in cluster.entity_ids[:2]
        ]
        if not anchors:
            return []
        response = self.backend.generate_json(
            task="variation",
            system=(
                "Generate scientifically meaningful low-frequency concepts that vary the "
                "selected entity clusters without merely renaming existing entities."
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
        owned = {
            entity_id
            for cluster in self.graph.sync_clusters(discipline)
            for entity_id in cluster.entity_ids
        }
        for item in response.get("entities", [])[: self.config.variation_count]:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            host = min(
                anchor_clusters,
                key=lambda cluster: (len(cluster.entity_ids), cluster.id),
            )
            parent = self.graph.entities[host.entity_ids[0]]
            entity = self.graph.add_entity(Entity(
                id=stable_id("ent", discipline, name),
                name=name,
                discipline=discipline,
                kind=str(item.get("kind") or "EvolvedConcept"),
                description=str(item.get("description") or ""),
                source="variation",
                fitness=max(0.25, host.fitness * 0.8),
                generation=generation,
                parents=[parent.id],
            ))
            # add_entity upserts on a duplicate name, so it can hand back an entity that
            # another cluster already owns. Appending it anyway puts one id in two clusters,
            # and sync_clusters resolves that by stripping it from the second — silently
            # shrinking a cluster below min_cluster_size.
            if entity is None or entity.id in owned:
                continue
            host.entity_ids.append(entity.id)
            owned.add(entity.id)
            created.append(entity.id)
        return list(dict.fromkeys(created))

    def _prune(self, discipline: str) -> list[str]:
        """Remove the discipline's weakest entities without emptying a cluster.

        The floor is relative to the discipline's own mean because an absolute one cannot
        fire: seeds start at 0.5, variation offspring at ≥ 0.25, and fitness now only moves
        for credited clusters — so at ``min_fitness`` 0.15 nothing was ever pruned inside a
        3–10 round run and the population could only grow. ``min_fitness`` is kept as a hard
        floor beneath the relative one.

        A cluster at ``min_cluster_size`` is exempt even under the population cap, so the
        cap is a soft target. Honouring it exactly would mean deleting a line of inquiry to
        satisfy a budget, and an emptied cluster reaches the mentor prompt as an empty
        entity list.
        """
        clusters = self.graph.sync_clusters(discipline)
        population = self.graph.entities_for(discipline)
        if not population:
            return []
        ranked = sorted(population, key=lambda entity: (entity.fitness, entity.name))
        average = sum(entity.fitness for entity in population) / len(population)
        floor = max(self.config.min_fitness, average * self.config.prune_floor_ratio)
        budget = max(1, round(len(ranked) * self.config.prune_ratio))
        candidates = [entity for entity in ranked[:budget] if entity.fitness < floor]
        overflow = len(ranked) - self.config.max_population_per_discipline
        if overflow > len(candidates):
            candidates = ranked[:overflow]
        room = {cluster.id: len(cluster.entity_ids) for cluster in clusters}
        host = {
            entity_id: cluster.id
            for cluster in clusters
            for entity_id in cluster.entity_ids
        }
        removed: list[str] = []
        for entity in candidates:
            owner = host.get(entity.id)
            if owner is not None:
                if room[owner] <= self.graph.config.min_cluster_size:
                    continue
                room[owner] -= 1
            removed.append(entity.id)
            self.graph.entities.pop(entity.id, None)
            self.graph.edges.pop(entity.id, None)
            for neighbors in self.graph.edges.values():
                neighbors.pop(entity.id, None)
        for cluster in clusters:
            cluster.entity_ids = [
                entity_id for entity_id in cluster.entity_ids
                if entity_id in self.graph.entities
            ]
        return removed
