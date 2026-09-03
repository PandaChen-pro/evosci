from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
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


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


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


def containment(left: str, right: str) -> float:
    """Fraction of the shorter name's words that also appear in the longer one.

    Asymmetric-by-length on purpose: this separates a name that merely *contains*
    another ("a + b" against "a", scoring 1.0) from a genuinely related name
    ("statistical mechanics" against "critical phenomena", scoring 0.0), which
    ``text_similarity`` cannot do.
    """
    a = _words(left)
    b = _words(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def selection_score(cluster: EntityCluster, round_index: int) -> float:
    """Rank a cluster for the mentor's problem space (paper §3.1 Eq 1, §3.4 Selection).

    The exploration term is load-bearing: on relevance and fitness alone the score locks
    onto the two clusters that won round 1, so most clusters never receive any review
    feedback and their fitness stays at its initial value forever.
    """
    exploration = math.sqrt(
        math.log(1 + max(0, round_index)) / (1 + cluster.times_selected)
    )
    return 0.50 * cluster.score + 0.30 * cluster.fitness + 0.20 * exploration


class KnowledgeGraph:
    CLUSTER_FLOOR = 3

    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, dict[str, float]] = defaultdict(dict)
        self._clusters: dict[str, list[EntityCluster]] = defaultdict(list)
        self._structural: dict[str, int] = {}

    def add_entity(self, entity: Entity, check_containment: bool = True) -> Entity | None:
        """Insert an entity, returning the stored instance, or None if it was vetoed."""
        existing = self.find_by_name(entity.name, entity.discipline)
        if existing:
            existing.fitness = max(existing.fitness, entity.fitness)
            if entity.description and not existing.description:
                existing.description = entity.description
            return existing
        if check_containment and self._containment_veto(entity):
            return None
        self.entities[entity.id] = entity
        return entity

    def _containment_veto(self, entity: Entity) -> bool:
        """Whether this name is a relabelling of names the discipline already holds.

        Three distinct failures, so three rules.

        *Recombination:* a name subsuming two or more existing entities is a join label
        rather than a concept. ``text_similarity`` cannot see this — a concatenated name
        is a near-perfect match for each of its parts — so admitting such names is what
        fused a discipline into a single connected component.

        *Redundancy:* a name introducing no word the subsuming entity lacks is an empty
        rename — ``working memory Uncertainty Uncertainty``, or a reordering. Containment
        alone cannot separate this from a legitimate variation, since both score 1.0
        against the parent; the discriminator is whether a *new* word appears.

        *Drift:* the two rules above still admit a chain, each link adding one genuinely
        new word — ``working memory`` → ``… Uncertainty`` → ``… Uncertainty Dynamics`` →
        and so on. Measured over 200 offline rounds the chain saturated at 6 words only
        because the offline backend owns 5 suffixes; a real model's vocabulary is
        unbounded, so the ceiling has to be explicit rather than incidental. The budget is
        relative to the discipline's own seed complexity — see ``_name_word_budget``.

        Only the recombination rule waits for a population floor, so variation is not
        starved while a discipline is still filling up. The other two apply always: a
        redundant or runaway name is never worth admitting, and redundancy must still
        fire once the shared parent has been pruned away — exactly when the recombination
        rule finds only one subsumed name and would pass.
        """
        population = self.entities_for(entity.discipline)
        words = _words(entity.name)
        if any(words <= _words(other.name) for other in population):
            return True
        if len(entity.name.split()) > self._name_word_budget(entity.discipline):
            return True
        if len(population) < self.config.cluster_limit * self.config.min_cluster_size:
            return False
        subsumed = sum(
            1 for other in population
            if containment(entity.name, other.name) >= self.config.containment_threshold
        )
        return subsumed >= 2

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

    def _name_word_budget(self, discipline: str) -> int:
        """The longest name, in words, this discipline may admit.

        Measured against the discipline's *seed* vocabulary rather than a global constant
        because seed complexity varies by an order of magnitude — cognitive science tops
        out at two words (``working memory``) while a vulnerability-detection discipline
        seeds five (``weak supervision and vulnerability localization``). One constant
        either lets the first drift or blocks the second's legitimate concepts outright.

        Read from ``DISCIPLINE_SEEDS`` and not from the live population, which loses seeds
        to pruning: a budget derived from survivors would ratchet downward mid-run and make
        admission depend on how long the run had been going.
        """
        normalized = discipline.strip().lower()
        seeds = DISCIPLINE_SEEDS.get(normalized) or [f"{normalized} mechanism"]
        longest = max(len(name.split()) for name in seeds)
        return longest + self.config.max_name_word_growth

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
                ), check_containment=False)
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
        """The discipline's persistent clusters, rescored against the topic."""
        clusters = self.sync_clusters(discipline)
        for cluster in clusters:
            cluster.score = self._cluster_relevance(cluster, topic)
        return sorted(clusters, key=lambda cluster: (-cluster.score, cluster.id))[
            : self.config.cluster_limit
        ]

    def _cluster_relevance(self, cluster: EntityCluster, topic: str) -> float:
        if not cluster.entity_ids:
            return 0.0
        return sum(
            0.65 * text_similarity(topic, self.entities[entity_id].name)
            + 0.35 * self.entities[entity_id].fitness
            for entity_id in cluster.entity_ids
        ) / len(cluster.entity_ids)

    def cluster_by_id(self, cluster_id: str) -> EntityCluster | None:
        for cluster in self.all_clusters():
            if cluster.id == cluster_id:
                return cluster
        return None

    def all_clusters(self) -> list[EntityCluster]:
        return [cluster for clusters in self._clusters.values() for cluster in clusters]

    def sync_clusters(
        self, discipline: str, generation: int | None = None
    ) -> list[EntityCluster]:
        """Reconcile a discipline's persistent clusters with its current members.

        Clusters survive across rounds — that is what makes them the unit of evolution —
        so membership is repaired in place rather than recomputed: dangling ids are
        dropped, new entities join their nearest cluster, and emptied clusters are
        discarded. Only a discipline with no clusters at all is built from scratch.

        Reconciliation is idempotent and every caller runs it, including read paths like
        ``clusters()``. Births and deaths are not: they run only when ``generation`` is
        supplied and only once per generation per discipline, which is why
        ``EntityEvolution.evolve`` is the sole caller that passes one.

        Gating matters because a round reads the clusters many times — fitness attribution,
        selection, pruning, diagnostics — and structural change on every read compounds
        instead of stepping. Ungated, fission fired 377 times across a 30-round run,
        oscillating one discipline between ``[2, 8, 2]`` and ``[2, 8, 2, 2]`` forever, while
        repeated dissolution dumped orphans into whichever cluster was already largest and
        ratcheted it from 7 to 11 members.

        Within a generation dissolution precedes the reassignment pass, so a retired
        cluster's members are rehomed in the same call, and it frees a slot under
        ``cluster_limit`` that fission can then use — letting a run replace one line of
        inquiry with another rather than filling to the cap and stopping.
        """
        normalized = discipline.strip().lower()
        members = {entity.id: entity for entity in self.entities_for(normalized)}
        clusters = self._clusters.get(normalized, [])
        if not clusters:
            clusters = self._build_clusters(normalized, members)
            self._clusters[normalized] = clusters
            return clusters
        structural = (
            generation is not None and self._structural.get(normalized) != generation
        )
        if structural:
            self._structural[normalized] = generation
        assigned: set[str] = set()
        for cluster in clusters:
            cluster.entity_ids = [
                entity_id for entity_id in cluster.entity_ids
                if entity_id in members and entity_id not in assigned
            ]
            assigned.update(cluster.entity_ids)
        clusters = [cluster for cluster in clusters if cluster.entity_ids]
        if structural:
            clusters = self._dissolve(clusters)
        assigned = {
            entity_id for cluster in clusters for entity_id in cluster.entity_ids
        }
        for entity_id in sorted(set(members) - assigned):
            host = self._nearest_cluster(clusters, members[entity_id])
            if host is None:
                clusters = self._build_clusters(normalized, members)
                break
            host.entity_ids.append(entity_id)
        if structural:
            clusters = self._fission(normalized, clusters)
        for cluster in clusters:
            cluster.label = self._cluster_label(cluster)
        self._clusters[normalized] = clusters
        return clusters

    def _dissolve(self, clusters: list[EntityCluster]) -> list[EntityCluster]:
        """Retire a cluster no idea has touched for ``cluster_max_stale`` rounds.

        Extinction is the other half of a moving cluster count. Fission alone fills the
        discipline to ``cluster_limit`` in three rounds and then stops — measured
        ``born=2 died=0`` over 15 rounds — so the count settles at the cap and the *set* of
        lines of inquiry freezes even though membership keeps churning.

        Staleness rather than low fitness is the trigger. An unexamined cluster's fitness
        never moves (that is deliberate — see ``EntityEvolution._apply_fitness``), so it
        sits at its initial value forever and cannot express failure; measured over 15
        rounds neglected clusters held ``f=0.50`` while ``stale`` climbed to 15. Sustained
        neglect is the only honest evidence available.

        Members are not deleted — they rejoin via the reassignment pass on the next sync,
        so dissolution retires a *grouping*, not the concepts in it.

        Two limits, because extinction that is merely permitted becomes the collapse this
        rewrite exists to prevent.

        *One per call, the stalest.* Retiring every cluster over the threshold wiped physics
        from 4 clusters to 1 in a single round — with no evaluated ideas ``stale`` advances in
        lockstep, so the threshold is crossed by all of them at once — and left
        ``_crossover`` with no receiver distinct from its donor. Uniform staleness means the
        discipline got no feedback at all, which is not evidence that any *particular* line
        of inquiry failed.

        *A floor of ``CLUSTER_FLOOR``.* Crossover needs two clusters and the no-collapse
        criterion asks for three, so a discipline at the floor stops retiring even when every
        cluster is stale. Below ``cluster_limit == CLUSTER_FLOOR`` dissolution never fires at
        all — correctly, since a discipline with no spare cluster has no room to replace one.
        """
        floor = min(self.config.cluster_limit, self.CLUSTER_FLOOR)
        if len(clusters) <= floor:
            return clusters
        stalest = max(clusters, key=lambda cluster: (cluster.stale, cluster.id))
        if stalest.stale < self.config.cluster_max_stale:
            return clusters
        return [cluster for cluster in clusters if cluster is not stalest]

    def _fission(
        self, discipline: str, clusters: list[EntityCluster]
    ) -> list[EntityCluster]:
        """Split a cluster that has grown into two unrelated groups.

        Without this the cluster *count* is a constant: ``_build_clusters`` emits exactly
        ``cluster_limit`` groups and nothing afterwards creates one, so "clusters don't
        collapse" would be guaranteed by arithmetic rather than by dynamics. Fission is the
        counterweight to extinction — together they let the count move and, more
        importantly, let *which* lines of inquiry exist change over a run.

        A split is accepted only when it improves size-weighted mean intra-cluster
        similarity by ``fission_min_gain``. On the seed populations the best available
        splits are semantically clean — cognitive science parts into
        ``[memory consolidation, working memory]`` and
        ``[cognitive load, metacognition, skill acquisition]`` for a gain of 0.157 — while
        a coherent cluster offers no qualifying split at all, so the criterion is what
        stops fission shredding good clusters.

        The child inherits the parent's fitness and staleness: it represents the same
        history, and starting it at a default would hand it an unearned advantage in
        selection. Its id is derived from the parent *and* its own membership so a resumed
        run rebuilds the same cluster identity. Membership alone is not enough — a split
        producing a set some earlier cluster already held mints a duplicate id, and two
        clusters sharing an id made ``_crossover``'s "every cluster but the donor" filter
        empty. A collision that survives the parent key means the same parent already split
        this way, so the split is declined rather than renamed.
        """
        if len(clusters) >= self.config.cluster_limit:
            return clusters
        splittable = [
            cluster for cluster in clusters
            if len(cluster.entity_ids) >= 2 * self.config.min_cluster_size
        ]
        if not splittable:
            return clusters
        best: tuple[float, EntityCluster, list[str], list[str]] | None = None
        for cluster in splittable:
            candidate = self._best_split(cluster)
            if candidate is None:
                continue
            gain, left, right = candidate
            if best is None or (gain, left) > (best[0], best[2]):
                best = (gain, cluster, left, right)
        if best is None or best[0] < self.config.fission_min_gain:
            return clusters
        _, parent, left, right = best
        taken = {cluster.id for cluster in clusters}
        child_id = stable_id("cluster", discipline, "fission", parent.id, *sorted(right))
        if child_id in taken:
            return clusters
        parent.entity_ids = left
        child = EntityCluster(
            id=child_id,
            discipline=discipline,
            entity_ids=right,
            fitness=parent.fitness,
            stale=parent.stale,
            generation=parent.generation,
        )
        index = clusters.index(parent)
        return clusters[: index + 1] + [child] + clusters[index + 1 :]

    def _best_split(
        self, cluster: EntityCluster
    ) -> tuple[float, list[str], list[str]] | None:
        """Best two-way split of a cluster, or None if every part would be undersized.

        Two medoids, not an exhaustive subset search. Enumerating subsets is tempting
        because clusters are usually near ``min_cluster_size``, but a graph loaded from a
        pre-cluster ``graph.json`` bootstraps whole disciplines into one group: at 28
        members that is 10^8 subsets, and the test suite hung outright on it.

        Seeds are the least similar pair, then each remaining member joins the closer seed;
        an undersized part is topped up from the other's furthest members. Ties break on
        the sorted id list so the layout never depends on dict ordering.
        """
        ids = sorted(cluster.entity_ids)
        minimum = self.config.min_cluster_size
        if len(ids) < 2 * minimum:
            return None
        similarity = {
            (a, b): text_similarity(self.entities[a].name, self.entities[b].name)
            for a, b in itertools.combinations(ids, 2)
        }

        def pair(a: str, b: str) -> float:
            return 1.0 if a == b else similarity[tuple(sorted((a, b)))]  # type: ignore[index]

        left_seed, right_seed = min(
            itertools.combinations(ids, 2), key=lambda ab: (pair(*ab), ab)
        )
        left, right = [left_seed], [right_seed]
        for entity_id in ids:
            if entity_id in (left_seed, right_seed):
                continue
            to_left = pair(entity_id, left_seed)
            to_right = pair(entity_id, right_seed)
            if (to_left, right_seed) >= (to_right, left_seed):
                left.append(entity_id)
            else:
                right.append(entity_id)
        for short, long_side, seed in ((left, right, left_seed), (right, left, right_seed)):
            while len(short) < minimum and len(long_side) > minimum:
                donor = min(long_side, key=lambda entity_id: (pair(entity_id, seed), entity_id))
                long_side.remove(donor)
                short.append(donor)
        if len(left) < minimum or len(right) < minimum:
            return None
        left, right = sorted(left), sorted(right)
        whole = self._cohesion(ids)
        weighted = (
            self._cohesion(left) * len(left) + self._cohesion(right) * len(right)
        ) / len(ids)
        return weighted - whole, left, right

    def _cohesion(self, entity_ids: list[str]) -> float:
        if len(entity_ids) < 2:
            return 1.0
        pairs = list(itertools.combinations(entity_ids, 2))
        return sum(
            text_similarity(self.entities[a].name, self.entities[b].name)
            for a, b in pairs
        ) / len(pairs)

    def _nearest_cluster(
        self, clusters: list[EntityCluster], entity: Entity
    ) -> EntityCluster | None:
        if not clusters:
            return None
        return min(
            clusters,
            key=lambda cluster: (
                -self._affinity(cluster, entity),
                len(cluster.entity_ids),
                cluster.id,
            ),
        )

    def _affinity(self, cluster: EntityCluster, entity: Entity) -> float:
        if not cluster.entity_ids:
            return 0.0
        return sum(
            text_similarity(entity.name, self.entities[entity_id].name)
            for entity_id in cluster.entity_ids
        ) / len(cluster.entity_ids)

    def _cluster_label(self, cluster: EntityCluster) -> str:
        if not cluster.entity_ids:
            return ""
        best = min(
            cluster.entity_ids,
            key=lambda entity_id: (
                -self.entities[entity_id].fitness,
                self.entities[entity_id].name,
            ),
        )
        return self.entities[best].name

    def _build_clusters(
        self, discipline: str, members: dict[str, Entity]
    ) -> list[EntityCluster]:
        """Partition a discipline into at most ``cluster_limit`` clusters of ``min_cluster_size``.

        Connected components are not usable here: the similarity matrix is nearly empty
        at the configured threshold (physics: 0 of 28 seed pairs), so components degenerate
        into singletons until crossover fuses everything into one blob. Seeding on the
        mutually least similar entities and growing under a minimum size gives balanced
        clusters from the same sparse matrix.
        """
        ordered = sorted(members.values(), key=lambda entity: (-entity.fitness, entity.name))
        if not ordered:
            return []
        wanted = max(1, min(self.config.cluster_limit, len(ordered) // max(1, self.config.min_cluster_size)))
        seeds = [ordered[0]]
        while len(seeds) < wanted:
            remaining = [entity for entity in ordered if entity not in seeds]
            if not remaining:
                break
            seeds.append(min(
                remaining,
                key=lambda entity: (
                    max(text_similarity(entity.name, seed.name) for seed in seeds),
                    entity.name,
                ),
            ))
        clusters = [
            EntityCluster(
                id=stable_id("cluster", discipline, str(index)),
                discipline=discipline,
                entity_ids=[seed.id],
                label=seed.name,
                generation=seed.generation,
            )
            for index, seed in enumerate(seeds)
        ]
        for entity in ordered:
            if any(entity.id in cluster.entity_ids for cluster in clusters):
                continue
            host = min(
                clusters,
                key=lambda cluster: (
                    len(cluster.entity_ids) >= self.config.min_cluster_size,
                    -self._affinity(cluster, entity),
                    len(cluster.entity_ids),
                    cluster.id,
                ),
            )
            host.entity_ids.append(entity.id)
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
            "clusters": [cluster.to_dict() for cluster in self.all_clusters()],
        }

    @classmethod
    def from_dict(cls, config: GraphConfig, data: dict[str, object]) -> "KnowledgeGraph":
        graph = cls(config)
        for item in data.get("entities", []):
            graph.add_entity(Entity.from_dict(item), check_containment=False)
        graph.edges = defaultdict(dict, {
            str(key): {str(other): float(score) for other, score in value.items()}
            for key, value in data.get("edges", {}).items()
        })
        for item in data.get("clusters", []):
            cluster = EntityCluster.from_dict(item)
            cluster.entity_ids = [
                entity_id for entity_id in cluster.entity_ids if entity_id in graph.entities
            ]
            if cluster.entity_ids:
                graph._clusters[cluster.discipline.strip().lower()].append(cluster)
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
