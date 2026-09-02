from __future__ import annotations

import itertools
from typing import Any

from .knowledge import KnowledgeGraph, stable_id, text_similarity
from .llm import LLMBackend
from .models import (
    EvaluatedIdea,
    ResearchIdea,
    ResearchProblem,
    Review,
    ScientistProfile,
)
from .parsing import as_string_list, clamp_score, require_fields


PROBLEM_SCHEMA = {
    "problems": [{
        "cluster_name": "string",
        "discipline": "string",
        "problem": "string",
        "description": "string",
        "guidance": "string",
        "entities": ["string"],
    }]
}

IDEA_SCHEMA = {
    "ideas": [{
        "title": "string",
        "hypothesis": "string",
        "rationale": "string",
        "method": "string",
        "experiment": "string",
        "expected_outcome": "string",
        "risks": ["string"],
        "entities": ["string"],
    }]
}

REVIEW_SCHEMA = {
    "novelty": "number 1-10",
    "feasibility": "number 1-10",
    "validity": "number 1-10",
    "excitement": "number 1-10",
    "overall": "number 1-10",
    "confidence": "number 0-1",
    "strengths": ["string"],
    "weaknesses": ["string"],
    "suggestions": ["string"],
}


DEFAULT_SCIENTISTS = [
    ScientistProfile(
        "scientist_program_analysis",
        [
            "static analysis", "interprocedural data flow", "taint analysis",
            "control flow", "program dependence graph", "vulnerability localization",
        ],
    ),
    ScientistProfile(
        "scientist_code_representation",
        [
            "CodeBERT", "GraphCodeBERT", "code language models", "multimodal fusion",
            "repository graphs", "contrastive learning", "long context",
        ],
    ),
    ScientistProfile(
        "scientist_software_security",
        [
            "software vulnerability detection", "CWE", "cross-function vulnerabilities",
            "repository-level security", "patch history", "security evaluation",
        ],
    ),
    ScientistProfile("scientist_learning", ["machine learning", "optimization", "generalization"]),
    ScientistProfile("scientist_causal", ["causal inference", "statistics", "experiments"]),
    ScientistProfile("scientist_physics", ["physics", "phase transition", "dynamical systems"]),
    ScientistProfile("scientist_cognitive", ["cognitive science", "memory", "learning"]),
    ScientistProfile("scientist_biology", ["biology", "evolution", "population dynamics"]),
    ScientistProfile("scientist_systems", ["computer systems", "scaling", "benchmarking"]),
    ScientistProfile("scientist_math", ["mathematics", "topology", "spectral analysis"]),
    ScientistProfile("scientist_domain", ["simulation", "measurement", "external validity"]),
]


class MentorAgent:
    def __init__(self, backend: LLMBackend, graph: KnowledgeGraph) -> None:
        self.backend = backend
        self.graph = graph

    def construct_problem_space(
        self,
        topic: str,
        disciplines: list[str],
        count: int,
        round_index: int,
    ) -> list[ResearchProblem]:
        candidates: list[tuple[str, list[str], float]] = []
        for discipline in disciplines:
            clusters = self.graph.clusters(discipline, topic)
            if not clusters:
                continue
            for cluster in clusters[:2]:
                names = [self.graph.entities[entity_id].name for entity_id in cluster.entity_ids]
                candidates.append((discipline, names, cluster.score))
        candidates.sort(key=lambda item: item[2], reverse=True)

        problems: list[ResearchProblem] = []
        for discipline, cluster_names, _ in candidates:
            if len(problems) >= count:
                break
            entity_names = cluster_names or [
                entity.name for entity in self.graph.top_entities(discipline, topic, limit=4)
            ]
            if len(entity_names) < 2:
                entity_names.extend(
                    entity.name for entity in self.graph.top_entities(discipline, topic, limit=4)
                    if entity.name not in entity_names
                )
            response = self.backend.generate_json(
                task="problem_generation",
                system=(
                    "You are the mentor of an interdisciplinary research team. Construct "
                    "specific, falsifiable problem clusters rather than broad themes."
                ),
                user=(
                    f"Topic: {topic}\nDiscipline: {discipline}\n"
                    f"Entity cluster: {entity_names}\nRound: {round_index}"
                ),
                schema=PROBLEM_SCHEMA,
                context={
                    "topic": topic,
                    "discipline": discipline,
                    "entities": entity_names,
                    "count": 1,
                    "round_index": round_index,
                },
            )
            for index, item in enumerate(response.get("problems", [])):
                require_fields(
                    item,
                    ["cluster_name", "problem", "description", "guidance"],
                    "problem_generation",
                )
                referenced = as_string_list(item.get("entities"))
                entity_ids = self._resolve_entities(referenced, discipline)
                problems.append(ResearchProblem(
                    id=stable_id(
                        "problem", topic, str(round_index), str(len(problems)), item["problem"]
                    ),
                    cluster_name=str(item["cluster_name"]),
                    discipline=str(item.get("discipline") or discipline).lower(),
                    problem=str(item["problem"]),
                    description=str(item["description"]),
                    guidance=str(item["guidance"]),
                    entity_ids=entity_ids,
                ))
                if len(problems) >= count:
                    break
        return problems

    def _resolve_entities(self, names: list[str], discipline: str) -> list[str]:
        resolved = []
        for name in names:
            entity = self.graph.find_by_name(name, discipline)
            if entity:
                resolved.append(entity.id)
        if not resolved:
            resolved = [
                entity.id for entity in self.graph.top_entities(discipline, " ".join(names), 2)
            ]
        return resolved


class ScientistRegistry:
    def __init__(self, profiles: list[ScientistProfile] | None = None) -> None:
        self.profiles = profiles or DEFAULT_SCIENTISTS

    def select(self, query: str, count: int) -> list[ScientistProfile]:
        ranked = sorted(
            self.profiles,
            key=lambda profile: text_similarity(query, " ".join(profile.topics)),
            reverse=True,
        )
        if count <= len(ranked):
            return ranked[:count]
        return list(itertools.islice(itertools.cycle(ranked), count))


class ResearchTeam:
    def __init__(
        self,
        backend: LLMBackend,
        graph: KnowledgeGraph,
        registry: ScientistRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.graph = graph
        self.registry = registry or ScientistRegistry()

    def explore(
        self,
        *,
        topic: str,
        problems: list[ResearchProblem],
        team_size: int,
        idea_count: int,
        round_index: int,
        prior_feedback: list[str] | None = None,
        literature: list[dict[str, str]] | None = None,
    ) -> list[ResearchIdea]:
        if not problems:
            raise ValueError("ResearchTeam requires at least one problem")
        query = " ".join(
            [topic] + [problem.discipline + " " + problem.problem for problem in problems]
        )
        profiles = self.registry.select(query, team_size)
        notes = []
        for profile in profiles:
            problem = problems[len(notes) % len(problems)]
            response = self.backend.generate_json(
                task="research_notes",
                system=(
                    "You are an assistant research scientist. Analyze the assigned problem "
                    "from your domain background and propose falsifiable next steps."
                ),
                user=(
                    f"Profile topics: {profile.topics}\nTopic: {topic}\n"
                    f"Problem: {problem.problem}\nGuidance: {problem.guidance}\n"
                    f"Literature: {literature or []}"
                ),
                schema={
                    "findings": ["string"],
                    "gaps": ["string"],
                    "recommendations": ["string"],
                },
                context={
                    "role": profile.id,
                    "topic": topic,
                    "problem": problem.problem,
                    "discipline": problem.discipline,
                    "literature": literature or [],
                },
            )
            notes.append({"scientist": profile.id, **response})

        entity_ids = list(dict.fromkeys(
            entity_id for problem in problems for entity_id in problem.entity_ids
        ))
        for problem in problems:
            entity_ids.extend(
                entity.id
                for entity in self.graph.top_entities(problem.discipline, topic, limit=3)
                if entity.id not in entity_ids
            )
        entity_names = [self.graph.entities[entity_id].name for entity_id in entity_ids]
        response = self.backend.generate_json(
            task="idea_generation",
            system=(
                "You are the prime research scientist. Integrate the team's findings into "
                "novel but feasible ideas. Each idea must state a falsifiable hypothesis, "
                "matched baselines, measurements, ablations, risks, and expected outcomes."
            ),
            user=(
                f"Topic: {topic}\nProblems: {[problem.to_dict() for problem in problems]}\n"
                f"Team notes: {notes}\nPrior reviewer feedback: {prior_feedback or []}\n"
                f"Available entities: {entity_names}\nGenerate {idea_count} ideas."
            ),
            schema=IDEA_SCHEMA,
            context={
                "topic": topic,
                "problems": [problem.to_dict() for problem in problems],
                "notes": notes,
                "prior_feedback": prior_feedback or [],
                "entities": entity_names,
                "round_index": round_index,
                "count": idea_count,
            },
        )
        ideas = []
        for index, item in enumerate(response.get("ideas", [])):
            require_fields(
                item,
                ["title", "hypothesis", "rationale", "method", "experiment", "expected_outcome"],
                "idea_generation",
            )
            referenced_names = as_string_list(item.get("entities"))
            referenced_ids = [
                entity.id for name in referenced_names
                if (entity := self.graph.find_by_name(name)) is not None
            ]
            if not referenced_ids:
                referenced_ids = entity_ids[:2]
            draft = {
                "title": str(item["title"]),
                "hypothesis": str(item["hypothesis"]),
                "rationale": str(item["rationale"]),
                "method": str(item["method"]),
                "experiment": str(item["experiment"]),
                "expected_outcome": str(item["expected_outcome"]),
                "risks": as_string_list(item.get("risks")),
                "entities": referenced_names,
            }
            team_suggestions = list(dict.fromkeys(
                suggestion
                for note in notes
                for suggestion in as_string_list(note.get("recommendations"))
            ))
            refinement = self.backend.generate_json(
                task="refinement",
                system=(
                    "You are the prime researcher refining a seed idea after team discussion. "
                    "Preserve the core hypothesis, but strengthen controls, measurements, "
                    "ablations, uncertainty reporting, and failure criteria."
                ),
                user=(
                    f"Draft: {draft}\nTeam recommendations: {team_suggestions}\n"
                    f"Prior reviewer feedback: {prior_feedback or []}"
                ),
                schema={"idea": IDEA_SCHEMA["ideas"][0]},
                context={
                    "idea": draft,
                    "suggestions": team_suggestions + (prior_feedback or []),
                    "round_index": round_index,
                },
            )
            refined = refinement.get("idea", draft)
            require_fields(
                refined,
                ["title", "hypothesis", "rationale", "method", "experiment", "expected_outcome"],
                "refinement",
            )
            ideas.append(ResearchIdea(
                id=stable_id("idea", topic, str(round_index), str(index), str(item["title"])),
                title=str(refined["title"]),
                hypothesis=str(refined["hypothesis"]),
                rationale=str(refined["rationale"]),
                method=str(refined["method"]),
                experiment=str(refined["experiment"]),
                expected_outcome=str(refined["expected_outcome"]),
                risks=as_string_list(refined.get("risks")),
                entity_ids=list(dict.fromkeys(referenced_ids)),
                round_index=round_index,
                authors=[profile.id for profile in profiles],
            ))
        if not ideas:
            raise ValueError("The model returned no research ideas")
        return ideas


class ReviewerPanel:
    def __init__(self, backend: LLMBackend, reviewer_count: int = 3) -> None:
        self.backend = backend
        self.reviewer_count = reviewer_count

    def evaluate(self, idea: ResearchIdea) -> EvaluatedIdea:
        reviews = [self._review(idea, index) for index in range(self.reviewer_count)]
        return EvaluatedIdea(
            idea=idea, reviews=reviews, meta_review=self._meta_review(idea, reviews)
        )

    def _review(self, idea: ResearchIdea, index: int) -> Review:
        response = self.backend.generate_json(
            task="review",
            system=(
                "You are an independent conference reviewer. Evaluate the proposal itself, "
                "not its writing style. Check falsifiability, novelty, controls, feasibility, "
                "and whether the claimed mechanism can be distinguished from alternatives."
            ),
            user=f"Reviewer index: {index}\nProposal: {idea.to_dict()}",
            schema=REVIEW_SCHEMA,
            context={"reviewer_index": index, "idea": idea.to_dict()},
        )
        return Review(
            reviewer_id=f"reviewer_{index + 1}",
            novelty=clamp_score(response.get("novelty")),
            feasibility=clamp_score(response.get("feasibility")),
            validity=clamp_score(response.get("validity")),
            excitement=clamp_score(response.get("excitement")),
            overall=clamp_score(response.get("overall")),
            confidence=max(0.0, min(1.0, float(response.get("confidence", 0.5)))),
            strengths=as_string_list(response.get("strengths")),
            weaknesses=as_string_list(response.get("weaknesses")),
            suggestions=as_string_list(response.get("suggestions")),
        )

    def _meta_review(self, idea: ResearchIdea, reviews: list[Review]) -> Review:
        response = self.backend.generate_json(
            task="meta_review",
            system=(
                "You are a meta-reviewer. Reconcile independent reviews without blindly "
                "averaging them. Resolve factual disagreements, account for reviewer "
                "confidence, and produce a single calibrated assessment."
            ),
            user=(
                f"Proposal: {idea.to_dict()}\n"
                f"Independent reviews: {[review.to_dict() for review in reviews]}"
            ),
            schema=REVIEW_SCHEMA,
            context={
                "idea": idea.to_dict(),
                "reviews": [review.to_dict() for review in reviews],
            },
        )
        return Review(
            reviewer_id="meta_reviewer",
            novelty=clamp_score(response.get("novelty")),
            feasibility=clamp_score(response.get("feasibility")),
            validity=clamp_score(response.get("validity")),
            excitement=clamp_score(response.get("excitement")),
            overall=clamp_score(response.get("overall")),
            confidence=max(0.0, min(1.0, float(response.get("confidence", 0.5)))),
            strengths=as_string_list(response.get("strengths")),
            weaknesses=as_string_list(response.get("weaknesses")),
            suggestions=as_string_list(response.get("suggestions")),
        )


class TournamentRanker:
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def rank(self, ideas: list[ResearchIdea], rounds: int = 5) -> list[tuple[str, int]]:
        if len(ideas) < 2:
            return [(idea.id, 1) for idea in ideas]
        scores = {idea.id: 1 for idea in ideas}
        for round_index in range(rounds):
            ordered = sorted(ideas, key=lambda idea: (scores[idea.id], idea.id), reverse=True)
            if round_index % 2:
                ordered = list(reversed(ordered))
            for pair_index in range(0, len(ordered) - 1, 2):
                left, right = ordered[pair_index : pair_index + 2]
                response = self.backend.generate_json(
                    task="pairwise_compare",
                    system=(
                        "Compare two research proposals. Choose the one more likely to be "
                        "accepted at a top-tier research venue. Return left or right."
                    ),
                    user=f"Left: {left.to_dict()}\nRight: {right.to_dict()}",
                    schema={"winner": "left|right", "reason": "string"},
                    context={
                        "round": round_index,
                        "left": left.to_dict(),
                        "right": right.to_dict(),
                    },
                )
                winner = left if response.get("winner") == "left" else right
                scores[winner.id] += 1
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
