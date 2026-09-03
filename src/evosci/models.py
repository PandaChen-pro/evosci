from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Entity:
    id: str
    name: str
    discipline: str
    kind: str = "Concept"
    description: str = ""
    source: str = "seed"
    fitness: float = 0.5
    generation: int = 0
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        return cls(**data)


@dataclass(slots=True)
class EntityCluster:
    """A persistent semantic cluster: the paper's basic unit of evolution (§3.4)."""

    id: str
    discipline: str
    entity_ids: list[str]
    score: float = 0.0
    label: str = ""
    fitness: float = 0.5
    times_selected: int = 0
    stale: int = 0
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityCluster":
        return cls(**data)


@dataclass(slots=True)
class ResearchProblem:
    id: str
    cluster_name: str
    discipline: str
    problem: str
    description: str
    guidance: str
    entity_ids: list[str] = field(default_factory=list)
    cluster_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchProblem":
        return cls(**data)


@dataclass(slots=True)
class ScientistProfile:
    id: str
    topics: list[str]
    affiliations: list[str] = field(default_factory=list)
    paper_count: int = 0
    citation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScientistProfile":
        return cls(**data)


@dataclass(slots=True)
class ResearchIdea:
    id: str
    title: str
    hypothesis: str
    rationale: str
    method: str
    experiment: str
    expected_outcome: str
    risks: list[str]
    entity_ids: list[str]
    round_index: int
    authors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchIdea":
        return cls(**data)


@dataclass(slots=True)
class Review:
    reviewer_id: str
    novelty: float
    feasibility: float
    validity: float
    excitement: float
    overall: float
    confidence: float
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Review":
        return cls(**data)


@dataclass(slots=True)
class EvaluatedIdea:
    idea: ResearchIdea
    reviews: list[Review]
    meta_review: Review

    @property
    def fitness(self) -> float:
        score = self.meta_review
        weighted = (
            0.30 * score.novelty
            + 0.20 * score.feasibility
            + 0.25 * score.validity
            + 0.15 * score.excitement
            + 0.10 * score.overall
        )
        return max(0.0, min(1.0, weighted / 10.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea.to_dict(),
            "reviews": [review.to_dict() for review in self.reviews],
            "meta_review": self.meta_review.to_dict(),
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluatedIdea":
        return cls(
            idea=ResearchIdea.from_dict(data["idea"]),
            reviews=[Review.from_dict(item) for item in data["reviews"]],
            meta_review=Review.from_dict(data["meta_review"]),
        )


@dataclass(slots=True)
class RoundResult:
    round_index: int
    problems: list[ResearchProblem]
    evaluated_ideas: list[EvaluatedIdea]
    evolution_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "problems": [problem.to_dict() for problem in self.problems],
            "evaluated_ideas": [item.to_dict() for item in self.evaluated_ideas],
            "evolution_summary": self.evolution_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoundResult":
        return cls(
            round_index=data["round_index"],
            problems=[ResearchProblem.from_dict(item) for item in data["problems"]],
            evaluated_ideas=[
                EvaluatedIdea.from_dict(item) for item in data["evaluated_ideas"]
            ],
            evolution_summary=data["evolution_summary"],
        )


@dataclass(slots=True)
class RunState:
    topic: str
    disciplines: list[str]
    rounds: list[RoundResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "disciplines": self.disciplines,
            "rounds": [round_result.to_dict() for round_result in self.rounds],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(
            topic=data["topic"],
            disciplines=data["disciplines"],
            rounds=[RoundResult.from_dict(item) for item in data["rounds"]],
        )
