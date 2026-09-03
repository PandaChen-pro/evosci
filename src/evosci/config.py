from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LLMConfig:
    provider: str = "heuristic"
    model: str = "offline-heuristic"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    stream: bool = False
    timeout_seconds: int = 120
    max_retries: int = 3

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass(slots=True)
class GraphConfig:
    similarity_threshold: float = 0.38
    max_entities_per_discipline: int = 40
    cluster_limit: int = 4
    min_cluster_size: int = 2
    containment_threshold: float = 0.90
    max_name_word_growth: int = 1
    fission_min_gain: float = 0.05
    cluster_max_stale: int = 6
    use_wikipedia: bool = False


@dataclass(slots=True)
class EvolutionConfig:
    selection_ratio: float = 0.5
    crossover_count: int = 2
    variation_count: int = 2
    max_population_per_discipline: int = 30
    fitness_decay: float = 0.92
    min_fitness: float = 0.15
    prune_ratio: float = 0.15
    prune_floor_ratio: float = 0.80


@dataclass(slots=True)
class RetrievalConfig:
    enabled: bool = False
    provider: str = "arxiv"
    max_results: int = 5
    timeout_seconds: int = 20


@dataclass(slots=True)
class RunConfig:
    rounds: int = 3
    team_size: int = 5
    ideas_per_round: int = 5
    problem_count: int = 3
    reviewer_count: int = 3
    random_seed: int = 42
    output_dir: str = "runs"


@dataclass(slots=True)
class EvoSciConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "EvoSciConfig":
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvoSciConfig":
        return cls(
            llm=LLMConfig(**data.get("llm", {})),
            graph=GraphConfig(**data.get("graph", {})),
            evolution=EvolutionConfig(**data.get("evolution", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            run=RunConfig(**data.get("run", {})),
        )

    def validate(self) -> None:
        if self.run.rounds < 1:
            raise ValueError("run.rounds must be at least 1")
        if self.run.team_size < 1:
            raise ValueError("run.team_size must be at least 1")
        if self.run.ideas_per_round < 1:
            raise ValueError("run.ideas_per_round must be at least 1")
        if self.run.reviewer_count < 1:
            raise ValueError("run.reviewer_count must be at least 1")
        if self.run.problem_count < 1:
            raise ValueError("run.problem_count must be at least 1")
        if not 0.0 < self.evolution.selection_ratio <= 1.0:
            raise ValueError("evolution.selection_ratio must be in (0, 1]")
        if not 0.0 < self.evolution.prune_ratio <= 1.0:
            raise ValueError("evolution.prune_ratio must be in (0, 1]")
        if not 0.0 < self.evolution.prune_floor_ratio <= 1.0:
            raise ValueError("evolution.prune_floor_ratio must be in (0, 1]")
        if not 0.0 <= self.graph.similarity_threshold <= 1.0:
            raise ValueError("graph.similarity_threshold must be in [0, 1]")
        if self.graph.max_entities_per_discipline < 2:
            raise ValueError("graph.max_entities_per_discipline must be at least 2")
        if self.graph.cluster_limit < 1:
            raise ValueError("graph.cluster_limit must be at least 1")
        if self.graph.min_cluster_size < 1:
            raise ValueError("graph.min_cluster_size must be at least 1")
        if not 0.0 < self.graph.containment_threshold <= 1.0:
            raise ValueError("graph.containment_threshold must be in (0, 1]")
        if self.graph.max_name_word_growth < 0:
            raise ValueError("graph.max_name_word_growth must be non-negative")
        if self.graph.fission_min_gain <= 0.0:
            raise ValueError("graph.fission_min_gain must be positive")
        if self.graph.cluster_max_stale < 1:
            raise ValueError("graph.cluster_max_stale must be at least 1")
        if self.retrieval.max_results < 0:
            raise ValueError("retrieval.max_results must be non-negative")
        if self.llm.provider == "openai-compatible" and not self.llm.api_key:
            raise ValueError(
                f"Missing API key in environment variable {self.llm.api_key_env}"
            )
        if (
            self.llm.max_completion_tokens is not None
            and self.llm.max_completion_tokens < 64
        ):
            raise ValueError("llm.max_completion_tokens must be at least 64")
        if self.llm.reasoning_effort not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError(
                "llm.reasoning_effort must be minimal, low, medium, high, or omitted"
            )
