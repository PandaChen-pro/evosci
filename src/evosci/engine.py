from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .agents import MentorAgent, ResearchTeam, ReviewerPanel, TournamentRanker
from .config import EvoSciConfig
from .diagnostics import diagnose
from .evolution import EntityEvolution
from .knowledge import KnowledgeGraph
from .llm import LLMBackend, build_backend
from .models import RoundResult, RunState
from .retrieval import LiteratureRetriever


ProgressCallback = Callable[[str], None]


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:60] or "research-topic"


class EvoSciEngine:
    def __init__(
        self,
        config: EvoSciConfig | None = None,
        *,
        backend: LLMBackend | None = None,
        graph: KnowledgeGraph | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config or EvoSciConfig()
        self.config.validate()
        self.backend = backend or build_backend(
            self.config.llm, seed=self.config.run.random_seed
        )
        self.graph = graph or KnowledgeGraph(self.config.graph)
        self.progress = progress or (lambda _: None)
        self.retriever = LiteratureRetriever(self.config.retrieval)

    def run(
        self,
        topic: str,
        disciplines: list[str],
        *,
        run_dir: str | Path | None = None,
        state: RunState | None = None,
    ) -> tuple[RunState, Path]:
        disciplines = list(dict.fromkeys(item.strip().lower() for item in disciplines if item.strip()))
        if not topic.strip():
            raise ValueError("topic must not be empty")
        if not disciplines:
            raise ValueError("at least one target discipline is required")
        if state is None:
            state = RunState(topic=topic.strip(), disciplines=disciplines)
        elif state.topic != topic or state.disciplines != disciplines:
            raise ValueError("Resume state does not match topic and disciplines")

        destination = Path(run_dir) if run_dir else self._new_run_dir(topic)
        destination.mkdir(parents=True, exist_ok=True)
        self._write_config(destination)

        if not self.graph.entities:
            self.progress("Initializing discipline-entity knowledge graph")
            self.graph.initialize(disciplines)

        mentor = MentorAgent(self.backend, self.graph)
        team = ResearchTeam(self.backend, self.graph)
        reviewers = ReviewerPanel(self.backend, self.config.run.reviewer_count)
        evolution = EntityEvolution(
            self.graph,
            self.backend,
            self.config.evolution,
            seed=self.config.run.random_seed,
        )

        for round_index in range(len(state.rounds) + 1, self.config.run.rounds + 1):
            self.progress(f"Round {round_index}: constructing problem space")
            problems = mentor.construct_problem_space(
                topic,
                disciplines,
                self.config.run.problem_count,
                round_index,
            )
            if not problems:
                raise RuntimeError("Mentor could not construct any research problems")

            literature_query = f"{topic} {problems[0].problem}"
            literature = [paper.to_dict() for paper in self.retriever.search(literature_query)]
            prior_feedback = self._prior_feedback(state)
            self.progress(f"Round {round_index}: running research team")
            ideas = team.explore(
                topic=topic,
                problems=problems,
                team_size=self.config.run.team_size,
                idea_count=self.config.run.ideas_per_round,
                round_index=round_index,
                prior_feedback=prior_feedback,
                literature=literature,
            )
            self.progress(f"Round {round_index}: reviewing {len(ideas)} ideas")
            evaluated = [reviewers.evaluate(idea) for idea in ideas]
            self.progress(f"Round {round_index}: evolving entity population")
            evolution_summary = evolution.evolve(
                topic, evaluated, disciplines, generation=round_index
            )
            state.rounds.append(RoundResult(
                round_index=round_index,
                problems=problems,
                evaluated_ideas=evaluated,
                evolution_summary=evolution_summary,
            ))
            self._checkpoint(destination, state)

        self._write_report(destination, state)
        return state, destination

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> "EvoSciEngine":
        directory = Path(run_dir)
        config = EvoSciConfig.from_dict(json.loads((directory / "config.json").read_text()))
        graph = KnowledgeGraph.from_dict(
            config.graph, json.loads((directory / "graph.json").read_text())
        )
        return cls(config, graph=graph, progress=progress)

    @staticmethod
    def load_state(run_dir: str | Path) -> RunState:
        data = json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))
        return RunState.from_dict(data)

    def _new_run_dir(self, topic: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(self.config.run.output_dir) / f"{timestamp}-{_slug(topic)}"

    def _prior_feedback(self, state: RunState) -> list[str]:
        if not state.rounds:
            return []
        ranked = sorted(
            state.rounds[-1].evaluated_ideas,
            key=lambda item: item.fitness,
            reverse=True,
        )
        return list(dict.fromkeys(
            suggestion
            for item in ranked[:3]
            for suggestion in item.meta_review.suggestions
        ))[:8]

    def _write_config(self, destination: Path) -> None:
        (destination / "config.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _checkpoint(self, destination: Path, state: RunState) -> None:
        (destination / "state.json").write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.graph.save(destination / "graph.json")
        report = diagnose(state, self.graph)
        (destination / "diagnostics.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (destination / "diagnostics.md").write_text(report.render(), encoding="utf-8")

    def _write_report(self, destination: Path, state: RunState) -> None:
        all_evaluated = [item for round_result in state.rounds for item in round_result.evaluated_ideas]
        ranking = TournamentRanker(self.backend).rank(
            [item.idea for item in all_evaluated], rounds=5
        )
        tournament_scores = dict(ranking)
        ordered = sorted(
            all_evaluated,
            key=lambda item: (tournament_scores.get(item.idea.id, 0), item.fitness),
            reverse=True,
        )
        lines = [
            f"# EvoSci Run: {state.topic}",
            "",
            f"- Disciplines: {', '.join(state.disciplines)}",
            f"- Completed rounds: {len(state.rounds)}",
            f"- Generated ideas: {len(all_evaluated)}",
            "",
            "## Ranked Ideas",
            "",
        ]
        for rank, item in enumerate(ordered, 1):
            review = item.meta_review
            lines.extend([
                f"### {rank}. {item.idea.title}",
                "",
                f"- Round: {item.idea.round_index}",
                f"- Tournament points: {tournament_scores.get(item.idea.id, 0)}",
                f"- Evolution fitness: {item.fitness:.3f}",
                f"- Scores: novelty {review.novelty:.2f}, feasibility {review.feasibility:.2f}, "
                f"validity {review.validity:.2f}, excitement {review.excitement:.2f}, overall {review.overall:.2f}",
                f"- Hypothesis: {item.idea.hypothesis}",
                f"- Method: {item.idea.method}",
                f"- Experiment: {item.idea.experiment}",
                f"- Weaknesses: {'; '.join(review.weaknesses)}",
                f"- Suggestions: {'; '.join(review.suggestions)}",
                "",
            ])
        (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
