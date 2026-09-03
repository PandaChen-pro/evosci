from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .config import LLMConfig
from .parsing import parse_json_object


class LLMBackend(ABC):
    @abstractmethod
    def generate_json(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleBackend(LLMBackend):
    # Observation seam for the web UI: retries happen inside generate_json, so a wrapping
    # decorator cannot see them. Set on the class or the instance; never let it raise.
    on_attempt: Callable[[dict[str, Any]], None] | None = None

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _report_attempt(self, **payload: Any) -> None:
        hook = self.on_attempt
        if hook is None:
            return
        try:
            hook(payload)
        except Exception:
            pass

    def generate_json(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del context
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system
                    + "\nReturn only valid JSON matching this schema:\n"
                    + json.dumps(schema, ensure_ascii=False),
                },
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.config.max_completion_tokens
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if self.config.stream:
            payload["stream"] = True
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    if self.config.stream:
                        content = self._read_stream(response)
                    else:
                        body = json.loads(response.read().decode("utf-8"))
                        content = body["choices"][0]["message"]["content"]
                parsed = parse_json_object(content)
                self._report_attempt(task=task, attempt=attempt + 1, ok=True, error=None)
                return parsed
            except urllib.error.HTTPError as exc:
                try:
                    details = exc.read().decode("utf-8", errors="replace")[:1000]
                except OSError:
                    details = ""
                last_error = RuntimeError(f"HTTP {exc.code}: {details or exc.reason}")
                self._report_attempt(
                    task=task, attempt=attempt + 1, ok=False, error=str(last_error)
                )
                if attempt + 1 < self.config.max_retries:
                    time.sleep(2**attempt)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                self._report_attempt(
                    task=task, attempt=attempt + 1, ok=False, error=str(exc)
                )
                if attempt + 1 < self.config.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Model request failed after retries: {last_error}")

    @staticmethod
    def _read_stream(response: Any) -> str:
        pieces: list[str] = []
        diagnostics: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if event.get("error"):
                diagnostics.append(f"error={event['error']}")
            choices = event.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                diagnostics.append(f"finish_reason={choices[0]['finish_reason']}")
            message = choices[0].get("message") or {}
            if isinstance(message.get("content"), str):
                pieces.append(message["content"])
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                pieces.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        pieces.append(item["text"])
        if not pieces:
            suffix = "; ".join(diagnostics[-3:]) or "no finish/error metadata"
            raise ValueError(
                f"Streaming response did not contain assistant content ({suffix})"
            )
        return "".join(pieces)


class HeuristicBackend(LLMBackend):
    """Deterministic offline backend for demos and integration tests."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _rng(self, task: str, context: dict[str, Any]) -> random.Random:
        encoded = json.dumps(context, sort_keys=True, ensure_ascii=True, default=str)
        digest = hashlib.sha1(f"{self.seed}:{task}:{encoded}".encode()).hexdigest()
        return random.Random(int(digest[:12], 16))

    def generate_json(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del system, user, schema
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ValueError(f"Heuristic backend does not support task {task!r}")
        return handler(context, self._rng(task, context))

    def _task_problem_generation(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        topic = context["topic"]
        entities = context["entities"]
        discipline = context["discipline"]
        count = context["count"]
        templates = [
            "How can {a} and {b} jointly explain or improve {topic}?",
            "Under which conditions does {a} change the behavior of {topic} through {b}?",
            "Can a measurable link between {a}, {b}, and {topic} be established?",
            "What causal mechanism connects {a} with {topic}, and how can {b} test it?",
        ]
        problems = []
        for index in range(count):
            a = entities[index % len(entities)]
            b = entities[(index + 1) % len(entities)]
            question = templates[index % len(templates)].format(a=a, b=b, topic=topic)
            problems.append(
                {
                    "cluster_name": f"{discipline}: {a} x {b}",
                    "discipline": discipline,
                    "problem": question,
                    "description": (
                        f"Investigate {topic} through the complementary lenses of {a} "
                        f"and {b}, separating correlation from mechanism."
                    ),
                    "guidance": (
                        "Survey adjacent work, define a falsifiable hypothesis, identify "
                        "baselines, and specify a controlled experiment."
                    ),
                    "entities": [a, b],
                }
            )
        rng.shuffle(problems)
        return {"problems": problems}

    def _task_research_notes(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        role = context["role"]
        topic = context["topic"]
        problem = context["problem"]
        lenses = ["mechanistic", "causal", "empirical", "representation-level"]
        lens = rng.choice(lenses)
        return {
            "findings": [
                f"A {lens} account should distinguish the proposed effect from scale and data leakage.",
                f"The {role} perspective suggests measuring both average performance and transition dynamics.",
            ],
            "gaps": [
                f"Existing work on {topic} rarely tests the full intervention implied by: {problem}",
                "Ablations that isolate each cross-disciplinary component are often missing.",
            ],
            "recommendations": [
                "Use a preregistered factorial design with matched-compute baselines.",
                "Report uncertainty, negative results, and sensitivity to random seeds.",
            ],
        }

    def _task_idea_generation(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        topic = context["topic"]
        entities = context["entities"]
        round_index = context["round_index"]
        count = context["count"]
        methods = [
            "a controlled intervention study",
            "a causal representation probe",
            "a multi-scale simulation",
            "a contrastive counterfactual benchmark",
            "an adaptive curriculum experiment",
        ]
        ideas = []
        for index in range(count):
            first = entities[index % len(entities)]
            second = entities[(index + round_index + 1) % len(entities)]
            method = methods[(index + round_index) % len(methods)]
            ideas.append(
                {
                    "title": f"{first.title()}-Guided {topic.title()} via {second.title()}",
                    "hypothesis": (
                        f"Explicitly controlling {first} will produce a reproducible change "
                        f"in {topic} mediated by {second}."
                    ),
                    "rationale": (
                        f"The combination of {first} and {second} yields a testable bridge "
                        "between the target phenomenon and an independent explanatory lens."
                    ),
                    "method": method,
                    "experiment": (
                        f"Construct matched conditions with and without {first}; measure {second}, "
                        "task performance, calibration, and dynamics across at least five seeds."
                    ),
                    "expected_outcome": (
                        "A dose-dependent effect that survives compute-matched controls and "
                        "is partially removed by the proposed mediation ablation."
                    ),
                    "risks": [
                        "The cross-disciplinary analogy may be superficial.",
                        "Observed gains may be attributable to added compute.",
                    ],
                    "entities": [first, second],
                }
            )
        rng.shuffle(ideas)
        return {"ideas": ideas}

    def _task_refinement(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Offline stand-in for refinement, mirroring the live prompt's length budget.

        The refinement clause replaces any earlier one rather than stacking: appending it
        every round made each idea monotonically longer, and ``_task_review`` scores partly
        on word count, so rounds appeared to improve when only the text had grown.
        """
        del rng
        idea = dict(context["idea"])
        suggestions = context.get("suggestions", [])
        if suggestions:
            base = str(idea["experiment"]).split(" Refinement: ")[0]
            idea["experiment"] = base + " Refinement: " + " ".join(suggestions[:2])
        idea["risks"] = list(dict.fromkeys(idea.get("risks", []) + [
            "External validity may vary across datasets or domains."
        ]))
        return {"idea": idea}

    def _task_review(self, context: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        idea = context["idea"]
        text = " ".join(str(value) for value in idea.values())
        specificity = min(2.0, len(text.split()) / 180.0)
        has_controls = 0.8 if "control" in text.lower() else 0.0
        has_seeds = 0.5 if "seed" in text.lower() else 0.0
        jitter = lambda: rng.uniform(-0.45, 0.45)
        novelty = 5.4 + specificity + jitter()
        feasibility = 5.8 + has_seeds + jitter()
        validity = 5.5 + has_controls + has_seeds + jitter()
        excitement = 5.2 + specificity + jitter()
        overall = (novelty + feasibility + validity + excitement) / 4
        return {
            "novelty": novelty,
            "feasibility": feasibility,
            "validity": validity,
            "excitement": excitement,
            "overall": overall,
            "confidence": 0.72 + rng.uniform(-0.08, 0.08),
            "strengths": [
                "The proposal states a falsifiable intervention.",
                "The experimental plan includes meaningful controls.",
            ],
            "weaknesses": [
                "The literature grounding has not been independently verified.",
                "The proposed mediator may not be uniquely identifiable.",
            ],
            "suggestions": [
                "Add a negative-control intervention.",
                "Predefine a mediation analysis and a stopping criterion.",
            ],
        }

    def _task_meta_review(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        del rng
        reviews = context["reviews"]
        weights = [max(0.1, float(item.get("confidence", 0.5))) for item in reviews]
        total = sum(weights)

        def aggregate(field: str) -> float:
            return sum(
                float(item.get(field, 5.0)) * weight
                for item, weight in zip(reviews, weights)
            ) / total

        return {
            "novelty": aggregate("novelty"),
            "feasibility": aggregate("feasibility"),
            "validity": aggregate("validity"),
            "excitement": aggregate("excitement"),
            "overall": aggregate("overall"),
            "confidence": sum(
                float(item.get("confidence", 0.5)) for item in reviews
            ) / len(reviews),
            "strengths": list(dict.fromkeys(
                value for item in reviews for value in item.get("strengths", [])
            )),
            "weaknesses": list(dict.fromkeys(
                value for item in reviews for value in item.get("weaknesses", [])
            )),
            "suggestions": list(dict.fromkeys(
                value for item in reviews for value in item.get("suggestions", [])
            )),
        }

    def _task_variation(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        discipline = context["discipline"]
        topic = context["topic"]
        anchors = context["anchors"]
        suffixes = ["Dynamics", "Intervention", "Topology", "Uncertainty", "Scaling"]
        entities = []
        for index in range(context["count"]):
            anchor = anchors[index % len(anchors)]
            suffix = suffixes[(index + rng.randrange(len(suffixes))) % len(suffixes)]
            entities.append(
                {
                    "name": f"{anchor} {suffix}",
                    "kind": "EvolvedConcept",
                    "description": f"A {discipline} concept evolved for studying {topic}.",
                }
            )
        return {"entities": entities}

    def _task_pairwise_compare(
        self, context: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        left = context["left"]
        right = context["right"]
        left_score = len(left.get("experiment", "")) + rng.uniform(0, 10)
        right_score = len(right.get("experiment", "")) + rng.uniform(0, 10)
        return {
            "winner": "left" if left_score >= right_score else "right",
            "reason": "Selected for the more specific and testable experimental plan.",
        }


def build_backend(config: LLMConfig, seed: int = 42) -> LLMBackend:
    if config.provider == "heuristic":
        return HeuristicBackend(seed=seed)
    if config.provider == "openai-compatible":
        return OpenAICompatibleBackend(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
