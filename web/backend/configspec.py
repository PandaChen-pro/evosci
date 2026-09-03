"""Derive the config form's field list from the dataclasses themselves.

Reading ``dataclasses.fields`` at request time means the form cannot drift from
``EvoSciConfig``: a new knob in ``config.py`` appears in the UI with no frontend change,
and a removed one disappears. ``api_key`` is a property rather than a field, so it is
structurally absent from everything here.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, asdict, fields, is_dataclass
from typing import Any, get_args, get_origin

from evosci.config import EvoSciConfig

from .keys import name_error

SECRET_HINTS = ("api_key", "apikey", "secret", "password", "token")

# Hints match whole underscore-delimited words, not substrings: `token` must not swallow
# `max_completion_tokens`, which is an ordinary integer knob and was being refused as a
# secret.
SECRET_HINT_PATTERN = re.compile(
    r"(?:^|_)(?:" + "|".join(re.escape(hint) for hint in SECRET_HINTS) + r")(?:_|$)"
)

# Fields whose annotation is `str` but whose accepted values are a closed set. Without
# this the form renders a free-text box pre-filled with "heuristic", and submitting it
# unchanged runs the offline backend while the model and base_url fields say otherwise.
CHOICES = {
    "llm.provider": ["openai-compatible", "heuristic"],
    "retrieval.provider": ["arxiv"],
    "llm.reasoning_effort": ["", "low", "medium", "high"],
}


def _kind(annotation: Any) -> tuple[str, bool]:
    """Return a JS-friendly type name and whether the field accepts None."""
    text = annotation if isinstance(annotation, str) else str(annotation)
    optional = "None" in text
    for name, kind in (("bool", "bool"), ("int", "int"), ("float", "float"), ("str", "str")):
        if name in text:
            return kind, optional
    if get_origin(annotation) is not None and get_args(annotation):
        return "str", optional
    return "str", optional


def field_spec() -> list[dict[str, Any]]:
    spec: list[dict[str, Any]] = []
    for section in fields(EvoSciConfig):
        if section.default_factory is MISSING:
            continue
        nested = section.default_factory()
        if not is_dataclass(nested):
            continue
        for item in fields(nested):
            kind, optional = _kind(item.type)
            default = getattr(nested, item.name)
            path = f"{section.name}.{item.name}"
            entry = {
                "path": path,
                "section": section.name,
                "name": item.name,
                "type": kind,
                "optional": optional,
                "default": default,
            }
            if path in CHOICES:
                entry["choices"] = CHOICES[path]
            spec.append(entry)
    return spec


def default_config() -> dict[str, Any]:
    return asdict(EvoSciConfig())


def known_paths() -> set[str]:
    return {item["path"] for item in field_spec()}


def is_secret_path(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].lower()
    if leaf == "api_key_env":
        return False
    return SECRET_HINT_PATTERN.search(leaf) is not None


def value_error(path: str, value: Any) -> str | None:
    """Reject a value the dataclass would happily accept but a human clearly misfilled.

    ``api_key_env`` is a legitimate field, so the key-name blocklist above cannot catch a
    real secret pasted into it — and a secret there is written verbatim into config.json,
    which the API serves. The check therefore has to be on the value.
    """
    if path == "llm.api_key_env":
        return name_error(str(value or ""))
    choices = CHOICES.get(path)
    if choices is not None:
        text = "" if value is None else str(value)
        if text not in choices:
            allowed = ", ".join(item or "（空）" for item in choices)
            return f"只能是以下之一：{allowed}"
    return None


def apply_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Fold dotted overrides into a nested config dict, reporting unusable paths.

    ``EvoSciConfig.from_dict`` raises a bare ``TypeError`` naming an internal ``__init__``
    for an unknown key, which is not something to hand a browser — so paths are checked
    against the dataclass field set first.
    """
    valid = known_paths()
    invalid = [
        path for path in overrides
        if path not in valid or is_secret_path(path)
    ]
    rejected: dict[str, str] = {}
    data: dict[str, Any] = {}
    for path, value in overrides.items():
        if path in invalid:
            continue
        problem = value_error(path, value)
        if problem:
            rejected[path] = problem
            continue
        section, _, name = path.partition(".")
        data.setdefault(section, {})[name] = value
    return data, invalid, rejected
