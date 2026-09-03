"""Derive the config form's field list from the dataclasses themselves.

Reading ``dataclasses.fields`` at request time means the form cannot drift from
``EvoSciConfig``: a new knob in ``config.py`` appears in the UI with no frontend change,
and a removed one disappears. ``api_key`` is a property rather than a field, so it is
structurally absent from everything here.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, fields, is_dataclass
from typing import Any, get_args, get_origin

from evosci.config import EvoSciConfig

SECRET_HINTS = ("api_key", "apikey", "secret", "password", "token")


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
            spec.append({
                "path": f"{section.name}.{item.name}",
                "section": section.name,
                "name": item.name,
                "type": kind,
                "optional": optional,
                "default": default,
            })
    return spec


def default_config() -> dict[str, Any]:
    return asdict(EvoSciConfig())


def known_paths() -> set[str]:
    return {item["path"] for item in field_spec()}


def is_secret_path(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].lower()
    if leaf == "api_key_env":
        return False
    return any(hint in leaf for hint in SECRET_HINTS)


def apply_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
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
    data: dict[str, Any] = {}
    for path, value in overrides.items():
        if path in invalid:
            continue
        section, _, name = path.partition(".")
        data.setdefault(section, {})[name] = value
    return data, invalid
