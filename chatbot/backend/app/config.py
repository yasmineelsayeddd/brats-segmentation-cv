from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH) as f:
            _cache = yaml.safe_load(f)
    return _cache


def get(section: str, key: str, default: Any = None) -> Any:
    return _load().get(section, {}).get(key, default)


def get_nested(section: str, *keys: str, default: Any = None) -> Any:
    node = _load().get(section, {})
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


def llm_config() -> dict[str, Any]:
    return _load().get("llm", {})


def backend_config() -> dict[str, Any]:
    return _load().get("backend", {})


def ml_config() -> dict[str, Any]:
    return _load().get("ml", {})


def agent_config() -> dict[str, Any]:
    return _load().get("agent", {})
