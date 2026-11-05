"""Config loader for the candidate version. Does not read E0 thresholds as defaults."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from I2.mcsd.errors import ConfigError


def load_config(path):
    text = Path(path).read_text()
    try:
        import yaml
    except ModuleNotFoundError:
        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    data["_config_path"] = str(path)
    data["_config_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return data


def config_sha256(config):
    if config.get("_config_sha256"):
        return config["_config_sha256"]
    payload = {key: value for key, value in config.items() if not str(key).startswith("_")}
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require_keys(mapping, keys, label):
    if not isinstance(mapping, dict):
        raise ConfigError("%s config is missing" % label)
    missing = [key for key in keys if mapping.get(key) is None]
    if missing:
        raise ConfigError("%s is missing required fields: %s" % (label, ", ".join(missing)))
    return mapping


def _load_simple_yaml(text):
    """Indent-based YAML subset: maps, scalars, and inline lists. No anchors."""
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if "#" in raw:
            raw = raw[: raw.index("#")]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            raise ConfigError("block lists are not accepted; use an inline list")
        key, sep, rest = line.partition(":")
        if not sep:
            raise ConfigError("expected key: value, got %r" % line)
        rest = rest.strip()
        if rest == "":
            if not isinstance(parent, dict):
                raise ConfigError("cannot nest under %r" % parent)
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if not isinstance(parent, dict):
                raise ConfigError("cannot assign under %r" % parent)
            parent[key] = _parse_scalar(rest)
    return root


def _parse_scalar(text):
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if _is_int(text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def _is_int(text):
    if text.startswith("-"):
        text = text[1:]
    return text.isdigit()
