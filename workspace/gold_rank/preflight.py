"""Fail closed before any model or dataset download."""

import os
from pathlib import Path


def enable_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def local_snapshot_issue(path):
    if not path or not Path(path).is_dir():
        return f"local model directory missing: {path}"
    if not (Path(path) / "config.json").is_file():
        return f"model directory has no config.json: {path}"
    return None


def preflight_issues(model_path, data_path, split="forget", retain="neighbor", reason_path=None, lock_path=None, need_reason=False, need_lock=False):
    enable_offline()
    issues = []
    model_issue = local_snapshot_issue(model_path)
    if model_issue:
        issues.append(model_issue)
    data = Path(data_path) if data_path else None
    forget = None if data is None else data / f"{split}.json"
    retain_file = None if data is None else data / f"{retain}.json"
    if forget is None or not forget.is_file():
        issues.append(f"forget split missing: {forget}")
    if retain_file is None or not retain_file.is_file():
        issues.append(f"retain split missing: {retain_file}")
    if need_reason and (not reason_path or not Path(reason_path).is_file()):
        issues.append(f"GSM8K reason JSONL missing: {reason_path}")
    if need_lock and (not lock_path or not Path(lock_path).is_file()):
        issues.append(f"gold-rank lock missing: {lock_path}")
    return issues
