"""Fail closed before any weight download. Fixtures are temporary and local."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from gold_rank.launch import GROUPS, command


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _model(root, with_config=True):
    model = root / "model"
    model.mkdir(parents=True)
    if with_config:
        _write(model / "config.json", "{}\n")
    return model


def _data(root, forget=True, retain=True):
    data = root / "real_world"
    data.mkdir(parents=True)
    if forget:
        _write(data / "forget.json", "[]\n")
    if retain:
        _write(data / "neighbor.json", "[]\n")
    return data


def _lock(root, status="frozen"):
    return _write(root / "lock.json", json.dumps({"status": status, "test_metrics_read": False}))


def _verdict(root, decision="proceed"):
    return _write(root / "verdict.json", json.dumps({"decision": decision}))


def _reason(root):
    return _write(root / "reason.jsonl", '{"question": "q", "answer": "a"}\n')


def _call(model, data, root, group, reason=None, lock=None, verdict=None):
    return command(
        str(model),
        str(data),
        str(root / "save"),
        1001,
        group,
        None if reason is None else str(reason),
        None if lock is None else str(lock),
        None if verdict is None else str(verdict),
    )


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_missing_model_directory_exits(tmp_path, group):
    with pytest.raises(SystemExit, match="local model directory missing"):
        _call(tmp_path / "absent", tmp_path / "data", tmp_path, group, tmp_path / "reason.jsonl", tmp_path / "lock.json")


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_missing_config_exits(tmp_path, group):
    model = _model(tmp_path, with_config=False)
    data = _data(tmp_path)
    with pytest.raises(SystemExit, match="no config.json"):
        _call(model, data, tmp_path, group, _reason(tmp_path), _lock(tmp_path), _verdict(tmp_path))


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_missing_real_world_splits_exit(tmp_path, group):
    model = _model(tmp_path)
    data = _data(tmp_path, forget=False, retain=False)
    with pytest.raises(SystemExit, match="forget split missing[\\s\\S]*retain split missing"):
        _call(model, data, tmp_path, group, _reason(tmp_path), _lock(tmp_path), _verdict(tmp_path))


def test_missing_gsm8k_blocks_anchor_groups_only(tmp_path):
    model = _model(tmp_path)
    data = _data(tmp_path)
    lock = _lock(tmp_path)
    verdict = _verdict(tmp_path)
    for group in ("B", "D"):
        with pytest.raises(SystemExit, match="GSM8K"):
            _call(model, data, tmp_path, group, reason=None, lock=lock, verdict=verdict)
    overrides = _call(model, data, tmp_path, "A")
    assert "gold_rank.enabled=false" in overrides
    assert "da_asu.enabled=false" in overrides
    assert f"model_path={model}" in overrides
    assert f"data_path={data}" in overrides


def test_demotion_without_lock_exits(tmp_path):
    model = _model(tmp_path)
    data = _data(tmp_path)
    with pytest.raises(SystemExit, match="requires a frozen gold-rank lock"):
        _call(model, data, tmp_path, "C")


def test_unfrozen_lock_exits(tmp_path):
    model = _model(tmp_path)
    data = _data(tmp_path)
    lock = _lock(tmp_path, status="preregistered_grid")
    with pytest.raises(SystemExit, match="not a frozen pre-test lock"):
        _call(model, data, tmp_path, "C", lock=lock)


def test_non_proceed_verdict_blocks_anchor(tmp_path):
    model = _model(tmp_path)
    data = _data(tmp_path)
    reason = _reason(tmp_path)
    verdict = _verdict(tmp_path, decision="stop_anchor")
    with pytest.raises(SystemExit, match="Anchor training stays off"):
        _call(model, data, tmp_path, "B", reason=reason, verdict=verdict)


def test_scripts_import_gold_rank_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for script in ("launch.py", "diagnose.py"):
        completed = subprocess.run(
            [sys.executable, str(WORKSPACE / "gold_rank" / script), "--help"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_default_config_exits_before_download(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "real_world_unlearn.py",
            f"hydra.run.dir={tmp_path / 'hydra'}",
            "hydra.output_subdir=null",
            "hydra.job.chdir=false",
        ],
        cwd=REPO / "Right-to-be-forgotten",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "Refusing to download 'NousResearch/Meta-Llama-3-8B-Instruct'" in output
    assert "Start Training" not in output
    assert "from_pretrained" not in output
