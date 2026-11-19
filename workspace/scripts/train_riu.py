"""Launch OpenUnlearning training with RIU plug-ins registered."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OU = REPO / "open-unlearning"
WS = REPO / "workspace"
sys.path[:0] = [str(OU / "src"), str(WS / "src")]

from open_unlearning_plugins.register import register_workspace_trainers  # noqa: E402

register_workspace_trainers()

runpy.run_path(str(OU / "src" / "train.py"), run_name="__main__")
