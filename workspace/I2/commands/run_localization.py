"""Validate the candidate config and refuse a real-model launch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mcsd_candidate.yaml"
    if arguments:
        config_path = Path(arguments[0])
    config = load_config(config_path)
    if config.get("run_real_model"):
        sys.stderr.write(
            "run_real_model is set, but this entry does not download weights or start a GPU run\n"
        )
        return 2
    report = {
        "real_model_executed": False,
        "student_e1_started": False,
        "flash_attention_boundary": "not_run",
        "status": "real_model_disabled",
        "config_version": config.get("version"),
    }
    sys.stdout.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
