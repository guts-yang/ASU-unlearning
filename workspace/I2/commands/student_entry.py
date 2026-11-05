"""Student entry. The default and E0 configs cannot start training."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.gates import student_block_reason  # noqa: E402

TRAINING_LAUNCHES = 0


def launch_student_e1(*_args, **_kwargs):
    """Present so the training call has one site. This candidate never invokes it."""
    global TRAINING_LAUNCHES
    TRAINING_LAUNCHES += 1
    raise RuntimeError("Student E1 is not launched by this candidate entry")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mcsd_candidate.yaml"
    if arguments:
        config_path = Path(arguments[0])
    config = load_config(config_path)
    reason = student_block_reason(config)
    if reason:
        sys.stderr.write(reason + "\n")
        return 2
    sys.stdout.write("Student E1 gate is open; training was not started\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
