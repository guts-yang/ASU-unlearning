"""Write a perturbation q- sidecar. The forget file is only read.

q- replaces one name slot with a fixed pseudonym. It has no y-.
Token alignment uses the recorded Llama-2 tokenizer when that tokenizer is present.
This command does not run the model and does not approve localization.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_REPO = _WORKSPACE.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.errors import ConfigError  # noqa: E402
from I2.mcsd.perturbation import run_perturbation  # noqa: E402
from I2.mcsd.same_split import require_locked_threshold  # noqa: E402
from I2.mcsd.storage import measurement_dir  # noqa: E402
from I2.mcsd.token_record import llama2_tags, load_tokenizer  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qminus_split.yaml"
_MODEL_CONFIG = _REPO / "Right-to-be-forgotten" / "config" / "model_config.yaml"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    config = load_config(_CONFIG)
    require_locked_threshold(config)
    forget_path = Path(arguments[0]) if arguments else _REPO / config["forget_file"]
    if not forget_path.is_file():
        raise ConfigError("forget file is missing: %s" % forget_path)
    output_dir = measurement_dir("perturbation_qminus")
    tokenizer, meta = load_tokenizer()
    tags = llama2_tags(_MODEL_CONFIG)
    summary, _rows = run_perturbation(
        forget_path,
        config,
        output_dir,
        tokenizer=tokenizer,
        tags=tags,
        tokenizer_meta=meta,
    )
    sys.stdout.write(str(output_dir) + "\n")
    return 0 if summary["forget_bytes_unchanged"] and summary["approved_for_localization"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
