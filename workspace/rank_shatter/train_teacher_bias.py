#!/usr/bin/env python3
"""Train a vocab logit bias on a frozen teacher.

DO NOT RUN this round. No optimizer steps, no 7B load, no checkpoint writes.
Next round: read workspace/configs/asu_frozen_2026-09-27.yaml and asu_gd_rs.yaml.
"""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rank-shattering teacher bias (deferred; do not run this round)."
    )
    parser.add_argument(
        "--config",
        default="workspace/configs/asu_gd_rs.yaml",
        help="Recipe aligned with frozen ASU+GD hparams.",
    )
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args(argv)
    sys.stderr.write(
        "DO NOT RUN this round. Refusing to train.\n"
        f"Would have used config={args.config} margin={args.margin}\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
