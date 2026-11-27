#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-pilot}"
shift || true
exec "${PYTHON_BIN:-python3}" "$HERE/run.py" --profile "$PROFILE" "$@"
