#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${PROFILE:-pilot}"
if [[ "${INSTALL:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$HERE/requirements.txt"
fi
if [[ "${PREFLIGHT:-1}" == "1" ]]; then
  "$PYTHON_BIN" "$HERE/validate_all.py" --operators-only
  "$PYTHON_BIN" "$HERE/preflight.py" --profile "$PROFILE" "$@"
fi
exec "$PYTHON_BIN" "$HERE/run_all.py" --profile "$PROFILE" "$@"
