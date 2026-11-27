#!/usr/bin/env bash
#SBATCH --job-name=cf3-unlearning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=cf3-%j.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROFILE="${PROFILE:-smoke}"
export GPUS="${GPUS:-0}"
exec bash "$HERE/start.sh" "$@"
