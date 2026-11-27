#!/usr/bin/env bash
#SBATCH --job-name=cf2-methods-1127
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=cf2-methods-%j.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROFILE="${PROFILE:-pilot}"
export GPUS="${GPUS:-0}"
export INSTALL="${INSTALL:-0}"
export PREFLIGHT="${PREFLIGHT:-1}"
bash "$HERE/start.sh"
