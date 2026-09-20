#!/usr/bin/env bash
# Use from PowerShell: wsl -d Ubuntu -- bash /mnt/d/sage/examples/sage_mas/run_alfworld_wsl.sh setup
# Then replace "setup" with "run". No Windows Python environment is modified.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${SAGE_ALFWORLD_VENV:-${HOME}/.venvs/sage-alfworld}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
MODE="${1:-check}"
cd "$REPO_ROOT"

if [[ "$MODE" == "setup" ]]; then
    if [[ "$(id -u)" == "0" ]]; then
        APT=(apt-get)
    else
        APT=(sudo apt-get)
    fi
    "${APT[@]}" update
    "${APT[@]}" install -y python3-venv python3-dev build-essential libffi-dev git curl
    python3 -m venv "$VENV_PATH"
    "$VENV_PATH/bin/python" -m pip install --upgrade pip
    "$VENV_PATH/bin/python" -m pip install -r examples/prompt_agent/requirements.txt
    if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ||
          ! -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen" ||
          ! -f "$ALFWORLD_DATA/logic/alfred.pddl" ]]; then
        "$VENV_PATH/bin/alfworld-download"
    fi
elif [[ "$MODE" != "check" && "$MODE" != "run" ]]; then
    echo "Usage: bash $0 setup|check|run" >&2
    exit 2
fi

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Missing ALFWorld virtual environment: $VENV_PATH. Run this script with setup first." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SAGE_ALFWORLD_REPO="$REPO_ROOT"
"$VENV_PATH/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml
import omegaconf
import ray
import gymnasium
import alfworld
import textworld
import openai
from sage_mas.online_evolution import OnlineAlfWorldEvolution
from agent_system.environments.env_package.alfworld.envs import build_alfworld_envs

root = Path(os.environ['SAGE_ALFWORLD_REPO'])
data = Path(os.environ['ALFWORLD_DATA']).expanduser().resolve()
train = list((data / 'json_2.1.1/train').rglob('game.tw-pddl'))
ood = list((data / 'json_2.1.1/valid_unseen').rglob('game.tw-pddl'))
if len(train) < 60 or len(ood) < 12 or not (data / 'logic/alfred.pddl').exists():
    raise SystemExit(f'Incomplete ALFWorld data at {data}: train={len(train)}, OOD={len(ood)}. Run setup or set ALFWORLD_DATA to the existing dataset root.')
source = root / 'examples/prompt_agent/llm_config.alfworld.local.yaml'
config = yaml.safe_load(source.read_text(encoding='utf-8'))
if not config.get('openai', {}).get('api_key'):
    raise SystemExit('Missing API key in the ALFWorld-only configuration.')
config['alfworld']['data_path'] = str(data)
target = root / 'examples/prompt_agent/llm_config.alfworld.wsl.local.yaml'
target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
print(f'Preflight passed: data={data}, train={len(train)}, OOD={len(ood)}')
print('ALFWorld-only WSL config prepared; no API request has been made.')
PY

if [[ "$MODE" != "run" ]]; then
    echo "Setup/check completed. To start evolution, run this script with run."
    exit 0
fi

# Refuse a second invocation of this launcher while its experiment is active.
exec 9>"$VENV_PATH/alfworld-evolution.lock"
if ! flock -n 9; then
    echo "An ALFWorld evolution launched by this script is already running." >&2
    exit 1
fi
RUN_OUTPUT="$REPO_ROOT/logs/sage_mas/evolve_6x10_val12_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_OUTPUT"
echo "Starting ALFWorld: 60 train episodes, 12 fixed OOD validation tasks per segment."
echo "Output: $RUN_OUTPUT"
"$VENV_PATH/bin/python" -u -m sage_mas.runners.online_alfworld \
    --llm-config "$REPO_ROOT/examples/prompt_agent/llm_config.alfworld.wsl.local.yaml" \
    --sage-config "$REPO_ROOT/examples/sage_mas/sage_config.online60_6x10.yaml" \
    --output "$RUN_OUTPUT" 2>&1 | tee "$RUN_OUTPUT/run.log"
