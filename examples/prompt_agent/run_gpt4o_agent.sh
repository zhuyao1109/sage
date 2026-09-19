#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="alfoworld"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export ALFWORLD_DATA="${ALFWORLD_DATA:-${REPO_ROOT}/data/alfworld}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import omegaconf
import yaml
import openai
import ray
import gymnasium
PY
then
  echo "Installing prompt-agent dependencies into ${PYTHON}..."
  "${PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

if [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld prompt agent (SkillMAS: 70 unseen guidance + 134 unseen test)..."
  echo "ALFWORLD_DATA=${ALFWORLD_DATA}"
  echo "PYTHON=${PYTHON}"
  cd "${REPO_ROOT}"
  "${PYTHON}" -m examples.prompt_agent.gpt4o_alfworld
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  exit 1
fi
