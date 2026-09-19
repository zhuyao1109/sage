#!/usr/bin/env bash
# One-shot WebShop SAGE online smoke launcher.
#
# Handles: JAVA_HOME, hostname, Python selection (must have WebShop deps),
# dependency preflight, API-key precedence, and the online_webshop runner.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LLM_CONFIG="${LLM_CONFIG:-examples/prompt_agent/llm_config.yaml}"
SAGE_CONFIG="${SAGE_CONFIG:-examples/sage_mas/sage_config.webshop_smoke.yaml}"
OUTPUT="${OUTPUT:-logs/sage_mas/webshop_smoke}"

JAVA_CANDIDATES=(
  "${JAVA_HOME:-}"
  "${CONDA_PREFIX:-}/lib/jvm"
  "/home/zhuyao/miniconda3/envs/webshop/lib/jvm"
)
for cand in "${JAVA_CANDIDATES[@]}"; do
  if [[ -n "${cand}" && -x "${cand}/bin/javac" ]]; then
    export JAVA_HOME="${cand}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
    break
  fi
done
if ! command -v javac >/dev/null 2>&1; then
  echo "ERROR: javac not found. Install openjdk=11 or set JAVA_HOME." >&2
  exit 1
fi

HOST="$(hostname -s 2>/dev/null || hostname || true)"
if [[ -n "${HOST}" ]] && ! grep -qE "[[:space:]]${HOST}([[:space:]]|\$)" /etc/hosts 2>/dev/null; then
  echo "WARN: ${HOST} missing from /etc/hosts (pyserini/log4j may warn)." >&2
  echo "      Fix with: echo '127.0.0.1 ${HOST}' | sudo tee -a /etc/hosts" >&2
fi

_webshop_python_ok() {
  local py="$1"
  [[ -x "${py}" ]] || return 1
  "${py}" - <<'PY' >/dev/null 2>&1
import importlib
for mod in ("faiss", "gym", "pyserini", "spacy", "cleantext", "rank_bm25", "thefuzz", "yaml"):
    importlib.import_module(mod)
import spacy
spacy.load("en_core_web_sm")
from pyserini.search.lucene import LuceneSearcher  # noqa: F401
PY
}

# Prefer an interpreter that already has WebShop deps.
# Do NOT blindly prefer repo .venv — it often lacks gym/pyserini/faiss.
PYTHON_CANDIDATES=()
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CANDIDATES+=("${PYTHON}")
fi
PYTHON_CANDIDATES+=(
  "/home/zhuyao/miniconda3/bin/python"
  "/home/zhuyao/miniconda3/envs/webshop/bin/python"
  "$(command -v python3 || true)"
  "$(command -v python || true)"
  "${REPO_ROOT}/.venv/bin/python"
)

PYTHON=""
for cand in "${PYTHON_CANDIDATES[@]}"; do
  [[ -n "${cand}" ]] || continue
  if _webshop_python_ok "${cand}"; then
    PYTHON="${cand}"
    break
  fi
done

if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: no Python with WebShop deps found." >&2
  echo "Tried:" >&2
  for cand in "${PYTHON_CANDIDATES[@]}"; do
    [[ -n "${cand}" ]] && echo "  - ${cand}" >&2
  done
  echo "Fix: use conda base (where faiss/gym/pyserini are installed), e.g." >&2
  echo "  PYTHON=/home/zhuyao/miniconda3/bin/python ./examples/sage_mas/run_webshop_smoke.sh" >&2
  echo "Or install into the chosen env:" >&2
  echo "  pip install faiss-cpu gym==0.24.0 pyserini cleantext rank_bm25 thefuzz" >&2
  echo "  python -m spacy download en_core_web_sm" >&2
  exit 1
fi

echo "[preflight] PYTHON=${PYTHON}"
echo "[preflight] JAVA_HOME=${JAVA_HOME}"
"${PYTHON}" - <<'PY'
print("[preflight] imports OK")
PY

# Prefer llm_config.yaml api_key over a stale shell OPENAI_API_KEY.
eval "$(
  "${PYTHON}" - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${LLM_CONFIG}").read_text()) or {}
key = ((cfg.get("openai") or {}).get("api_key")) or ""
if key:
    import shlex
    print(f"export OPENAI_API_KEY={shlex.quote(str(key))}")
    print("echo '[preflight] using api_key from llm_config.yaml'")
else:
    print("echo '[preflight] llm_config.yaml has no api_key; using ambient OPENAI_API_KEY'")
PY
)"

mkdir -p "${OUTPUT}"
echo "[run] online_webshop -> ${OUTPUT}"
exec env PYTHONUNBUFFERED=1 "${PYTHON}" -m sage_mas.runners.online_webshop \
  --llm-config "${LLM_CONFIG}" \
  --sage-config "${SAGE_CONFIG}" \
  --output "${OUTPUT}"
