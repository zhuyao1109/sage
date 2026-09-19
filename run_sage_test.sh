#!/bin/bash
# SAGE τ²-bench Fix Verification Test
# Small-scale experiment: 50 tasks (5 segments × 10 tasks)

set -e

echo "========================================"
echo "SAGE τ²-bench Fix Verification Test"
echo "========================================"

cd /home/zhuyao/verl-agent/tau2-bench

# Setup environment
export OPENAI_API_KEY="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['api_key'])")"
export OPENAI_API_BASE="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['base_url'])")"
export OPENAI_BASE_URL="$OPENAI_API_BASE"

RUN=telecom_test_small_noseed
OUTPUT=../logs/sage_tau2/$RUN

echo ""
echo "Run ID: $RUN"
echo "Output: $OUTPUT"
echo ""

# Clean previous run
rm -rf $OUTPUT
rm -rf data/simulations/sage_tau2_${RUN}_*
rm -rf data/simulations/sage_tau2_probe_*
mkdir -p $OUTPUT

echo "Starting test run..."
echo "Config: telecom_test_small.yaml"
echo "Tasks: 50 (5 segments × 10 tasks)"
echo "Domain: telecom"
echo ""

# Run with output logging
PYTHONPATH=.. uv run python -m sage_tau2.runners.online_tau2 \
  --config ../sage_tau2/configs/telecom_test_small.yaml \
  --output $OUTPUT \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  2>&1 | tee $OUTPUT/run.log

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo ""
echo "Check results:"
echo "  1. Log file: $OUTPUT/run.log"
echo "  2. Summary: $OUTPUT/online_summary.json"
echo "  3. Skills: $OUTPUT/skill_bank.json"
echo "  4. Organization: $OUTPUT/organization.json"
echo "  5. Dispatch: $OUTPUT/dispatch_journal.jsonl"
echo ""
echo "Key metrics to check:"
echo "  - Are specialists marked as eligible?"
echo "  - Are skills used as primary agent?"
echo "  - Do skill credit scores increase?"
echo "  - Are any specialists created/nominated?"
echo ""
