#!/bin/bash
# Analyze SAGE test results

set -e

RUN=telecom_test_small_fix_verification
OUTPUT=/home/zhuyao/verl-agent/logs/sage_tau2/$RUN

echo "========================================"
echo "SAGE Test Results Analysis"
echo "========================================"
echo ""

if [ ! -d "$OUTPUT" ]; then
    echo "Error: Output directory not found: $OUTPUT"
    exit 1
fi

cd /home/zhuyao/verl-agent

echo "【1. Dispatch Analysis】"
echo ""
python3 << 'PYEOF'
import json
import sys

OUTPUT = "/home/zhuyao/verl-agent/logs/sage_tau2/telecom_test_small_fix_verification"

try:
    with open(f"{OUTPUT}/dispatch_journal.jsonl") as f:
        dispatches = [json.loads(line) for line in f]

    print(f"Total dispatches: {len(dispatches)}")

    # Layer breakdown
    layers = {}
    for d in dispatches:
        layer = d.get('layer', 'unknown')
        layers[layer] = layers.get(layer, 0) + 1

    print("\nDispatch layers:")
    for layer, count in sorted(layers.items(), key=lambda x: -x[1]):
        pct = count / len(dispatches) * 100
        print(f"  {layer}: {count} ({pct:.1f}%)")

    # Primary agents
    primaries = {}
    for d in dispatches:
        primary = d.get('primary', 'unknown')
        primaries[primary] = primaries.get(primary, 0) + 1

    print("\nPrimary agents:")
    for agent, count in sorted(primaries.items(), key=lambda x: -x[1]):
        pct = count / len(dispatches) * 100
        print(f"  {agent}: {count} ({pct:.1f}%)")

    # Check for specialists
    non_executor = [d for d in dispatches if d.get('primary') != 'Executor']
    if non_executor:
        print(f"\n✅ SUCCESS: {len(non_executor)} tasks used specialist as primary!")
    else:
        print("\n❌ FAILURE: All tasks used Executor (no specialists used)")

    # Eligible count
    eligible_count = sum(1 for d in dispatches if d.get('eligible', []))
    print(f"\nTasks with eligible specialists: {eligible_count}/{len(dispatches)}")

except FileNotFoundError:
    print(f"Dispatch journal not found")
except Exception as e:
    print(f"Error: {e}")

PYEOF

echo ""
echo "【2. Skill Bank Analysis】"
echo ""

python3 << 'PYEOF'
import json

OUTPUT = "/home/zhuyao/verl-agent/logs/sage_tau2/telecom_test_small_fix_verification"

try:
    with open(f"{OUTPUT}/skill_bank.json") as f:
        skills = json.load(f)

    print(f"Total skills: {len(skills)}")

    for i, skill in enumerate(skills, 1):
        print(f"\n{i}. {skill['skill_name'][:50]}")
        print(f"   Status: {skill['status']}")
        print(f"   Domain: {skill['domain']}")

        credit = skill.get('metadata', {}).get('skill_credit', {})
        uses = credit.get('uses', 0)
        successes = credit.get('successes', 0)
        score = credit.get('score', 0)

        print(f"   Credit: uses={uses}, successes={successes}, score={score:.3f}")

        if uses > 0:
            print(f"   ✅ Skill was used {uses} times")
            if score > 0.35:
                print(f"   ✅ Score {score:.3f} above nomination threshold (0.35)")
        else:
            print(f"   ❌ Skill never used")

except FileNotFoundError:
    print("Skill bank not found")
except Exception as e:
    print(f"Error: {e}")

PYEOF

echo ""
echo "【3. Organization Analysis】"
echo ""

python3 << 'PYEOF'
import json

OUTPUT = "/home/zhuyao/verl-agent/logs/sage_tau2/telecom_test_small_fix_verification"

try:
    with open(f"{OUTPUT}/organization.json") as f:
        org = json.load(f)

    agents = org.get('agents', [])
    print(f"Total agents: {len(agents)}")

    specialists = [a for a in agents if a['name'] != 'Executor']

    for agent in agents:
        print(f"\n  {agent['name']}:")
        print(f"    Status: {agent.get('acting_status')}")
        print(f"    Capabilities: {agent.get('capability_keys', [])}")
        print(f"    Skills: {len(agent.get('assigned_skills', []))}")

    if specialists:
        print(f"\n✅ {len(specialists)} specialist(s) created")
    else:
        print("\n❌ No specialists created")

except FileNotFoundError:
    print("Organization file not found")
except Exception as e:
    print(f"Error: {e}")

PYEOF

echo ""
echo "【4. Summary】"
echo ""

python3 << 'PYEOF'
import json

OUTPUT = "/home/zhuyao/verl-agent/logs/sage_tau2/telecom_test_small_fix_verification"

try:
    with open(f"{OUTPUT}/online_summary.json") as f:
        summary = json.load(f)

    print(f"Total tasks: {summary.get('total_tasks', 'N/A')}")
    print(f"Mean segment reward: {summary.get('mean_segment_reward', 0):.3f}")
    print(f"Skills in bank: {summary.get('n_skills', 0)}")
    print(f"Specialists: {summary.get('specialists', [])}")

    segments = summary.get('segments', [])
    print(f"\nSegments completed: {len(segments)}")
    for seg in segments:
        print(f"  Segment {seg['segment']}: reward={seg.get('mean_domain_reward', 0):.3f}")

except FileNotFoundError:
    print("Summary not found")
except Exception as e:
    print(f"Error: {e}")

PYEOF

echo ""
echo "========================================"
echo "Analysis complete"
echo "========================================"
