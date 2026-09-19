# SAGE Skill / MAS Ablation Checklist

Acting model default: **`gemini-2.5-flash`** (swap whole table to `gpt-4o-mini` if the relay is too slow — do not mix models inside one table).

Code fixes already applied for these arms:
- ALFWorld respects `online.gate_injected_skills` (no longer hard-forced `true`)
- τ² reads `require_scope_match` / `enable_executor_dispatch` from YAML

## Naming

| Arm | Meaning |
|-----|---------|
| **S0** | No skill |
| **S1** | credit ON + on-demand retrieval (**main**) |
| **S2** | credit ON + fixed prefix |
| **S3** | credit OFF + retrieval |
| **S4** | credit OFF + fixed prefix |
| **M0** | Fixed bank + freeze org |
| **M1** | Evolve org, **no** shift/novelty gate |
| **M2** | Evolve org + shift/novelty gate |
| **M3** | M2 + org MU (ALF shadow / τ² Spec-vs-Exec) |
| **M2a** | M2 without meta dispatch |

**Order:** run Skill table → copy S1 `skill_bank.json` → run MAS table.

---

## 1) ALFWorld — Skill table (MAS frozen)

```bash
cd ~/verl-agent
LLM=examples/prompt_agent/llm_config.yaml

run_alf () {
  local cfg="$1" out="$2"
  mkdir -p "$out"
  env -u OPENAI_API_KEY PYTHONUNBUFFERED=1 python -m sage_mas.runners.online_alfworld \
    --llm-config "$LLM" \
    --sage-config "$cfg" \
    --output "$out" \
    2>&1 | tee "$out/run.log"
}

run_alf examples/sage_mas/ablation/skill_S0_noskill.yaml            logs/sage_mas/ablate_skill_S0_noskill
run_alf examples/sage_mas/ablation/skill_S1_creditON_retr.yaml       logs/sage_mas/ablate_skill_S1_creditON_retr
run_alf examples/sage_mas/ablation/skill_S2_creditON_fixed.yaml      logs/sage_mas/ablate_skill_S2_creditON_fixed
run_alf examples/sage_mas/ablation/skill_S3_creditOFF_retr.yaml      logs/sage_mas/ablate_skill_S3_creditOFF_retr
run_alf examples/sage_mas/ablation/skill_S4_creditOFF_fixed.yaml     logs/sage_mas/ablate_skill_S4_creditOFF_fixed
```

Fill Pick/Look/Clean/Heat/Cool/Pick2/Avg from each `online_summary.json`.

Optional faster actor (whole Skill table):

```bash
# add to each command:
  --executor-model gpt-4o-mini --specialist-model gpt-4o-mini --acceptor-model gpt-4o-mini
```

---

## 2) ALFWorld — MAS table (skill bank fixed from S1)

```bash
# MAS yamls already point at:
#   sage.seed_skill_bank_path: logs/sage_mas/ablate_skill_S1_creditON_retr/skill_bank.json
test -f logs/sage_mas/ablate_skill_S1_creditON_retr/skill_bank.json

run_alf examples/sage_mas/ablation/mas_M0_fixed_org.yaml             logs/sage_mas/ablate_mas_M0_fixed_org
run_alf examples/sage_mas/ablation/mas_M1_evolve_noshift.yaml        logs/sage_mas/ablate_mas_M1_evolve_noshift
run_alf examples/sage_mas/ablation/mas_M2_evolve_shift.yaml          logs/sage_mas/ablate_mas_M2_evolve_shift
run_alf examples/sage_mas/ablation/mas_M3_evolve_shift_mu.yaml       logs/sage_mas/ablate_mas_M3_evolve_shift_mu
run_alf examples/sage_mas/ablation/mas_M2a_evolve_shift_nometa.yaml  logs/sage_mas/ablate_mas_M2a_evolve_shift_nometa
```

`max_skills_per_round: 0` + seeded bank ≈ freeze skill birth (credit may still update scores).

---

## 3) τ² — Skill table (telecom primary; org frozen)

```bash
cd ~/verl-agent/tau2-bench
export OPENAI_API_KEY="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['api_key'])")"
export OPENAI_API_BASE="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['base_url'])")"
export OPENAI_BASE_URL="$OPENAI_API_BASE"
LLM=../examples/prompt_agent/llm_config.yaml

run_tau () {
  local cfg="$1" out="$2"
  rm -rf "../logs/sage_tau2/$out"
  mkdir -p "../logs/sage_tau2/$out"
  PYTHONPATH=.. uv run python -m sage_tau2.runners.online_tau2 \
    --config "../sage_tau2/configs/ablation/$cfg" \
    --output "../logs/sage_tau2/$out" \
    --llm-config "$LLM" \
    2>&1 | tee "../logs/sage_tau2/$out/run.log"
}

run_tau telecom_skill_S0_noskill.yaml       ablate_telecom_skill_S0_noskill
run_tau telecom_skill_S1_creditON_retr.yaml ablate_telecom_skill_S1_creditON_retr
run_tau telecom_skill_S2_creditON_fixed.yaml ablate_telecom_skill_S2_creditON_fixed
run_tau telecom_skill_S3_creditOFF_retr.yaml ablate_telecom_skill_S3_creditOFF_retr
run_tau telecom_skill_S4_creditOFF_fixed.yaml ablate_telecom_skill_S4_creditOFF_fixed
```

**Retail / Airline columns (same Skill-S1 knobs, domain swap):**

```bash
# retail uses online_multidomain when domain_weights is set — prefer:
PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \
  --config ../sage_tau2/configs/ablation/retail_skill_S1_creditON_retr.yaml \
  --output ../logs/sage_tau2/ablate_retail_skill_S1_creditON_retr \
  --llm-config "$LLM"

run_tau airline_skill_S1_creditON_retr.yaml ablate_airline_skill_S1_creditON_retr
```

For a full retail/airline Skill 2×2, copy the telecom `S0–S4` yamls and change `domain` / `domain_weights` / `segment_size` the same way as the S1 companions.

---

## 4) τ² — MAS table (telecom; bank fixed from S1)

```bash
# Seed each MAS run with the Skill-S1 bank (no new skill birth: max_new_skills=0).
seed_bank=../logs/sage_tau2/ablate_telecom_skill_S1_creditON_retr/skill_bank.json
test -f "$seed_bank"

run_tau_mas () {
  local cfg="$1" out="$2"
  rm -rf "../logs/sage_tau2/$out"
  mkdir -p "../logs/sage_tau2/$out"
  cp "$seed_bank" "../logs/sage_tau2/$out/skill_bank.json"
  PYTHONPATH=.. uv run python -m sage_tau2.runners.online_tau2 \
    --config "../sage_tau2/configs/ablation/$cfg" \
    --output "../logs/sage_tau2/$out" \
    --llm-config "$LLM" \
    2>&1 | tee "../logs/sage_tau2/$out/run.log"
}

run_tau_mas telecom_mas_M0_fixed_org.yaml            ablate_telecom_mas_M0_fixed_org
run_tau_mas telecom_mas_M1_evolve_noshift.yaml       ablate_telecom_mas_M1_evolve_noshift
run_tau_mas telecom_mas_M2_evolve_shift.yaml         ablate_telecom_mas_M2_evolve_shift
run_tau_mas telecom_mas_M3_evolve_shift_mu.yaml      ablate_telecom_mas_M3_evolve_shift_mu
run_tau_mas telecom_mas_M2a_evolve_shift_nometa.yaml ablate_telecom_mas_M2a_evolve_shift_nometa
```

---

## Table → config map

### Skill evolution (fixed MAS)

| Row | ALFWorld yaml | τ² telecom yaml |
|-----|---------------|-----------------|
| credit✓ + 按需 | `skill_S1_creditON_retr.yaml` | `telecom_skill_S1_creditON_retr.yaml` |
| credit✓ + 固定开头 | `skill_S2_creditON_fixed.yaml` | `telecom_skill_S2_creditON_fixed.yaml` |
| credit✗ + 按需 | `skill_S3_creditOFF_retr.yaml` | `telecom_skill_S3_creditOFF_retr.yaml` |
| credit✗ + 固定开头 | `skill_S4_creditOFF_fixed.yaml` | `telecom_skill_S4_creditOFF_fixed.yaml` |
| (opt) 无 skill | `skill_S0_noskill.yaml` | `telecom_skill_S0_noskill.yaml` |

### MAS evolution (fixed skill bank)

| Row | ALFWorld yaml | τ² telecom yaml |
|-----|---------------|-----------------|
| 不演化 | `mas_M0_fixed_org.yaml` | `telecom_mas_M0_fixed_org.yaml` |
| 演化无分布门 | `mas_M1_evolve_noshift.yaml` | `telecom_mas_M1_evolve_noshift.yaml` |
| 演化+分布门 | `mas_M2_evolve_shift.yaml` | `telecom_mas_M2_evolve_shift.yaml` |
| + MAS MU | `mas_M3_evolve_shift_mu.yaml` | `telecom_mas_M3_evolve_shift_mu.yaml` |
| 无 meta | `mas_M2a_evolve_shift_nometa.yaml` | `telecom_mas_M2a_evolve_shift_nometa.yaml` |

---

## Notes

1. ALFWorld OOD134: `valid_unseen`, `seed=1`, `game_selection=first`, `segment_size=10`.
2. τ² telecom default: `segment_size=50`, `num_segments=1` (smoke-scale); raise `num_segments` for paper curves.
3. Do **not** mix Pro-coldstart / noseed-org Multi-agent numbers into these ablation cells.
4. Keep `enable_action_guards=false` and `use_visited_location_memory=false`.
