# Frozen OOD134 ablations (Approach A)

**Protocol (all arms identical):** `valid_unseen` 134, `seed=1`, `game_selection=first`,
`gemini-2.5-flash`, `temperature=0.0`, **one segment / no distill** (`segment_size=134`,
`max_skills_per_round=0`). Only the loaded bank / org / inject mode changes.

Online Skill arms (S0–S3) are **training** only. This folder is **frozen eval** for the paper table.

| Arm | Meaning |
|-----|---------|
| **F0** | EO: no skill, single Executor |
| **F1** | Executor + **S1 bank** + on-demand retrieval |
| **F2** | Executor + **same S1 bank** + fixed-prefix inject |
| **F3** | Executor + **S3 bank** (credit-off training) + retrieval |
| **F4** | Executor + **S2 bank** + fixed-prefix |
| **F5** | **Same S1 skills** + multi-agent org + dispatch |

Existing reference (same protocol as F0): `logs/sage_mas/executor_only_test134/` → 104/134 = 77.6%.  
You may skip re-running F0 and reuse that number if the config matches.

---

## 0) Build multi-agent org from S1 bank (needed for F5)

```bash
cd ~/verl-agent
mkdir -p logs/sage_mas/ablate_freeze

python examples/sage_mas/build_pro_coldstart_seed_org.py \
  --skill-bank logs/sage_mas/ablate_skill_S1_creditON_retr/skill_bank.json \
  --org-output logs/sage_mas/ablate_freeze/S1_seed_organization.json \
  --bank-output logs/sage_mas/ablate_freeze/S1_skill_bank_verified_for_agents.json
```

---

## 1) Run frozen evals (serial recommended)

```bash
cd ~/verl-agent

run_freeze () {
  local cfg="$1" out="$2"
  mkdir -p "$out"
  env -u OPENAI_API_KEY PYTHONUNBUFFERED=1 nohup python -m sage_mas.runners.online_alfworld \
    --llm-config examples/prompt_agent/llm_config.yaml \
    --sage-config "$cfg" \
    --output "$out" \
    > "$out/run.log" 2>&1 &
  echo "$out pid=$!"
}

# F0 optional if you already trust executor_only_test134
run_freeze examples/sage_mas/ablation/freeze/F0_executor_only.yaml \
  logs/sage_mas/ablate_freeze_F0_executor_only

run_freeze examples/sage_mas/ablation/freeze/F1_S1bank_retr.yaml \
  logs/sage_mas/ablate_freeze_F1_S1bank_retr

run_freeze examples/sage_mas/ablation/freeze/F2_S1bank_fixed.yaml \
  logs/sage_mas/ablate_freeze_F2_S1bank_fixed

run_freeze examples/sage_mas/ablation/freeze/F3_S3bank_retr.yaml \
  logs/sage_mas/ablate_freeze_F3_S3bank_retr

run_freeze examples/sage_mas/ablation/freeze/F4_S2bank_fixed.yaml \
  logs/sage_mas/ablate_freeze_F4_S2bank_fixed

# after step 0
run_freeze examples/sage_mas/ablation/freeze/F5_S1bank_multiagent.yaml \
  logs/sage_mas/ablate_freeze_F5_S1bank_multiagent
```

Monitor:

```bash
tail -f logs/sage_mas/ablate_freeze_F1_S1bank_retr/run.log
```

---

## 2) Summarize (paper table)

```bash
cd ~/verl-agent
python3 - <<'PY'
import json
from pathlib import Path
order=[('Pick','pick_and_place'),('Look','look_at_obj_in_light'),('Clean','pick_clean_then_place_in_recep'),
       ('Heat','pick_heat_then_place_in_recep'),('Cool','pick_cool_then_place_in_recep'),('Pick2','pick_two_obj_and_place')]
arms=[
  ('F0', 'logs/sage_mas/ablate_freeze_F0_executor_only'),
  ('F0_ref', 'logs/sage_mas/executor_only_test134'),  # existing EO
  ('F1', 'logs/sage_mas/ablate_freeze_F1_S1bank_retr'),
  ('F2', 'logs/sage_mas/ablate_freeze_F2_S1bank_fixed'),
  ('F3', 'logs/sage_mas/ablate_freeze_F3_S3bank_retr'),
  ('F4', 'logs/sage_mas/ablate_freeze_F4_S2bank_fixed'),
  ('F5', 'logs/sage_mas/ablate_freeze_F5_S1bank_multiagent'),
]
print(f"{'arm':<8} {'Avg':>7}  Pick Look Clean Heat Cool Pick2")
for name, root in arms:
  p=Path(root)/'online_summary.json'
  if not p.exists():
    print(f"{name:<8} pending")
    continue
  s=json.load(p.open()); bf=s['by_family']
  cells=' '.join(f"{bf[k]['success_rate']*100:5.1f}" for _,k in order)
  print(f"{name:<8} {s['success_rate']*100:6.1f}%  {cells}  ({s['wins']}/{s['num_games']})")
PY
```

---

## Notes

- Do **not** compare these cells to online S0–S3 Avg; those are a different protocol.
- F1 vs F2 isolates **retrieval vs fixed prefix** on the **same bank**.
- F1 vs F3 isolates **credit-on vs credit-off training** of the bank (same inject mode).
- F1 vs F5 isolates **single Executor vs multi-agent** on the **same skills**.
- Keep `enable_action_guards=false`.
