# SAGE-τ² for τ²-bench (isolated package)

This folder is **separate from** `sage_mas/` and ALFWorld. It does not modify
existing SAGE configs or `logs/sage_mas/*` results.

## Alignment with `sage_mas` online600_noseed

| Knob | ALFWorld online600 | SAGE-τ² default |
|------|--------------------|-----------------|
| Org path | OrganizationEditor ADD | `editor_commit` (Spec OFF) |
| Spec / actor_promotion_probe | **false** | `enable_spec_vs_exec: false` |
| Max inject | 2 | 2 |
| Provisional org edits | false | `allow_provisional_org_edits: false` |
| Probation onboarding | min_games=3, min_wins=2 | same |
| Probation primary | quota=8 | quota=8 |
| Credit verify / prune | 0.60 / 0.20 | same |
| Cluster novelty | ≈0.05 | 0.05 |

```text
τ² segment collect
  → ExecutorDispatcher sticky primary (probation quota / accepted)
  → distill / credit / skill bank
  → capability cluster (novelty≈0.05)
  → nominate uncovered **verified** same-domain capability
  → editor_commit ADD → probation
  → onboarding promotes after ≥3 primary games & ≥2 wins
```

Optional: set `enable_spec_vs_exec: true` to restore Spec-vs-Exec (ALFWorld
`actor_promotion_probe`).

## Train sampling (official 178)

- Default: `task_split_name: train`, `allow_task_resampling: false`
- Each train task id appears **at most once** per evolve run (pools shuffled once per domain).
- Multi-segment runs consume the next unseen tasks; the last segment may be partial.
- Full train pass = **178** tasks (airline 30 + retail 74 + telecom 74).
- Recommended fresh evolve: `segment_size=34`, `num_segments=6` → 5 full + 1 partial segment.
- Set `allow_task_resampling: true` only for legacy/debug runs.

### Fixed val retest (optional)

- `val_size: 5` + `val_split_name: train` samples a fixed set once (`val_seed` or `seed+7919`).
- Default `val_exclude_from_train: true` holds them out of the train schedule (airline → 25 train / 5 val).
- After each segment evolve, the same 5 are re-run (no distill); metrics under `segment_XXX/val/`.
- Example: `sage_tau2/configs/airline_train25_seg5_val5.yaml`.
- Optional `retry_write_fails: true` + `retry_k: 2`: after train collect, re-run
  write-gold fails up to K times and keep the best sim per task before distill
  (`airline_train25_seg5_val5_retry.yaml`).
  Set `retry_force_executor: true` so retries use Executor primary.

## Run (fresh multidomain 178, 34×6 segments)

Config: `configs/multidomain_fresh_178.yaml` — `segment_size=34`, `num_segments=6`,
no resampling, full official train (178 = 5×34 + partial 8).

```bash
cd ~/verl-agent/tau2-bench

export OPENAI_API_KEY="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['api_key'])")"
export OPENAI_API_BASE="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['base_url'])")"
export OPENAI_BASE_URL="$OPENAI_API_BASE"

RUN=multidomain_fresh_178
rm -rf ../logs/sage_tau2/$RUN
rm -rf data/simulations/sage_tau2_${RUN}_* data/simulations/sage_tau2_probe_*
mkdir -p ../logs/sage_tau2/$RUN

PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \
  --config ../sage_tau2/configs/multidomain_fresh_178.yaml \
  --output ../logs/sage_tau2/$RUN \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  2>&1 | tee ../logs/sage_tau2/$RUN/run.log
```

Resume after interrupt (same output dir; reads `online_state.json`):

```bash
cd ~/verl-agent/tau2-bench
RUN=multidomain_fresh_178

PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \
  --config ../sage_tau2/configs/multidomain_fresh_178.yaml \
  --checkpoint ../logs/sage_tau2/$RUN \
  --output ../logs/sage_tau2/$RUN \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  2>&1 | tee -a ../logs/sage_tau2/$RUN/run.log
```

## Run (aligned quality gates — preferred)

Config: `configs/multidomain_fresh_178_aligned.yaml`

- Distill: `min_support=2`, protocol canonicalize + write-spine trim
- Credit: **birth-support seeding** → VERIFIED without waiting on inject (fixes nominate deadlock)
- Org: Spec-vs-Exec ON, no provisional org edits, `probation_primary_quota=0`
- Inject: provisional allowed for ongoing credit; nominate still verified-only
- Frozen eval: `--accepted-only --no-provisional-inject`

```bash
cd ~/verl-agent/tau2-bench
RUN=multidomain_fresh_178_aligned
rm -rf ../logs/sage_tau2/$RUN data/simulations/sage_tau2_${RUN}_* data/simulations/sage_tau2_probe_*
mkdir -p ../logs/sage_tau2/$RUN
PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \
  --config ../sage_tau2/configs/multidomain_fresh_178_aligned.yaml \
  --output ../logs/sage_tau2/$RUN \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  2>&1 | tee ../logs/sage_tau2/$RUN/run.log
```

Smoke (no LLM):

```bash
cd ~/verl-agent
python -m pytest tests/sage_tau2/test_quality_gates_smoke.py -q
```

## Run (relaxed gates for τ² small train)

Config: `configs/multidomain_fresh_178_relaxed.yaml` — same 34×6 / 178 train sampling;
relaxed: `allow_provisional_org_edits`, lower verify/coverage, shorter probation.

Stop any in-flight strict run first (Ctrl+C), then:

```bash
cd ~/verl-agent/tau2-bench

export OPENAI_API_KEY="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['api_key'])")"
export OPENAI_API_BASE="$(python3 -c "import yaml; print(yaml.safe_load(open('../examples/prompt_agent/llm_config.yaml'))['openai']['base_url'])")"
export OPENAI_BASE_URL="$OPENAI_API_BASE"

RUN=multidomain_fresh_178_relaxed
rm -rf ../logs/sage_tau2/$RUN
rm -rf data/simulations/sage_tau2_${RUN}_* data/simulations/sage_tau2_probe_*
mkdir -p ../logs/sage_tau2/$RUN

PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \
  --config ../sage_tau2/configs/multidomain_fresh_178_relaxed.yaml \
  --output ../logs/sage_tau2/$RUN \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  2>&1 | tee ../logs/sage_tau2/$RUN/run.log
```

Resume:

```bash
cd ~/verl-agent/tau2-bench
RUN=multidomain_fresh_178_relaxed

PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \
  --config ../sage_tau2/configs/multidomain_fresh_178_relaxed.yaml \
  --checkpoint ../logs/sage_tau2/$RUN \
  --output ../logs/sage_tau2/$RUN \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  2>&1 | tee -a ../logs/sage_tau2/$RUN/run.log
```

Monitor specialists:

```bash
tail -f ../logs/sage_tau2/multidomain_fresh_178_relaxed/run.log \
  | grep -E 'add=\[|accepted_agents|segment [0-9]+ done|specialists='
```

## Frozen OOD eval

Official **test** split (100 tasks: 20/40/40):

```bash
cd ~/verl-agent/tau2-bench
CKPT=../logs/sage_tau2/multidomain_fresh_178

PYTHONPATH=.. uv run python -m sage_tau2.runners.eval_frozen \
  --checkpoint $CKPT \
  --output ${CKPT}_test100 \
  --task-split-name test \
  --domains airline:20,retail:40,telecom:40 \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  --max-concurrency 6 \
  2>&1 | tee ${CKPT}_test100/run.log
```

Executor-only baseline (same test split):

```bash
PYTHONPATH=.. uv run python -m sage_tau2.runners.eval_frozen \
  --executor-only \
  --output ../logs/sage_tau2/executor_test100 \
  --task-split-name test \
  --domains airline:20,retail:40,telecom:40 \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  --max-concurrency 6 \
  2>&1 | tee ../logs/sage_tau2/executor_test100/run.log
```

**Base** split OOD (~278 = 50/114/114):

```bash
PYTHONPATH=.. uv run python -m sage_tau2.runners.eval_frozen \
  --checkpoint $CKPT \
  --output ${CKPT}_base278 \
  --task-split-name base \
  --llm-config ../examples/prompt_agent/llm_config.yaml \
  --max-concurrency 6 \
  2>&1 | tee ${CKPT}_base278/run.log
```

## Layout

| Path | Role |
|------|------|
| `executor_dispatch.py` | sticky primary (sage_mas-aligned quota) |
| `nominate_admit.py` | nominate + editor_commit / optional Spec |
| `onboarding.py` | probation → accepted |
| `admission_probe.py` | paired forced-primary (optional) |
| `agent.py` | HalfDuplex + dispatch journal |
| `pipeline.py` | segment update |
| `runners/online_multidomain.py` | multi-domain evolve |
| `runners/eval_frozen.py` | frozen OOD |
