# SAGE-MAS

The online pipeline has one skill lifecycle:

```text
ALFWorld trajectories
  -> trajectory-grounded candidate Skill
  -> evidence check
  -> provisional SkillBank entry
  -> online effect-success rate
  -> promotion at success-rate threshold
  -> AgentRoleCompiler proposes a specialist
  -> paired organization shadow
  -> next segment
```

The online main loop does not launch a separate Discovery fork. Skills come
only from collected trajectories that contain either a successful episode or
an observed positive `goal_progress_delta`, together with a grounded action
protocol. A failed trajectory without positive environment progress cannot
create a Skill or trigger an isolated environment experiment.

Credit is the Laplace-smoothed effect-success rate
`(successes + 1) / (uses + 2)`. A use is counted only after the learned anchor
matches, the contract is selected, and the matching environment action is
executed. A success means that the Skill's expected local environment effect
is observed; final episode outcome is diagnostic only. Promotion requires at
least one use and score `0.60`. Skills with at least three uses and score
below `0.20` are pruned. Non-use is neutral. At most two matching grounded
contracts are injected into an episode.

Credit promotion only proves a Skill is usable. Organization expansion still
requires an independent cluster novelty signal against other historical Skills
plus an assignment gap showing the current Agents cannot cover the capability.
Promotion never forces distribution shift to `1.0`.

The scored 134-task stream is never reordered to create Skill opportunities.
Optional organization shadow uses a separate mechanism pool.

New specialists are committed only as dispatch-only probation agents. Even
when organization shadow is skipped, they remain in probation until they
collect at least three in-contract primary dispatches and at least two
complete-task wins, with success rate not worse than Executor on the same
CapabilityContract scope. A single lucky win is not enough for acceptance.

For every new task, Executor compares the task text with each available
Agent's learned CapabilityContract and verified Skills. The roster prompt also
includes full-task wins, dispatches, and a Laplace-smoothed success rate for
the current task scope. Executor remains a candidate fallback even when only
one specialist is available; the selected primary Agent owns the full episode.

The removed held-out Skill verifier, ProbeAnchor, ITT probe runner, and separate
self-evolution loop are not part of the pipeline. Paired evaluation remains
only for old/new organization comparison.

Each online round persists:

- trajectories and segment metrics;
- candidate Skills and the canonical SkillBank;
- an `ExperienceDistributionSnapshot`;
- compiled capability contracts and organization edits;
- old/new organization shadow results when an edit is proposed.

## Run all 134 valid_unseen tasks

```bash
env -u OPENAI_API_KEY python -m sage_mas.runners.online_alfworld \
  --llm-config examples/prompt_agent/llm_config.yaml \
  --sage-config examples/sage_mas/sage_config.online134.yaml \
  --output logs/sage_mas/online134_credit_v1
```

Use a fresh output directory after changing task-pool configuration. An
interrupted run resumes from `online_state.json` when its configuration is
unchanged.

## Online 500 learning curve (train / val / test)

Fixed 500-task train stream with periodic held-out probes every segment:

- train: `collection_dataset=train`, `num_games=500`, `segment_size=20`
- val: fixed 10 `valid_seen` tasks after every segment
- test: fixed 10 `valid_unseen` tasks after every segment

```bash
PYTHONUNBUFFERED=1 python -m sage_mas.runners.online_alfworld \
  --llm-config examples/prompt_agent/llm_config.yaml \
  --sage-config examples/sage_mas/sage_config.online500_curve.yaml \
  --output logs/sage_mas/online500_curve_bmu_seed1

python -m sage_mas.runners.plot_learning_curve \
  --input logs/sage_mas/online500_curve_bmu_seed1/online_state.json \
  --output-dir logs/sage_mas/online500_curve_bmu_seed1 \
  --watch --interval 60
```

`val_eval` / `test_eval` with `every_segment: true` are independent of
organization shadow. The plotter draws three curves from task 0 when periodic
pools or `freeze_organization` are set. Note: this config also keeps B_mu-style
paired MU probes, so wall-clock cost is higher than the raw 500+500 episode
count.

## Causal 134 ablations (A/B/C/D)

Lock the same task stream (`valid_unseen`, `game_selection=first`,
`seed=1`, `segment_size=10`) and vary only skill/org/shadow switches:

| Arm | Config | Skill update/inject | Org commit | Shadow |
|-----|--------|---------------------|------------|--------|
| A | `sage_config.causal134_A_executor.yaml` | no | no | no |
| B | `sage_config.causal134_B_skill.yaml` | yes | frozen (`freeze_organization`) | no |
| C | `sage_config.causal134_C_full.yaml` | yes | yes | 8 paired tasks |
| D | `sage_config.causal134_D_noshadow.yaml` | yes | yes | skipped |

```bash
PYTHONUNBUFFERED=1 python -m sage_mas.runners.online_alfworld \
  --llm-config examples/prompt_agent/llm_config.yaml \
  --sage-config examples/sage_mas/sage_config.causal134_A_executor.yaml \
  --output logs/sage_mas/causal134_A_executor_seed1
```

Compare `online_summary.json` fields `success_rate`, `by_family`,
`ablation`, `adaptation`, and `dispatch`. Keep scaffold settings identical
across arms in `llm_config.yaml` (`max_steps` especially).

### B_gemini (strong distill, fixed Executor)

Do **not** change the Executor acting model in this arm. Keep
`openai.model=gpt-4o-mini` and only swap the distill model. Otherwise
skill gains are confounded with a stronger actor.

Same skill-only / freeze-org protocol as heuristic B, but:
- distill mode `trajectory_enriched` with `gemini-2.5-pro` rewrite layer
- protocol remains trajectory-aligned (`clean`/`cool`/`heat`/`use desklamp`
  markers kept; lamp is never collapsed to `use <entity>`)
- Gemini JSON rewrite retries on parse failure (`max_retries`)
- broad `execution.*` / `recovery.*` seeds are excluded; only
  `transform.*` / `track.*` / `inspect.*` may be injected
- **paired MU gate**: each new skill is dual-evaluated with vs without
  injection; only \(\Delta SR>0\) (or local-effect gain) skills become
  injectable. Credit successes are relative to a no-skill control pass.

```bash
PYTHONUNBUFFERED=1 python -m sage_mas.runners.online_alfworld \
  --llm-config examples/prompt_agent/llm_config.yaml \
  --sage-config examples/sage_mas/sage_config.causal134_B_gemini.yaml \
  --output logs/sage_mas/causal134_B_gemini_seed1
```

Compare `B_gemini` vs `A` / `B_heuristic` using `success_rate`,
`ablation.distillation_model`, and `ablation.executor_model`
(`executor_model` must stay `gpt-4o-mini`).

## Offline distillation

```bash
python -m sage_mas.runners.evolve evolve \
  --config examples/sage_mas/sage_config.example.yaml \
  --trajectories logs/alfworld/trajectories/<run-id>/trajectories.jsonl \
  --output logs/sage_mas/evolution
```

Offline candidates also enter SkillBank as provisional credit-managed Skills;
they do not directly expand the organization.

## WebShop online evolution

WebShop uses the same play → distill → credit → org loop with
``webshop://goal/{idx}`` task ids (test stream ``0..499``).

**One-shot smoke** (sets ``JAVA_HOME``, picks a Python that already has
WebShop deps — prefers conda base over empty ``.venv`` — and prefers
``llm_config.yaml`` api_key over a stale ``OPENAI_API_KEY``):

```bash
./examples/sage_mas/run_webshop_smoke.sh
```

Or manually:

```bash
export JAVA_HOME=/home/zhuyao/miniconda3/envs/webshop/lib/jvm
export PATH="$JAVA_HOME/bin:$PATH"
# Avoid stale ambient tokens overriding llm_config.yaml:
env -u OPENAI_API_KEY PYTHONUNBUFFERED=1 python -m sage_mas.runners.online_webshop \
  --llm-config examples/prompt_agent/llm_config.yaml \
  --sage-config examples/sage_mas/sage_config.webshop_smoke.yaml \
  --output logs/sage_mas/webshop_smoke
```

Fuller evolution config (skills + optional ADD_AGENT, shadow off by default):

```bash
SAGE_CONFIG=examples/sage_mas/sage_config.webshop_online.yaml \
OUTPUT=logs/sage_mas/webshop_online \
  ./examples/sage_mas/run_webshop_smoke.sh
```

Ensure ``llm_config.yaml`` has a ``webshop:`` block (see
``llm_config.example.yaml``). WebShop product data must already be installed
under the env package. First-time machine deps: JDK 11 (``javac``), ``faiss``,
and ``python -m spacy download en_core_web_sm``.

## Organization shadow

```bash
python -m sage_mas.runners.shadow_alfworld \
  --llm-config examples/prompt_agent/llm_config.yaml \
  --sage-config examples/sage_mas/sage_config.example.yaml \
  --old-organization logs/sage_mas/evolution/<run>/active_organization.json \
  --new-organization logs/sage_mas/evolution/<run>/candidate_organization.json \
  --skill-bank logs/sage_mas/skill_bank.json \
  --candidate-skills logs/sage_mas/evolution/<run>/candidate_skills.json \
  --output logs/sage_mas/shadow_result.json \
  --active-output logs/sage_mas/active_organization.json
```

Keep credentials outside YAML:

```bash
export OPENAI_API_KEY="..."
```
