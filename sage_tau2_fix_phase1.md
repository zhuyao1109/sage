# SAGE τ²-bench 修复 Phase 1：加强task_family匹配

## 修改文件

### 1. 修改 executor_dispatch.py

在 `_contract_matches_episode` 方法中添加task_family检查：

```python
@classmethod
def _contract_matches_episode(
    cls,
    skill: Tau2Skill,
    *,
    agent: AgentSpec,
    task: str,
    domain: str,
    task_id: str,
    episode_scope: str,
) -> bool:
    """Gate routing with learned contract scope (sage_mas-aligned)."""
    skill_domain = cls._normalized_scope(getattr(skill, "domain", "") or "")
    episode_domain = cls._normalized_scope(domain)
    if skill_domain and episode_domain and not domains_match(
        skill_domain, episode_domain
    ):
        return False

    # === 新增：提取task_family ===
    task_family = ""
    if task_id.startswith('[') and ']' in task_id:
        task_family = task_id[1:task_id.index(']')]
    
    # 如果技能有明确的task_families，检查匹配
    skill_families = skill.metadata.get('task_families', [])
    if task_family and skill_families:
        # 归一化比较
        normalized_family = cls._normalized_scope(task_family)
        normalized_skill_families = [cls._normalized_scope(f) for f in skill_families]
        
        if normalized_family not in normalized_skill_families:
            # Family不匹配，直接拒绝
            return False
    # === 新增结束 ===

    scopes = cls._skill_scopes(skill, agent)
    current_scope = cls._normalized_scope(episode_scope)
    if not current_scope:
        current_scope = episode_scope_from_task_id(task_id)

    primary = cls._normalized_scope(skill.metadata.get("primary_task_family"))

    # If we have a current_scope and skill has primary/scopes, use strict matching
    if current_scope and (primary or scopes):
        if primary and current_scope == primary:
            return True
        if scopes and current_scope in scopes:
            return True
        # Don't immediately reject - fall through to semantic matching

    # Fallback: semantic token matching
    # This handles cases where task_id is numeric or doesn't extract a scope
    contract_text = " ".join(
        [
            skill.capability_key,
            skill.skill_name,
            skill.description,
            skill.precondition,
            str(skill.expected_effect or ""),
        ]
    )
    contract_tokens = cls._semantic_tokens(contract_text)
    task_tokens = cls._semantic_tokens(
        f"{episode_scope} {task_id} {task}"
    )
    return bool(contract_tokens & task_tokens)
```

### 2. 创建简化测试配置

创建 `sage_tau2/configs/telecom_easy_test.yaml`:

```yaml
# 使用更简单的任务，积累成功信号
domain: telecom
model: openai/gemini-2.5-flash
user_model: openai/gemini-2.5-flash

# 使用标准train，不用train_large
task_split_name: train  
allow_task_resampling: false
segment_size: 15
num_segments: 3
seed: 42
max_concurrency: 3

val_size: 0

# 更宽松的蒸馏设置
max_inject_skills: 2
allow_provisional_inject: true
freeze_organization: false
inject_same_domain_only: true

min_support: 2
require_success: true
min_protocol_len: 2
max_new_skills: 12
verify_score: 0.35
prune_score: 0.12
min_protocol_coverage: 0.45

# 更宽松的提名
cluster_novelty_threshold: 0.35
nominate_min_cluster_support: 2
nominate_min_utility: 0.30
max_new_agents_per_round: 1
dispatch_only_new_agents: true
probation_games: 3
allow_provisional_org_edits: false
require_same_domain_for_nominate: true
no_spec_mode: editor_commit

enable_spec_vs_exec: true
admit_num_tasks: 6
admit_min_advantage: 0.0
admit_accept_ties: false
admit_on_fail_action: remove
admit_min_utility: 0.35
admit_min_support: 2

probation_min_games: 2
probation_min_wins: 1
remove_after_rejected_windows: 2
probation_primary_quota: 0

agent_name: sage_tau2
```

## 测试命令

### 运行修复后的测试
```bash
cd /home/zhuyao/verl-agent

# 清理旧数据
rm -rf logs/sage_tau2/telecom_phase1_test

# 运行测试
cd tau2-bench && PYTHONPATH=.. uv run python -m sage_tau2.runners.online_tau2 \
  --config ../sage_tau2/configs/telecom_easy_test.yaml \
  --output ../logs/sage_tau2/telecom_phase1_test \
  --llm-config ../examples/prompt_agent/llm_config.yaml
```

### 分析结果
```bash
cd /home/zhuyao/verl-agent
./analyze_sage_test.sh
# 修改OUTPUT变量为 telecom_phase1_test
```

## 预期改进

- task_family不匹配的注入被阻止
- enable_roaming不再注入到MMS任务
- 技能成功率从10.5%提升到30-40%
- 技能存活时间更长
