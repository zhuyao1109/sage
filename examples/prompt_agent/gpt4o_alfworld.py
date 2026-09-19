import os
import time
import json
import yaml
import logging
from datetime import datetime
from openai import OpenAI
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_system.environments.env_manager import AlfWorldEnvironmentManager

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class _CfgNode:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "llm_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path


def _list_unseen_gamefiles(data_path: str):
    unseen_root = os.path.join(data_path, "json_2.1.1", "valid_unseen")
    gamefiles = []
    for root, _, files in os.walk(unseen_root):
        if "game.tw-pddl" in files:
            gamefiles.append(os.path.join(root, "game.tw-pddl"))
    gamefiles.sort()
    if not gamefiles:
        raise RuntimeError(f"No ALFWorld unseen game files found under {unseen_root}")
    return gamefiles


def build_alfworld_env_manager(
    *,
    env_num: int,
    seed: int,
    eval_dataset: str,
    game_files_list,
    num_cpus_per_worker: float,
):
    from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs

    alf_config_path = os.path.join(
        REPO_ROOT,
        "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    env_kwargs = {
        "eval_dataset": eval_dataset,
        "game_files_list": list(game_files_list),
    }
    resources_per_worker = {"num_cpus": float(num_cpus_per_worker), "num_gpus": 0.0}
    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    # AlfWorldEnvironmentManager expects a config object with `.env.history_length`.
    cfg = _CfgNode(env=_CfgNode(history_length=0))
    return AlfWorldEnvironmentManager(envs, alfworld_projection, cfg)

class Agent:
    def __init__(self, model_name="gpt-4o", *, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.4):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.client = OpenAI(
            api_key=(api_key or os.environ.get("OPENAI_API_KEY", "")),
            base_url=base_url,
        )
        
    def get_action_from_gpt(self, obs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": obs
                }
            ],
            temperature=self.temperature,
            n=1,
            stop=None
        )
        action = response.choices[0].message.content.strip()
        return action


def _extract_task_from_obs(obs_text: str) -> str:
    marker = "Your task is to: "
    idx = obs_text.find(marker)
    return obs_text[idx + len(marker) :].strip() if idx != -1 else ""


HARD_TASK_FAMILIES = (
    "pick_two_obj_and_place",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def _task_family_from_gamefile(gamefile: str) -> str:
    families = [
        "pick_and_place",
        *HARD_TASK_FAMILIES,
        "look_at_obj_in_light",
    ]
    for f in families:
        if f in gamefile:
            return f
    return "other"


def _select_hard_gamefiles(all_gamefiles: list[str], num_games: int) -> list[str]:
    by_family = defaultdict(list)
    for gf in all_gamefiles:
        fam = _task_family_from_gamefile(gf)
        if fam in HARD_TASK_FAMILIES:
            by_family[fam].append(gf)
    for fam in by_family:
        by_family[fam].sort()

    selected = []
    families = sorted(by_family.keys())
    while len(selected) < num_games and any(by_family[f] for f in families):
        for fam in families:
            if by_family[fam]:
                selected.append(by_family[fam].pop(0))
                if len(selected) >= num_games:
                    break
    if len(selected) < num_games:
        raise RuntimeError(
            f"Requested {num_games} hard games but only found {len(selected)} across {HARD_TASK_FAMILIES}."
        )
    return selected


def run_direct_unseen(cfg: dict):
    os.makedirs("logs/alfworld", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fp = os.path.join("logs/alfworld", f"run_log_{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    openai_cfg = cfg["openai"]
    alf_cfg = cfg["alfworld"]
    phases = cfg.get("phases", {})
    test_cfg = phases.get("test", {}) or {}

    data_path = alf_cfg["data_path"]
    max_steps = int(alf_cfg.get("max_steps", 50))
    parallel_envs = int(alf_cfg.get("parallel_envs", 10))
    api_concurrency = int(alf_cfg.get("api_concurrency", 10))
    num_cpus_per_worker = float(alf_cfg.get("num_cpus_per_worker", 0.05))
    save_trajectories = bool(alf_cfg.get("save_trajectories", True))
    history_length = int(alf_cfg.get("history_length", 0))

    num_games = int(test_cfg.get("num_games", 134))
    game_selection = str(test_cfg.get("game_selection", "first")).lower()
    use_guidance_examples = bool(test_cfg.get("use_guidance_examples", False))
    if use_guidance_examples:
        logging.warning("use_guidance_examples=true but direct run ignores guidance; forcing off.")

    model = openai_cfg["model"]
    temperature = float(openai_cfg.get("temperature", 0.4))

    logging.info(f"Model: {model}")
    logging.info(f"ALFWORLD_DATA={data_path}")
    logging.info(f"parallel_envs={parallel_envs}, api_concurrency={api_concurrency}")

    traj_dir = os.path.join("logs/alfworld/trajectories", run_id)
    traj_path = os.path.join(traj_dir, "trajectories.jsonl")
    if save_trajectories:
        os.makedirs(traj_dir, exist_ok=True)
        logging.info(f"Trajectory log: {traj_path}")

    all_gamefiles = _list_unseen_gamefiles(data_path)
    exclude_from = test_cfg.get("exclude_from")
    excluded = set()
    if exclude_from:
        exclude_path = exclude_from if os.path.isabs(exclude_from) else os.path.join(REPO_ROOT, exclude_from)
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded.add(json.loads(line)["gamefile"])
        logging.info(f"exclude_from={exclude_path}: skipping {len(excluded)} games")

    if game_selection == "hard":
        gamefiles = _select_hard_gamefiles(all_gamefiles, num_games)
        logging.info(f"game_selection=hard: {num_games} games from {HARD_TASK_FAMILIES}")
    elif game_selection == "first":
        candidates = [g for g in all_gamefiles if g not in excluded]
        gamefiles = candidates[:num_games]
        if len(gamefiles) != num_games:
            raise RuntimeError(
                f"Requested num_games={num_games} but only found {len(candidates)} unseen games "
                f"after excluding {len(excluded)}."
            )
    else:
        raise RuntimeError(f"Unknown game_selection={game_selection!r}; use 'first' or 'hard'.")

    logging.info(
        f"========== Direct Test ({num_games} unseen, no guidance): game_selection={game_selection} | eval_dataset=eval_out_of_distribution =========="
    )

    agent = Agent(
        model_name=model,
        base_url=openai_cfg.get("base_url"),
        api_key=os.environ.get("OPENAI_API_KEY") or openai_cfg.get("api_key"),
        temperature=temperature,
    )

    def write_traj(obj: dict):
        if not save_trajectories:
            return
        with open(traj_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    start_time = time.time()
    wins = 0
    task_wins = defaultdict(int)
    task_total = defaultdict(int)

    batch_idx = 0
    for offset in range(0, num_games, parallel_envs):
        batch_games = gamefiles[offset : offset + parallel_envs]
        env_manager = build_alfworld_env_manager(
            env_num=len(batch_games),
            seed=1,
            eval_dataset="eval_out_of_distribution",
            game_files_list=batch_games,
            num_cpus_per_worker=num_cpus_per_worker,
        )
        # patch env history_length in manager config
        try:
            env_manager.config.env.history_length = history_length
        except Exception:
            pass

        obs, infos = env_manager.reset({})
        env_dones = [False] * len(batch_games)
        env_steps = [[] for _ in batch_games]
        tasks = []
        for i in range(len(batch_games)):
            tasks.append(_extract_task_from_obs(obs["anchor"][i]))

        for _step_idx in range(max_steps):
            actions = ["None"] * len(batch_games)
            prompts = [(i, obs["text"][i]) for i in range(len(batch_games)) if not env_dones[i]]

            with ThreadPoolExecutor(max_workers=api_concurrency) as ex:
                futs = {ex.submit(agent.get_action_from_gpt, p): i for i, p in prompts}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        actions[i] = fut.result()
                    except Exception as e:
                        logging.warning(f"LLM call failed for env {i}: {e}")
                        actions[i] = "<think>error</think><action>look</action>"

            next_obs, rewards, dones, infos = env_manager.step(actions)

            for i in range(len(batch_games)):
                if env_dones[i]:
                    continue
                env_steps[i].append({"observation": next_obs["anchor"][i], "action": actions[i]})

                if bool(dones[i]):
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    wins += int(won)

                    fam = _task_family_from_gamefile(batch_games[i])
                    task_total[fam] += 1
                    task_wins[fam] += int(won)

                    write_traj(
                        {
                            "phase": "phase2_test",
                            "phase_name": "Direct Test (unseen, no guidance)",
                            "game_index": offset + i,
                            "eval_dataset": "eval_out_of_distribution",
                            "gamefile": batch_games[i],
                            "task": tasks[i],
                            "won": won,
                            "num_steps": len(env_steps[i]),
                            "steps": env_steps[i],
                        }
                    )

            obs = next_obs
            if all(env_dones):
                break

        sr = wins / (offset + len(batch_games))
        logging.info(
            f"Direct Test (134 unseen) batch {batch_idx}: finished {len(batch_games)} games, cumulative SR={sr:.4f}"
        )
        batch_idx += 1

    sr = wins / num_games
    logging.info(f"Direct Test (134 unseen) overall success: {sr:.4f} ({wins}/{num_games})")
    for k in sorted(task_total.keys()):
        logging.info(f"    {k:<35s}: {task_wins[k]/task_total[k]:.4f} ({task_wins[k]}/{task_total[k]})")
    logging.info("=============== Final Summary ===============")
    logging.info(f"Direct Test (134 unseen, no guidance): {sr:.4f} ({wins}/{num_games})")
    logging.info(f"Total elapsed: {time.time() - start_time:.2f}s")
    if save_trajectories:
        logging.info(f"Trajectories saved to: {traj_path}")

if __name__ == "__main__":
    cfg, _cfg_path = _load_cfg()
    run_direct_unseen(cfg)
