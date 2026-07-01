from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from v25.rl_env_disruptive import GuidedDroneEnvV25


# ============================================================
# Default experiment settings
# Edit here for "code-first" workflow.
# CLI arguments can still override all defaults.
# ============================================================
DEFAULT_RUN_NAME = "v25_stage3_disruptive"
DEFAULT_SEED = 42
DEFAULT_OBS_MODE = "full"
DEFAULT_TOTAL_TIMESTEPS = 200000
DEFAULT_N_ENVS = 1
DEFAULT_CURRICULUM_STAGE = 3

DEFAULT_FROM_SCRATCH = False
DEFAULT_ENABLE_APAS = False
DEFAULT_APAS_USE_DISRUPTION = True

DEFAULT_MODEL_SAVE_ROOT = "v25/artifacts/models/ppo_v25_stage3"
DEFAULT_LOG_ROOT = "v25/artifacts/logs"
DEFAULT_LOAD_MODEL_PATH = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class SuccessFirstEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_path: str,
        eval_csv_path: str,
        eval_seed_base: int = 42,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.eval_csv_path = eval_csv_path
        self.eval_seed_base = eval_seed_base
        self.best_success_rate = -1.0
        self.best_mean_length = float("inf")
        self.best_mean_reward = -float("inf")
        self.eval_round = 0
        self.recent_results = deque(maxlen=3)

        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)
        if not os.path.exists(self.eval_csv_path):
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timesteps",
                        "mean_reward",
                        "mean_ep_length",
                        "success_rate",
                        "seed_start",
                        "robust_mean_reward",
                        "robust_mean_ep_length",
                        "robust_success_rate",
                        "best_success_rate",
                        "best_mean_length",
                        "best_mean_reward",
                    ]
                )

    def _evaluate_once(self):
        rewards = []
        lengths = []
        successes = []
        seed_start = self.eval_seed_base + self.eval_round * self.n_eval_episodes
        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset(seed=seed_start + ep)
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            final_info = info
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, final_info = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1
            rewards.append(ep_reward)
            lengths.append(ep_len)
            successes.append(1.0 if final_info.get("is_success", False) else 0.0)
        self.eval_round += 1
        return (
            float(np.mean(rewards)),
            float(np.mean(lengths)),
            float(np.mean(successes)),
            seed_start,
        )

    def _append_eval_csv(
        self,
        mean_reward,
        mean_length,
        success_rate,
        seed_start,
        robust_mean_reward,
        robust_mean_length,
        robust_success_rate,
    ):
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.num_timesteps,
                    mean_reward,
                    mean_length,
                    success_rate,
                    seed_start,
                    robust_mean_reward,
                    robust_mean_length,
                    robust_success_rate,
                    self.best_success_rate,
                    self.best_mean_length,
                    self.best_mean_reward,
                ]
            )

    def _is_better(self, success_rate, mean_length, mean_reward):
        if success_rate > self.best_success_rate:
            return True
        if success_rate == self.best_success_rate and mean_length < self.best_mean_length:
            return True
        if (
            success_rate == self.best_success_rate
            and mean_length == self.best_mean_length
            and mean_reward > self.best_mean_reward
        ):
            return True
        return False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            mean_reward, mean_length, success_rate, seed_start = self._evaluate_once()
            self.recent_results.append((mean_reward, mean_length, success_rate))
            robust_mean_reward = float(np.mean([result[0] for result in self.recent_results]))
            robust_mean_length = float(np.mean([result[1] for result in self.recent_results]))
            robust_success_rate = float(np.mean([result[2] for result in self.recent_results]))
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/robust_success_rate", robust_success_rate)
            if self._is_better(robust_success_rate, robust_mean_length, robust_mean_reward):
                self.best_success_rate = robust_success_rate
                self.best_mean_length = robust_mean_length
                self.best_mean_reward = robust_mean_reward
                self.model.save(self.best_model_path)
            self._append_eval_csv(
                mean_reward,
                mean_length,
                success_rate,
                seed_start,
                robust_mean_reward,
                robust_mean_length,
                robust_success_rate,
            )
        return True


def create_envs(config: SimulationConfig, n_envs: int = 1):
    def make_env():
        return Monitor(GuidedDroneEnvV25(config))

    return make_vec_env(make_env, n_envs=n_envs)


def build_new_model(env, log_dir: str, seed: int):
    tensorboard_log = log_dir if importlib.util.find_spec("tensorboard") is not None else None
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.6,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device="auto",
        seed=seed,
    )


def evaluate_model(model_path: str, config: SimulationConfig, n_episodes: int, eval_seed: int, raw_csv_path: Optional[str] = None):
    model = PPO.load(model_path, device="cpu")
    env = GuidedDroneEnvV25(config)
    raw_rows = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=eval_seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        final_info = info
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
        raw_rows.append(
            {
                "episode": episode,
                "seed": eval_seed + episode,
                "success": final_info.get("is_success", False),
                "terminated_reason": final_info.get("terminated_reason", "unknown"),
                "max_p_crash": getattr(env, "telemetry_max_p_crash", 0.0),
                "ep_length": episode_length,
                "total_reward": episode_reward,
                "residual_action_magnitude": final_info.get("residual_action_magnitude", 0.0),
                "disturbance_severity": final_info.get("disturbance_severity", 0.0),
                "episode_disturbance_max": getattr(env, "episode_disturbance_max", 0.0),
                "episode_disturbance_mean": (
                    getattr(env, "episode_disturbance_sum", 0.0) / max(episode_length, 1)
                ),
                "episode_disturbance_steps": getattr(env, "episode_disturbance_steps", 0),
                "episode_residual_action_sum": getattr(env, "episode_residual_action_sum", 0.0),
                "episode_residual_heading_abs_sum": getattr(env, "episode_residual_heading_abs_sum", 0.0),
                "episode_residual_speed_abs_sum": getattr(env, "episode_residual_speed_abs_sum", 0.0),
                "episode_residual_agl_abs_sum": getattr(env, "episode_residual_agl_abs_sum", 0.0),
                "episode_action_delta_sum": getattr(env, "episode_action_delta_sum", 0.0),
                "episode_intervention_need_mean": (
                    getattr(env, "episode_intervention_need_sum", 0.0) / max(episode_length, 1)
                ),
                "episode_unneeded_residual_sum": getattr(env, "episode_unneeded_residual_sum", 0.0),
                "episode_needed_residual_sum": getattr(env, "episode_needed_residual_sum", 0.0),
                "episode_apas_interventions": getattr(env, "episode_apas_interventions", 0),
                "episode_destructive_core_hits": getattr(env, "episode_destructive_core_hits", 0),
                "episode_unproductive_residual_cost_sum": getattr(
                    env, "episode_unproductive_residual_cost_sum", 0.0
                ),
                "episode_progress_shortfall_mean": (
                    getattr(env, "episode_progress_shortfall_sum", 0.0) / max(episode_length, 1)
                ),
                "episode_apas_intervention_cost_sum": getattr(env, "episode_apas_intervention_cost_sum", 0.0),
                "episode_speed_residual_cost_sum": getattr(env, "episode_speed_residual_cost_sum", 0.0),
                "episode_residual_gate_mean": (
                    getattr(env, "episode_residual_gate_sum", 0.0) / max(episode_length, 1)
                ),
                "episode_do_no_harm_events": getattr(env, "episode_do_no_harm_events", 0),
                "episode_do_no_harm_suppressed_steps": getattr(env, "episode_do_no_harm_suppressed_steps", 0),
                "episode_do_no_harm_cooldown_steps": getattr(env, "episode_do_no_harm_cooldown_steps", 0),
                "episode_local_hazard_need_mean": (
                    getattr(env, "episode_local_hazard_need_sum", 0.0) / max(episode_length, 1)
                ),
                "episode_local_hazard_cost_sum": getattr(env, "episode_local_hazard_cost_sum", 0.0),
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    if raw_csv_path:
        Path(raw_csv_path).parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "mean_reward": float(raw_df["total_reward"].mean()) if not raw_df.empty else 0.0,
        "std_reward": float(raw_df["total_reward"].std()) if len(raw_df) > 1 else 0.0,
        "mean_length": float(raw_df["ep_length"].mean()) if not raw_df.empty else 0.0,
        "success_rate": float(raw_df["success"].mean()) if not raw_df.empty else 0.0,
        "storm_risk_fail_rate": float((raw_df["terminated_reason"] == "storm_risk_too_high").mean()) if not raw_df.empty else 0.0,
        "destructive_core_fail_rate": float((raw_df["terminated_reason"] == "destructive_storm_core").mean()) if not raw_df.empty else 0.0,
        "max_p_crash_mean": float(raw_df["max_p_crash"].mean()) if not raw_df.empty else 0.0,
        "max_p_crash_max": float(raw_df["max_p_crash"].max()) if not raw_df.empty else 0.0,
        "residual_action_mean": float(raw_df["residual_action_magnitude"].mean()) if not raw_df.empty else 0.0,
        "disturbance_severity_mean": float(raw_df["disturbance_severity"].mean()) if not raw_df.empty else 0.0,
        "episode_disturbance_max_mean": float(raw_df["episode_disturbance_max"].mean()) if not raw_df.empty else 0.0,
        "episode_disturbance_mean_mean": float(raw_df["episode_disturbance_mean"].mean()) if not raw_df.empty else 0.0,
        "episode_disturbance_steps_mean": float(raw_df["episode_disturbance_steps"].mean()) if not raw_df.empty else 0.0,
        "episode_residual_action_sum_mean": float(raw_df["episode_residual_action_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_residual_heading_abs_sum_mean": float(raw_df["episode_residual_heading_abs_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_residual_speed_abs_sum_mean": float(raw_df["episode_residual_speed_abs_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_residual_agl_abs_sum_mean": float(raw_df["episode_residual_agl_abs_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_action_delta_sum_mean": float(raw_df["episode_action_delta_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_intervention_need_mean": float(raw_df["episode_intervention_need_mean"].mean()) if not raw_df.empty else 0.0,
        "episode_unneeded_residual_sum_mean": float(raw_df["episode_unneeded_residual_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_needed_residual_sum_mean": float(raw_df["episode_needed_residual_sum"].mean()) if not raw_df.empty else 0.0,
        "episode_apas_interventions_mean": float(raw_df["episode_apas_interventions"].mean()) if not raw_df.empty else 0.0,
        "episode_destructive_core_hits_mean": float(raw_df["episode_destructive_core_hits"].mean()) if not raw_df.empty else 0.0,
        "episode_unproductive_residual_cost_sum_mean": (
            float(raw_df["episode_unproductive_residual_cost_sum"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_progress_shortfall_mean": (
            float(raw_df["episode_progress_shortfall_mean"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_apas_intervention_cost_sum_mean": (
            float(raw_df["episode_apas_intervention_cost_sum"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_speed_residual_cost_sum_mean": (
            float(raw_df["episode_speed_residual_cost_sum"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_residual_gate_mean": (
            float(raw_df["episode_residual_gate_mean"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_do_no_harm_events_mean": (
            float(raw_df["episode_do_no_harm_events"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_do_no_harm_suppressed_steps_mean": (
            float(raw_df["episode_do_no_harm_suppressed_steps"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_do_no_harm_cooldown_steps_mean": (
            float(raw_df["episode_do_no_harm_cooldown_steps"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_local_hazard_need_mean": (
            float(raw_df["episode_local_hazard_need_mean"].mean()) if not raw_df.empty else 0.0
        ),
        "episode_local_hazard_cost_sum_mean": (
            float(raw_df["episode_local_hazard_cost_sum"].mean()) if not raw_df.empty else 0.0
        ),
    }
    return summary, raw_df


def model_file_exists(model_path: str) -> bool:
    path = Path(model_path)
    return path.exists() or path.with_suffix(".zip").exists()


def choose_available_model(*model_paths: Optional[str]) -> str:
    for model_path in model_paths:
        if model_path and model_file_exists(model_path):
            return model_path
    candidates = ", ".join(str(path) for path in model_paths if path)
    raise FileNotFoundError(f"No model file exists among: {candidates}")


def build_config_from_args(args) -> SimulationConfig:
    config = SimulationConfig()
    config.curriculum_stage = args.curriculum_stage
    config.wind_seed = args.seed
    config.obs_ablation_mode = args.obs_mode
    config.enable_single_agent_gusts = False
    config.collect_ablation_telemetry = True
    config.rl_enable_apas = bool(args.enable_apas)
    config.v25_sensor_mode = str(args.sensor_mode)
    config.v25_disruption_stress_level = str(args.stress)
    # v2.5-only toggle consumed by GuidedDroneEnvV25.
    config.v25_apas_use_disruption = bool(args.apas_use_disruption)
    return config


def train_ppo(
    config: SimulationConfig,
    total_timesteps: int,
    n_envs: int,
    model_save_root: str,
    log_root: str,
    run_name: str,
    seed: int,
    load_model_path: Optional[str] = None,
    from_scratch: bool = False,
):
    env = create_envs(config, n_envs=n_envs)
    eval_env = Monitor(GuidedDroneEnvV25(config))

    model_root = Path(model_save_root)
    log_dir = Path(log_root)
    model_root.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if not from_scratch and load_model_path and os.path.exists(load_model_path + ".zip"):
        try:
            model = PPO.load(load_model_path, env=env, device="auto")
            expected_obs = 31 + GuidedDroneEnvV25._sensor_feature_count(config)
            model_obs = int(np.prod(model.observation_space.shape))
            if model_obs != expected_obs:
                raise ValueError(
                    f"Loaded model observation size is {model_obs}; upgraded v2.5 requires {expected_obs}."
                )
            new_lr = 5e-5
            model.learning_rate = new_lr
            model.lr_schedule = get_schedule_fn(new_lr)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr
            if importlib.util.find_spec("tensorboard") is None:
                model.tensorboard_log = None
        except Exception as exc:
            raise ValueError(
                "Unable to continue from the requested model. Upgraded v2.5 models must be retrained "
                "with the shared true-world observation space."
            ) from exc

    if model is None:
        model = build_new_model(env, str(log_dir), seed=seed)

    eval_interval = max(1, min(20_000, int(total_timesteps)))
    checkpoint_interval = max(1, min(20_000, int(total_timesteps)))
    eval_csv_path = str(model_root) + "_eval/eval_metrics.csv"
    if from_scratch:
        Path(eval_csv_path).unlink(missing_ok=True)
    eval_callback = SuccessFirstEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_interval,
        n_eval_episodes=min(config.ablation_eval_episodes, 10),
        best_model_path=str(model_root) + "_best/best_model",
        eval_csv_path=eval_csv_path,
        eval_seed_base=seed,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=str(model_root) + "_checkpoints",
        name_prefix=run_name,
    )

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback], progress_bar=True)
    final_model_path = str(model_root) + "_final"
    model.save(final_model_path)
    best_model_path = str(model_root) + "_best/best_model"
    evaluation_model_path = choose_available_model(best_model_path, final_model_path)
    return final_model_path, best_model_path, evaluation_model_path


def ensure_no_unintended_overwrite(model_save_root: str, from_scratch: bool, load_model_path: Optional[str]) -> None:
    """
    Prevent accidental artifact overwrite when reusing the same model root.
    """
    root = Path(model_save_root)
    collision_targets = [
        Path(str(root) + "_final.zip"),
        Path(str(root) + "_best") / "best_model.zip",
        Path(str(root) + "_eval") / "eval_metrics.csv",
    ]
    has_existing = any(p.exists() for p in collision_targets)
    if has_existing and not from_scratch and not load_model_path:
        raise FileExistsError(
            f"Artifacts already exist under '{model_save_root}'. "
            "Use a different --model-save-root, or pass --load-model-path to continue training, "
            "or use --from-scratch if you intentionally want to overwrite checkpoints."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for v2.5 disruptive execution setting")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--obs-mode", choices=["full", "no_future", "no_radar"], default=DEFAULT_OBS_MODE)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--curriculum-stage", type=int, default=DEFAULT_CURRICULUM_STAGE)
    parser.add_argument("--sensor-mode", choices=["circle_oracle", "sector_radar"], default="circle_oracle")
    parser.add_argument("--stress", choices=["normal", "hard", "extreme", "fragile"], default="normal")
    parser.add_argument("--from-scratch", action="store_true", default=DEFAULT_FROM_SCRATCH)
    parser.add_argument("--enable-apas", action="store_true", default=DEFAULT_ENABLE_APAS)
    parser.add_argument(
        "--apas-use-disruption",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_APAS_USE_DISRUPTION,
        help="When APAS is enabled, include v2.5 disruptive winds/risk in APAS checks (default: true).",
    )
    parser.add_argument("--model-save-root", default=DEFAULT_MODEL_SAVE_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--load-model-path", default=DEFAULT_LOAD_MODEL_PATH)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument(
        "--eval-model-path",
        default=None,
        help="Model path (without .zip) used in eval-only mode. "
        "If omitted, fallback order is --load-model-path then <model-save-root>_best/best_model.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episode count in eval-only mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    logger = setup_logger(args.log_root, args.run_name)
    logger.info("Starting v2.5 PPO training run %s", args.run_name)
    config = build_config_from_args(args)

    if args.eval_only:
        eval_model_path = choose_available_model(
            args.eval_model_path,
            args.load_model_path,
            str(args.model_save_root) + "_best/best_model",
            str(args.model_save_root) + "_final",
        )
        eval_episodes = args.eval_episodes if args.eval_episodes is not None else config.ablation_eval_episodes
        eval_raw_path = str(Path(args.model_save_root).parent / f"{args.run_name}_eval_only_raw.csv")
        summary, _ = evaluate_model(
            eval_model_path,
            config,
            n_episodes=eval_episodes,
            eval_seed=args.seed,
            raw_csv_path=eval_raw_path,
        )
        summary_path = str(Path(args.model_save_root).parent / f"{args.run_name}_eval_only_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Eval-only finished. Model: %s", eval_model_path)
        logger.info("Eval-only raw: %s", eval_raw_path)
        logger.info("Eval-only summary: %s", summary_path)
    else:
        ensure_no_unintended_overwrite(
            model_save_root=args.model_save_root,
            from_scratch=args.from_scratch,
            load_model_path=args.load_model_path,
        )
        final_model_path, best_model_path, evaluation_model_path = train_ppo(
            config=config,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            model_save_root=args.model_save_root,
            log_root=args.log_root,
            run_name=args.run_name,
            seed=args.seed,
            load_model_path=args.load_model_path,
            from_scratch=args.from_scratch,
        )

        final_eval_episodes = (
            min(config.ablation_eval_episodes, 10)
            if args.total_timesteps < 20_000
            else config.ablation_eval_episodes
        )
        eval_raw_path = str(Path(args.model_save_root).parent / f"{args.run_name}_final_eval_raw.csv")
        summary, _ = evaluate_model(
            evaluation_model_path,
            config,
            n_episodes=final_eval_episodes,
            eval_seed=args.seed,
            raw_csv_path=eval_raw_path,
        )
        summary_path = str(Path(args.model_save_root).parent / f"{args.run_name}_final_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Training finished. Final model: %s", final_model_path)
        logger.info("Best model: %s", best_model_path)
        logger.info("Evaluated model: %s", evaluation_model_path)
        logger.info("Final eval raw: %s", eval_raw_path)
        logger.info("Final eval summary: %s", summary_path)
