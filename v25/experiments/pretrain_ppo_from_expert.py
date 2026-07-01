from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - training dependency
    raise SystemExit("PyTorch is required for behavior cloning pretraining.") from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from configs.config import SimulationConfig
from v25.rl_env_disruptive import GuidedDroneEnvV25


def _build_config(args: argparse.Namespace) -> SimulationConfig:
    config = SimulationConfig()
    config.curriculum_stage = int(args.curriculum_stage)
    config.max_steps = int(args.env_max_steps)
    config.v25_disruption_stress_level = str(args.stress)
    config.v25_apas_segment_check_enabled = True
    config.v25_replan_enabled = False
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    config.planner_time_mode = "4d"
    config.rl_enable_apas = bool(args.enable_apas)
    return config


def _build_model(env, args: argparse.Namespace) -> PPO:
    tensorboard_log = str(args.log_dir) if importlib.util.find_spec("tensorboard") is not None else None
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(args.learning_rate),
        n_steps=2048,
        batch_size=int(args.batch_size),
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.6,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=args.device,
        seed=int(args.seed),
    )


def _policy_mean_action(policy, obs_tensor: torch.Tensor) -> torch.Tensor:
    distribution = policy.get_distribution(obs_tensor)
    dist = distribution.distribution
    if hasattr(dist, "mean"):
        return dist.mean
    return distribution.mode()


def pretrain(model: PPO, observations: np.ndarray, actions: np.ndarray, args: argparse.Namespace) -> dict:
    device = model.policy.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    dataset = TensorDataset(obs_tensor, action_tensor)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(args.learning_rate))
    mse = torch.nn.MSELoss()
    history: list[dict] = []

    model.policy.train()
    for epoch in range(int(args.epochs)):
        losses: list[float] = []
        for batch_obs, batch_actions in loader:
            optimizer.zero_grad(set_to_none=True)
            pred_actions = _policy_mean_action(model.policy, batch_obs)
            loss = mse(pred_actions, batch_actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": int(epoch + 1), "mse": epoch_loss})
        print(f"[epoch {epoch + 1:03d}] bc_mse={epoch_loss:.6f}", flush=True)

    model.policy.eval()
    with torch.no_grad():
        pred = _policy_mean_action(model.policy, obs_tensor)
        final_mse = float(mse(pred, action_tensor).detach().cpu().item())
        mae = float(torch.mean(torch.abs(pred - action_tensor)).detach().cpu().item())
    return {"final_mse": final_mse, "final_mae": mae, "history": history}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavior-clone PPO policy from v2.5 Expert demonstrations.")
    parser.add_argument("--demo-npz", required=True, type=Path)
    parser.add_argument("--run-name", default="ppo_v25_expert_bc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--stress", choices=("normal", "hard", "extreme", "fragile"), default="fragile")
    parser.add_argument("--env-max-steps", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-apas", action="store_true", default=False)
    parser.add_argument("--model-save-root", type=Path, default=Path("v25") / "artifacts" / "models")
    parser.add_argument("--log-dir", type=Path, default=Path("v25") / "artifacts" / "logs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = np.load(args.demo_npz)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Demonstration arrays must be 2D: observations=(N, obs_dim), actions=(N, action_dim).")
    if len(observations) != len(actions):
        raise ValueError("Observation/action counts do not match.")

    config = _build_config(args)
    env = Monitor(GuidedDroneEnvV25(config))
    expected_obs_dim = int(np.prod(env.observation_space.shape))
    expected_action_dim = int(np.prod(env.action_space.shape))
    if observations.shape[1] != expected_obs_dim:
        raise ValueError(f"Demo obs dim {observations.shape[1]} != env obs dim {expected_obs_dim}")
    if actions.shape[1] != expected_action_dim:
        raise ValueError(f"Demo action dim {actions.shape[1]} != env action dim {expected_action_dim}")

    model = _build_model(env, args)
    stats = pretrain(model, observations, actions, args)

    output_root = Path(args.model_save_root) / str(args.run_name)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_root))

    summary = {
        "run_name": str(args.run_name),
        "demo_npz": str(args.demo_npz),
        "samples": int(len(observations)),
        "obs_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "model_path": str(output_root) + ".zip",
        **stats,
    }
    summary_path = output_root.parent / f"{args.run_name}_bc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== BC Pretrain Summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, ensure_ascii=False))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
