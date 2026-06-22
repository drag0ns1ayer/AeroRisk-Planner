# v2.5 Shared True-World Workspace

This folder isolates the upgraded two-layer execution model from legacy files.

## World Model

- Predictable layer: terrain, NFZs, and forecast wind. A* can only use this layer.
- Random true-world layer: seeded storm mutation and local wind pulses. It is
  hidden before execution and affects actual ground motion, energy, and risk.
- Sensor layer: RL receives noisy local measurements of forecast error,
  tracking velocity error, and a local disturbance view. The default
  `circle_oracle` mode gives a causal circular summary of the current hidden
  random layer around the aircraft; legacy `sector_radar` remains available.
- Control: RL outputs bounded residuals around A* heading, airspeed, and AGL.

Both A* baseline and A*+RL must execute in the same `GuidedDroneEnvV25` true
world. The A* baseline is represented by the zero residual action.

## Entry Points

- Single-drone A* disruptive validation:
  - `python launcher.py astar25`
  - or `python v25/experiments/single_astar_disruptive.py`
- RL training (A* teacher + disruptive execution):
  - `python launcher.py rl25 -- --run-name v25_run1 --total-timesteps 200000 --n-envs 1`
  - or `python v25/train_rl_disruptive.py --run-name v25_run1`

## Design

- A* plans only in the predictable layer.
- The random layer is reproducible per episode seed but unknown to A*.
- True wind changes ground velocity and therefore causes real track drift.
- The policy observes sensors, not simulator-only disturbance parameters.
- The policy action is a true residual around the local A* command.

## Files

- `v25/disruptions.py`: disturbance models and builder.
- `v25/true_world_dynamics.py`: constrained point-mass true-world dynamics.
- `v25/rl_env_disruptive.py`: RL env with disruptive execution and autonomy reward.
- `v25/train_rl_disruptive.py`: PPO training/eval CLI for v2.5.
- `v25/experiments/single_astar_disruptive.py`: single-drone A* disruptive experiment.
- `v25/命令行使用说明.md`: 中文命令速查与训练流程。
- `v25/scripts/v25-run.cmd`: 一键命令入口（Windows）。
