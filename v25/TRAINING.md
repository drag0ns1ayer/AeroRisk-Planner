# v2.5 Training Guide

## Compatibility Note

The shared true-world upgrade changes observations from the legacy 31-D base
state to a sensor-augmented state and changes actions to true A* residual
commands. The current default `circle_oracle` local-view sensor uses 49
observation dimensions. Older v2.5 checkpoints with different observation
spaces are not compatible and must not be used for continued training or
comparison.

## Should APAS Be Enabled?

Recommended strategy:

1. Early/mid training: `APAS OFF`
- Let policy learn disturbance perception + autonomous correction.
- Avoid over-reliance on APAS intervention too early.

2. Late fine-tuning: `APAS ON`
- Improve safety envelope and reduce terminal failures.
- Keep same disturbance setting, only switch APAS to test/control robustness.

Use `--enable-apas` when needed.

## Do We Need Short/Mid/Long Curriculum?

Yes, recommended.

- `short`: fast sanity check, reward/logic validation
- `mid`: main behavior shaping
- `long`: convergence and final model selection

## Suggested Commands

### 1) Short (smoke + reward sanity)
```bash
python launcher.py rl25 -- --from-scratch --run-name v25_trueworld_short_s3 --curriculum-stage 3 --total-timesteps 50000 --n-envs 1 --seed 42 --model-save-root v25/artifacts/models/ppo_v25_trueworld_short_s3
```

### 2) Mid (main training)
```bash
python launcher.py rl25 -- --from-scratch --run-name v25_trueworld_mid_s3 --curriculum-stage 3 --total-timesteps 200000 --n-envs 2 --seed 42 --model-save-root v25/artifacts/models/ppo_v25_trueworld_mid_s3
```

### 3) Long (convergence)
```bash
python launcher.py rl25 -- --from-scratch --run-name v25_trueworld_long_s3 --curriculum-stage 3 --total-timesteps 500000 --n-envs 4 --seed 42 --model-save-root v25/artifacts/models/ppo_v25_trueworld_long_s3
```

### 4) APAS fine-tuning (optional)
```bash
python launcher.py rl25 -- --run-name v25_long_s3_apas --curriculum-stage 3 --total-timesteps 150000 --n-envs 2 --seed 42 --enable-apas --load-model-path v25/artifacts/models/ppo_v25_long_s3_best/best_model --model-save-root v25/artifacts/models/ppo_v25_long_s3_apas
```

### 5) APAS mode ablation (optional)
- APAS senses disruptive layer (recommended default):
```bash
python launcher.py rl25 -- --run-name v25_apas_disruptive --enable-apas --apas-use-disruption --model-save-root v25/artifacts/models/ppo_v25_apas_disruptive
```
- APAS senses predictable layer only (control baseline):
```bash
python launcher.py rl25 -- --run-name v25_apas_predictable_only --enable-apas --no-apas-use-disruption --model-save-root v25/artifacts/models/ppo_v25_apas_predictable
```

### 6) Eval-only (no training, telemetry refresh)
```bash
python launcher.py rl25 -- --eval-only --eval-model-path v25/artifacts/models/ppo_v25_mid_s3_best/best_model --run-name v25_eval_mid --seed 42 --model-save-root v25/artifacts/models/ppo_v25_eval
```

## What To Watch In Eval

- `success_rate` (primary)
- `storm_risk_fail_rate` (safety)
- `max_p_crash_mean/max` (risk envelope)
- `autonomy_bonus_mean` (whether disturbance adaptation is actually used)
- `mean_reward` only as a secondary signal
