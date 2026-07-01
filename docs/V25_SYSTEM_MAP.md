# V2.5 System Map

本文档用于记录当前 v2.5 稳定版本的系统边界。它不是重构后的目标结构，而是重构前的“地图”：先知道哪些模块在干活，再决定怎么拆。

## 1. 当前稳定叙事

v2.5 的核心不再是早期的“在 A* 路线上做 RL 优化”，而是：

```text
可预测层 A* 全局规划
    ↓
真实执行层遭遇随机扰动
    ↓
局部风险观测器 / 风险膜
    ↓
Expert / PPO residual 给出局部修正
    ↓
do-no-harm gate 防止上层策略帮倒忙
    ↓
APAS 做最终安全过滤
    ↓
waypoint-skip 修复路径点推进卡死
```

这个版本的主要工程结论是：

- `A* + APAS + waypoint-skip` 已经是强基线，能完成大部分随机层避险。
- `Expert` 的主要价值不是单纯拉高成功率，而是降低 APAS 介入负担、降低风险、让系统少靠最后一层硬救。
- `PPO/BC` 的定位是压缩 Expert 策略，降低在线计算成本，而不是无保护地替代安全系统。
- `do-no-harm gate` 是必要保险，用于发现 Expert/PPO residual 连续无效时回退到 A* + APAS。

## 2. 主要入口

### 测试入口

```powershell
.\v25\scripts\v25-run.cmd test
```

### 最终消融

```powershell
python v25\experiments\final_ablation_v25.py --episodes 30 --seed 200 --curriculum-stage 3 --stress fragile --planner-max-steps 15000
```

### A* / Expert / RL 对比

```powershell
python v25\experiments\compare_astar_rl_disruptive.py --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip --episodes 30 --seed 200 --curriculum-stage 3 --stress fragile --expert --astar-apas --expert-apas --rl-apas --planner-max-steps 15000
```

### 失败轨迹诊断

```powershell
python v25\experiments\diagnose_failure_traces.py --seeds 43 --methods astar_apas expert_apas rl_apas --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip --curriculum-stage 3 --stress fragile --planner-max-steps 15000 --last-steps 120
```

### Expert 演示采集与 PPO 行为克隆

```powershell
python v25\experiments\collect_expert_demonstrations.py --episodes 200 --seed 42 --curriculum-stage 3 --stress fragile --output-dir v25\artifacts\expert_demos
python v25\experiments\pretrain_ppo_from_expert.py --demo-npz v25\artifacts\expert_demos\expert_demo_20260628_202504.npz --run-name ppo_v25_expert_bc_s200 --epochs 25 --batch-size 256 --curriculum-stage 3 --stress fragile
```

## 3. 代码区域

### 配置

- `configs/config.py`

当前几乎所有 v2.5 参数都集中在这里，包括：

- 飞行器/物理参数
- 地图参数
- A* 参数
- 随机层参数
- APAS 参数
- Expert 参数
- waypoint-skip 参数
- do-no-harm gate 参数
- PPO/训练参数

重构时不要第一步就拆配置。配置是全系统耦合最重的地方，应该先通过适配层稳定读取方式，再分组拆分。

### 可预测规划与物理模型

- `core/planner.py`
- `core/estimator.py`
- `core/physics.py`

这部分属于早期项目的核心资产，负责 A* 路径搜索、能耗估计和飞行动力学近似。重构后应归入：

```text
aerorisk/planning/
aerorisk/control/
```

### 地图与环境基础

- `environment/map_manager.py`
- `environment/wind_models.py`

这部分负责地形、基础地图、可预测风场等。重构后应归入：

```text
aerorisk/maps/
aerorisk/world/
```

### V2.5 真实世界与随机扰动

- `v25/disruptions.py`
- `v25/true_world_dynamics.py`

这部分是 v2.5 的关键升级：把随机层从 A* 可见规划层中剥离出来，让执行阶段面对真实扰动。

重构后建议归入：

```text
aerorisk/world/random_layer.py
aerorisk/execution/dynamics.py
```

### V2.5 执行环境

- `v25/rl_env_disruptive.py`
- `v25/control_helpers.py`
- `v25/episode_metrics.py`
- `v25/local_hazard.py`
- `v25/risk_membrane.py`
- `v25/sensors.py`
- `v25/apas_safety.py`
- `v25/expert_policy.py`

`v25/rl_env_disruptive.py` 仍然是 Gym 环境和真实执行主循环，但 v3.0 前已经抽出了下列稳定边界：

- `control_helpers.py`: waypoint-skip、do-no-harm gate、评估代价。
- `episode_metrics.py`: episode 级计数器与运行时 tracker 初始化。
- `local_hazard.py`: 局部风险历史、趋势预警。
- `risk_membrane.py`: 稀疏观测到连续风险膜、风险带绕行建议。
- `sensors.py`: local risk observer / circle oracle / sector radar 观测特征。
- `apas_safety.py`: APAS 候选动作、segment probe、硬安全检查相关 helper。
- `expert_policy.py`: Expert 候选动作、动作选择、rollout scoring。

`rl_env_disruptive.py` 保留的职责是：

- Gym 环境接口；
- reset / step 主流程；
- A* 参考指令和真实世界执行；
- 各 helper 模块之间的状态编排；
- reward、终止条件、日志字段汇总。

这一状态足够支撑 v3.0 地图任务层。后续不建议在没有明确业务痛点时继续搬空环境类。

目标归属：

```text
aerorisk/execution/env.py
aerorisk/control/expert.py
aerorisk/control/apas.py
aerorisk/control/waypoints.py
aerorisk/control/rollback_gate.py
aerorisk/world/risk_observer.py
aerorisk/learning/rewards.py
```

### RL 训练与评估

- `v25/train_rl_disruptive.py`
- `v25/experiments/collect_expert_demonstrations.py`
- `v25/experiments/pretrain_ppo_from_expert.py`
- `v25/experiments/compare_astar_rl_disruptive.py`
- `v25/experiments/final_ablation_v25.py`
- `v25/experiments/diagnose_failure_traces.py`

这部分目前既有“实验入口”，也有不少评估公共逻辑。重构后应分为：

```text
aerorisk/learning/
aerorisk/evaluation/
experiments/
```

## 4. 稳定模块与高风险模块

### 先不要动

- `core/planner.py`
- `core/physics.py`
- `environment/map_manager.py`
- `v25/scripts/v25-run.cmd`
- 已经能复现实验结论的 `v25/experiments/*.py`

这些模块先作为稳定依赖保留。第一阶段只在外层加文档和脚本。

### 可以优先整理

- `v25/rl_env_disruptive.py`
- `configs/config.py`
- `v25/experiments` 内重复的 summary / aggregation / method 命名逻辑

### 明确不建议第一刀做

- 大规模移动目录
- 同时改参数和改结构
- 同时重命名实验方法和重写评估逻辑
- 把 v1.0/v2.5 混成一个统一大抽象

## 5. 命名边界

最终报告和代码命名建议统一：

- `astar_raw`: 裸 A*
- `astar_wp`: A* + waypoint-skip
- `astar_apas`: A* + APAS
- `astar_apas_wp`: A* + APAS + waypoint-skip
- `expert_apas_wp`: A* + Expert + APAS + waypoint-skip
- `rl_apas_wp`: A* + PPO residual + APAS + waypoint-skip

不要再把 `A* + APAS` 写成 `A* only`。这会让比较关系变脏。

## 6. 重构验收标准

每次重构提交至少满足：

```powershell
.\v25\scripts\v25-run.cmd test
```

若改动触及 v2.5 控制/环境逻辑，还应补跑：

```powershell
.\scripts\run_v25_final_validation.ps1 -Episodes 30
```

判断标准：

- 测试通过。
- 关键实验能正常产出 summary。
- 方法命名不混乱。
- 结果文件仍写入 `results/`。
- 不在同一个提交里同时大改结构和大改参数。
