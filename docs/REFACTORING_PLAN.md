# AeroRisk Planner 重构路线图

更新时间：2026-07-01

这份文档用于指导 v2.5 收口之后的工程重构。当前目标不是重写算法，也不是继续堆新功能，而是把已经验证过的系统整理成清晰、可维护、可复现的工程结构。

## 1. 重构原则

### 1.1 不做大爆炸式重构

当前 `main` 分支已经包含一套可运行、可测试、可复现实验的 v2.5/v2.6 收口版本。重构必须保留这条稳定线，避免一次性移动大量文件导致实验口径、导入路径、训练脚本全部失效。

推荐方式：

```text
先封存稳定入口
再整理外围文档和脚本
再逐步拆分核心模块
每拆一层都跑测试和关键实验
```

### 1.2 先建立边界，再移动代码

现在项目的问题不是某个函数单独写坏了，而是探索过程留下了很多混合边界：

```text
legacy v1.0
v2.5 true-world
实验脚本
训练脚本
诊断脚本
桌面/UI
结果文件
文档
```

重构第一阶段只做边界梳理，不急着移动核心逻辑。

### 1.3 以可复现为第一优先级

只要涉及实验命令、seed、模型路径、输出路径，就必须保留可复现入口。重构后至少要能稳定复现：

```text
单元测试
v2.5 final ablation
compare A* / Expert / RL
failure trace seed 43
PPO behavior cloning 流程
```

## 2. 当前系统分层

当前 v2.5/v2.6 收口后的系统可以理解为：

```text
Predictable Layer
    A* global planning
    terrain / NFZ / forecast wind

Random True-World Layer
    local wind disturbance
    destructive storm core + halo
    true execution risk

Execution Stack
    A* reference command
    optional Expert residual
    optional PPO residual
    do-no-harm rollback gate
    APAS safety filter
    true-world dynamics
    metrics / traces / evaluation
```

对应职责：

```text
A*:             全局可解释规划
waypoint-skip:  路径进度修复
APAS:           单步预测安全盾
Expert:         局部风险决策器
PPO student:    Expert residual 的轻量学习策略
do-no-harm:     防止上层策略伤害 A*+APAS 基线
failure trace:  病理诊断
```

## 3. 推荐目标结构

长期目标结构可以设计为：

```text
aerorisk/
    config/
    maps/
    planning/
    world/
    control/
    execution/
    learning/
    evaluation/
    visualization/

experiments/
    v25_final/
    diagnostics/
    ablations/
    paper/

scripts/
    run_tests.ps1
    run_v25_final_validation.ps1
    collect_expert_demos.ps1
    train_bc_ppo.ps1

docs/
    REFACTORING_PLAN.md
    V25_SYSTEM_MAP.md
    EXPERIMENTS.md
    INTERVIEW_STORY.md

legacy/
    v1/
    swarm/
    old_rl/
```

### 3.1 `aerorisk/planning`

放全局规划与路径表示：

```text
AStarPlanner
Node / path model
path tracking helpers
```

### 3.2 `aerorisk/world`

放真实世界与扰动层：

```text
predictable world
random true-world layer
wind models
storm/disruption models
true-world dynamics
```

### 3.3 `aerorisk/control`

放执行时局部控制与安全：

```text
APASSafetyFilter
ExpertController
RiskMembraneObserver
DoNoHarmGate
WaypointProgressTracker
```

### 3.4 `aerorisk/learning`

放学习相关：

```text
GuidedDroneEnv / V25Env adapter
PPO training
Expert demo collection
BC pretraining
fine-tuning
```

### 3.5 `aerorisk/evaluation`

放评估和诊断：

```text
comparison runner
final ablation
failure trace
hazard classification
summary metrics
```

## 4. 分阶段执行计划

## Phase 0：封存稳定版本

目标：确认当前主线是 v2.5/v2.6 收口版。

动作：

```text
1. 保留当前 main 作为稳定基线。
2. 给当前版本打 tag，例如 v2.5-closure 或 v2.6-rollback-closure。
3. 确认 README / v25 文档能说明当前系统。
4. 保留最终实验结果路径。
```

验收：

```text
.\v25\scripts\v25-run.cmd test
python v25\experiments\compare_astar_rl_disruptive.py ...
```

## Phase 1：外围整理

目标：不动核心逻辑，只整理文档、脚本、入口说明。

动作：

```text
1. 新增 docs/REFACTORING_PLAN.md。
2. 新增 docs/V25_SYSTEM_MAP.md。
3. 新增 scripts/run_v25_final_validation.ps1。
4. 明确哪些入口是稳定入口，哪些入口是历史/探索入口。
5. 不移动核心 Python 文件。
```

验收：

```text
git diff 只包含 docs/ scripts/ 或极少量 README 更新。
核心测试不需要因文档变化而重新训练。
```

## Phase 2：拆分 `GuidedDroneEnvV25`

目标：把当前巨型环境拆成若干职责明确的小模块。

当前 `v25/rl_env_disruptive.py` 同时负责：

```text
observation
residual command
APAS
Expert
risk membrane
do-no-harm
waypoint-skip
reward
metrics
replan memory
```

建议拆分顺序：

```text
1. MetricsRecorder
2. DoNoHarmGate
3. APASSafetyFilter
4. WaypointProgressTracker
5. RiskObserver / RiskMembrane
6. ExpertController
7. ObservationBuilder
```

每一步原则：

```text
先复制小块逻辑到新模块
旧 env 调新模块
保持外部行为不变
跑测试
再删旧内联代码
```

## Phase 3：配置拆分

目标：拆分巨型 `SimulationConfig`。

当前 `SimulationConfig` 同时包含：

```text
地图参数
无人机物理参数
A* 参数
RL 参数
APAS 参数
Expert 参数
v2.5 random layer 参数
评估代价参数
UI/实验参数
```

推荐拆为：

```text
MapConfig
VehicleConfig
PlannerConfig
WorldConfig
APASConfig
ExpertConfig
RLConfig
EvaluationConfig
```

但这一阶段要谨慎，因为引用点非常多。建议先做兼容层：

```python
SimulationConfig
    .map
    .vehicle
    .planner
    .world
    .apas
    .expert
    .rl
    .eval
```

短期内保留旧字段，避免一次性修改所有调用点。

## Phase 4：legacy 隔离

目标：把 v1.0、多机、旧 RL、旧 UI 等历史探索内容明确隔离。

建议：

```text
legacy/v1_single_agent
legacy/swarm_state_machine
legacy/old_rl
legacy/desktop_ui
```

注意：不要删除历史成果。它们有叙事价值，也可能还有复试/答辩价值。隔离即可。

## 5. 风险清单

### 风险 1：移动文件导致导入路径崩溃

应对：

```text
先新增模块，再逐步改调用。
保留兼容 wrapper。
每次移动后运行测试。
```

### 风险 2：实验口径变化

应对：

```text
所有对比脚本保留 seed、stress、curriculum、model path。
最终实验命令写进 scripts。
```

### 风险 3：配置拆分导致隐性默认值变化

应对：

```text
先做只读 dataclass 分组。
再替换引用。
最后清理旧字段。
```

### 风险 4：重构过程中又开始调参

应对：

```text
重构阶段不改策略参数。
所有行为变化必须单独标记为功能改动，不混进结构重构 commit。
```

## 6. Commit 建议

重构期间建议小提交：

```text
docs: add refactoring plan
scripts: add v25 validation runner
refactor: extract do-no-harm gate
refactor: extract APAS safety filter
refactor: extract expert controller
refactor: extract metrics recorder
```

不要把“改策略参数”和“移动代码”放进同一个 commit。

## 7. 当前推荐下一步

Phase 1 已完成，并且 Phase 2 的 v2.5 核心边界已经完成到可支撑 v3.0 地图任务的程度：

```text
1. 固化重构文档与最终验证脚本；
2. 抽出 waypoint-skip / do-no-harm / evaluation costs；
3. 抽出 episode metrics reset；
4. 抽出 local hazard history 与 risk membrane；
5. 抽出 APAS segment probe / candidate / scoring helper；
6. 抽出 Expert candidate / selection / rollout scoring；
7. 抽出 local risk observer / sensor feature helpers。
```

因此，v3.0 前不建议继续为了“干净”而深拆 `GuidedDroneEnvV25`。剩余大块包括 true-world step 主循环、rejoin/replan、reward 结算和 Gym 接口，它们彼此耦合较强，继续拆容易在没有新业务收益时引入回归。

当前停止线：

```text
先进入 v3.0 地图任务执行层：
地图标定 / 巡检点 / 任务权重 / 时间约束 / 充电点 / 避难点 / 多任务执行。
```

如果 v3.0 实现过程中发现某个边界反复阻碍开发，再做针对性重构。不要先把重构本身变成新项目。
