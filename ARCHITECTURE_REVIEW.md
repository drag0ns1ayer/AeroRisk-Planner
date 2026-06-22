# 项目架构总结、当前更新与未来拓展方向

> 撰写日期：2026-05-15 · 分支：`feature/m1-my-update` · 版本：v2.5

---

## 一、项目概览

本项目是一个**基于4D时空风险模型的无人机路径规划与强化学习仿真系统**，面向山地复杂气象环境中的无人机自主导航研究。系统以瑞士伯尔尼高地（Bernese Oberland）真实DEM高程图为底图，集成了传统A\*规划算法与PPO深度强化学习两条技术路线，并支持异构四机蜂群协同（Master / Scout / Relay / Support）、动态移动风暴、TKE极值概率风险模型、以及Streamlit交互式Web指挥中心。

**技术栈**：Python 3.8+ · PyTorch · Stable-Baselines3 (PPO) · Gymnasium · Streamlit · Plotly · OpenCV · pywebview / PyInstaller

---

## 二、系统架构

### 2.1 分层架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    表现层 (Presentation)                  │
│  ui/drone_ui.py  │  desktop_launcher.py  │  main.py     │
│  utils/visualizer_core.py  │  utils/animation_builder.py │
├─────────────────────────────────────────────────────────┤
│                    分析层 (Analysis)                      │
│  analysis/benchmark_*.py  │  analysis/exp*_*.py          │
│  analysis/mission_metrics.py  │  analysis/render_*.py    │
├─────────────────────────────────────────────────────────┤
│                  RL训练层 (RL Pipeline)                   │
│  rl_env/drone_env.py  │  rl_training/train_ppo.py       │
│  adapters/rl_adapter.py  │  v25/rl_env_disruptive.py    │
├─────────────────────────────────────────────────────────┤
│                   仿真执行层 (Simulation)                  │
│  simulation/mission_executor.py                          │
│  simulation/swarm_mission_executor.py                    │
│  simulation/swarm_disturbance.py                         │
├─────────────────────────────────────────────────────────┤
│                   核心算法层 (Core)                        │
│  core/planner.py  │  core/physics.py  │  core/estimator.py │
│  core/battery_manager.py                                  │
├─────────────────────────────────────────────────────────┤
│                   环境建模层 (Environment)                 │
│  environment/map_manager.py  │  environment/wind_models.py │
├─────────────────────────────────────────────────────────┤
│                   配置层 (Config)                         │
│  configs/config.py  │  configs/eval_config.py            │
│  models/mission_models.py                                │
└─────────────────────────────────────────────────────────┘
```

### 2.2 各层职责

| 层级 | 目录 | 职责 |
|------|------|------|
| **配置层** | `configs/`, `models/` | `SimulationConfig` dataclass 统一管理 ~100 个参数，覆盖地图、风场、风暴、物理、电池、规划、RL、v2.5 全部调优项；`mission_models.py` 定义核心数据模型（`MissionResult` 等） |
| **环境层** | `environment/` | DEM高程图加载与地形分析（梯度、粗糙度、碰撞检测、NFZ禁飞区）；多类型风场模型（坡度风、对数廓线、时变背景风、动态风暴系统 `StormWindManager`） |
| **核心层** | `core/` | 4D时空感知A\*规划器（26邻域3D搜索，时间传播预测）；气动物理学引擎（阻力功率+爬升功率+降级速度）；TKE湍动能→坠毁概率风险估计器；电池/能量管理器 |
| **仿真层** | `simulation/` | 单机动态任务执行器（含周期性重规划）；四机异构蜂群执行器（Master/Scout/Relay/Support + 主动护盾模式）；随机阵风扰动注入 |
| **RL层** | `rl_env/`, `rl_training/`, `adapters/` | 31维观测 × 3维动作的Gymnasium环境（A\*教师引导+RL局部修正+16线雷达扫描）；PPO课程训练四阶段；RL→MissionResult适配桥接 |
| **分析层** | `analysis/`, `experiments/` | 固定/长距离基准测试；消融实验；风场/风险场敏感性分析；案例渲染；性能指标统计 |
| **表现层** | `ui/`, `utils/`, `desktop_launcher.py` | Streamlit蜂群指挥中心Web UI；Matplotlib/Plotly 2D/3D可视化；飞行过程GIF动画生成；pywebview桌面壳 + PyInstaller打包 |
| **实验区** | `v25/` | v2.5独立迭代沙箱：突变风暴扰动模型、脉冲扰动、扰动感知RL环境与训练脚本 |

### 2.3 核心数据流

```
Config ──→ MapManager(加载DEM) ──→ WindModels(生成风场+风暴序列)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                         ▼
           AStarPlanner (4D)                        GuidedDroneEnv (RL)
           代价 = 预期能耗 + 风险惩罚                  obs(31) → action(3)
                    │                                         │
                    └────────────┬────────────────────────────┘
                                 ▼
                      MissionExecutor / SwarmExecutor
                                 │
                                 ▼
                          MissionResult ──→ Visualizer / Animation / UI
```

关键设计决策：
- **A\*与RL非竞争而为互补**：A\*负责全局路径规划（给出waypoint序列），RL负责局部微观修正（航向、速度、高度微调）
- **4D时空感知**：规划器对每个搜索节点的**未来到达时间**查询风场/风暴状态，实现真正的预测性避障，而非事后反应
- **统一结果模型**：`rl_adapter.py` 将RL训练轨迹桥接为 `MissionResult`，使可视化工具链对两条技术路线通用

---

## 三、当前分支 (`feature/m1-my-update`) 更新内容

该分支相对于 `main` 经历了一次**全项目重构+大规模功能扩张**（280文件变动，+17,334/-640行），核心变化如下：

### 3.1 从简单仿真到完整研究平台

| 维度 | 原状态 (main) | 当前状态 |
|------|--------------|---------|
| 规划器 | 基础3D A\* | **4D时空感知A\***（时间传播+未来风场预测+NFZ+加权启发式） |
| 风场 | 静态/简单时变 | 坡度风+对数廓线+时变周期风+**动态移动风暴**（3个风暴，半径500~1500m，15m/s风速） |
| 风险模型 | 无 | **TKE极值概率模型**（剪切+尾流+坡度贡献→阵风超限→坠毁概率） |
| 物理引擎 | 基础运动学 | 向量级气动阻力功率计算+重力爬升功率+**可行速度降级** |
| 电池 | 简单线性 | 容量/储备比/路径级可行性验证 |
| 仿真 | 单次路径规划 | 周期性重规划动态任务执行+四机异构蜂群 |
| RL | 无 | 31维观测PPO+4阶段课程训练+消融实验框架 |
| UI | 无 | Streamlit蜂群指挥中心+桌面端pywebview打包 |
| 分析 | 无 | 基准测试+消融实验+敏感性分析+案例渲染全套体系 |

### 3.2 关键新增模块

1. **`rl_env/drone_env.py`** (742行) — 自定义Gym环境，核心创新是"A\*教师引导+RL局部修正"双模架构，观测空间含16线雷达扫描（`rl_scan_distance_m=220m`），支持消融模式（`no_future`/`no_radar`）
2. **`rl_training/train_ppo.py`** (349行) — 多环境并行PPO训练，内置 `SuccessFirstEvalCallback`，每20k步存检查点，4阶段渐进课程
3. **`simulation/swarm_mission_executor.py`** (937行) — 项目最大单体模块，Master/Scout/Relay/Support四角色协同，含FANET通信、Scout巡逻回传、Relay中继桥接、Support主动护盾
4. **`ui/drone_ui.py`** (915行) — 交互式Web仪表盘，参数调节/地图预览/模型加载/3D轨迹导出
5. **`v25/` 实验沙箱** — 突变风暴（方向/速度可变）、脉冲扰动、扰动感知RL奖励塑形

### 3.3 配置体系升级

`SimulationConfig` 从 ~60行 扩展到 ~460行，新增参数群：
- 风暴序列参数（数量/半径/速度/寿命/强度）
- TKE风险系数（剪切0.2 / 尾流0.002 / 坡度0.05）
- NFZ禁飞区定义
- 4阶段RL课程参数（目标距离/NFZ数量/教师路径长度/奖励权重）
- v2.5扰动调优参数（功率节省/风险增益/绕路惩罚等9项增益系数）
- 完整 `__post_init__` 参数校验（45+规则）

---

## 四、未来可拓展方向

### 4.1 算法层面

#### 4.1.1 多智能体强化学习 (MARL)
当前蜂群采用"A\*几何跟踪+RL微观修正"的混合控制，Master/Scout/Relay/Support的协同策略是手工规则的（如Support固定在Master上风方向400m处）。可以升级为：
- **CTDE架构**（集中训练/分散执行）：用MAPPO/QMIX训练四个角色的协同策略
- **通信学习**：让RL学习何时传递信息、传递什么，而非固定FANET协议
- **角色自适应切换**：训练一个通用策略网络，根据场景动态切换角色行为

#### 4.1.2 更好的探索策略与离线RL
当前PPO训练依赖A\*教师提供waypoint引导，本质是模仿学习+微调。可探索：
- **无教师纯RL**：移除A\*依赖，完全从原始感知端到端学习——难度极高但学术价值大
- **好奇心驱动探索**（ICM/RND）：在稀疏奖励场景提升探索效率
- **离线RL（Batch RL）**：从已有benchmark数据中学一个策略，避免在线交互成本

#### 4.1.3 元学习与泛化
当前模型针对固定DEM地图（伯尔尼高地）训练，泛化能力有限：
- **MAML/RL²**：在多个地形（阿尔卑斯/喜马拉雅/落基山脉）上元训练，快速适配新地形
- **域随机化**：训练时随机化地形参数（坡度/粗糙度/海拔范围），提升鲁棒性

#### 4.1.4 规划器升级
- **混合规划**：A\*全局粗粒度 + MPC（模型预测控制）局部精粒度
- **概率规划**：考虑风场/风暴预测不确定性的信念空间规划（Belief Space Planning）
- **RRT\*/PRM\*替代**：在高维配置空间中采样规划可能比栅格搜索更高效

### 4.2 建模层面

#### 4.2.1 更多环境动力学
- **降雨/降雪/结冰**：影响升力系数和传感器有效性
- **昼夜光照模型**：视觉传感器（如搭载的雷达模拟）受光照条件影响
- **大气密度分层**：非均匀空气密度 `ρ(h)` 而非当前常数 `1.225 kg/m³`

#### 4.2.2 更真实的风暴模型
当前风暴是"高斯风场圆柱体匀速平移"。可升级为：
- **气象动力学**：基于Navier-Stokes简化解的真实风暴演化
- **风暴合并/分裂**：多风暴相互作用
- **地形-风暴耦合**：山谷对风暴路径的引导效应（现实中显著存在）

#### 4.2.3 传感器建模
当前RL观测的"16线雷达扫描"直接查询模拟器真值。可加入：
- **传感器噪声模型**（高斯/非高斯）
- **传感器故障/降级**（部分扇区失效、精度下降）
- **多传感器融合**（雷达+视觉+IMU的卡尔曼滤波模拟）

#### 4.2.4 多地图与真实场景
- 接入**全球DEM数据源**（SRTM/ASTER），实现任意地形一键加载
- 导入**真实气象再分析数据**（ERA5）作为风场/风暴初始化
- 支持**城市峡谷场景**（建筑物3D模型替代高程图）

### 4.3 工程与产品化

#### 4.3.1 训练基础设施
- **Docker化**：标准化训练环境，可重复实验
- **实验追踪**：集成MLflow/Weights & Biases，自动记录超参/指标/模型
- **超参优化**：Optuna/Ray Tune 自动搜索PPO超参和奖励权重
- **分布式训练**：Ray RLLib 支持多GPU/多节点PPO训练

#### 4.3.2 可视化与交互
- **实时3D渲染**：从Plotly静态图升级为Three.js/Cesium.js WebGL实时渲染
- **数字孪生模式**：连接真实无人机飞控数据，实时对比仿真预测
- **回放与分析**：存储完整仿真轨迹到数据库（TimescaleDB），支持历史回放和对比分析

#### 4.3.3 CI/CD与测试
- **GitHub Actions**：自动运行单元测试+pytest覆盖率报告
- **性能回归检测**：每次PR自动跑轻量benchmark，检测规划/RL性能退化
- **类型检查**：已配置mypy，可强制CI门禁

### 4.4 应用方向

#### 4.4.1 从仿真到实物迁移
- **Sim-to-Real Transfer**：域随机化+系统辨识，将仿真策略部署到真实无人机
- **硬件在环 (HIL)**：接入PX4/ArduPilot SITL，用真实飞控运行仿真规划结果

#### 4.4.2 物流与巡检场景
- **多配送站路径优化**：VRPTW（带时间窗的车辆路径问题）+ 风场约束
- **电力线/管道巡检**：沿目标路径自动生成巡检航点，在规避风险的同时保持覆盖率

#### 4.4.3 应急响应
- **灾区快速评估**：多机协同搜索+覆盖，最大化信息增益
- **通信中继部署**：Relay角色的自动化位置优化（当前为固定几何位置）

---

## 五、架构改进建议

### 5.1 当前架构的技术债务

| 问题 | 位置 | 建议 |
|------|------|------|
| `SimulationConfig` 过于庞大（460行） | `configs/config.py` | 拆分为分层配置类（`EnvConfig`, `DroneConfig`, `RLConfig`），用组合替代单一dataclass |
| `swarm_mission_executor.py` 937行过于臃肿 | `simulation/` | 拆分为4个角色独立的策略模块 + 1个协调器 |
| `drone_ui.py` 915行，逻辑与UI混杂 | `ui/` | Streamlit最佳实践：分离数据处理层和UI渲染层 |
| 全局随机种子硬编码 | `main.py` 等多处 | 统一通过Config注入seed，支持可重复实验 |
| 生成产物（logs/ results/ models/）混入仓库 | 根目录 | 已在 `.gitignore` 排除，但存量文件仍占用空间，建议用 Git LFS 管理模型文件 |

### 5.2 模块化深化方向

```
当前:  configs/ → core/ → simulation/ → analysis/
          ↓         ↓          ↓
        environment/       rl_env/ → rl_training/ → adapters/

建议:  configs/           # 分层配置
       core/              # 算法层（不变）
       environment/       # 环境层（不变）
         ├── terrain/     # DEM/地图独立
         ├── weather/     # 风/风暴/降水独立
         └── sensors/     # 新增传感器模型
       agents/            # 智能体层（新增）
         ├── astar/       # A*规划
         ├── rl/          # RL策略
         └── hybrid/      # 混合策略
       swarm/             # 蜂群层（从simulation拆分）
         ├── roles/       # 各角色独立模块
         ├── comms/       # 通信模型
         └── coordinator/ # 协调器
       ui/                # 表现层（不变）
       experiments/       # 实验层（统一analysis+experiments+v25）
```

---

## 六、总结

`feature/m1-my-update` 分支将一个基础路径规划demo发展成了一个**功能完备的无人机路径规划与RL训练研究平台**。核心创新点——4D时空感知A\*、TKE极值概率风险模型、A\*教师引导RL、异构四机蜂群——构成了一个有学术发表潜力的完整系统。

**短期优先级建议**：
1. 拆分超大型模块（`swarm_mission_executor.py`、`drone_ui.py`、`SimulationConfig`），降低维护成本
2. 集成MLflow/W&B实验追踪，使训练过程可复现可对比
3. 补全单元测试覆盖率（当前仅有8个测试文件）

**中期学术方向建议**：
1. MARL替代手工蜂群规则——这是最自然的升级路径
2. 多地形泛化实验——验证方法在不同地形的迁移能力
3. 与PX4 SITL对接——迈向sim-to-real的第一步

---

*本文档基于项目代码和 git 历史自动分析生成，建议随项目演进而持续更新。*
