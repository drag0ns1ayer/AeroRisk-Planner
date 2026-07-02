# AeroRisk Planner

面向复杂山地环境的无人机四维路径规划、随机风险感知、局部安全决策与巡检任务仿真系统。

这个项目从最初的 **A\* + PPO 路径修正**，逐步发展成了一个更完整的 UAV 任务规划实验平台：它既能在真实地形和风场上做四维能耗规划，也能在“可预测层 + 随机风险层”的设定下比较 A\*、APAS 安全盾、Expert 局部决策器和 PPO residual 策略，还能在 V3.0 中对巡检点、充电点和任务优先级进行语义级任务执行仿真。

> 简单说：A\* 负责“往哪走”，APAS 负责“别硬撞”，Expert/RL 负责“看见局部随机风险后怎么更聪明地绕”，V3.0 则把这些能力接进了巡检任务地图。

---

## 当前完成状态

项目现在大致分为三代能力：

### V1.0：基础路径规划与多机雏形

- 基于真实地形/高程图的 A\* 四维路径规划。
- 风场、能耗、爬升、速度、禁飞区、雷暴区等基础建模。
- PPO 对 A\* 路径进行 residual 修正。
- 早期多机协同/状态机式 FANET 执行示意。

这一阶段更像“把很多规划和仿真模块拼起来”，工程展示效果较强，但 RL 的任务分工不够清晰。

### V2.5：可预测层 + 随机层 + 局部安全决策

V2.5 是当前项目的核心技术路线。

- 地图拆分为：
  - **可预测层**：地形、高程、禁飞区、可预测风场、可预测雷暴/风险。
  - **随机层**：局部随机风扰动、不可预测高风险区、破坏性 storm core。
- A\* 只基于可预测层做全局规划。
- 执行阶段在真实地图中暴露随机层。
- 飞机通过有限范围的 local risk observer 获取周围风险信息。
- APAS 作为最终安全盾，对危险动作进行段级安全检查和替换。
- Expert 局部决策器提供 risk membrane、band avoidance、emergency、do-no-harm gate 等局部避险逻辑。
- PPO/BC 策略可以学习 Expert 的 residual 行为，在降低在线计算量的同时保留 APAS 兜底。

### V3.0：地图巡检任务模式

V3.0 在 V2.5 的控制能力上增加了任务语义：

- 可标定起点、返航点、巡检点和充电点。
- 巡检点支持：
  - 权重 / 优先级
  - 服务时间
  - 风险值
  - 可选截止时间
- 充电点支持：
  - 充电速率
  - 服务时间
- 任务执行器会按任务规划顺序飞行、执行巡检、必要时补能。
- 支持把 V2.5 的 A\*、Expert、RL 控制模式接入每一段巡检飞行。
- UI 中新增 “V3.0 地图巡检模式”，支持交互式编辑任务点并生成地形图、风场图、轨迹图和 GIF。

---

## 核心思想

项目最终形成的结构可以概括为：

```text
真实地形 / 风场 / 风险层
        ↓
可预测层建图
        ↓
A* 全局四维路径规划
        ↓
局部执行阶段暴露随机层
        ↓
local risk observer / risk membrane
        ↓
Expert 或 PPO residual 局部修正
        ↓
do-no-harm gate
        ↓
APAS safety shield
        ↓
无人机执行 / 能耗 / 风险 / 任务状态更新
```

其中：

- **A\***：负责可解释、稳定、全局的宏观路径。
- **APAS**：负责最后一层强安全约束，不追求优雅，只负责尽可能避免危险动作。
- **Expert**：负责上层局部避险，目标是提前绕开风险、减少 APAS 强制介入。
- **PPO / BC**：负责学习 Expert 的 residual 策略，作为更轻量的在线控制策略。
- **Waypoint-skip**：解决路径跟踪过程中追逐过期 waypoint 导致 timeout 的问题。
- **Do-no-harm gate**：当 Expert/RL 连续无进展且风险未下降时，回退到 A\* + APAS，避免上层策略“帮倒忙”。

---

## 功能特性

### 地形与地图

- 支持从图片读取地形高程图。
- 默认包含 `Bernese_Oberland_46.6241_8.0413.png` 作为复杂山地地形示例。
- 支持禁飞区、雷暴区、巡检点、充电点、起点、终点/返航点。
- 地图坐标使用米制坐标，UI 中通常以 km 显示。

### 风场与风险

- 可预测风场用于规划。
- 真实执行风场可叠加随机扰动。
- 支持局部风扰动区域。
- 支持破坏性 storm core 和外围 halo 风险预兆。
- 支持基于局部采样的风险观测器。
- 支持 risk membrane / 风险外延层，用于把稀疏风险采样组织成连续风险带。

### 路径规划与控制模式

当前 V3.0 可视化脚本和 UI 支持以下控制模式：

| 模式 | 含义 |
| --- | --- |
| `legacy_astar` | 传统 A\* 分段执行，主要用于兼容早期版本和基础展示 |
| `v25_astar` | V2.5 A\* 基准，使用 A\* reference + APAS + waypoint-skip |
| `v25_expert` | 完整 Expert 方案，使用局部风险决策 + APAS + waypoint-skip |
| `v25_rl` | PPO residual 方案，加载训练好的 PPO 模型并由 APAS 兜底 |

### 安全与评估

- APAS 最终安全盾。
- 段级候选动作检查。
- no-valid candidate 统计。
- 安全干预负担 `safety_intervention_burden`。
- 调整后能耗 `eval_adjusted_energy_j`，用于体现 APAS / emergency 并非“免费动作”。
- failure trace / hazard classification 分析脚本，用于定位失败类型。

### 可视化

V3.0 可输出：

- 可观测层风场图。
- 可观测层 + 随机层真实风场图。
- 地形 + 巡检任务轨迹图。
- 局部 zoom 轨迹图。
- 风场叠加轨迹图。
- 高程剖面图。
- 飞行动画 GIF。
- JSON summary。

### UI

`ui/drone_ui.py` 使用 Streamlit 提供界面：

- 保留早期 FANET / A\* / RL 展示功能。
- 新增 `V3.0 地图巡检模式`。
- 可上传地图。
- 可设置自然环境参数。
- 可编辑巡检点、充电点、起点和返航点。
- 可切换 `legacy_astar` / `v25_astar` / `v25_expert` / `v25_rl`。
- 可展示生成的地形图、风场图、轨迹图和 GIF。

---

## 项目目录

主要目录说明如下：

```text
project1/
├─ core/                         # 基础物理、A* 规划、能耗等核心模块
├─ environment/                  # 地图管理、风场模型
├─ simulation/                   # 早期任务执行器
├─ rl_env/                       # 早期 RL 环境
├─ rl_training/                  # 早期 PPO 训练脚本
├─ analysis/                     # 早期分析与 benchmark
├─ ui/                           # Streamlit 图形界面
│  └─ drone_ui.py
├─ tests/                        # 单元测试与回归测试
├─ v25/                          # V2.5：随机层、APAS、Expert、PPO 实验
│  ├─ experiments/               # 比较实验、消融实验、失败诊断
│  ├─ artifacts/                 # 本地模型和 demo，默认被 .gitignore 忽略
│  └─ scripts/
├─ v30/                          # V3.0：巡检任务地图、语义任务执行、可视化
│  ├─ examples/                  # 巡检任务 JSON 示例
│  ├─ experiments/               # V3.0 命令行可视化入口
│  ├─ segment_executor.py        # 任务段执行器，接入 legacy/v25/RL 控制模式
│  ├─ task_map.py                # 巡检点 / 充电点 / 任务地图数据结构
│  ├─ task_scheduler.py          # 任务调度
│  ├─ mission_executor.py        # 巡检任务执行
│  └─ visualization.py           # 地形、风场、轨迹、GIF 输出
├─ docs/                         # 设计文档和阶段记录
├─ results/                      # 实验输出，默认不提交
├─ Bernese_Oberland_46.6241_8.0413.png
├─ requirements.txt
└─ README.md
```

---

## 环境安装

建议使用 Conda 创建独立环境。

```powershell
conda create -n aerorisk python=3.11 -y
conda activate aerorisk
pip install -r requirements.txt
```

如果你继续使用已有环境，例如：

```powershell
conda activate bcp_env
```

也可以直接运行。

注意：当前依赖中可能出现 Gym 的提示：

```text
Gym has been unmaintained since 2022...
```

这是 Stable-Baselines3 / Gym 版本兼容提示，不代表程序必然失败。后续如果要长期维护，可以把 Gym 迁移到 Gymnasium。

---

## 快速运行测试

推荐先跑完整测试：

```powershell
.\v25\scripts\v25-run.cmd test
```

或者直接使用 unittest：

```powershell
python -m unittest discover -s tests
```

测试中看到类似：

```text
警告: 无法读取地图文件: nonexistent.file，将生成虚拟高斯地形。
```

这是测试故意触发的 fallback 分支，不是错误。

---

## 启动 UI

在项目根目录运行：

```powershell
python -m streamlit run ui\drone_ui.py
```

或者：

```powershell
streamlit run ui\drone_ui.py
```

打开浏览器后，可以进入：

```text
V3.0 地图巡检模式
```

推荐使用流程：

1. 选择 `展示级 Bernese 山地巡检`。
2. 检查起点、返航点、巡检点、充电点。
3. 选择控制模式：
   - 快速展示：`legacy_astar`
   - V2.5 完整 A\* 基线：`v25_astar`
   - 当前推荐工程方案：`v25_expert`
   - PPO 模型方案：`v25_rl`
4. 如果选择 `v25_rl`，填写本地模型路径。
5. 点击运行。
6. 查看 summary、地形轨迹图、风场图、高程图和 GIF。

---

## 命令行运行 V3.0 巡检可视化

### 1. 展示级 Bernese 巡检任务

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_visuals_showcase `
  --gif-frames 12 `
  --max-replans 35 `
  --max-mission-time 3600
```

默认控制模式是 `legacy_astar`。

### 2. 使用 V2.5 A\* + APAS + waypoint-skip

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_visuals_v25_astar `
  --control-mode v25_astar `
  --gif-frames 12 `
  --max-replans 35 `
  --max-mission-time 3600
```

### 3. 使用完整 Expert 方案

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_visuals_v25_expert `
  --control-mode v25_expert `
  --gif-frames 12 `
  --max-replans 35 `
  --max-mission-time 3600
```

### 4. 使用 PPO residual 方案

PPO 模型文件默认位于本地 `v25\artifacts\models\...` 下。这个目录默认被 `.gitignore` 忽略，不会随 GitHub 仓库提交。

示例：

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_visuals_v25_rl `
  --control-mode v25_rl `
  --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip `
  --gif-frames 12 `
  --max-replans 35 `
  --max-mission-time 3600
```

如果没有本地模型，请先使用 `legacy_astar`、`v25_astar` 或 `v25_expert`。

---

## 输出文件说明

运行 V3.0 可视化脚本后，输出目录通常包含：

| 文件 | 含义 |
| --- | --- |
| `summary.json` | 本次任务执行摘要 |
| `observable_wind_field.png` | 可观测层风场 |
| `true_wind_field.png` | 可观测层 + 随机层真实风场 |
| `observable_wind_trajectory.png` | 可观测层风场 + 轨迹 |
| `true_wind_trajectory.png` | 真实风场 + 轨迹 |
| `mission_terrain_trajectory.png` | 地形图 + 巡检点 + 充电点 + 轨迹 |
| `mission_terrain_zoom.png` | 地形轨迹局部放大图 |
| `mission_elevation_profile.png` | 飞行高程剖面 |
| `true_wind_trajectory.gif` | 轨迹动画 |

这些输出默认位于 `results/` 下，通常不建议提交到 Git。

---

## 巡检任务地图 JSON

V3.0 的任务地图可以参考：

```text
v30/examples/mission_map_showcase.json
```

一个简化示例如下：

```json
{
  "map_file": "Bernese_Oberland_46.6241_8.0413.png",
  "start": {
    "x_m": -7800,
    "y_m": -7800
  },
  "home": {
    "x_m": -7800,
    "y_m": -7800
  },
  "inspection_points": [
    {
      "id": "central-pass-sensor",
      "x_m": 700,
      "y_m": 2100,
      "priority": 1.4,
      "service_time_s": 55,
      "risk": 0.18,
      "deadline_s": 1800
    }
  ],
  "charging_stations": [
    {
      "id": "south-charger",
      "x_m": -4200,
      "y_m": -3600,
      "charge_rate_w": 850,
      "service_time_s": 90
    }
  ]
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `map_file` | 地形图路径 |
| `start` | 起点 |
| `home` | 返航点 |
| `inspection_points` | 巡检点列表 |
| `priority` | 巡检优先级，越高越重要 |
| `service_time_s` | 到点后执行巡检所需时间 |
| `risk` | 巡检点自身风险，可用于调度和评估 |
| `deadline_s` | 可选截止时间 |
| `charging_stations` | 充电点列表 |
| `charge_rate_w` | 充电功率 |

---

## V2.5 常用实验命令

### 最终消融实验

```powershell
python v25\experiments\final_ablation_v25.py --episodes 30 --seed 200 --stress fragile
```

最终消融通常比较：

- 裸 A\*
- A\* + waypoint-skip
- A\* + APAS
- A\* + APAS + waypoint-skip
- A\* + Expert + APAS + waypoint-skip

### Hazard classification

```powershell
python v25\experiments\hazard_type_classification.py `
  --episodes 100 `
  --seed 42 `
  --methods astar expert `
  --curriculum-stage 3 `
  --stress fragile `
  --planner-max-steps 15000
```

### Failure trace

```powershell
python v25\experiments\failure_trace.py `
  --episodes 100 `
  --seed 42 `
  --methods astar_apas expert_apas `
  --curriculum-stage 3 `
  --stress fragile `
  --planner-max-steps 15000
```

### PPO 从 Expert demo 行为克隆预训练

```powershell
python v25\experiments\pretrain_ppo_from_expert.py `
  --demo-npz v25\artifacts\expert_demos\expert_demo_20260628_202504.npz `
  --run-name ppo_v25_expert_bc_s200 `
  --epochs 25 `
  --batch-size 256 `
  --curriculum-stage 3 `
  --stress fragile
```

### PPO fine-tune

```powershell
python v25\train_rl_disruptive.py `
  --run-name ppo_v25_expert_bc_s200_ft50k `
  --total-timesteps 50000 `
  --curriculum-stage 3 `
  --stress fragile `
  --init-model-path v25\artifacts\models\ppo_v25_expert_bc_s200.zip
```

具体参数可能随实验脚本更新而变化，可以用：

```powershell
python v25\train_rl_disruptive.py --help
```

查看当前版本支持的参数。

---

## 关键指标怎么读

| 指标 | 含义 |
| --- | --- |
| `success_rate` | 任务成功率 |
| `energy_used_j_mean` | 原始飞行能耗 |
| `eval_adjusted_energy_j_mean` | 加入安全干预负担后的评估能耗 |
| `safety_intervention_burden_mean` | 安全干预负担 |
| `avg_risk_mean` | 平均风险 |
| `peak_risk_mean` | 峰值风险 |
| `apas_interventions_mean` | APAS 强制接管次数 |
| `apas_segment_rejections_mean` | APAS 拒绝危险候选段次数 |
| `apas_no_valid_candidates_mean` | APAS 没有完全安全候选动作的次数 |
| `stale_waypoint_skips_mean` | 过期 waypoint 跳过次数 |
| `expert_band_avoidance_steps_mean` | Expert 风险带绕行步数 |
| `expert_emergency_steps_mean` | Expert emergency 步数 |
| `destructive_core_hits_mean` | 破坏性 storm core 命中情况 |

一个重要结论是：不要只看成功率。

例如 A\* + APAS 可能成功率很高，但 APAS 介入次数和 segment rejection 很多；Expert 的价值经常体现在：

- 成功率持平或略高。
- 平均/峰值风险下降。
- APAS 介入显著减少。
- 调整后任务代价更低。

---

## 重要设计记录

### 为什么不是只用 A\*？

A\* 在可预测层上非常可靠，但它不知道执行阶段暴露出来的随机层。如果随机层足够温和，A\* + APAS 就已经很强；如果随机层出现连续高风险带、局部风扰动、破坏性 core，单纯 A\* 会缺少提前规避能力。

### 为什么 APAS 很强，但仍然需要 Expert？

APAS 是最后安全盾，它能救命，但不应该被当成免费主控制器。频繁 APAS 代表：

- 上层路径/动作不够安全。
- 控制动作更突兀。
- 任务中断和安全负担更高。
- 评估能耗和安全干预代价应当增加。

Expert 的价值不是永远大幅提高成功率，而是尽量让系统少依赖最终兜底。

### 为什么 PPO 没有直接成为最强？

这个项目里的 PPO 更适合学习 residual 局部策略，例如更平滑、更轻量、更低在线计算。避障不是“奖励一写就会”的问题，它需要：

- 看得见风险。
- 奖励能提前区分风险。
- 训练集中反复出现类似危险。
- APAS/安全盾兜底。
- do-no-harm gate 防止上层策略帮倒忙。

因此当前更成熟的叙事是：

```text
A* 提供全局可解释路径；
Expert 提供可靠但计算更重的局部决策；
PPO/BC 学习 Expert，提供轻量化 residual 策略；
APAS 负责最终安全边界。
```

### 为什么要 waypoint-skip？

路径跟踪中可能出现飞机已经越过某个 waypoint，但系统仍要求它回去追旧 waypoint 的情况。这会导致：

- 任务进度不涨。
- 飞机在局部来回磨。
- 最后 timeout。

Waypoint-skip 通过最近路径点投影和过期 waypoint 判断，让路径进度可以向前跳，解决了一类非策略性 timeout。

### 为什么要 do-no-harm gate？

Expert/RL 并不总是有益。如果上层策略连续几步没有带来进展，而且风险没有下降，同时 APAS 已经频繁介入，就说明上层策略可能正在帮倒忙。此时系统会衰减 residual 或回退到 A\* + APAS。

---

## GitHub 提交指南

### 1. 查看当前状态

```powershell
git status
```

### 2. 确认远程仓库

```powershell
git remote -v
```

如果你要换成自己账号下的新仓库：

```powershell
git remote set-url origin https://github.com/<你的用户名>/<你的仓库名>.git
```

例如：

```powershell
git remote set-url origin https://github.com/your-name/AeroRisk-Planner.git
```

### 3. 添加本次改动

如果只想提交本次主要代码和 README：

```powershell
git add README.md `
  core/planner.py `
  tests/test_v30_task_scheduler.py `
  ui/drone_ui.py `
  v30/experiments/render_task_map_visuals.py `
  v30/segment_executor.py `
  v30/visualization.py `
  v30/examples/mission_map_showcase.json
```

如果你确认所有修改都要提交，也可以：

```powershell
git add .
```

### 4. 提交 commit

```powershell
git commit -m "feat: finalize v30 inspection mission workflow"
```

或者如果这次主要是文档：

```powershell
git commit -m "docs: add complete project readme"
```

### 5. 推送当前分支

当前分支如果是：

```text
refactor/v25-structure
```

推送：

```powershell
git push -u origin refactor/v25-structure
```

### 6. 合并到 main

如果你想把当前分支合进主分支：

```powershell
git checkout main
git pull origin main
git merge refactor/v25-structure
git push origin main
```

如果 GitHub 上习惯走 Pull Request，则推送分支后在 GitHub 页面创建 PR，再合并到 `main`。

---

## Git 提交注意事项

`.gitignore` 默认忽略了很多实验输出和模型文件：

- `results/`
- `logs/`
- `mission_outputs/`
- `v25/artifacts/`
- `*.zip`
- GIF / MP4 等大输出文件

因此：

- 训练好的 PPO 模型通常不会被普通 git push 上传。
- 大型可视化结果不会污染仓库。
- 如果确实要发布模型，建议使用 GitHub Releases 或 Git LFS，不建议直接塞进普通 Git 历史。

查看哪些文件会被提交：

```powershell
git status --short
```

查看改动概要：

```powershell
git diff --stat
```

查看 staged 内容：

```powershell
git diff --cached --stat
```

---

## 常见问题

### 1. 为什么 RL 模式找不到模型？

因为 `v25/artifacts/` 默认被 `.gitignore` 忽略。你需要在本地保留模型，并通过 `--rl-model-path` 或 UI 输入框指定。

### 2. 为什么测试里有地图读取警告？

测试故意使用 `nonexistent.file` 检查 fallback 地形生成逻辑，只要最终 `OK` 就没问题。

### 3. 为什么 A\* + APAS 已经很强，还要 Expert？

A\* + APAS 可以作为强基线。Expert 的价值更多体现在降低风险、降低 APAS 介入、降低安全干预负担，而不一定每次都显著提高 raw success rate。

### 4. 这个项目能直接用于真实无人机吗？

不能直接用于真实飞控。当前系统是研究和仿真平台，不是经过实机安全认证的飞行控制软件。真实部署还需要传感器建模、控制器接口、通信链路、实机动力学辨识、冗余安全机制和大量验证。

### 5. ROS2 仿真难吗？

可行，但工作量不小。更合理的路线是：

1. 先把当前规划/决策模块稳定成 Python API。
2. 再封装 ROS2 node。
3. 接 Gazebo / PX4 / AirSim 等仿真平台。
4. 最后做话题接口、坐标变换、实时性和安全边界。

---

## 后续可以继续做的方向

- 在 UI 中支持点击地图直接添加巡检点和充电点。
- V3.0 支持多无人机任务分配。
- 更正式的电池模型和充电策略。
- 更真实的传感器模型，而不是抽象 risk observer。
- PPO 使用 RNN / history observation 处理动态风险。
- ROS2 / Gazebo / PX4 联合仿真。
- 将模型权重发布到 GitHub Releases。
- 整理最终答辩版图表和论文式实验表格。

---

## 一句话总结

AeroRisk Planner 现在已经从一个“A\* + RL 修正”的路径规划原型，演化成了一个包含真实地形、风场风险、随机层感知、局部安全决策、PPO residual 学习和语义巡检任务执行的完整无人机规划仿真平台。

它不只是能跑一条路径，而是能讲清楚：

```text
全局规划怎么来，
随机风险怎么暴露，
局部决策怎么介入，
安全盾怎么兜底，
任务点怎么执行，
失败原因怎么诊断。
```

这就是这个项目最有价值的地方。
