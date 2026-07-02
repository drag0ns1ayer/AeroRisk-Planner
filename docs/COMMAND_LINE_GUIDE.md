# AeroRisk Planner 命令行使用说明

这份文档是给自己和后续维护者看的“照着抄就能跑”版本。项目现在命令比较多，核心原则是：

```text
日常展示：跑 v30/experiments/render_task_map_visuals.py
方法对比：切换 --control-mode
想要 GIF 更顺：调 --gif-frames
想用 PPO：额外传 --rl-model-path
```

---

## 0. 进入项目目录

PowerShell：

```powershell
cd C:\Users\20340\Desktop\project1
conda activate bcp_env
```

Git Bash：

```bash
cd /c/Users/20340/Desktop/project1
conda activate bcp_env
```

如果你用的是另一个环境，把 `bcp_env` 换成自己的环境名即可。

---

## 1. 启动 UI

最简单的图形界面启动方式：

```powershell
python -m streamlit run ui\drone_ui.py
```

进入 UI 后使用：

```text
V3.0 地图巡检模式
```

UI 里能调的核心参数：

- 控制模式：`legacy_astar` / `v25_astar` / `v25_expert` / `v25_rl`
- GIF 帧数：建议 60
- APAS 是否开启
- stress 难度
- 最大重规划次数
- 最大任务时间
- PPO 模型路径

---

## 2. V3.0 巡检任务最常用命令

### 2.1 展示级 Bernese 巡检任务，传统 A\*

这个适合快速生成完整图：

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_showcase_legacy `
  --control-mode legacy_astar `
  --gif-frames 60 `
  --max-replans 40 `
  --max-mission-time 4200
```

一行版：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_legacy --control-mode legacy_astar --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

### 2.2 V2.5 A\* + APAS + waypoint-skip

这个是当前强基线：

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_showcase_v25_astar `
  --control-mode v25_astar `
  --gif-frames 60 `
  --max-replans 40 `
  --max-mission-time 4200
```

一行版：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_v25_astar --control-mode v25_astar --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

### 2.3 完整 Expert 方案

这是当前工程叙事最完整的方案：

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_showcase_v25_expert `
  --control-mode v25_expert `
  --gif-frames 60 `
  --max-replans 40 `
  --max-mission-time 4200
```

一行版：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_v25_expert --control-mode v25_expert --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

### 2.4 PPO / RL 方案

这个需要本地有模型文件。模型通常在 `v25\artifacts\models\...`，默认不会提交到 GitHub。

```powershell
python v30\experiments\render_task_map_visuals.py `
  --mission-map v30\examples\mission_map_showcase.json `
  --output-dir results\v30_showcase_v25_rl `
  --control-mode v25_rl `
  --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip `
  --gif-frames 60 `
  --max-replans 40 `
  --max-mission-time 4200
```

一行版：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_v25_rl --control-mode v25_rl --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

---

## 3. 四种 control-mode 怎么选

| 模式 | 说明 | 什么时候用 |
| --- | --- | --- |
| `legacy_astar` | 旧版 A\* 分段巡检执行，考虑地形、NFZ、可预测风场、V1.0 可预测移动风暴 | 快速展示、V1.0 路线展示 |
| `v25_astar` | A\* reference + APAS + waypoint-skip | 强基线，对比 Expert/RL |
| `v25_expert` | Expert 局部风险决策 + APAS + waypoint-skip | 当前推荐方案，展示完整工程闭环 |
| `v25_rl` | PPO residual + APAS + waypoint-skip | 展示 RL 学习 Expert 后的轻量策略 |

一句话：

```text
想稳妥展示：legacy_astar
想展示 v2.5 安全盾：v25_astar
想展示完整成果：v25_expert
想展示 RL：v25_rl
```

---

## 4. 参数解释

### `--mission-map`

任务地图 JSON。

推荐：

```powershell
--mission-map v30\examples\mission_map_showcase.json
```

### `--output-dir`

输出目录。

例如：

```powershell
--output-dir results\v30_showcase_v25_expert
```

输出目录里会生成：

```text
mission_map.json
summary.json
mission_terrain_trajectory.png
mission_terrain_zoom.png
observable_wind_field.png
true_wind_field.png
observable_wind_trajectory.png
true_wind_trajectory.png
mission_elevation_profile.png
true_wind_trajectory.gif
```

### `--control-mode`

控制模式：

```text
legacy_astar
v25_astar
v25_expert
v25_rl
```

### `--gif-frames`

GIF 帧数。

建议：

| 场景 | 帧数 |
| --- | --- |
| 快速测试 | 4 ~ 10 |
| 普通查看 | 30 |
| 展示推荐 | 60 |
| 更顺滑展示 | 90 |

例如：

```powershell
--gif-frames 60
```

注意：之前 smoke test 里用过 `--gif-frames 4`，所以 GIF 只有 4 帧。那不是坏了，是为了快。

### `--max-replans`

每段任务允许的最大重规划次数。

巡检任务比较长，推荐：

```powershell
--max-replans 40
```

如果任务特别长，可以加到 60。

### `--max-mission-time`

最大任务时间，单位秒。

展示级任务推荐：

```powershell
--max-mission-time 4200
```

如果加返航后失败原因是时间不够，可以调大，比如：

```powershell
--max-mission-time 5400
```

### `--mission-update-interval`

任务段执行时每次推进/重规划的间隔，单位秒。

默认一般够用。想更细腻但更慢，可以：

```powershell
--mission-update-interval 45
```

想更快但更粗，可以：

```powershell
--mission-update-interval 90
```

### `--seed`

随机种子。

同一个 seed 会生成同样的可预测移动风暴和随机层。

例如：

```powershell
--seed 42
```

换 seed 可以换一套风场/随机层：

```powershell
--seed 200
```

### `--stress`

V2.5 随机层压力等级：

```text
normal
hard
extreme
fragile
```

推荐展示：

```powershell
--stress fragile
```

如果只是想先跑通：

```powershell
--stress normal
```

### `--rl-model-path`

仅 `--control-mode v25_rl` 需要。

例如：

```powershell
--rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip
```

如果模型不存在，`v25_rl` 会报错。可以先用 `v25_expert`。

### `--no-apas`

关闭 APAS。

一般不建议展示时关闭。只在消融实验中使用：

```powershell
--no-apas
```

---

## 5. 图层说明

V3.0 可视化里现在有两层风险叙事。

### 可预测层

| 图层 | 含义 |
| --- | --- |
| `Predictable NFZ` | 预设静态禁飞区，红色斜线圆，A\* 会硬避开 |
| `Forecast moving storm` | V1.0 旧版可预测移动风暴，蓝紫色圆 + 虚线箭头，A\* 通过预测风场/TKE/风险考虑它 |

### 随机层

| 图层 | 含义 |
| --- | --- |
| `Random wind region` | V2.5 随机局部风扰区，青色圆，能穿但有扰动代价 |
| `Random storm halo` | V2.5 破坏性随机雷暴外围预警层 |
| `Destructive storm core` | V2.5 破坏性随机雷暴核心，最危险 |

一句话：

```text
Forecast moving storm 是旧版可预测雷暴。
Random storm halo/core 是 v2.5 随机层破坏性雷暴。
```

---

## 6. PowerShell 和 Git Bash 换行区别

PowerShell 用反引号换行：

```powershell
python script.py `
  --arg1 value1 `
  --arg2 value2
```

Git Bash 用反斜杠换行：

```bash
python script.py \
  --arg1 value1 \
  --arg2 value2
```

如果在 Git Bash 里复制 PowerShell 的反引号写法，可能会出现一堆：

```text
import: command not found
syntax error near unexpected token
```

所以不确定终端类型时，最稳的是用“一行版命令”。

---

## 7. 推荐日常命令集合

### 快速生成一套完整展示图

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_legacy --control-mode legacy_astar --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

### 生成 Expert 完整方案

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_v25_expert --control-mode v25_expert --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

### 生成 RL 方案

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_showcase_v25_rl --control-mode v25_rl --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

### 只想快速确认程序没坏

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_smoke --control-mode legacy_astar --gif-frames 4 --max-replans 40 --max-mission-time 4200
```

---

## 8. 测试命令

完整测试：

```powershell
.\v25\scripts\v25-run.cmd test
```

只测 V3.0：

```powershell
python -m unittest tests.test_v30_task_scheduler
```

语法检查：

```powershell
python -m py_compile ui\drone_ui.py v30\visualization.py v30\task_executor.py v30\experiments\render_task_map_visuals.py
```

---

## 9. 常见问题

### GIF 为什么只有 4 帧？

因为命令里写了：

```powershell
--gif-frames 4
```

改成：

```powershell
--gif-frames 60
```

### 图上为什么有两种 storm？

因为项目现在有两套雷暴：

```text
Forecast moving storm：V1.0 可预测移动雷暴，属于可预测层。
Random storm halo/core：V2.5 随机破坏性雷暴，属于随机层。
```

### 为什么青色 Random wind region 可以穿过去？

因为它不是硬禁飞区，只是随机风扰区。穿过去会有扰动/风险/能耗影响，但不是必死。

### 为什么 PPO 模式报模型不存在？

模型在 `v25/artifacts/`，这个目录默认被 `.gitignore` 忽略。需要本地有模型文件，或者先用 `v25_expert`。

### 为什么命令很慢？

可能原因：

- `--gif-frames` 太大。
- `v25_expert` / `v25_rl` 比 `legacy_astar` 更重。
- `max-replans` 大，任务长。
- 真实 Bernese 地形比小地图复杂。

先用：

```powershell
--gif-frames 10
```

确认没问题，再生成 60 帧展示版。

---

## 10. 最推荐的展示组合

如果只放一套结果：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_final_expert_showcase --control-mode v25_expert --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

如果要对比：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_final_legacy --control-mode legacy_astar --gif-frames 60 --max-replans 40 --max-mission-time 4200

python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_final_expert --control-mode v25_expert --gif-frames 60 --max-replans 40 --max-mission-time 4200
```

如果还要加 RL：

```powershell
python v30\experiments\render_task_map_visuals.py --mission-map v30\examples\mission_map_showcase.json --output-dir results\v30_final_rl --control-mode v25_rl --rl-model-path v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip --gif-frames 60 --max-replans 40 --max-mission-time 4200
```
