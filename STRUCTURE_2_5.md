# 2.5 结构规范（仓库治理）

## 1) 启动入口规范

- 主入口统一为 `launcher.py`
- 推荐命令：
  - `python launcher.py sim`
  - `python launcher.py rl`
  - `python launcher.py rl25`
  - `python launcher.py ui`
  - `python launcher.py desktop`
  - `python launcher.py swarm`
  - `python launcher.py astar25`
  - `python launcher.py ablation --exp all`
  - `python launcher.py test`

## 2) 实验脚本收拢

- 已将根目录实验脚本迁移到 `experiments/`：
  - `experiments/check_coords.py`
  - `experiments/smoke_test.py`
  - `experiments/test_swarm_standalone.py`
- 根目录保留同名兼容壳文件，避免旧命令失效。

## 3) 目录边界

- 核心运行代码：`core/` `environment/` `simulation/` `rl_env/` `rl_training/` `ui/`
- 分析与论文实验：`analysis/` `experiments/`
- v2.5 独立迭代区：`v25/`（新扰动模型、新RL环境、新训练脚本）
- 自动产出目录（不入库）：`logs/` `results/` `mission_outputs/`

## 4) Git 规范补充

- `.gitignore` 已新增运行产物和训练产物规则。
- 日常提交建议只包含：代码、配置、文档、必要测试。
