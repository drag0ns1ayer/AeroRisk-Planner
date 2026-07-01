# Scripts

这个目录放项目级别的运行包装脚本。脚本只负责把稳定命令串起来，不承载核心算法逻辑。

## V2.5 最小验证

只跑编译检查和单元测试：

```powershell
.\scripts\run_v25_final_validation.ps1 -SkipSlow
```

跑测试、seed 43 病理检查和一轮 30 episodes 对比：

```powershell
.\scripts\run_v25_final_validation.ps1 -Episodes 30
```

如果模型路径不同，可以显式传入：

```powershell
.\scripts\run_v25_final_validation.ps1 -ModelPath "v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip"
```

## 原则

- 脚本可以改命令组合，但不要把实验逻辑写进脚本。
- 算法变化应发生在 `v25/` 或后续重构后的包目录中。
- 脚本默认从仓库根目录运行。

