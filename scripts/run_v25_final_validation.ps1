param(
    [int]$Episodes = 30,
    [int]$Seed = 200,
    [string]$ModelPath = "v25\artifacts\models\ppo_v25_expert_bc_s200_ft50k_best\best_model.zip",
    [switch]$SkipSlow
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "== V2.5 validation =="
Write-Host "Project root: $ProjectRoot"
Write-Host "Episodes:     $Episodes"
Write-Host "Seed:         $Seed"
Write-Host "Model path:   $ModelPath"
Write-Host ""

Write-Host "== Python syntax check =="
python -m py_compile `
    configs\config.py `
    v25\rl_env_disruptive.py `
    v25\train_rl_disruptive.py `
    v25\experiments\compare_astar_rl_disruptive.py `
    v25\experiments\final_ablation_v25.py `
    v25\experiments\diagnose_failure_traces.py

Write-Host ""
Write-Host "== Unit tests =="
.\v25\scripts\v25-run.cmd test

if ($SkipSlow) {
    Write-Host ""
    Write-Host "SkipSlow enabled; slow validation commands were not run."
    exit 0
}

if (-not (Test-Path $ModelPath)) {
    throw "Model path does not exist: $ModelPath"
}

Write-Host ""
Write-Host "== Seed 43 do-no-harm pathology check =="
python v25\experiments\diagnose_failure_traces.py `
    --seeds 43 `
    --methods astar_apas expert_apas rl_apas `
    --rl-model-path $ModelPath `
    --curriculum-stage 3 `
    --stress fragile `
    --planner-max-steps 15000 `
    --last-steps 120 `
    --output-dir results\failure_traces_do_no_harm_seed43

Write-Host ""
Write-Host "== V2.5 comparison smoke run =="
python v25\experiments\compare_astar_rl_disruptive.py `
    --rl-model-path $ModelPath `
    --episodes $Episodes `
    --seed $Seed `
    --curriculum-stage 3 `
    --stress fragile `
    --expert `
    --astar-apas `
    --expert-apas `
    --rl-apas `
    --planner-max-steps 15000

Write-Host ""
Write-Host "V2.5 validation finished."
