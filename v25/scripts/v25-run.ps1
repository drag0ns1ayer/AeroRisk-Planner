param(
    [Parameter(Position = 0)]
    [string]$Mode = "help",

    [Parameter(Position = 1)]
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

$ModelRoot = "v25/artifacts/models"
$LogRoot = "v25/artifacts/logs"

function Show-Help {
    @"
v2.5 shared true-world commands

  v25\scripts\v25-run.cmd test
  v25\scripts\v25-run.cmd astar
  v25\scripts\v25-run.cmd train-smoke
  v25\scripts\v25-run.cmd train
  v25\scripts\v25-run.cmd train-apas
  v25\scripts\v25-run.cmd train-long
  v25\scripts\v25-run.cmd eval <model_path_without_zip>
  v25\scripts\v25-run.cmd compare <model_path_without_zip>

Older 31-dimensional v2.5 models are incompatible with this protocol.
"@
}

function Require-ModelPath {
    if ([string]::IsNullOrWhiteSpace($ModelPath)) {
        throw "Mode '$Mode' requires a model path without .zip."
    }
}

switch ($Mode.ToLower()) {
    "help" {
        Show-Help
    }
    "test" {
        python -m unittest discover -s tests -v
    }
    "astar" {
        python launcher.py astar25
    }
    "train-smoke" {
        python launcher.py rl25 -- --from-scratch --run-name v25_random_v3_smoke --curriculum-stage 1 --total-timesteps 5000 --n-envs 1 --seed 42 --model-save-root "$ModelRoot/ppo_v25_random_v3_smoke" --log-root $LogRoot
    }
    "train" {
        python launcher.py rl25 -- --from-scratch --run-name v25_random_v3_main --curriculum-stage 3 --total-timesteps 200000 --n-envs 2 --seed 42 --model-save-root "$ModelRoot/ppo_v25_random_v3_main" --log-root $LogRoot
    }
    "train-apas" {
        python launcher.py rl25 -- --run-name v25_random_v3_apas --curriculum-stage 3 --total-timesteps 100000 --n-envs 2 --seed 42 --enable-apas --load-model-path "$ModelRoot/ppo_v25_random_v3_main_best/best_model" --model-save-root "$ModelRoot/ppo_v25_random_v3_apas" --log-root $LogRoot
    }
    "train-long" {
        python launcher.py rl25 -- --from-scratch --run-name v25_random_v3_long --curriculum-stage 4 --total-timesteps 500000 --n-envs 4 --seed 42 --model-save-root "$ModelRoot/ppo_v25_random_v3_long" --log-root $LogRoot
    }
    "eval" {
        Require-ModelPath
        python launcher.py rl25 -- --eval-only --eval-model-path $ModelPath --run-name v25_trueworld_eval --seed 42 --model-save-root "$ModelRoot/ppo_v25_trueworld_eval"
    }
    "compare" {
        Require-ModelPath
        python v25/experiments/compare_astar_rl_disruptive.py --rl-model-path $ModelPath --episodes 30 --curriculum-stage 3
    }
    default {
        Show-Help
        throw "Unknown v2.5 mode: $Mode"
    }
}
