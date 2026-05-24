# Example PowerShell script for running ablation study
# This script demonstrates how to run the ablation study with different configurations

# Activate virtual environment (adjust path as needed)
# & .venv-1\Scripts\Activate.ps1

Write-Host "GTNet Ablation Study Examples" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

# Example 1: Quick smoke test
Write-Host "Example 1: Quick Smoke Test (5 epochs, 200 samples)" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Green
Write-Host "python scripts/run_ablation_study.py --data-dir data/processed/multi_agent --out-dir ablation_results/quick_test --quick-ablation" -ForegroundColor White
Write-Host ""

# Example 2: Full ablation study on single town
Write-Host "Example 2: Full Ablation Study (30 epochs, all samples)" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Green
Write-Host "python scripts/run_ablation_study.py --data-dir data/processed/town01 --out-dir ablation_results/town01 --epochs 30 --batch-size 16 --seed 42" -ForegroundColor White
Write-Host ""

# Example 3: Ablation study with multiple towns
Write-Host "Example 3: Multi-Town Ablation Study" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Green
Write-Host "python scripts/run_ablation_study.py --data-dir data/processed/town01 data/processed/town02 data/processed/town03 --out-dir ablation_results/multi_town --epochs 30" -ForegroundColor White
Write-Host ""

# Example 4: GPU training with larger batch size
Write-Host "Example 4: GPU Training with Larger Batch" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Green
Write-Host "python scripts/run_ablation_study.py --data-dir data/processed/multi_agent --out-dir ablation_results/gpu --device cuda --batch-size 32 --epochs 30" -ForegroundColor White
Write-Host ""

# Example 5: Limited samples for testing
Write-Host "Example 5: Limited Samples Test" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Green
Write-Host "python scripts/run_ablation_study.py --data-dir data/processed/multi_agent --out-dir ablation_results/limited --limit-samples 500 --epochs 10" -ForegroundColor White
Write-Host ""

# Prompt user to run an example
Write-Host "Select an example to run (1-5), or press Enter to exit:" -ForegroundColor Cyan
$choice = Read-Host

switch ($choice) {
    "1" {
        Write-Host "Running Example 1: Quick Smoke Test..." -ForegroundColor Green
        python scripts/run_ablation_study.py `
            --data-dir data/processed/multi_agent `
            --out-dir ablation_results/quick_test `
            --quick-ablation
    }
    "2" {
        Write-Host "Running Example 2: Full Ablation Study..." -ForegroundColor Green
        python scripts/run_ablation_study.py `
            --data-dir data/processed/town01 `
            --out-dir ablation_results/town01 `
            --epochs 30 `
            --batch-size 16 `
            --seed 42
    }
    "3" {
        Write-Host "Running Example 3: Multi-Town Ablation Study..." -ForegroundColor Green
        python scripts/run_ablation_study.py `
            --data-dir data/processed/town01 data/processed/town02 data/processed/town03 `
            --out-dir ablation_results/multi_town `
            --epochs 30
    }
    "4" {
        Write-Host "Running Example 4: GPU Training..." -ForegroundColor Green
        python scripts/run_ablation_study.py `
            --data-dir data/processed/multi_agent `
            --out-dir ablation_results/gpu `
            --device cuda `
            --batch-size 32 `
            --epochs 30
    }
    "5" {
        Write-Host "Running Example 5: Limited Samples Test..." -ForegroundColor Green
        python scripts/run_ablation_study.py `
            --data-dir data/processed/multi_agent `
            --out-dir ablation_results/limited `
            --limit-samples 500 `
            --epochs 10
    }
    default {
        Write-Host "Exiting..." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
