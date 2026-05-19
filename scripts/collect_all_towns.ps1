# Script thu thập dữ liệu GTNet cho tất cả towns
# Mỗi town: 5000 samples (500 giây = 8.3 phút ở 10 FPS)

$towns = @("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07")
$duration = 500  # 500 giây = 5000 frames ở 10 FPS
$npc_vehicles = 40

Write-Host "=== Bắt đầu thu thập dữ liệu GTNet ===" -ForegroundColor Green
Write-Host "Towns: $($towns -join ', ')" -ForegroundColor Cyan
Write-Host "Duration per town: $duration seconds (5000 frames)" -ForegroundColor Cyan
Write-Host "NPC vehicles: $npc_vehicles" -ForegroundColor Cyan
Write-Host ""

foreach ($town in $towns) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Thu thập dữ liệu cho $town" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    python collect_multi_agent_data.py `
        --town $town `
        --duration $duration `
        --npc-vehicles $npc_vehicles `
        --output-dir data/multi_agent `
        --seed 42
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "LỖI: Thu thập $town thất bại!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Hoàn thành $town" -ForegroundColor Green
    Write-Host ""
    
    # Nghỉ 5 giây giữa các towns
    Start-Sleep -Seconds 5
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Hoàn thành thu thập tất cả towns!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Build dataset
Write-Host ""
Write-Host "Bắt đầu build dataset..." -ForegroundColor Cyan

python scripts/build_multi_agent_dataset.py `
    --csv data/multi_agent/raw/*.csv `
    --output data/multi_agent/processed `
    --adaptive-radius `
    --radius-base 20.0 `
    --radius-alpha 0.5

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Build dataset thành công!" -ForegroundColor Green
} else {
    Write-Host "LỖI: Build dataset thất bại!" -ForegroundColor Red
    exit 1
}
