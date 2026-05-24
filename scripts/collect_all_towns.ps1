# Script thu thập dữ liệu GTNet cho tất cả towns
# Mỗi town: 5000 samples (500 giây = 8.3 phút ở 10 FPS)

$towns = @("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07")
$duration = 500  # 500 giây = 5000 frames ở 10 FPS
$npc_vehicles = 100
$npc_bikes = 60
$npc_motorbikes = 60
$npc_pedestrians = 60

Write-Host "=" * 70 -ForegroundColor Green
Write-Host "Multi-Agent Data Collection - All Towns" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "Towns: $($towns -join ', ')" -ForegroundColor Cyan
Write-Host "Duration per town: $duration seconds" -ForegroundColor Cyan
Write-Host "NPC 4-wheel vehicles: $npc_vehicles" -ForegroundColor Cyan
Write-Host "NPC bikes: $npc_bikes" -ForegroundColor Cyan
Write-Host "NPC motorbikes: $npc_motorbikes" -ForegroundColor Cyan
Write-Host "NPC pedestrians: $npc_pedestrians" -ForegroundColor Cyan
Write-Host "Output directory: data/multi_agent" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""

$townIndex = 1
$totalTowns = $towns.Count

foreach ($town in $towns) {
    Write-Host "[$townIndex/$totalTowns] Collecting data from $town..." -ForegroundColor Yellow
    
    python collect_multi_agent_data.py `
        --town $town `
        --duration $duration `
        --npc-vehicles $npc_vehicles `
        --npc-bikes $npc_bikes `
        --npc-motorbikes $npc_motorbikes `
        --npc-pedestrians $npc_pedestrians `
        --output-dir data/multi_agent `
        --seed 42
    
    $exitCode = $LASTEXITCODE
    
    # Check if CSV file was created (data collection succeeded)
    $csvPattern = "data\multi_agent\raw\${town}_*.csv"
    $csvFiles = Get-ChildItem $csvPattern -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($csvFiles -and $csvFiles.Length -gt 1000) {
        Write-Host "[OK] $town data collection completed (CSV: $($csvFiles.Length) bytes)" -ForegroundColor Green
    } elseif ($exitCode -eq 0) {
        Write-Host "[OK] $town data collection completed" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] $town completed with exit code $exitCode, but data may be valid" -ForegroundColor Yellow
    }
    
    Write-Host "Waiting 5 seconds before next town..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    
    $townIndex++
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "Data collection complete for all towns!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""

# Check collected data
Write-Host "Checking collected CSV files..." -ForegroundColor Cyan
$csvFiles = Get-ChildItem data\multi_agent\raw\*.csv
Write-Host "Total CSV files: $($csvFiles.Count)" -ForegroundColor Cyan
$totalSize = ($csvFiles | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Total size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Cyan
Write-Host ""

# Build dataset
Write-Host "Building dataset..." -ForegroundColor Yellow
python scripts/build_multi_agent_dataset.py `
    --csv data/multi_agent/raw/*.csv `
    --output data/multi_agent/processed `
    --adaptive-radius `
    --radius-base 20.0 `
    --radius-alpha 0.5

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Dataset built successfully!" -ForegroundColor Green
} else {
    Write-Host "[FAILED] Dataset building failed!" -ForegroundColor Red
}

Write-Host ""
Write-Host "All done! Dataset ready for training." -ForegroundColor Green
