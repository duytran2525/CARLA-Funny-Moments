# PowerShell script to collect multi-agent data from all CARLA towns
# Usage: .\collect_all_towns.ps1

$ErrorActionPreference = "Stop"

$TOWNS = @("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD")
$DURATION = 600  # 10 minutes per town
$NPC_VEHICLES = 40
$OUTPUT_DIR = "data/multi_agent"

Write-Host "=" * 70
Write-Host "Multi-Agent Data Collection - All Towns"
Write-Host "=" * 70
Write-Host "Towns: $($TOWNS -join ', ')"
Write-Host "Duration per town: $DURATION seconds"
Write-Host "NPC vehicles: $NPC_VEHICLES"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host "=" * 70
Write-Host ""

$total_towns = $TOWNS.Count
$completed = 0
$failed = 0

foreach ($town in $TOWNS) {
    $town_num = $completed + 1
    Write-Host ""
    Write-Host "[$town_num/$total_towns] Collecting data from $town..."
    Write-Host ""
    
    try {
        python collect_multi_agent_data.py `
            --town $town `
            --duration $DURATION `
            --npc-vehicles $NPC_VEHICLES `
            --output-dir $OUTPUT_DIR
        
        if ($LASTEXITCODE -eq 0) {
            $completed++
            Write-Host ""
            Write-Host "[SUCCESS] $town data collection completed" -ForegroundColor Green
        } else {
            $failed++
            Write-Host ""
            Write-Host "[FAILED] $town data collection failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }
    catch {
        $failed++
        Write-Host ""
        Write-Host "[ERROR] $town data collection failed: $_" -ForegroundColor Red
    }
    
    # Brief pause between towns
    if ($town_num -lt $total_towns) {
        Write-Host ""
        Write-Host "Waiting 5 seconds before next town..."
        Start-Sleep -Seconds 5
    }
}

Write-Host ""
Write-Host "=" * 70
Write-Host "Collection Summary"
Write-Host "=" * 70
Write-Host "Total towns: $total_towns"
Write-Host "Completed: $completed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host "=" * 70

if ($failed -gt 0) {
    exit 1
}
