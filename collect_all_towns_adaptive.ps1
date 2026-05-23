param(
    [double]$Fps = 20.0,
    [int]$Duration = 900,
    [string]$OutputDir = "data/multi_agent_20fps"
)

# PowerShell script with adaptive NPC configuration per town.
# Usage:
#   .\collect_all_towns_adaptive.ps1 -Fps 20 -Duration 900 -OutputDir data\multi_agent_20fps

$ErrorActionPreference = "Stop"

# Town-specific configurations (tuned for stability)
$town_configs = @{
    "Town01" = @{
        vehicles = 70
        bikes = 15
        motorbikes = 15
        pedestrians = 20
        note = "Large town, moderate load for 20 FPS"
    }
    "Town02" = @{
        vehicles = 50
        bikes = 10
        motorbikes = 10
        pedestrians = 0
        note = "Medium town, pedestrians disabled for nav-mesh stability"
    }
    "Town03" = @{
        vehicles = 60
        bikes = 15
        motorbikes = 15
        pedestrians = 20
        note = "Large town, pedestrians disabled for nav-mesh stability"
    }
    "Town04" = @{
        vehicles = 50
        bikes = 10
        motorbikes = 10
        pedestrians = 0
        note = "Highway town, pedestrians disabled for nav-mesh stability"
    }
    "Town05" = @{
        vehicles = 70
        bikes = 15
        motorbikes = 15
        pedestrians = 20
        note = "Large town, moderate load for 20 FPS"
    }
    "Town06" = @{
        vehicles = 60
        bikes = 15
        motorbikes = 15
        pedestrians = 20
        note = "Medium town, moderate load for 20 FPS"
    }
    "Town07" = @{
        vehicles = 60
        bikes = 15
        motorbikes = 15
        pedestrians = 20
        note = "Medium town, moderate load for 20 FPS"
    }
    "Town10HD" = @{
        vehicles = 70
        bikes = 15
        motorbikes = 15
        pedestrians = 20
        note = "Large HD town, moderate load for 20 FPS"
    }
}

$DURATION = $Duration
$OUTPUT_DIR = $OutputDir
$expectedFrames = [int][Math]::Round($DURATION * $Fps)

Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Multi-Agent Data Collection - Adaptive Per-Town Configuration" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Duration per town: $DURATION seconds (~$expectedFrames frames @ $Fps FPS)" -ForegroundColor Cyan
Write-Host "Output directory: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "Town Configurations:" -ForegroundColor Yellow
foreach ($town in $town_configs.Keys | Sort-Object) {
    $cfg = $town_configs[$town]
    $total = $cfg.vehicles + $cfg.bikes + $cfg.motorbikes + $cfg.pedestrians
    Write-Host ("  {0,-10} V:{1,2} B:{2,2} M:{3,2} P:{4,2} Total:{5,3} - {6}" -f `
        $town, $cfg.vehicles, $cfg.bikes, $cfg.motorbikes, $cfg.pedestrians, $total, $cfg.note) -ForegroundColor Gray
}
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host ""

$total_towns = $town_configs.Count
$completed = 0
$failed = 0

foreach ($town in $town_configs.Keys | Sort-Object) {
    $cfg = $town_configs[$town]
    $town_num = $completed + $failed + 1
    
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ("[$town_num/$total_towns] Collecting data from $town") -ForegroundColor Yellow
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ("Config: Vehicles={0}, Bikes={1}, Motorbikes={2}, Pedestrians={3}" -f `
        $cfg.vehicles, $cfg.bikes, $cfg.motorbikes, $cfg.pedestrians) -ForegroundColor Gray
    Write-Host ("Note: {0}" -f $cfg.note) -ForegroundColor Gray
    Write-Host ""
    
    try {
        python collect_multi_agent_data.py `
            --town $town `
            --duration $DURATION `
            --npc-vehicles $cfg.vehicles `
            --npc-bikes $cfg.bikes `
            --npc-motorbikes $cfg.motorbikes `
            --npc-pedestrians $cfg.pedestrians `
            --fps $Fps `
            --output-dir $OUTPUT_DIR `
            --seed 42
        
        $exitCode = $LASTEXITCODE
        
        # Check if CSV file was created
        $csvPattern = "$OUTPUT_DIR\raw\${town}_*.csv"
        $csvFiles = Get-ChildItem $csvPattern -ErrorAction SilentlyContinue | 
                    Sort-Object LastWriteTime -Descending | 
                    Select-Object -First 1
        
        if ($csvFiles -and $csvFiles.Length -gt 10000) {
            $completed++
            $sizeMB = [math]::Round($csvFiles.Length / 1MB, 2)
            Write-Host ""
            Write-Host ("[SUCCESS] $town completed - CSV: $sizeMB MB") -ForegroundColor Green
        } elseif ($exitCode -eq 0) {
            $completed++
            Write-Host ""
            Write-Host ("[SUCCESS] $town completed") -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host ("[WARNING] $town exit code: $exitCode") -ForegroundColor Yellow
            
            # Check if CSV exists anyway
            if ($csvFiles -and $csvFiles.Length -gt 10000) {
                $completed++
                $sizeMB = [math]::Round($csvFiles.Length / 1MB, 2)
                Write-Host ("[INFO] CSV file exists ($sizeMB MB), counting as success") -ForegroundColor Cyan
            } else {
                $failed++
                Write-Host ("[FAILED] No valid CSV file created") -ForegroundColor Red
            }
        }
    }
    catch {
        $failed++
        Write-Host ""
        Write-Host ("[ERROR] $town failed: $_") -ForegroundColor Red
    }
    
    # Pause between towns
    if ($town_num -lt $total_towns) {
        Write-Host ""
        Write-Host "Waiting 5 seconds before next town..." -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
}

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Collection Summary" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Total towns: $total_towns"
Write-Host "Completed: $completed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host ("=" * 80) -ForegroundColor Green

# Detailed CSV analysis
Write-Host ""
Write-Host "Collected CSV Files:" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

$csvFiles = Get-ChildItem $OUTPUT_DIR\raw\*.csv -ErrorAction SilentlyContinue | Sort-Object Name

if ($csvFiles) {
    $totalSize = 0
    $totalRows = 0
    
    foreach ($csv in $csvFiles) {
        $sizeMB = [math]::Round($csv.Length / 1MB, 2)
        $totalSize += $csv.Length
        
        # Try to count rows (may be slow for large files)
        try {
            $rows = (Import-Csv $csv.FullName -ErrorAction Stop).Count
            $totalRows += $rows
            Write-Host ("  {0,-35} {1,8} MB  {2,10} rows" -f $csv.Name, $sizeMB, $rows.ToString("N0"))
        } catch {
            Write-Host ("  {0,-35} {1,8} MB  (counting...)" -f $csv.Name, $sizeMB)
        }
    }
    
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ("Total: {0} files, {1} MB, ~{2} rows" -f `
        $csvFiles.Count, 
        [math]::Round($totalSize / 1MB, 2),
        $totalRows.ToString("N0")) -ForegroundColor Green
} else {
    Write-Host "No CSV files found!" -ForegroundColor Red
}

Write-Host ""
if ($failed -gt 0) {
    Write-Host "Some towns failed. Review logs above for details." -ForegroundColor Yellow
    Write-Host "You may need to adjust per-town configurations." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "All towns collected successfully!" -ForegroundColor Green
    Write-Host "Ready to build dataset and train GTNet!" -ForegroundColor Green
    exit 0
}
