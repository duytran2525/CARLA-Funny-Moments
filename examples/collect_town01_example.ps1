# Example: Collect multi-agent data from Town01
# This is a simple example showing how to use the data collection script

Write-Host "Multi-Agent Data Collection Example - Town01" -ForegroundColor Cyan
Write-Host "=" * 60
Write-Host ""

# Configuration
$TOWN = "Town01"
$DURATION = 120  # 2 minutes for quick test
$NPC_VEHICLES = 35
$OUTPUT_DIR = "data/multi_agent"
$SEED = 42

Write-Host "Configuration:"
Write-Host "  Town: $TOWN"
Write-Host "  Duration: $DURATION seconds (2 minutes)"
Write-Host "  NPC vehicles: $NPC_VEHICLES"
Write-Host "  Output directory: $OUTPUT_DIR"
Write-Host "  Random seed: $SEED"
Write-Host ""

# Check if CARLA is running
Write-Host "Checking CARLA connection..." -ForegroundColor Yellow
$carlaRunning = $false
try {
    $connection = Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        $carlaRunning = $true
        Write-Host "CARLA server is running on port 2000" -ForegroundColor Green
    }
}
catch {
    Write-Host "Could not check CARLA connection" -ForegroundColor Yellow
}

if (-not $carlaRunning) {
    Write-Host ""
    Write-Host "WARNING: CARLA server may not be running!" -ForegroundColor Red
    Write-Host "Please start CARLA before running this script:" -ForegroundColor Yellow
    Write-Host "  cd C:\path\to\CARLA" -ForegroundColor Yellow
    Write-Host "  .\CarlaUE4.exe -carla-rpc-port=2000" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        Write-Host "Exiting..."
        exit 1
    }
}

Write-Host ""
Write-Host "Starting data collection..." -ForegroundColor Cyan
Write-Host ""

# Run data collection
python collect_multi_agent_data.py `
    --town $TOWN `
    --duration $DURATION `
    --npc-vehicles $NPC_VEHICLES `
    --output-dir $OUTPUT_DIR `
    --seed $SEED

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "SUCCESS! Data collection completed" -ForegroundColor Green
    Write-Host "=" * 60
    Write-Host ""
    Write-Host "Output location:"
    Write-Host "  $OUTPUT_DIR\raw\${TOWN}_*.csv"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Inspect the CSV file to verify data quality"
    Write-Host "  2. Run dataset builder to process into training samples"
    Write-Host "  3. Collect data from other towns using collect_all_towns.ps1"
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "FAILED! Data collection failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "=" * 60
    Write-Host ""
    Write-Host "Troubleshooting:"
    Write-Host "  - Check that CARLA server is running"
    Write-Host "  - Verify PYTHONPATH includes CARLA Python API"
    Write-Host "  - Check logs above for specific error messages"
    Write-Host ""
    exit 1
}
