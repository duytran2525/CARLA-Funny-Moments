param(
    [string]$CarlaRoot = "E:\Carla",
    [string]$Config = "configs/carla_env.yaml",
    [string]$Model = "D:\AI\CARLA-Funny-Moments\models\cnn_steering Train Loss 0.0074 Val Loss 0.0151.pth",
    [double]$TargetSpeedKmh = 40,
    [double]$MaxThrottle = 0.85,
    [double]$MaxBrake = 0.80,
    [int]$Ticks = 0,
    [string]$Device = "auto",
    [switch]$NoRandomWeather,
    [int]$NpcVehicleCount = 0,
    [int]$NpcBikeCount = 0,
    [int]$NpcMotorbikeCount = 0,
    [int]$NpcPedestrianCount = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found at $pythonExe. Create .venv first."
}

if ([System.IO.Path]::IsPathRooted($Config)) {
    $configPath = $Config
}
else {
    $configPath = Join-Path $repoRoot $Config
}
if (-not (Test-Path $configPath)) {
    throw "Config file not found: $configPath"
}

if ($Model -eq "auto") {
    $modelPath = "auto"
}
else {
    if ([System.IO.Path]::IsPathRooted($Model)) {
        $resolvedModel = $Model
    }
    else {
        $resolvedModel = Join-Path $repoRoot $Model
    }
    if (-not (Test-Path $resolvedModel)) {
        throw "Model file not found: $resolvedModel"
    }
    $modelPath = $resolvedModel
}

$pythonApi = Join-Path $CarlaRoot "PythonAPI"
if (-not (Test-Path $pythonApi)) {
    throw "CARLA PythonAPI not found: $pythonApi"
}

$env:CARLA_ROOT = $CarlaRoot
$env:CARLA_PYTHONAPI = $pythonApi
$carlaAgentsPath = Join-Path $pythonApi "carla"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $carlaAgentsPath
}
else {
    $env:PYTHONPATH = "$carlaAgentsPath;$env:PYTHONPATH"
}

# Check navigation agent availability to avoid surprising fallback behavior.
& $pythonExe -c "from agents.navigation.basic_agent import BasicAgent" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "BasicAgent import failed. CIL will use heuristic command fallback."
    Write-Warning "Install dependency in venv if needed: .\\.venv\\Scripts\\python.exe -m pip install shapely"
}

$args = @(
    "run_agents.py",
    "--agent", "cil",
    "--config", $configPath,
    "--cil-model-path", $modelPath,
    "--device", $Device,
    "--target-speed-kmh", $TargetSpeedKmh,
    "--max-throttle", $MaxThrottle,
    "--max-brake", $MaxBrake,
    "--npc-vehicle-count", $NpcVehicleCount,
    "--npc-bike-count", $NpcBikeCount,
    "--npc-motorbike-count", $NpcMotorbikeCount,
    "--npc-pedestrian-count", $NpcPedestrianCount,
    "--ticks", $Ticks
)

if ($NoRandomWeather) {
    $args += "--no-random-weather"
}

Write-Host "Python: $pythonExe"
Write-Host "Config: $configPath"
Write-Host "Model : $modelPath"
Write-Host "CARLA : $CarlaRoot"
Write-Host "----- Running CIL agent -----"

& $pythonExe @args
exit $LASTEXITCODE
