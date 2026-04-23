param(
    [string]$CarlaRoot = "E:\Carla",
    [string]$Config = "configs/carla_env.yaml",
    [string]$Model = "auto",
    [Nullable[double]]$TargetSpeedKmh = $null,
    [Nullable[double]]$MaxThrottle = $null,
    [Nullable[double]]$MaxBrake = $null,
    [Nullable[int]]$Ticks = $null,
    [string]$Device = "auto",
    [switch]$NoRandomWeather,
    [Nullable[int]]$NpcVehicleCount = $null,
    [Nullable[int]]$NpcBikeCount = $null,
    [Nullable[int]]$NpcMotorbikeCount = $null,
    [Nullable[int]]$NpcPedestrianCount = $null
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

Write-Host "Python: $pythonExe"
Write-Host "Config: $configPath"
if ($modelPath -eq "auto") {
    Write-Host "Model : auto (run_agents.py will auto-select a checkpoint)"
}
else {
    Write-Host "Model : $modelPath"
}
Write-Host "CARLA : $CarlaRoot"
Write-Host "----- Running CIL agent -----"

$runnerArgs = @(
    "--agent", "cil",
    "--config", $configPath,
    "--cil-model-path", $modelPath,
    "--device", $Device
)

if ($null -ne $TargetSpeedKmh) {
    $runnerArgs += @("--target-speed-kmh", [string]$TargetSpeedKmh)
}
if ($null -ne $MaxThrottle) {
    $runnerArgs += @("--max-throttle", [string]$MaxThrottle)
}
if ($null -ne $MaxBrake) {
    $runnerArgs += @("--max-brake", [string]$MaxBrake)
}
if ($null -ne $NpcVehicleCount) {
    $runnerArgs += @("--npc-vehicle-count", [string]$NpcVehicleCount)
}
if ($null -ne $NpcBikeCount) {
    $runnerArgs += @("--npc-bike-count", [string]$NpcBikeCount)
}
if ($null -ne $NpcMotorbikeCount) {
    $runnerArgs += @("--npc-motorbike-count", [string]$NpcMotorbikeCount)
}
if ($null -ne $NpcPedestrianCount) {
    $runnerArgs += @("--npc-pedestrian-count", [string]$NpcPedestrianCount)
}
if ($null -ne $Ticks) {
    $runnerArgs += @("--ticks", [string]$Ticks)
}
if ($NoRandomWeather) {
    $runnerArgs += "--no-random-weather"
}

& $pythonExe "run_agents.py" @runnerArgs
exit $LASTEXITCODE
