# ============================================================================
# collect_recovery.ps1 — Launcher cho Intentional Drift Recovery Collector
# ============================================================================
# Sử dụng:
#   .\collect_recovery.ps1
#   .\collect_recovery.ps1 -Map Town04 -Ticks 30000 -NoiseIntensity 0.35
# ============================================================================

param(
    [string]$CarlaRoot = "E:\Carla",
    [string]$Config = "configs/carla_env.yaml",
    [string]$OutputDir = "data/recovery",
    [string]$Map = "",
    [int]$Ticks = 15000,
    [int]$SpawnPoint = -1,
    [double]$NoiseIntensity = 0.30,
    [double]$NoiseDuration = 1.5,
    [double]$RecoveryDuration = 3.0,
    [double]$CruiseDuration = 5.0,
    [double]$TargetSpeedKmh = 35.0,
    [int]$NpcVehicleCount = 0,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found at $pythonExe"
}

# Setup CARLA PythonAPI
$pythonApi = Join-Path $CarlaRoot "PythonAPI"
if (-not (Test-Path $pythonApi)) {
    throw "CARLA PythonAPI not found: $pythonApi"
}
$env:CARLA_ROOT = $CarlaRoot
$carlaPath = Join-Path $pythonApi "carla"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $carlaPath
} else {
    $env:PYTHONPATH = "$carlaPath;$env:PYTHONPATH"
}

$scriptArgs = @(
    "scripts/collect_recovery_data.py",
    "--config", $Config,
    "--out-dir", $OutputDir,
    "--ticks", [string]$Ticks,
    "--spawn-point", [string]$SpawnPoint,
    "--noise-intensity", [string]$NoiseIntensity,
    "--noise-duration", [string]$NoiseDuration,
    "--recovery-duration", [string]$RecoveryDuration,
    "--cruise-duration", [string]$CruiseDuration,
    "--target-speed-kmh", [string]$TargetSpeedKmh,
    "--npc-vehicle-count", [string]$NpcVehicleCount,
    "--seed", [string]$Seed
)

if ($Map -ne "") {
    $scriptArgs += @("--map", $Map)
}

Write-Host "Python  : $pythonExe"
Write-Host "Config  : $Config"
Write-Host "Output  : $OutputDir"
Write-Host "Noise   : intensity=$NoiseIntensity duration=$NoiseDuration"
Write-Host "Recovery: duration=$RecoveryDuration"
Write-Host "Cruise  : duration=$CruiseDuration"
Write-Host "----- Starting Recovery Data Collection -----"

& $pythonExe @scriptArgs
