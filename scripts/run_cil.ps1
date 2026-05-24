param(
    [string]$CarlaRoot = "E:\Carla",
    [string]$Config = "D:\AI\CARLA-Funny-Moments\configs\carla_env.yaml",
    [ValidateSet("cil", "cil_yolo")]
    [string]$Agent = "cil_yolo",
    [string]$Model = "models\waypoint_predictor_h5.pth",
    [string]$YoloModel = "D:\AI\CARLA-Funny-Moments\models\rtdetr-l.engine",
    [Nullable[double]]$TargetSpeedKmh = $null,
    [Nullable[double]]$MaxThrottle = $null,
    [Nullable[double]]$MaxBrake = $null,
    [Nullable[int]]$Ticks = $null,
    [Nullable[double]]$Fps = $null,
    [string]$Device = "auto",
    [Nullable[int]]$YoloEveryNTicks = $null,
    [string]$GTNetModel = "D:\AI\CARLA-Funny-Moments\models\gtnet_full_best.pt",
    [Nullable[int]]$GTNetEveryNTicks = $null,
    [switch]$DisableGTNet,
    [switch]$GTNetDrawDebug,
    [switch]$NoYoloVisualize,
    [switch]$NoYoloDrawOverlay,
    [switch]$OpencvRouteMap,
    [switch]$NoOpencvRouteMap,
    [switch]$NoRandomWeather,
    [Nullable[int]]$NpcVehicleCount = $null,
    [Nullable[int]]$NpcBikeCount = $null,
    [Nullable[int]]$NpcMotorbikeCount = $null,
    [Nullable[int]]$NpcPedestrianCount = $null,
    [switch]$EvalOnline
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
    $modelExt = [System.IO.Path]::GetExtension($resolvedModel).ToLowerInvariant()
    if ($modelExt -eq ".pth" -or $modelExt -eq ".pt") {
        $modelPath = $resolvedModel
    }
    else {
        throw "CIL waypoint model must be a PyTorch checkpoint (.pth/.pt), not '$modelExt': $resolvedModel. Use -YoloModel for YOLO/RT-DETR .pt/.engine files."
    }
}

if ([System.IO.Path]::IsPathRooted($YoloModel)) {
    $resolvedYoloModel = $YoloModel
}
else {
    $resolvedYoloModel = Join-Path $repoRoot $YoloModel
}
if (-not (Test-Path $resolvedYoloModel)) {
    throw "YOLO/RT-DETR model file not found: $resolvedYoloModel"
}
$yoloModelExt = [System.IO.Path]::GetExtension($resolvedYoloModel).ToLowerInvariant()
if ($yoloModelExt -ne ".pt" -and $yoloModelExt -ne ".engine" -and $yoloModelExt -ne ".onnx") {
    throw "YOLO/RT-DETR model must be .pt, .engine, or .onnx, not '$yoloModelExt': $resolvedYoloModel"
}
$yoloModelPath = $resolvedYoloModel

# GTNet model path is always resolved so run_agents.py can load and log the
# checkpoint metadata even when gtnet.enabled=false in carla_env.yaml.
# The -DisableGTNet switch is a hard CLI override (passes --disable-gtnet) that
# overrides even a YAML enabled=true.  Without it, YAML controls the enabled flag.
if ([System.IO.Path]::IsPathRooted($GTNetModel)) {
    $resolvedGTNetModel = $GTNetModel
}
else {
    $resolvedGTNetModel = Join-Path $repoRoot $GTNetModel
}
if (-not (Test-Path $resolvedGTNetModel)) {
    throw "GTNet model file not found: $resolvedGTNetModel"
}
$gtnetModelExt = [System.IO.Path]::GetExtension($resolvedGTNetModel).ToLowerInvariant()
if ($gtnetModelExt -ne ".pt" -and $gtnetModelExt -ne ".pth") {
    throw "GTNet model must be .pt or .pth, not '$gtnetModelExt': $resolvedGTNetModel"
}
$gtnetModelPath = $resolvedGTNetModel

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
    Write-Host "CIL   : auto (run_agents.py will auto-select a waypoint checkpoint)"
}
else {
    Write-Host "CIL   : $modelPath"
}
Write-Host "YOLO  : $yoloModelPath"
Write-Host "GTNet : $gtnetModelPath"
if ($DisableGTNet) {
    Write-Host "GTNet : FORCE-DISABLED via -DisableGTNet switch (overrides YAML)"
}
Write-Host "Agent : $Agent"
if ($null -ne $Fps) {
    Write-Host "FPS   : $Fps (fixed_delta=$([double](1.0 / [double]$Fps)))"
}
Write-Host "CARLA : $CarlaRoot"
Write-Host "----- Running CIL + YOLO agent -----"

$runnerArgs = @(
    "--agent", $Agent,
    "--config", $configPath,
    "--cil-model-path", $modelPath,
    "--yolo-model-path", $yoloModelPath,
    "--device", $Device
)


$runnerArgs += @("--gtnet-model-path", $gtnetModelPath)
if ($DisableGTNet) {
    $runnerArgs += "--disable-gtnet"
}

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
if ($null -ne $Fps) {
    if ($Fps -le 0) {
        throw "Fps must be > 0, got $Fps"
    }
    $fixedDelta = 1.0 / [double]$Fps
    $runnerArgs += @("--fixed-delta", [string]$fixedDelta)
}
if ($null -ne $YoloEveryNTicks) {
    $runnerArgs += @("--yolo-every-n-ticks", [string]$YoloEveryNTicks)
}
if ($null -ne $GTNetEveryNTicks) {
    $runnerArgs += @("--gtnet-every-n-ticks", [string]$GTNetEveryNTicks)
}
if ($GTNetDrawDebug) {
    $runnerArgs += "--gtnet-draw-debug"
}
if ($NoYoloVisualize) {
    $runnerArgs += "--no-yolo-visualize"
}
if ($NoYoloDrawOverlay) {
    $runnerArgs += "--no-yolo-draw-overlay"
}
if ($OpencvRouteMap) {
    $runnerArgs += "--opencv-route-map"
}
if ($NoOpencvRouteMap) {
    $runnerArgs += "--no-opencv-route-map"
}
if ($NoRandomWeather) {
    $runnerArgs += "--no-random-weather"
}
if ($EvalOnline) {
    $runnerArgs += "--eval-online"
}

& $pythonExe "run_agents.py" @runnerArgs
exit $LASTEXITCODE
