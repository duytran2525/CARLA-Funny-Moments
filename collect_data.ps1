param(
    [string]$CarlaRoot = "E:\Carla",
    [string]$Config = "configs/carla_env.yaml",
    [string]$OutputRoot = "data/collected",
    [string[]]$Towns = @("Town01", "Town02", "Town03"),
    [int]$TargetImagesPerCamera = 30000,
    [int]$ChunkTicks = 15000,
    [int]$SaveEveryN = 3,
    [Nullable[double]]$TargetSpeedKmh = $null,
    [Nullable[int]]$NpcVehicleCount = $null,
    [Nullable[int]]$NpcBikeCount = $null,
    [Nullable[int]]$NpcMotorbikeCount = $null,
    [Nullable[int]]$NpcPedestrianCount = $null,
    [int]$SeedBase = -1
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$RepoRoot,
        [string]$PathValue
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $RepoRoot $PathValue)
}

function Get-CameraImageCounts {
    param(
        [string]$TownOutputDir
    )

    $counts = [ordered]@{
        center = 0
        left   = 0
        right  = 0
    }

    $folderMap = @{
        center = "images_center"
        left   = "images_left"
        right  = "images_right"
    }

    foreach ($key in $folderMap.Keys) {
        $folder = Join-Path $TownOutputDir $folderMap[$key]
        if (Test-Path $folder) {
            $counts[$key] = @(Get-ChildItem -Path $folder -Filter *.jpg -File).Count
        }
    }

    if (($counts.center -ne $counts.left) -or ($counts.center -ne $counts.right)) {
        throw (
            "Camera image counts are inconsistent in '$TownOutputDir'. " +
            "center=$($counts.center), left=$($counts.left), right=$($counts.right)"
        )
    }

    return [pscustomobject]$counts
}

function Invoke-CollectorChunk {
    param(
        [string]$PythonExe,
        [string]$ConfigPath,
        [string]$Town,
        [string]$TownOutputDir,
        [int]$Ticks,
        [int]$SaveEveryNValue,
        [Nullable[double]]$TargetSpeed,
        [Nullable[int]]$VehicleCount,
        [Nullable[int]]$BikeCount,
        [Nullable[int]]$MotorbikeCount,
        [Nullable[int]]$PedestrianCount,
        [Nullable[int]]$SeedValue
    )

    $runnerArgs = @(
        "run_agents.py",
        "--agent", "autopilot",
        "--config", $ConfigPath,
        "--collect-data",
        "--collect-data-dir", $TownOutputDir,
        "--map", $Town,
        "--spawn-point", "-1",
        "--destination-point", "-1",
        "--ticks", [string]$Ticks,
        "--save-every-n", [string]$SaveEveryNValue,
        "--nav-agent-type", "basic"
    )

    if ($null -ne $TargetSpeed) {
        $runnerArgs += @("--target-speed-kmh", [string]$TargetSpeed)
    }
    if ($null -ne $VehicleCount) {
        $runnerArgs += @("--npc-vehicle-count", [string]$VehicleCount)
    }
    if ($null -ne $BikeCount) {
        $runnerArgs += @("--npc-bike-count", [string]$BikeCount)
    }
    if ($null -ne $MotorbikeCount) {
        $runnerArgs += @("--npc-motorbike-count", [string]$MotorbikeCount)
    }
    if ($null -ne $PedestrianCount) {
        $runnerArgs += @("--npc-pedestrian-count", [string]$PedestrianCount)
    }
    if ($null -ne $SeedValue) {
        $runnerArgs += @("--seed", [string]$SeedValue)
    }

    & $PythonExe @runnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "run_agents.py exited with code $LASTEXITCODE for town '$Town'."
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found at $pythonExe. Create .venv first."
}

$configPath = Resolve-AbsolutePath -RepoRoot $repoRoot -PathValue $Config
if (-not (Test-Path $configPath)) {
    throw "Config file not found: $configPath"
}

$resolvedOutputRoot = Resolve-AbsolutePath -RepoRoot $repoRoot -PathValue $OutputRoot
New-Item -ItemType Directory -Force -Path $resolvedOutputRoot | Out-Null

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

& $pythonExe -c "from agents.navigation.basic_agent import BasicAgent" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "BasicAgent import failed. Check CARLA_PYTHONAPI and Python dependencies in .venv."
}

Write-Host "Python : $pythonExe"
Write-Host "Config : $configPath"
Write-Host "CARLA  : $CarlaRoot"
Write-Host "Output : $resolvedOutputRoot"
Write-Host "Towns  : $($Towns -join ', ')"
Write-Host "Target : $TargetImagesPerCamera images per camera per town"
Write-Host "Ticks  : $ChunkTicks per chunk | save_every_n=$SaveEveryN"
Write-Host "----- Collecting autopilot dataset -----"

$summary = New-Object System.Collections.Generic.List[object]

for ($townIndex = 0; $townIndex -lt $Towns.Count; $townIndex++) {
    $town = [string]$Towns[$townIndex]
    $townOutputDir = Join-Path $resolvedOutputRoot $town
    New-Item -ItemType Directory -Force -Path $townOutputDir | Out-Null

    $iteration = 0
    while ($true) {
        $countsBefore = Get-CameraImageCounts -TownOutputDir $townOutputDir
        if ([int]$countsBefore.center -ge $TargetImagesPerCamera) {
            Write-Host ("[{0}] already complete: {1} images/camera" -f $town, $countsBefore.center)
            break
        }

        $remaining = $TargetImagesPerCamera - [int]$countsBefore.center
        Write-Host (
            "[{0}] chunk #{1} | current={2} | remaining={3}" -f
            $town,
            ($iteration + 1),
            $countsBefore.center,
            $remaining
        )

        $seedValue = $null
        if ($SeedBase -ge 0) {
            $seedValue = [int]($SeedBase + ($townIndex * 1000) + $iteration)
        }

        Invoke-CollectorChunk `
            -PythonExe $pythonExe `
            -ConfigPath $configPath `
            -Town $town `
            -TownOutputDir $townOutputDir `
            -Ticks $ChunkTicks `
            -SaveEveryNValue $SaveEveryN `
            -TargetSpeed $TargetSpeedKmh `
            -VehicleCount $NpcVehicleCount `
            -BikeCount $NpcBikeCount `
            -MotorbikeCount $NpcMotorbikeCount `
            -PedestrianCount $NpcPedestrianCount `
            -SeedValue $seedValue

        $countsAfter = Get-CameraImageCounts -TownOutputDir $townOutputDir
        $delta = [int]$countsAfter.center - [int]$countsBefore.center
        Write-Host (
            "[{0}] chunk #{1} done | added={2} | total={3}" -f
            $town,
            ($iteration + 1),
            $delta,
            $countsAfter.center
        )

        if ($delta -le 0) {
            throw "No new images were added for '$town'. Stop to avoid an infinite loop."
        }

        $iteration += 1
    }

    $finalCounts = Get-CameraImageCounts -TownOutputDir $townOutputDir
    $summary.Add(
        [pscustomobject]@{
            Town   = $town
            Center = [int]$finalCounts.center
            Left   = [int]$finalCounts.left
            Right  = [int]$finalCounts.right
            Output = $townOutputDir
        }
    ) | Out-Null
}

Write-Host "----- Collection summary -----"
$summary | Format-Table -AutoSize
