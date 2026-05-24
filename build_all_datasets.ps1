param(
    [string]$InputDir = "data/multi_agent_20fps/raw",
    [string]$OutputDir = "data/multi_agent_20fps/processed_adaptive",
    [double]$Fps = 20.0,
    [switch]$AdaptiveRadius,
    [double]$RadiusBase = 20.0,
    [double]$RadiusAlpha = 0.5
)

# PowerShell script to build datasets from all collected CSV files.
# Usage:
#   .\build_all_datasets.ps1 -InputDir data\multi_agent_20fps\raw -OutputDir data\multi_agent_20fps\processed_adaptive -Fps 20 -AdaptiveRadius

$ErrorActionPreference = "Stop"

if ($Fps -le 0) {
    throw "Fps must be > 0, got $Fps"
}

$INPUT_DIR  = $InputDir
$OUTPUT_DIR = $OutputDir
$DT = 1.0 / $Fps
$HISTORY_FRAMES = [int][Math]::Round($Fps * 2.0)
$FUTURE_FRAMES  = [int][Math]::Round($Fps * 3.0)

# BUG FIX #8 (cosmetic): wrap "=" * N in parens so PowerShell evaluates string
# repetition correctly. Without parens, `Write-Host "=" * 80` prints "= 80".
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Building Multi-Agent Datasets from All Towns" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Input directory:  $INPUT_DIR"  -ForegroundColor Cyan
Write-Host "Output directory: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host ("FPS / dt:         {0} / {1:N6}s" -f $Fps, $DT) -ForegroundColor Cyan
Write-Host "History frames:   $HISTORY_FRAMES" -ForegroundColor Cyan
Write-Host "Future frames:    $FUTURE_FRAMES"  -ForegroundColor Cyan
Write-Host "Adaptive radius:  $([bool]$AdaptiveRadius)" -ForegroundColor Cyan
if ($AdaptiveRadius) {
    Write-Host ("Radius formula:   r = {0} + {1} * speed" -f $RadiusBase, $RadiusAlpha) -ForegroundColor Cyan
}
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host ""

# Get all CSV files
$csvFiles = Get-ChildItem "$INPUT_DIR\*.csv" -ErrorAction Stop | Sort-Object Name

if ($csvFiles.Count -eq 0) {
    Write-Host "ERROR: No CSV files found in $INPUT_DIR" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($csvFiles.Count) CSV files:" -ForegroundColor Yellow
foreach ($csv in $csvFiles) {
    $sizeMB = [math]::Round($csv.Length / 1MB, 2)
    Write-Host ("  {0,-35} {1,8} MB" -f $csv.Name, $sizeMB) -ForegroundColor Gray
}
Write-Host ""

$completed = 0
$failed    = 0

foreach ($csv in $csvFiles) {
    $csvNum   = $completed + $failed + 1
    $townName = $csv.BaseName -replace '_\d{8}_\d{6}$', ''

    # BUG FIX #2: each town writes into its own sub-directory so later towns
    # cannot overwrite sample_000000.pt … sample_XXXXXX.pt produced by earlier
    # towns.  The manifest.csv / build_summary.json inside each subdir is also
    # town-specific; a combined manifest is assembled after the loop (Bug Fix #7).
    $townOutDir = "$OUTPUT_DIR\$townName"

    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ("[$csvNum/$($csvFiles.Count)] Processing $($csv.Name)") -ForegroundColor Yellow
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "Town:       $townName"    -ForegroundColor Gray
    Write-Host "Output dir: $townOutDir" -ForegroundColor Gray
    Write-Host "Size:       $([math]::Round($csv.Length / 1MB, 2)) MB" -ForegroundColor Gray
    Write-Host ""

    try {
        $startTime = Get-Date

        $buildArgs = @(
            "scripts/build_multi_agent_dataset.py",
            "--raw-csv", $csv.FullName,
            "--out-dir", $townOutDir,
            "--history-frames", [string]$HISTORY_FRAMES,
            "--future-frames", [string]$FUTURE_FRAMES,
            "--stride", "1",
            "--dt", [string]$DT,
            "--adjacency-radius-m", "100.0",
            "--min-agents", "2",
            "--allow-missing"
        )
        if ($AdaptiveRadius) {
            $buildArgs += @(
                "--adaptive-radius",
                "--radius-base", [string]$RadiusBase,
                "--radius-alpha", [string]$RadiusAlpha
            )
        }

        python @buildArgs

        $exitCode = $LASTEXITCODE
        $elapsed  = ((Get-Date) - $startTime).TotalSeconds

        if ($exitCode -eq 0) {
            $completed++
            Write-Host ""
            Write-Host ("[SUCCESS] $townName processed in $([math]::Round($elapsed, 1))s") -ForegroundColor Green
        } else {
            $failed++
            Write-Host ""
            Write-Host ("[FAILED] $townName failed with exit code $exitCode") -ForegroundColor Red
        }
    }
    catch {
        $failed++
        Write-Host ""
        Write-Host ("[ERROR] $townName failed: $_") -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# BUG FIX #7: Merge all per-town manifest.csv files into a single
# $OUTPUT_DIR\manifest.csv so the training script sees every sample in one
# flat index.  sample_file paths are prefixed with the town sub-directory
# (e.g. "Town01\sample_000000.pt") so the loader can resolve them relative
# to $OUTPUT_DIR.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Merging per-town manifests …" -ForegroundColor Yellow

$mergedRows = [System.Collections.Generic.List[PSCustomObject]]::new()

Get-ChildItem "$OUTPUT_DIR" -Directory | Sort-Object Name | ForEach-Object {
    $dir          = $_
    $manifestFile = Join-Path $dir.FullName "manifest.csv"
    if (Test-Path $manifestFile) {
        Import-Csv $manifestFile | ForEach-Object {
            # Prefix sample_file with the town sub-directory name so paths are
            # relative to $OUTPUT_DIR. Use forward slash for cross-platform
            # compatibility (e.g. "Town01/sample_000000.pt").
            $_.sample_file = "$($dir.Name)/$($_.sample_file)"
            $mergedRows.Add($_)
        }
        Write-Host "  $($dir.Name): added $(( Import-Csv $manifestFile ).Count) rows" -ForegroundColor Gray
    } else {
        Write-Host "  WARNING: no manifest.csv in $($dir.Name)" -ForegroundColor Yellow
    }
}

if ($mergedRows.Count -gt 0) {
    $globalManifest = "$OUTPUT_DIR\manifest.csv"
    $mergedRows | Export-Csv $globalManifest -NoTypeInformation
    Write-Host "[OK] Global manifest: $globalManifest ($($mergedRows.Count) samples total)" -ForegroundColor Green
} else {
    Write-Host "WARNING: merged manifest is empty." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Dataset Building Summary" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Total CSV files: $($csvFiles.Count)"
Write-Host "Completed: $completed" -ForegroundColor Green
Write-Host "Failed:    $failed"    -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host ("=" * 80) -ForegroundColor Green

# BUG FIX #1: check for .pt files (PyTorch) not .npz (NumPy).
# The Python script writes sample_XXXXXX.pt via torch.save(); the old check
# for *.npz never matched anything and always printed "No .npz files found".
Write-Host ""
Write-Host "Generated Dataset Files (.pt) per town:" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

$totalPtCount = 0
$totalBytes   = 0

Get-ChildItem "$OUTPUT_DIR" -Directory | Sort-Object Name | ForEach-Object {
    $ptFiles = Get-ChildItem "$($_.FullName)\*.pt" -ErrorAction SilentlyContinue
    if ($ptFiles) {
        $dirBytes  = ($ptFiles | Measure-Object -Property Length -Sum).Sum
        $dirSizeMB = [math]::Round($dirBytes / 1MB, 2)
        Write-Host ("  {0,-20} {1,6} files   {2,8} MB" -f $_.Name, $ptFiles.Count, $dirSizeMB) -ForegroundColor Gray
        $totalPtCount += $ptFiles.Count
        $totalBytes   += $dirBytes
    }
}

if ($totalPtCount -gt 0) {
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ("Total: $totalPtCount .pt files, $([math]::Round($totalBytes / 1MB, 2)) MB") -ForegroundColor Green
} else {
    Write-Host "No .pt files found in $OUTPUT_DIR" -ForegroundColor Yellow
}

Write-Host ""
if ($failed -gt 0) {
    Write-Host "Some datasets failed to build. Review logs above for details." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "All datasets built successfully!" -ForegroundColor Green
    Write-Host "Ready to train GTNet!"            -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Train full GTNet: python scripts/kaggle_train_gtnet.py --data-dir $OUTPUT_DIR --out-dir models/gtnet_20fps_adaptive --mode full --num-modes 5" -ForegroundColor Gray
    exit 0
}
