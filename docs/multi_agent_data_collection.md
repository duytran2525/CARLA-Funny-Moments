# Multi-Agent Data Collection Guide

This guide explains how to use the enhanced data collection script for multi-town support in the GTNet improvements project.

## Overview

The `collect_multi_agent_data.py` script collects multi-agent trajectory data from CARLA for training the GTNet trajectory prediction model. It supports all major CARLA towns and records ego vehicle state along with all visible NPC vehicle states.

## Features

- **Multi-town support**: Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD
- **Configurable NPC density**: 30-50 vehicles per town for realistic traffic
- **10 FPS data collection**: 0.1 second intervals for trajectory prediction
- **Comprehensive state recording**: Position (x, y, z), velocity (vx, vy), and yaw for all agents
- **Visibility filtering**: Only records NPCs within 100m radius of ego vehicle
- **Robust error handling**: Automatic retry logic for CARLA connection failures (up to 3 retries)
- **Progress logging**: Updates every 100 frames

## Requirements

- CARLA Simulator (0.9.13 or later)
- Python 3.8+
- NumPy
- CARLA Python API (in PYTHONPATH)

## Installation

1. Ensure CARLA is installed and running:
```bash
cd /path/to/CARLA
./CarlaUE4.sh  # Linux
# or
CarlaUE4.exe  # Windows
```

2. Set PYTHONPATH to include CARLA Python API:
```bash
# Linux/Mac
export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla

# Windows PowerShell
$env:PYTHONPATH += ";C:\path\to\CARLA\PythonAPI\carla"
```

## Usage

### Basic Usage

Collect data from a single town:

```bash
python collect_multi_agent_data.py --town Town01 --duration 600
```

### Advanced Options

```bash
python collect_multi_agent_data.py \
    --town Town03 \
    --duration 600 \
    --npc-vehicles 40 \
    --host 127.0.0.1 \
    --port 2000 \
    --output-dir data/multi_agent \
    --seed 42
```

### Collect from All Towns

Use the PowerShell script to collect from all towns sequentially:

```powershell
.\collect_all_towns.ps1
```

This will collect 10 minutes of data from each of the 8 supported towns.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--town` | str | **required** | CARLA town name (Town01-Town07, Town10HD) |
| `--host` | str | 127.0.0.1 | CARLA server host |
| `--port` | int | 2000 | CARLA server port |
| `--timeout` | float | 10.0 | CARLA client timeout (seconds) |
| `--npc-vehicles` | int | 40 | Number of NPC vehicles (30-50) |
| `--duration` | float | 600 | Collection duration (seconds) |
| `--output-dir` | Path | data/multi_agent | Output directory |
| `--seed` | int | None | Random seed for reproducibility |

## Output Format

### Directory Structure

```
data/multi_agent/
└── raw/
    ├── Town01_20250101_120000.csv
    ├── Town02_20250101_130000.csv
    └── ...
```

### CSV Schema

Each CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | Unique identifier for this collection run |
| `town` | str | CARLA town name |
| `frame` | int | Frame number (sequential) |
| `timestamp` | float | Simulation timestamp (seconds) |
| `ego_id` | int | Ego vehicle actor ID |
| `ego_x` | float | Ego vehicle X position (meters) |
| `ego_y` | float | Ego vehicle Y position (meters) |
| `ego_z` | float | Ego vehicle Z position (meters) |
| `ego_vx` | float | Ego vehicle X velocity (m/s) |
| `ego_vy` | float | Ego vehicle Y velocity (m/s) |
| `ego_yaw` | float | Ego vehicle yaw angle (radians) |
| `actor_id` | int | NPC vehicle actor ID |
| `actor_type` | str | NPC vehicle type (e.g., "vehicle.audi.a2") |
| `actor_x` | float | NPC vehicle X position (meters) |
| `actor_y` | float | NPC vehicle Y position (meters) |
| `actor_z` | float | NPC vehicle Z position (meters) |
| `actor_vx` | float | NPC vehicle X velocity (m/s) |
| `actor_vy` | float | NPC vehicle Y velocity (m/s) |
| `actor_yaw` | float | NPC vehicle yaw angle (radians) |
| `distance_m` | float | Distance from ego to NPC (meters) |

### Data Format Notes

- **One row per NPC per frame**: Each frame generates N rows, where N is the number of visible NPCs
- **Global coordinates**: All positions are in CARLA world coordinates (not ego-centric)
- **Yaw in radians**: Converted from CARLA degrees to radians
- **Visibility filtering**: Only NPCs within 100m of ego are recorded

## Example Output

```csv
run_id,town,frame,timestamp,ego_id,ego_x,ego_y,ego_z,ego_vx,ego_vy,ego_yaw,actor_id,actor_type,actor_x,actor_y,actor_z,actor_vx,actor_vy,actor_yaw,distance_m
run_1704110400_a1b2c3d4,Town01,1,0.100000,123,100.500000,200.300000,0.500000,10.000000,5.000000,1.570000,456,vehicle.audi.a2,110.200000,205.100000,0.500000,8.500000,4.200000,1.520000,11.234
run_1704110400_a1b2c3d4,Town01,1,0.100000,123,100.500000,200.300000,0.500000,10.000000,5.000000,1.570000,789,vehicle.bmw.grandtourer,95.300000,198.700000,0.500000,12.000000,6.000000,1.600000,5.831
```

## Error Handling

### Connection Failures

The script automatically retries CARLA connection failures up to 3 times with exponential backoff:

```
[ERROR] Connection attempt 1 failed: timeout
[INFO] Retrying in 2.0 seconds...
[INFO] Successfully connected to CARLA server
```

### Insufficient NPCs

If fewer than 30 NPCs can be spawned, the script will fail with an error:

```
[ERROR] Only spawned 25 NPCs, minimum required is 30
```

**Solution**: Reduce `--npc-vehicles` or use a different spawn point.

### Map Load Failures

If the map fails to load after 3 retries, the script will exit:

```
[ERROR] All map load attempts failed
```

**Solution**: Check that CARLA server is running and the town name is correct.

## Performance Considerations

### Collection Speed

- **Target FPS**: 10 (0.1 second intervals)
- **Actual FPS**: Typically 9-11 depending on system performance
- **Duration**: 10 minutes = 6000 frames

### Disk Space

Approximate CSV file sizes:

- **10 minutes, 40 NPCs**: ~50-100 MB per town
- **All 8 towns**: ~400-800 MB total

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8+ GB
- **GPU**: Not required (CARLA can run in headless mode)

## Troubleshooting

### "Cannot import carla"

**Problem**: CARLA Python API not in PYTHONPATH

**Solution**:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla
```

### "Connection refused"

**Problem**: CARLA server not running

**Solution**: Start CARLA server before running the script:
```bash
cd /path/to/CARLA
./CarlaUE4.sh -carla-rpc-port=2000
```

### "Only spawned X NPCs"

**Problem**: Not enough spawn points available

**Solution**: Reduce `--npc-vehicles` to a lower value (e.g., 30)

### Slow collection (< 8 FPS)

**Problem**: System performance bottleneck

**Solution**:
- Reduce `--npc-vehicles`
- Run CARLA in headless mode: `./CarlaUE4.sh -RenderOffScreen`
- Close other applications

## Next Steps

After collecting data, use the dataset builder to process raw CSV logs into training samples:

```bash
python scripts/build_multi_agent_dataset.py \
    --input-dir data/multi_agent/raw \
    --output-dir data/multi_agent/processed \
    --adaptive-radius
```

See the [Dataset Building Guide](dataset_building.md) for details.

## References

- **Requirements**: See `requirements.md` section 5 (Data Collection Infrastructure)
- **Design**: See `design.md` section on Data Models
- **CARLA Documentation**: https://carla.readthedocs.io/
