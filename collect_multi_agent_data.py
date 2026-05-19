"""
collect_multi_agent_data.py — Multi-Agent Trajectory Data Collection
======================================================================

Collects multi-agent trajectory data from CARLA for GTNet training.
Supports multiple towns with configurable NPC density.

Usage:
    python collect_multi_agent_data.py --town Town01 --duration 600
    python collect_multi_agent_data.py --town Town03 --npc-vehicles 40
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "Cannot import carla. Set PYTHONPATH to <CARLA_ROOT>/PythonAPI/carla"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required.") from exc


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

SUPPORTED_TOWNS = [
    "Town01", "Town02", "Town03", "Town04", "Town05",
    "Town06", "Town07", "Town10HD"
]

DEFAULT_NPC_VEHICLES = 40
MIN_NPC_VEHICLES = 30
MAX_NPC_VEHICLES = 50

COLLECTION_FPS = 10
FIXED_DELTA_SECONDS = 1.0 / COLLECTION_FPS  # 0.1 seconds

DEFAULT_DURATION_SECONDS = 600  # 10 minutes
LOG_PROGRESS_EVERY_N_FRAMES = 100

MAX_CONNECTION_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0

VISIBILITY_RADIUS_METERS = 100.0  # Only record NPCs within this radius


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActorState:
    """State of a single actor at a given frame."""
    actor_id: int
    actor_type: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    yaw: float


@dataclass
class FrameData:
    """Complete frame data including ego and all visible NPCs."""
    frame: int
    timestamp: float
    ego_state: ActorState
    npc_states: List[ActorState]


# ═══════════════════════════════════════════════════════════════════════════
# CSV WRITER
# ═══════════════════════════════════════════════════════════════════════════

class MultiAgentCSVWriter:
    """Writes multi-agent trajectory data to CSV."""
    
    FIELDNAMES = [
        "run_id",
        "town",
        "frame",
        "timestamp",
        "ego_id",
        "ego_x",
        "ego_y",
        "ego_z",
        "ego_vx",
        "ego_vy",
        "ego_yaw",
        "actor_id",
        "actor_type",
        "actor_x",
        "actor_y",
        "actor_z",
        "actor_vx",
        "actor_vy",
        "actor_yaw",
        "distance_m",
    ]
    
    def __init__(self, output_path: Path, run_id: str, town: str):
        self.output_path = output_path
        self.run_id = run_id
        self.town = town
        self._file = None
        self._writer = None
        self._rows_written = 0
        
    def start(self) -> None:
        """Open CSV file and write header."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        logging.info("CSV writer started: %s", self.output_path)
        
    def write_frame(self, frame_data: FrameData) -> None:
        """Write all actors in a frame to CSV."""
        if self._writer is None:
            raise RuntimeError("CSV writer not started")
            
        ego = frame_data.ego_state
        
        # Write one row per NPC actor
        for npc in frame_data.npc_states:
            # Calculate distance from ego to NPC
            dx = npc.x - ego.x
            dy = npc.y - ego.y
            distance_m = math.sqrt(dx * dx + dy * dy)
            
            row = {
                "run_id": self.run_id,
                "town": self.town,
                "frame": frame_data.frame,
                "timestamp": f"{frame_data.timestamp:.6f}",
                "ego_id": ego.actor_id,
                "ego_x": f"{ego.x:.6f}",
                "ego_y": f"{ego.y:.6f}",
                "ego_z": f"{ego.z:.6f}",
                "ego_vx": f"{ego.vx:.6f}",
                "ego_vy": f"{ego.vy:.6f}",
                "ego_yaw": f"{ego.yaw:.6f}",
                "actor_id": npc.actor_id,
                "actor_type": npc.actor_type,
                "actor_x": f"{npc.x:.6f}",
                "actor_y": f"{npc.y:.6f}",
                "actor_z": f"{npc.z:.6f}",
                "actor_vx": f"{npc.vx:.6f}",
                "actor_vy": f"{npc.vy:.6f}",
                "actor_yaw": f"{npc.yaw:.6f}",
                "distance_m": f"{distance_m:.3f}",
            }
            self._writer.writerow(row)
            self._rows_written += 1
            
    def flush(self) -> None:
        """Flush CSV buffer to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self) -> None:
        """Close CSV file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
        logging.info("CSV writer closed. Total rows written: %d", self._rows_written)
        
    @property
    def rows_written(self) -> int:
        return self._rows_written


# ═══════════════════════════════════════════════════════════════════════════
# CARLA CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class CarlaConnectionManager:
    """Manages CARLA connection with retry logic."""
    
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        max_retries: int = MAX_CONNECTION_RETRIES,
        retry_delay: float = RETRY_DELAY_SECONDS,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        
    def connect(self) -> bool:
        """Connect to CARLA server with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logging.info(
                    "Connecting to CARLA server at %s:%d (attempt %d/%d)...",
                    self.host, self.port, attempt, self.max_retries
                )
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
                self.world = self.client.get_world()
                
                # Test connection by getting world info
                _ = self.world.get_map().name
                
                logging.info("Successfully connected to CARLA server")
                return True
                
            except RuntimeError as exc:
                logging.error("Connection attempt %d failed: %s", attempt, exc)
                if attempt < self.max_retries:
                    logging.info("Retrying in %.1f seconds...", self.retry_delay)
                    time.sleep(self.retry_delay)
                else:
                    logging.error("All connection attempts failed")
                    return False
                    
        return False
        
    def load_map(self, map_name: str) -> bool:
        """Load specified map with retry logic."""
        if self.client is None:
            logging.error("Cannot load map: not connected to CARLA")
            return False
            
        for attempt in range(1, self.max_retries + 1):
            try:
                logging.info(
                    "Loading map %s (attempt %d/%d)...",
                    map_name, attempt, self.max_retries
                )
                self.world = self.client.load_world(map_name)
                
                # Wait for map to fully load
                time.sleep(2.0)
                
                # Verify map loaded correctly
                loaded_map_name = self.world.get_map().name
                if map_name.lower() not in loaded_map_name.lower():
                    raise RuntimeError(
                        f"Map mismatch: requested {map_name}, got {loaded_map_name}"
                    )
                    
                logging.info("Successfully loaded map: %s", loaded_map_name)
                return True
                
            except RuntimeError as exc:
                logging.error("Map load attempt %d failed: %s", attempt, exc)
                if attempt < self.max_retries:
                    logging.info("Retrying in %.1f seconds...", self.retry_delay)
                    time.sleep(self.retry_delay)
                else:
                    logging.error("All map load attempts failed")
                    return False
                    
        return False


# ═══════════════════════════════════════════════════════════════════════════
# DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════

class MultiAgentDataCollector:
    """Collects multi-agent trajectory data from CARLA."""
    
    def __init__(
        self,
        world: carla.World,
        output_dir: Path,
        town: str,
        npc_vehicle_count: int,
        duration_seconds: float,
        fixed_delta: float = FIXED_DELTA_SECONDS,
    ):
        self.world = world
        self.output_dir = output_dir
        self.town = town
        self.npc_vehicle_count = npc_vehicle_count
        self.duration_seconds = duration_seconds
        self.fixed_delta = fixed_delta
        
        self.run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.npc_vehicles: List[carla.Vehicle] = []
        self.traffic_manager: Optional[carla.TrafficManager] = None
        
        self.frame_count = 0
        self.start_time = 0.0
        
        # CSV writer
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{town}_{timestamp_str}.csv"
        csv_path = output_dir / "raw" / csv_filename
        self.csv_writer = MultiAgentCSVWriter(csv_path, self.run_id, town)
        
    def setup(self) -> bool:
        """Setup CARLA world and spawn vehicles."""
        try:
            # Configure synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta
            self.world.apply_settings(settings)
            logging.info(
                "Synchronous mode enabled (fixed_delta=%.3fs, fps=%d)",
                self.fixed_delta, int(1.0 / self.fixed_delta)
            )
            
            # Get traffic manager
            self.traffic_manager = self.world.get_traffic_manager()
            self.traffic_manager.set_synchronous_mode(True)
            logging.info("Traffic manager configured")
            
            # Spawn ego vehicle
            if not self._spawn_ego_vehicle():
                return False
                
            # Spawn NPC vehicles
            if not self._spawn_npc_vehicles():
                return False
                
            # Start CSV writer
            self.csv_writer.start()
            
            # Warm-up ticks
            logging.info("Running warm-up ticks...")
            for _ in range(50):
                self.world.tick()
                
            logging.info("Setup complete")
            return True
            
        except Exception as exc:
            logging.error("Setup failed: %s", exc)
            return False
            
    def _spawn_ego_vehicle(self) -> bool:
        """Spawn ego vehicle at random spawn point."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                logging.error("No spawn points available")
                return False
                
            spawn_point = np.random.choice(spawn_points)
            
            self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
            if self.ego_vehicle is None:
                logging.error("Failed to spawn ego vehicle")
                return False
                
            # Enable autopilot
            self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            
            logging.info(
                "Ego vehicle spawned (id=%d) at (%.1f, %.1f)",
                self.ego_vehicle.id,
                spawn_point.location.x,
                spawn_point.location.y,
            )
            return True
            
        except Exception as exc:
            logging.error("Failed to spawn ego vehicle: %s", exc)
            return False
            
    def _spawn_npc_vehicles(self) -> bool:
        """Spawn NPC vehicles with autopilot enabled."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bps = blueprint_library.filter("vehicle.*")
            
            # Filter out non-vehicle blueprints
            vehicle_bps = [
                bp for bp in vehicle_bps
                if int(bp.get_attribute("number_of_wheels")) == 4
            ]
            
            spawn_points = self.world.get_map().get_spawn_points()
            if len(spawn_points) < self.npc_vehicle_count:
                logging.warning(
                    "Only %d spawn points available, requested %d NPCs",
                    len(spawn_points), self.npc_vehicle_count
                )
                
            # Shuffle spawn points
            np.random.shuffle(spawn_points)
            
            spawned_count = 0
            for i in range(min(self.npc_vehicle_count, len(spawn_points))):
                try:
                    bp = np.random.choice(vehicle_bps)
                    
                    # Randomize color if possible
                    if bp.has_attribute("color"):
                        color = np.random.choice(
                            bp.get_attribute("color").recommended_values
                        )
                        bp.set_attribute("color", color)
                        
                    npc = self.world.spawn_actor(bp, spawn_points[i])
                    if npc is not None:
                        npc.set_autopilot(True, self.traffic_manager.get_port())
                        self.npc_vehicles.append(npc)
                        spawned_count += 1
                        
                except RuntimeError:
                    # Spawn point occupied, skip
                    continue
                    
            if spawned_count < MIN_NPC_VEHICLES:
                logging.error(
                    "Only spawned %d NPCs, minimum required is %d",
                    spawned_count, MIN_NPC_VEHICLES
                )
                return False
                
            logging.info("Spawned %d NPC vehicles", spawned_count)
            return True
            
        except Exception as exc:
            logging.error("Failed to spawn NPC vehicles: %s", exc)
            return False
            
    def _get_actor_state(self, actor: carla.Vehicle) -> ActorState:
        """Extract state from CARLA actor."""
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        
        return ActorState(
            actor_id=actor.id,
            actor_type=actor.type_id,
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
            vx=velocity.x,
            vy=velocity.y,
            yaw=math.radians(transform.rotation.yaw),
        )
        
    def _get_visible_npcs(self, ego_state: ActorState) -> List[ActorState]:
        """Get states of NPCs within visibility radius."""
        visible_npcs = []
        
        for npc in self.npc_vehicles:
            try:
                npc_state = self._get_actor_state(npc)
                
                # Calculate distance from ego
                dx = npc_state.x - ego_state.x
                dy = npc_state.y - ego_state.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance <= VISIBILITY_RADIUS_METERS:
                    visible_npcs.append(npc_state)
                    
            except RuntimeError:
                # Actor may have been destroyed
                continue
                
        return visible_npcs
        
    def collect(self) -> bool:
        """Run data collection loop."""
        if self.ego_vehicle is None:
            logging.error("Cannot collect: ego vehicle not spawned")
            return False
            
        try:
            self.start_time = time.time()
            target_frames = int(self.duration_seconds / self.fixed_delta)
            
            logging.info("=" * 70)
            logging.info("Starting data collection:")
            logging.info("  Town: %s", self.town)
            logging.info("  Duration: %.1f seconds", self.duration_seconds)
            logging.info("  Target frames: %d", target_frames)
            logging.info("  FPS: %d", int(1.0 / self.fixed_delta))
            logging.info("  NPC vehicles: %d", len(self.npc_vehicles))
            logging.info("  Run ID: %s", self.run_id)
            logging.info("=" * 70)
            
            while self.frame_count < target_frames:
                # Tick world
                snapshot = self.world.tick()
                self.frame_count += 1
                
                # Get ego state
                ego_state = self._get_actor_state(self.ego_vehicle)
                
                # Get visible NPC states
                npc_states = self._get_visible_npcs(ego_state)
                
                # Create frame data
                frame_data = FrameData(
                    frame=self.frame_count,
                    timestamp=snapshot.timestamp.elapsed_seconds,
                    ego_state=ego_state,
                    npc_states=npc_states,
                )
                
                # Write to CSV
                self.csv_writer.write_frame(frame_data)
                
                # Periodic logging
                if self.frame_count % LOG_PROGRESS_EVERY_N_FRAMES == 0:
                    elapsed = time.time() - self.start_time
                    progress_pct = 100.0 * self.frame_count / target_frames
                    logging.info(
                        "Frame %d/%d (%.1f%%) | Elapsed: %.1fs | "
                        "Visible NPCs: %d | CSV rows: %d",
                        self.frame_count,
                        target_frames,
                        progress_pct,
                        elapsed,
                        len(npc_states),
                        self.csv_writer.rows_written,
                    )
                    
                # Flush CSV periodically
                if self.frame_count % (LOG_PROGRESS_EVERY_N_FRAMES * 5) == 0:
                    self.csv_writer.flush()
                    
            # Final flush
            self.csv_writer.flush()
            
            elapsed = time.time() - self.start_time
            logging.info("=" * 70)
            logging.info("Collection complete:")
            logging.info("  Frames collected: %d", self.frame_count)
            logging.info("  CSV rows written: %d", self.csv_writer.rows_written)
            logging.info("  Elapsed time: %.1f seconds", elapsed)
            logging.info("  Average FPS: %.1f", self.frame_count / elapsed)
            logging.info("=" * 70)
            
            return True
            
        except KeyboardInterrupt:
            logging.warning("Collection interrupted by user")
            return False
            
        except Exception as exc:
            logging.error("Collection failed: %s", exc)
            return False
            
    def cleanup(self) -> None:
        """Cleanup CARLA actors and close CSV writer."""
        try:
            # Close CSV writer
            self.csv_writer.close()
            
            # Destroy ego vehicle
            if self.ego_vehicle is not None:
                self.ego_vehicle.destroy()
                self.ego_vehicle = None
                
            # Destroy NPC vehicles
            for npc in self.npc_vehicles:
                try:
                    npc.destroy()
                except RuntimeError:
                    pass
            self.npc_vehicles.clear()
            
            # Restore asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(False)
                
            logging.info("Cleanup complete")
            
        except Exception as exc:
            logging.error("Cleanup failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect multi-agent trajectory data from CARLA"
    )
    
    parser.add_argument(
        "--town",
        type=str,
        required=True,
        choices=SUPPORTED_TOWNS,
        help="CARLA town to collect data from",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="CARLA server host (default: 127.0.0.1)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="CARLA client timeout in seconds (default: 10.0)",
    )
    
    parser.add_argument(
        "--npc-vehicles",
        type=int,
        default=DEFAULT_NPC_VEHICLES,
        help=f"Number of NPC vehicles to spawn (default: {DEFAULT_NPC_VEHICLES}, "
             f"range: {MIN_NPC_VEHICLES}-{MAX_NPC_VEHICLES})",
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Collection duration in seconds (default: {DEFAULT_DURATION_SECONDS})",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/multi_agent"),
        help="Output directory for collected data (default: data/multi_agent)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        logging.info("Random seed set to: %d", args.seed)
        
    # Validate NPC count
    if not (MIN_NPC_VEHICLES <= args.npc_vehicles <= MAX_NPC_VEHICLES):
        logging.error(
            "NPC vehicle count must be between %d and %d",
            MIN_NPC_VEHICLES, MAX_NPC_VEHICLES
        )
        return 1
        
    # Connect to CARLA
    connection_manager = CarlaConnectionManager(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    
    if not connection_manager.connect():
        logging.error("Failed to connect to CARLA server")
        return 1
        
    # Load map
    if not connection_manager.load_map(args.town):
        logging.error("Failed to load map: %s", args.town)
        return 1
        
    # Create data collector
    collector = MultiAgentDataCollector(
        world=connection_manager.world,
        output_dir=args.output_dir,
        town=args.town,
        npc_vehicle_count=args.npc_vehicles,
        duration_seconds=args.duration,
    )
    
    # Setup and collect
    success = False
    try:
        if collector.setup():
            success = collector.collect()
    finally:
        collector.cleanup()
        
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
