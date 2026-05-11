from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_control.carla_manager import CarlaManager, SpectatorConfig
from core_perception.multi_agent_trajectory import RAW_FIELDNAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw multi-agent CARLA actor states for trajectory prediction. "
            "The collector ego vehicle runs autopilot and logs nearby actors each synchronous tick."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--map", dest="map_name", default="Town03")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--fixed-delta", type=float, default=0.1, help="Synchronous dt; 0.1 = 10 FPS.")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--spawn-point", type=int, default=-1)
    parser.add_argument("--target-speed-kmh", type=float, default=35.0)
    parser.add_argument("--npc-vehicle-count", type=int, default=80)
    parser.add_argument("--npc-bike-count", type=int, default=10)
    parser.add_argument("--npc-motorbike-count", type=int, default=10)
    parser.add_argument("--npc-pedestrian-count", type=int, default=30)
    parser.add_argument("--disable-npc-autopilot", action="store_true")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument(
        "--include-walkers",
        action="store_true",
        help="Also log nearby walkers. By default walkers are spawned as traffic but only vehicles are logged.",
    )
    parser.add_argument("--out-dir", default="data/multi_agent/raw")
    parser.add_argument("--run-id", default="", help="Optional run id used for CSV and metadata filenames.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Flush raw CSV every N ticks to reduce data loss on interruption.",
    )
    parser.add_argument(
        "--lock-spectator-on-spawn",
        action="store_true",
        help="Place spectator behind the collector at startup.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _actor_kinematics(actor: Any) -> tuple[Any, Any, float]:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    return transform.location, velocity, float(transform.rotation.yaw)


def _distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def _configure_ego_autopilot(manager: CarlaManager, target_speed_kmh: float) -> None:
    ego = manager.ego_vehicle
    if ego is None:
        raise RuntimeError("Cannot enable autopilot because ego vehicle was not spawned.")
    try:
        ego.set_autopilot(True, manager.tm_port)
    except TypeError:
        ego.set_autopilot(True)

    tm = manager.tm
    if tm is None:
        return
    try:
        tm.auto_lane_change(ego, True)
    except Exception:
        pass
    try:
        tm.distance_to_leading_vehicle(ego, 2.5)
    except Exception:
        pass

    if target_speed_kmh > 0.0:
        try:
            speed_limit = max(1.0, float(ego.get_speed_limit()))
            speed_delta_pct = 100.0 * (speed_limit - float(target_speed_kmh)) / speed_limit
            tm.vehicle_percentage_speed_difference(ego, float(speed_delta_pct))
        except Exception:
            pass


def _nearby_actor_rows(
    *,
    run_id: str,
    town: str,
    frame_id: int,
    timestamp: float,
    ego: Any,
    actors: Iterable[Any],
    radius_m: float,
) -> List[Dict[str, Any]]:
    ego_loc, ego_vel, ego_yaw = _actor_kinematics(ego)
    ego_id = int(ego.id)
    rows: List[Dict[str, Any]] = []

    for actor in actors:
        try:
            actor_id = int(actor.id)
        except Exception:
            continue
        if actor_id == ego_id:
            continue
        try:
            loc, vel, yaw = _actor_kinematics(actor)
        except Exception:
            continue
        distance = _distance_xy(loc.x, loc.y, ego_loc.x, ego_loc.y)
        if distance > float(radius_m):
            continue
        rows.append(
            {
                "run_id": run_id,
                "town": town,
                "frame": int(frame_id),
                "timestamp": _format_float(timestamp),
                "ego_id": ego_id,
                "ego_x": _format_float(ego_loc.x),
                "ego_y": _format_float(ego_loc.y),
                "ego_z": _format_float(ego_loc.z),
                "ego_vx": _format_float(ego_vel.x),
                "ego_vy": _format_float(ego_vel.y),
                "ego_yaw": _format_float(ego_yaw),
                "actor_id": actor_id,
                "actor_type": str(getattr(actor, "type_id", "unknown")),
                "actor_x": _format_float(loc.x),
                "actor_y": _format_float(loc.y),
                "actor_z": _format_float(loc.z),
                "actor_vx": _format_float(vel.x),
                "actor_vy": _format_float(vel.y),
                "actor_yaw": _format_float(yaw),
                "distance_m": _format_float(distance),
            }
        )

    if rows:
        return rows

    return [
        {
            "run_id": run_id,
            "town": town,
            "frame": int(frame_id),
            "timestamp": _format_float(timestamp),
            "ego_id": ego_id,
            "ego_x": _format_float(ego_loc.x),
            "ego_y": _format_float(ego_loc.y),
            "ego_z": _format_float(ego_loc.z),
            "ego_vx": _format_float(ego_vel.x),
            "ego_vy": _format_float(ego_vel.y),
            "ego_yaw": _format_float(ego_yaw),
            "actor_id": -1,
            "actor_type": "__none__",
            "actor_x": "",
            "actor_y": "",
            "actor_z": "",
            "actor_vx": "",
            "actor_vy": "",
            "actor_yaw": "",
            "distance_m": "",
        }
    ]


def _actors_to_log(world: Any, include_walkers: bool) -> list[Any]:
    actors = world.get_actors()
    selected = list(actors.filter("vehicle.*"))
    if include_walkers:
        selected.extend(list(actors.filter("walker.pedestrian.*")))
    return selected


def _write_metadata(path: Path, args: argparse.Namespace, run_id: str, csv_path: Path) -> None:
    metadata = {
        "run_id": run_id,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_csv": str(csv_path),
        "map_name": str(args.map_name),
        "sync": True,
        "fixed_delta": float(args.fixed_delta),
        "sampling_fps": 1.0 / max(1e-6, float(args.fixed_delta)),
        "ticks": int(args.ticks),
        "seed": int(args.seed),
        "radius_m": float(args.radius_m),
        "include_walkers": bool(args.include_walkers),
        "vehicle_filter": str(args.vehicle_filter),
        "spawn_point": int(args.spawn_point),
        "target_speed_kmh": float(args.target_speed_kmh),
        "npc_vehicle_count": int(args.npc_vehicle_count),
        "npc_bike_count": int(args.npc_bike_count),
        "npc_motorbike_count": int(args.npc_motorbike_count),
        "npc_pedestrian_count": int(args.npc_pedestrian_count),
        "npc_enable_autopilot": not bool(args.disable_npc_autopilot),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if carla is None:
        raise RuntimeError("Python package 'carla' is required. Start CARLA and expose CARLA PythonAPI first.")
    if args.fixed_delta <= 0.0:
        raise ValueError("--fixed-delta must be positive.")
    if args.seed is not None:
        random.seed(int(args.seed))

    run_id = str(args.run_id).strip() or f"multi_agent_{args.map_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_id}_raw.csv"
    metadata_path = out_dir / f"{run_id}_metadata.json"

    manager = CarlaManager(
        host=str(args.host),
        port=int(args.port),
        tm_port=int(args.tm_port),
        timeout=float(args.timeout),
        map_name=str(args.map_name),
        sync=True,
        fixed_delta=float(args.fixed_delta),
        no_rendering=bool(args.no_rendering),
        vehicle_filter=str(args.vehicle_filter),
        spawn_point=int(args.spawn_point),
        spectator_cfg=SpectatorConfig(lock_on_spawn=bool(args.lock_spectator_on_spawn)),
        npc_vehicle_count=max(0, int(args.npc_vehicle_count)),
        npc_bike_count=max(0, int(args.npc_bike_count)),
        npc_motorbike_count=max(0, int(args.npc_motorbike_count)),
        npc_pedestrian_count=max(0, int(args.npc_pedestrian_count)),
        npc_enable_autopilot=not bool(args.disable_npc_autopilot),
        seed=int(args.seed) if args.seed is not None else None,
    )

    rows_written = 0
    try:
        manager.start()
        _configure_ego_autopilot(manager, target_speed_kmh=float(args.target_speed_kmh))
        world = manager.world
        ego = manager.ego_vehicle
        if world is None or ego is None:
            raise RuntimeError("CARLA world or ego vehicle is unavailable after manager.start().")
        town = str(world.get_map().name).replace("\\", "/").split("/")[-1]
        _write_metadata(metadata_path, args, run_id=run_id, csv_path=csv_path)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
            writer.writeheader()

            for step in range(1, int(args.ticks) + 1):
                manager.tick()
                snapshot = world.get_snapshot()
                frame_id = int(snapshot.frame)
                timestamp = float(snapshot.timestamp.elapsed_seconds)
                rows = _nearby_actor_rows(
                    run_id=run_id,
                    town=town,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    ego=ego,
                    actors=_actors_to_log(world, include_walkers=bool(args.include_walkers)),
                    radius_m=float(args.radius_m),
                )
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in RAW_FIELDNAMES})
                rows_written += len(rows)

                if step % max(1, int(args.flush_every)) == 0:
                    csv_file.flush()
                if step % max(1, int(args.log_every)) == 0:
                    actor_rows = sum(1 for row in rows if int(row.get("actor_id", -1)) >= 0)
                    logging.info(
                        "collect step=%d/%d frame=%d nearby_actors=%d rows_written=%d",
                        step,
                        int(args.ticks),
                        frame_id,
                        actor_rows,
                        rows_written,
                    )

        logging.info("Raw multi-agent CSV written: %s", csv_path)
        logging.info("Metadata written: %s", metadata_path)
        return 0
    finally:
        manager.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
