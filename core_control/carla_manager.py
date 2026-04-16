
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    carla = None


if TYPE_CHECKING:
    import carla as carla_api

    CarlaClient = carla_api.Client
    CarlaWorld = carla_api.World
    CarlaTrafficManager = carla_api.TrafficManager
    CarlaVehicle = carla_api.Vehicle
    CarlaWorldSettings = carla_api.WorldSettings
    CarlaTransform = carla_api.Transform
    CarlaActor = carla_api.Actor
    CarlaActorBlueprint = carla_api.ActorBlueprint
else:
    CarlaClient = Any
    CarlaWorld = Any
    CarlaTrafficManager = Any
    CarlaVehicle = Any
    CarlaWorldSettings = Any
    CarlaTransform = Any
    CarlaActor = Any
    CarlaActorBlueprint = Any


def map_basename(name: str) -> str:
    return name.replace("\\", "/").split("/")[-1].lower()


@dataclass
class SpectatorConfig:
    lock_on_spawn: bool = True
    keep_reapply_each_tick: bool = False
    follow_distance: float = 9.0
    height: float = 4.5
    pitch: float = -18.0


class CarlaManager:
    """Manage CARLA world settings, ego spawn, and spectator placement."""

    def __init__(
        self,
        host: str,
        port: int,
        tm_port: int,
        timeout: float,
        map_name: str,
        sync: bool,
        fixed_delta: float,
        no_rendering: bool,
        vehicle_filter: str,
        spawn_point: int,
        spectator_cfg: Optional[SpectatorConfig] = None,
        npc_vehicle_count: int = 30,
        npc_bike_count: int = 10,
        npc_motorbike_count: int = 10,
        npc_pedestrian_count: int = 50,
        npc_enable_autopilot: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.timeout = timeout
        self.map_name = map_name
        self.sync = sync
        self.fixed_delta = fixed_delta
        self.no_rendering = no_rendering
        self.vehicle_filter = vehicle_filter
        self.spawn_point = spawn_point
        self.spectator_cfg = spectator_cfg or SpectatorConfig()
        self.npc_vehicle_count = max(0, npc_vehicle_count)
        self.npc_bike_count = max(0, npc_bike_count)
        self.npc_motorbike_count = max(0, npc_motorbike_count)
        self.npc_pedestrian_count = max(0, npc_pedestrian_count)
        self.npc_enable_autopilot = npc_enable_autopilot

        self.client: Optional[CarlaClient] = None
        self.world: Optional[CarlaWorld] = None
        self.tm: Optional[CarlaTrafficManager] = None
        self.ego_vehicle: Optional[CarlaVehicle] = None
        self._original_settings: Optional[CarlaWorldSettings] = None
        self._spawn_transform: Optional[CarlaTransform] = None
        self._npc_actors: List[CarlaActor] = []
        self._walker_actors: List[CarlaActor] = []
        self._walker_controllers: List[CarlaActor] = []

    def _retry_rpc_call(
        self,
        op_name: str,
        fn,
        *,
        max_wait_seconds: float,
        per_attempt_timeout: float,
        retry_interval_seconds: float = 2.0,
    ):
        assert self.client is not None
        deadline = time.time() + max_wait_seconds
        attempt = 0
        last_exc: Optional[Exception] = None
        while time.time() < deadline:
            attempt += 1
            try:
                self.client.set_timeout(per_attempt_timeout)
                return fn()
            except RuntimeError as exc:
                last_exc = exc
                remaining = max(0.0, deadline - time.time())
                logging.warning(
                    "CARLA RPC %s failed (attempt %d, remain %.1fs): %s",
                    op_name,
                    attempt,
                    remaining,
                    exc,
                )
                if remaining <= 0.0:
                    break
                time.sleep(min(retry_interval_seconds, remaining))

        raise RuntimeError(
            f"CARLA RPC {op_name} failed after {max_wait_seconds:.0f}s on "
            f"{self.host}:{self.port}. Last error: {last_exc}"
        )

    def start(self) -> None:
        if carla is None:
            raise RuntimeError("Python package 'carla' is required for CarlaManager.")
        self.client = carla.Client(self.host, self.port)

        connect_wait = max(30.0, self.timeout)
        world = self._retry_rpc_call(
            "get_world",
            self.client.get_world,
            max_wait_seconds=connect_wait,
            per_attempt_timeout=min(30.0, max(5.0, connect_wait)),
        )
        current_map = world.get_map().name
        if map_basename(current_map) != map_basename(self.map_name):
            logging.info("Current map is %s. Loading %s...", current_map, self.map_name)
            load_wait = max(300.0, self.timeout * 5.0)
            world = self._retry_rpc_call(
                f"load_world({self.map_name})",
                lambda: self.client.load_world(self.map_name),
                max_wait_seconds=load_wait,
                per_attempt_timeout=min(load_wait, max(120.0, self.timeout * 2.0)),
                retry_interval_seconds=5.0,
            )
            logging.info("Map loaded: %s", world.get_map().name)

        self.client.set_timeout(self.timeout)
        self.world = world
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self._original_settings = self.world.get_settings()
        self._apply_world_settings()
        self._destroy_existing_actors()
        self._spawn_ego_vehicle()
        self._spawn_requested_traffic()
        self._spawn_pedestrians()
        self._log_static_traffic_actors()
        self.apply_spawn_locked_spectator(force=True)

    def _apply_world_settings(self) -> None:
        assert self.world is not None
        settings = self.world.get_settings()
        settings.synchronous_mode = self.sync
        settings.fixed_delta_seconds = self.fixed_delta if self.sync else 0.0
        settings.no_rendering_mode = self.no_rendering
        self.world.apply_settings(settings)
        if self.tm is not None:
            self.tm.set_synchronous_mode(self.sync)

    def _destroy_existing_actors(self) -> None:
        """Destroy all leftover vehicles, walkers, and controllers from previous sessions."""
        assert self.world is not None
        actors = self.world.get_actors()

        # Destroy walker controllers first
        controllers = list(actors.filter("controller.ai.walker"))
        for c in controllers:
            try:
                c.stop()
                c.destroy()
            except RuntimeError:
                pass

        # Destroy walkers
        walkers = list(actors.filter("walker.pedestrian.*"))
        for w in walkers:
            try:
                w.destroy()
            except RuntimeError:
                pass

        # Destroy vehicles
        vehicles = list(actors.filter("vehicle.*"))
        for v in vehicles:
            try:
                v.destroy()
            except RuntimeError:
                pass

        total = len(controllers) + len(walkers) + len(vehicles)
        if total > 0:
            logging.info(
                "Cleaned %d leftover actors (vehicles=%d, walkers=%d, controllers=%d).",
                total, len(vehicles), len(walkers), len(controllers),
            )

    def _spawn_ego_vehicle(self) -> None:
        assert self.world is not None
        blueprints = self.world.get_blueprint_library().filter(self.vehicle_filter)
        if not blueprints:
            raise RuntimeError(f"No vehicle blueprint matches '{self.vehicle_filter}'.")

        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in current map.")

        if self.spawn_point >= 0:
            ordered_points = [spawn_points[self.spawn_point % len(spawn_points)]]
        else:
            ordered_points = spawn_points.copy()
            random.shuffle(ordered_points)

        self.ego_vehicle = None
        self._spawn_transform = None
        for transform in ordered_points:
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                self.ego_vehicle = actor
                self._spawn_transform = transform
                break

        if self.ego_vehicle is None or self._spawn_transform is None:
            raise RuntimeError("Cannot spawn ego vehicle at available spawn points.")

        loc = self._spawn_transform.location
        rot = self._spawn_transform.rotation
        logging.info(
            "Spawned %s at (%.1f, %.1f, %.1f) yaw=%.1f",
            self.ego_vehicle.type_id, loc.x, loc.y, loc.z, rot.yaw,
        )

    @staticmethod
    def _wheel_count(bp: CarlaActorBlueprint) -> int:
        if bp.has_attribute("number_of_wheels"):
            try:
                return int(bp.get_attribute("number_of_wheels").as_int())
            except RuntimeError:
                pass
            except ValueError:
                pass
        return 4

    @staticmethod
    def _is_motorbike(bp: CarlaActorBlueprint) -> bool:
        bp_id = bp.id.lower()
        motor_keywords = (
            "harley",
            "kawasaki",
            "yamaha",
            "vespa",
            "ninja",
            "yzf",
            "motor",
        )
        return CarlaManager._wheel_count(bp) == 2 and any(k in bp_id for k in motor_keywords)

    @staticmethod
    def _is_bike(bp: CarlaActorBlueprint) -> bool:
        bp_id = bp.id.lower()
        bike_keywords = (
            "bike",
            "bicycle",
            "crossbike",
            "century",
            "omafiets",
            "diamondback",
            "gazelle",
            "bh.",
        )
        return CarlaManager._wheel_count(bp) == 2 and any(k in bp_id for k in bike_keywords)

    @staticmethod
    def _is_car(bp: CarlaActorBlueprint) -> bool:
        return CarlaManager._wheel_count(bp) >= 4

    def _spawn_npc_group(
        self,
        label: str,
        blueprints: List[CarlaActorBlueprint],
        desired_count: int,
    ) -> int:
        if desired_count <= 0:
            return 0
        if not blueprints:
            logging.warning("No blueprint found for %s.", label)
            return 0

        spawn_points = self.world.get_map().get_spawn_points().copy()
        random.shuffle(spawn_points)
        spawned = 0
        for transform in spawn_points:
            if spawned >= desired_count:
                break

            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", f"npc_{label}")
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue

            self._npc_actors.append(actor)
            if self.npc_enable_autopilot:
                try:
                    actor.set_autopilot(True, self.tm_port)
                except TypeError:
                    actor.set_autopilot(True)
            spawned += 1

        logging.info("Spawned %d/%d %s actors.", spawned, desired_count, label)
        return spawned

    def _spawn_requested_traffic(self) -> None:
        assert self.world is not None
        all_vehicle_bps = list(self.world.get_blueprint_library().filter("vehicle.*"))
        car_bps = [bp for bp in all_vehicle_bps if self._is_car(bp)]
        bike_bps = [bp for bp in all_vehicle_bps if self._is_bike(bp)]
        motorbike_bps = [bp for bp in all_vehicle_bps if self._is_motorbike(bp)]

        self._spawn_npc_group("vehicle", car_bps, self.npc_vehicle_count)
        self._spawn_npc_group("bike", bike_bps, self.npc_bike_count)
        self._spawn_npc_group("motobike", motorbike_bps, self.npc_motorbike_count)

    def _spawn_pedestrians(self) -> None:
        """Spawn walker (pedestrian) actors with AI controllers near the ego vehicle."""
        if self.npc_pedestrian_count <= 0:
            return
        assert self.world is not None

        walker_bps = list(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        if not walker_bps:
            logging.warning("No pedestrian blueprints found.")
            return

        # Collect random navigation locations and filter to those near ego vehicle
        ego_loc = self._spawn_transform.location if self._spawn_transform else None
        nearby_radius = 80.0  # metres
        spawn_points = []
        attempts = self.npc_pedestrian_count * 20
        for _ in range(attempts):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            if ego_loc is not None:
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                if (dx * dx + dy * dy) > nearby_radius * nearby_radius:
                    continue
            spawn_points.append(carla.Transform(location=loc))
            if len(spawn_points) >= self.npc_pedestrian_count * 3:
                break
        if not spawn_points:
            logging.warning("Could not find navigation spawn locations for pedestrians.")
            return

        spawned_walkers = []
        for i, transform in enumerate(spawn_points):
            if len(spawned_walkers) >= self.npc_pedestrian_count:
                break
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                spawned_walkers.append(actor)
                self._walker_actors.append(actor)

        # Spawn AI controllers for each walker
        controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        for walker in spawned_walkers:
            controller = self.world.try_spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker
            )
            if controller is not None:
                self._walker_controllers.append(controller)

        # Tick once so controllers are ready
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick(self.timeout)

        # Start AI controllers — send walkers toward nearby locations
        for controller in self._walker_controllers:
            # Try multiple times to find a target near ego
            target = None
            for _ in range(30):
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                if ego_loc is not None:
                    dx = loc.x - ego_loc.x
                    dy = loc.y - ego_loc.y
                    if (dx * dx + dy * dy) <= nearby_radius * nearby_radius:
                        target = loc
                        break
                else:
                    target = loc
                    break
            if target is None:
                target = self.world.get_random_location_from_navigation()
            if target is not None:
                controller.start()
                controller.go_to_location(target)
                controller.set_max_speed(1.0 + random.random() * 1.5)

        logging.info(
            "Spawned %d/%d pedestrians with %d AI controllers.",
            len(spawned_walkers),
            self.npc_pedestrian_count,
            len(self._walker_controllers),
        )

    def _log_static_traffic_actors(self) -> None:
        assert self.world is not None
        actors = self.world.get_actors()
        traffic_lights = len(actors.filter("traffic.traffic_light*"))
        traffic_signs = len(actors.filter("traffic.traffic_sign*"))
        logging.info(
            "Town static actors: traffic_light=%d, traffic_sign=%d",
            traffic_lights,
            traffic_signs,
        )

    def _spawn_spectator_transform(self) -> Optional[CarlaTransform]:
        if self._spawn_transform is None:
            return None

        spawn = self._spawn_transform
        forward = spawn.get_forward_vector()
        loc = spawn.location
        dist = self.spectator_cfg.follow_distance
        height = self.spectator_cfg.height
        target = carla.Location(
            x=loc.x - forward.x * dist,
            y=loc.y - forward.y * dist,
            z=loc.z + height,
        )

        return carla.Transform(
            target,
            carla.Rotation(
                pitch=self.spectator_cfg.pitch,
                yaw=spawn.rotation.yaw,
                roll=0.0,
            ),
        )

    def apply_spawn_locked_spectator(self, force: bool = False) -> None:
        if not self.spectator_cfg.lock_on_spawn and not force:
            return
        if self.world is None:
            return
        transform = self._spawn_spectator_transform()
        if transform is None:
            return
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)
        if force:
            loc = transform.location
            logging.info(
                "Spectator locked at (%.1f, %.1f, %.1f) pitch=%.1f yaw=%.1f",
                loc.x, loc.y, loc.z,
                transform.rotation.pitch, transform.rotation.yaw,
            )

    def tick(self) -> None:
        assert self.world is not None
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick(self.timeout)

        if self.spectator_cfg.keep_reapply_each_tick:
            self.apply_spawn_locked_spectator()

    def cleanup(self) -> None:
        # Stop and destroy walker controllers first
        for controller in self._walker_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        for controller in self._walker_controllers:
            try:
                controller.destroy()
            except RuntimeError:
                pass
        if self._walker_controllers:
            logging.info("Destroyed %d walker controllers.", len(self._walker_controllers))
            self._walker_controllers = []

        # Destroy walker actors
        for walker in self._walker_actors:
            try:
                walker.destroy()
            except RuntimeError:
                pass
        if self._walker_actors:
            logging.info("Destroyed %d walker actors.", len(self._walker_actors))
            self._walker_actors = []

        if self._npc_actors:
            for actor in self._npc_actors:
                try:
                    actor.destroy()
                except RuntimeError:
                    pass
            logging.info("Destroyed %d NPC actors.", len(self._npc_actors))
            self._npc_actors = []

        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except RuntimeError:
                pass
            self.ego_vehicle = None
            logging.info("Destroyed ego vehicle.")
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except RuntimeError:
                pass
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
                logging.info("Restored world settings.")
            except RuntimeError:
                logging.warning("Could not restore world settings (server may be gone).")
