from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Type

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    carla = None


@dataclass
class RunConfig:
    host: str
    port: int
    tm_port: int
    timeout: float
    sync: bool
    fixed_delta: float
    no_rendering: bool
    vehicle_filter: str
    spawn_point: int
    ticks: int
    tick_interval: float
    dry_run: bool
    seed: Optional[int]


class BaseSession:
    """Shared interface for CARLA and dry-run sessions."""

    def start(self) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError

    @property
    def ego_vehicle(self):
        return None


class CarlaSession(BaseSession):
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.client = None
        self.world = None
        self.tm = None
        self._original_settings = None
        self._ego_vehicle = None

    @property
    def ego_vehicle(self):
        return self._ego_vehicle

    def start(self) -> None:
        assert carla is not None
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.get_world()
        self.tm = self.client.get_trafficmanager(self.config.tm_port)
        self._original_settings = self.world.get_settings()
        self._apply_world_settings()
        self._spawn_ego_vehicle()
        logging.info("CARLA session is ready.")

    def _apply_world_settings(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = self.config.sync
        settings.fixed_delta_seconds = self.config.fixed_delta if self.config.sync else None
        settings.no_rendering_mode = self.config.no_rendering
        self.world.apply_settings(settings)
        if self.tm is not None:
            self.tm.set_synchronous_mode(self.config.sync)

    def _spawn_ego_vehicle(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        blueprints = blueprint_library.filter(self.config.vehicle_filter)
        if not blueprints:
            raise RuntimeError(
                f"No vehicle blueprint matches '{self.config.vehicle_filter}'."
            )
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in current map.")

        preferred_index = self.config.spawn_point
        if preferred_index >= 0:
            ordered_points = [spawn_points[preferred_index % len(spawn_points)]]
        else:
            ordered_points = spawn_points.copy()
            random.shuffle(ordered_points)

        self._ego_vehicle = None
        for transform in ordered_points:
            self._ego_vehicle = self.world.try_spawn_actor(blueprint, transform)
            if self._ego_vehicle is not None:
                break

        if self._ego_vehicle is None:
            raise RuntimeError("Cannot spawn ego vehicle at available spawn points.")
        logging.info("Spawned ego vehicle: %s", self._ego_vehicle.type_id)

    def tick(self) -> None:
        if self.config.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick(self.config.timeout)

    def cleanup(self) -> None:
        if self._ego_vehicle is not None:
            self._ego_vehicle.destroy()
            self._ego_vehicle = None
            logging.info("Destroyed ego vehicle.")
        if self.world is not None and self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            logging.info("Restored world settings.")


class DryRunSession(BaseSession):
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.step = 0

    def start(self) -> None:
        logging.info("Starting dry-run mode (CARLA not required).")

    def tick(self) -> None:
        self.step += 1
        time.sleep(self.config.tick_interval)

    def cleanup(self) -> None:
        logging.info("Dry-run session ended after %d ticks.", self.step)


class BaseAgent:
    name = "base"

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.session: Optional[BaseSession] = None

    def setup(self, session: BaseSession) -> None:
        self.session = session

    def run_step(self, step_idx: int) -> None:
        raise NotImplementedError

    def teardown(self) -> None:
        return


class AutopilotAgent(BaseAgent):
    name = "autopilot"

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        if vehicle is None:
            logging.info("No ego vehicle in this session; autopilot setup skipped.")
            return
        try:
            vehicle.set_autopilot(True, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(True)
        logging.info("Autopilot enabled for ego vehicle.")

    def run_step(self, step_idx: int) -> None:
        if step_idx % 20 == 0:
            logging.info("Agent tick %d", step_idx)

    def teardown(self) -> None:
        if self.session is None:
            return
        vehicle = self.session.ego_vehicle
        if vehicle is None:
            return
        try:
            vehicle.set_autopilot(False, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(False)


class NoopAgent(BaseAgent):
    name = "noop"

    def run_step(self, step_idx: int) -> None:
        if step_idx % 50 == 0:
            logging.info("Noop agent alive at tick %d", step_idx)


AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    AutopilotAgent.name: AutopilotAgent,
    NoopAgent.name: NoopAgent,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARLA agent loop.")
    parser.add_argument("--agent", choices=sorted(AGENT_REGISTRY), default="autopilot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=1000.0)
    parser.add_argument("--sync", action="store_true", help="Enable synchronous mode.")
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
        help="Fixed delta seconds (used when --sync is enabled).",
    )
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=-1,
        help="Spawn-point index. Negative means random.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=200,
        help="Number of ticks to run. Use 0 or negative for infinite loop.",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=0.05,
        help="Sleep interval per tick in dry-run mode.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        timeout=args.timeout,
        sync=args.sync,
        fixed_delta=args.fixed_delta,
        no_rendering=args.no_rendering,
        vehicle_filter=args.vehicle_filter,
        spawn_point=args.spawn_point,
        ticks=args.ticks,
        tick_interval=args.tick_interval,
        dry_run=args.dry_run,
        seed=args.seed,
    )


def build_session(config: RunConfig) -> BaseSession:
    if config.dry_run:
        return DryRunSession(config)
    if carla is None:
        raise RuntimeError(
            "Python package 'carla' is not installed. Use --dry-run or install CARLA API."
        )
    return CarlaSession(config)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(args)
    if config.seed is not None:
        random.seed(config.seed)

    session = build_session(config)
    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls(config)
    tick_limit = config.ticks

    try:
        session.start()
        agent.setup(session)
        step = 0
        while tick_limit <= 0 or step < tick_limit:
            step += 1
            agent.run_step(step)
            session.tick()
        logging.info("Finished %d ticks.", step)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logging.error("Agent run failed: %s", exc)
        return 1
    finally:
        try:
            agent.teardown()
        finally:
            session.cleanup()


if __name__ == "__main__":
    sys.exit(main())
