"""
Humanized Waypoint Perturbation – Anti-Cheat Driving Module.

Adds coherent, time-continuous lateral jitter and steering micro-oscillation
to make the autonomous vehicle drive like a real human instead of a perfect
center-line-tracking robot.

Key techniques:
* Smoothed lateral offset (low-frequency sine + tiny Gaussian touch)
* Speed-dependent amplitude damping (faster → straighter)
* Turn suppression (less jitter when turning)
* EMA smoothing (no sudden jumps between frames)
* Lane-width aware clamping
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b*."""
    return a + (b - a) * t


class WaypointHumanizer:
    """Add coherent lateral jitter to ego-frame waypoints.

    The jitter is Perlin-like (sine-wave based) so that it oscillates
    smoothly over time rather than jumping randomly each frame.

    Parameters
    ----------
    amplitude : float
        Peak lateral offset in **metres**.  0.25 m is "visibly noticeable
        but safe inside a 3.5 m lane".
    frequency : float
        Oscillation speed in Hz.  0.4–0.7 Hz feels human-like.
    ema_alpha : float
        Exponential Moving Average blending factor (0→very smooth, 1→raw).
    half_lane_width_m : float
        Hard clamp so the offset never exceeds half a lane width.
    """

    def __init__(
        self,
        amplitude: float = 0.25,
        frequency: float = 0.55,
        ema_alpha: float = 0.08,
        half_lane_width_m: float = 1.6,
    ) -> None:
        self._amplitude = float(amplitude)
        self._frequency = float(frequency)
        self._ema_alpha = float(ema_alpha)
        self._half_lane_width = float(half_lane_width_m)

        # Internal state – evolves across frames
        self._phase: float = random.uniform(0.0, 2.0 * math.pi)
        self._current_offset: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def humanize_waypoints(
        self,
        waypoints: Any,
        speed_kmh: float,
        command: int,
        dt: float,
        half_lane_width_m: Optional[float] = None,
    ) -> Any:
        """Apply coherent lateral perturbation to *waypoints* (ego-frame).

        Parameters
        ----------
        waypoints : ndarray (N, 2)
            Ego-centric waypoints – column 0 is forward (X), column 1 is
            lateral (Y, positive = left in CARLA).
        speed_kmh : float
            Current vehicle speed.
        command : int
            Navigation command (0 = lane follow, 1 = left, 2 = right,
            3 = straight).
        dt : float
            Time step since last call (seconds).

        Returns
        -------
        ndarray (N, 2)
            Perturbed waypoints (same shape, **new** array).
        """
        if np is None:
            return waypoints

        wp = np.array(waypoints, dtype=np.float32, copy=True)
        if wp.ndim != 2 or wp.shape[1] != 2:
            return wp

        # 1. Advance the oscillation phase
        self._phase += dt * self._frequency * 2.0 * math.pi
        # Keep phase bounded to avoid float overflow over very long runs
        if self._phase > 1e6:
            self._phase -= 1e6

        # 2. Base offset: sine wave + tiny Gaussian touch
        base_offset = math.sin(self._phase) * self._amplitude
        base_offset += random.gauss(0.0, 0.05 * self._amplitude)

        # 3. Speed damping – faster → less lateral wander
        speed_factor = max(0.3, 1.0 - float(speed_kmh) / 60.0)

        # 4. Turn suppression – don't jitter during active manoeuvres
        turn_factor = 0.25 if int(command) in (1, 2, 3) else 1.0

        # 5. Combine and EMA-smooth
        target_offset = base_offset * speed_factor * turn_factor
        self._current_offset = _lerp(
            self._current_offset, target_offset, self._ema_alpha,
        )

        # 6. Hard clamp to lane width
        lane_half_width = self._half_lane_width
        if (
            half_lane_width_m is not None
            and math.isfinite(float(half_lane_width_m))
            and float(half_lane_width_m) > 0.1
        ):
            lane_half_width = float(half_lane_width_m)

        self._current_offset = max(
            -lane_half_width,
            min(lane_half_width, self._current_offset),
        )

        # 7. Apply to each waypoint's Y axis with distance decay
        n = wp.shape[0]
        for i in range(n):
            # Further waypoints get progressively less offset so that
            # pure-pursuit sees a coherent, gentle curve – not a parallel
            # shift of the entire trajectory.
            decay = 1.0 - (i / max(1, n)) * 0.5
            wp[i, 1] += self._current_offset * decay

        return wp

    def reset(self) -> None:
        """Re-initialise state (e.g. after a respawn)."""
        self._phase = random.uniform(0.0, 2.0 * math.pi)
        self._current_offset = 0.0


class SteeringMicroOscillator:
    """Simulate tiny steering-wheel corrections that real drivers make.

    Overlays a low-amplitude, medium-frequency sine wave onto the final
    steering value – just enough to avoid perfectly straight driving.

    Parameters
    ----------
    amplitude : float
        Peak oscillation in normalised steering units (±1 range).
        0.008 ≈ barely visible wobble.
    frequency_hz : float
        Oscillation frequency.  ~2.5 Hz is similar to a human's micro-
        corrections.
    """

    def __init__(
        self,
        amplitude: float = 0.008,
        frequency_hz: float = 2.5,
    ) -> None:
        self._amplitude = float(amplitude)
        self._frequency_hz = float(frequency_hz)
        self._phase: float = random.uniform(0.0, 2.0 * math.pi)

    def apply(
        self,
        steering: float,
        speed_kmh: float,
        command: int,
        dt: float,
    ) -> float:
        """Add micro-oscillation to *steering* and return the result."""
        self._phase += dt * self._frequency_hz * 2.0 * math.pi
        if self._phase > 1e6:
            self._phase -= 1e6

        micro = math.sin(self._phase) * self._amplitude

        # Suppress during turns
        if int(command) in (1, 2, 3):
            micro *= 0.3

        # Suppress at high speed
        if float(speed_kmh) > 40.0:
            micro *= max(0.4, 1.0 - float(speed_kmh) / 80.0)

        return max(-1.0, min(1.0, float(steering) + micro))

    def reset(self) -> None:
        self._phase = random.uniform(0.0, 2.0 * math.pi)
