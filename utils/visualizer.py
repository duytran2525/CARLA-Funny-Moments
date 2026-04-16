from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping, Optional

try:
	import cv2
except ImportError:
	cv2 = None

try:
	import numpy as np
except ImportError:
	np = None


def _clamp(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def _to_float(value: Any, default: float = 0.0) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _to_int(value: Any, default: int = 0) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return default


class DrivingVisualizer:
	"""Render a lightweight driving HUD for CARLA runtime monitoring."""

	COMMAND_LABELS = {
		0: "LANE_FOLLOW",
		1: "LEFT",
		2: "RIGHT",
		3: "STRAIGHT",
	}

	def __init__(self, window_name: str = "CARLA Driving HUD") -> None:
		self.window_name = window_name
		self.enabled = cv2 is not None
		self._last_show_time: Optional[float] = None
		self._ema_fps: Optional[float] = None
		self._fps_alpha = 0.90

	def annotate_bgr(
		self,
		frame_bgr: Any,
		metrics: Mapping[str, Any],
		extra_lines: Optional[Iterable[str]] = None,
	) -> Any:
		if frame_bgr is None or cv2 is None:
			return frame_bgr

		output = frame_bgr.copy()
		height, width = output.shape[:2]

		panel_x = 12
		panel_y = 12
		panel_w = max(240, min(520, width - 2 * panel_x))

		lines, status_color = self._build_lines(metrics, extra_lines)
		line_h = 24
		panel_h = min(height - 2 * panel_y, 48 + line_h * len(lines) + 44)

		overlay = output.copy()
		cv2.rectangle(
			overlay,
			(panel_x, panel_y),
			(panel_x + panel_w, panel_y + panel_h),
			(18, 18, 18),
			thickness=-1,
		)
		cv2.addWeighted(overlay, 0.50, output, 0.50, 0.0, output)

		cv2.rectangle(
			output,
			(panel_x, panel_y),
			(panel_x + panel_w, panel_y + panel_h),
			status_color,
			thickness=2,
		)

		y = panel_y + 28
		for text, color in lines:
			cv2.putText(
				output,
				text,
				(panel_x + 12, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.58,
				color,
				2,
			)
			y += line_h

		steer = _clamp(_to_float(metrics.get("steer", 0.0)), -1.0, 1.0)
		self._draw_steering_bar(
			output,
			x=panel_x + 12,
			y=panel_y + panel_h - 26,
			width=panel_w - 24,
			steer=steer,
		)
		return output

	def show_bgr(
		self,
		frame_bgr: Any,
		metrics: Mapping[str, Any],
		extra_lines: Optional[Iterable[str]] = None,
	) -> None:
		if not self.enabled or cv2 is None or frame_bgr is None:
			return
		metrics_with_fps = dict(metrics)
		if "fps" not in metrics_with_fps:
			metrics_with_fps["fps"] = self._update_fps()
		annotated = self.annotate_bgr(frame_bgr, metrics_with_fps, extra_lines=extra_lines)
		cv2.imshow(self.window_name, annotated)
		cv2.waitKey(1)

	def show_rgb(
		self,
		frame_rgb: Any,
		metrics: Mapping[str, Any],
		extra_lines: Optional[Iterable[str]] = None,
	) -> None:
		if not self.enabled or cv2 is None or frame_rgb is None:
			return
		frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
		self.show_bgr(frame_bgr, metrics, extra_lines=extra_lines)

	def close(self) -> None:
		if not self.enabled or cv2 is None:
			return
		try:
			cv2.destroyWindow(self.window_name)
		except Exception:
			pass

	def _update_fps(self) -> float:
		now = time.perf_counter()
		if self._last_show_time is None:
			self._last_show_time = now
			return 0.0

		dt = now - self._last_show_time
		self._last_show_time = now
		if dt <= 1e-6:
			return float(self._ema_fps) if self._ema_fps is not None else 0.0

		instant_fps = 1.0 / dt
		if self._ema_fps is None:
			self._ema_fps = instant_fps
		else:
			self._ema_fps = (self._fps_alpha * self._ema_fps) + ((1.0 - self._fps_alpha) * instant_fps)
		return self._ema_fps

	def _build_lines(
		self,
		metrics: Mapping[str, Any],
		extra_lines: Optional[Iterable[str]],
	) -> tuple[list[tuple[str, tuple[int, int, int]]], tuple[int, int, int]]:
		agent = str(metrics.get("agent", "agent")).upper()
		tick = _to_int(metrics.get("tick", 0))
		speed_kmh = _to_float(metrics.get("speed_kmh", 0.0))
		target_speed_kmh = _to_float(metrics.get("target_speed_kmh", 0.0))
		fps = _to_float(metrics.get("fps", 0.0))
		steer = _clamp(_to_float(metrics.get("steer", 0.0)), -1.0, 1.0)
		steer_raw = metrics.get("steer_raw", None)
		throttle = _clamp(_to_float(metrics.get("throttle", 0.0)), 0.0, 1.0)
		brake = _clamp(_to_float(metrics.get("brake", 0.0)), 0.0, 1.0)
		yaw_deg = metrics.get("yaw_deg", None)
		command = metrics.get("command", None)
		emergency = bool(metrics.get("emergency", False))
		reason = str(metrics.get("reason", "")).strip()

		lines: list[tuple[str, tuple[int, int, int]]] = [
			(f"Agent: {agent} | Tick: {tick}", (255, 255, 255)),
			(
				f"Speed: {speed_kmh:5.1f} km/h | Target: {target_speed_kmh:5.1f} km/h",
				(255, 255, 255),
			),
		]

		if steer_raw is None:
			lines.append((f"Steering: {steer:+.3f}", (40, 220, 255)))
		else:
			lines.append(
				(
					f"Steering: {steer:+.3f} | Raw: {_clamp(_to_float(steer_raw), -1.0, 1.0):+.3f}",
					(40, 220, 255),
				)
			)

		lines.append((f"Throttle: {throttle:.2f} | Brake: {brake:.2f}", (210, 255, 210)))
		if yaw_deg is None:
			lines.append((f"FPS: {fps:5.1f}", (235, 235, 235)))
		else:
			yaw = _to_float(yaw_deg, 0.0)
			lines.append((f"FPS: {fps:5.1f} | Heading Yaw: {yaw:+6.1f} deg", (235, 235, 235)))

		if command is not None:
			command_id = _to_int(command)
			command_label = self.COMMAND_LABELS.get(command_id, f"CMD_{command_id}")
			lines.append((f"Command: {command_label} ({command_id})", (255, 235, 120)))

		if emergency:
			emg_text = "Emergency: ON"
			if reason:
				emg_text = f"{emg_text} | {reason}"
			lines.append((emg_text, (0, 85, 255)))
			status_color = (0, 85, 255)
		else:
			status_color = (0, 170, 80)

		if extra_lines is not None:
			for line in extra_lines:
				if line:
					lines.append((str(line), (235, 235, 235)))

		return lines, status_color

	def _draw_steering_bar(self, frame_bgr: Any, x: int, y: int, width: int, steer: float) -> None:
		if cv2 is None:
			return

		width = max(120, width)
		bar_h = 16
		bar_y0 = y - bar_h
		bar_y1 = y

		cv2.putText(
			frame_bgr,
			"Steering [-1 .. +1]",
			(x, bar_y0 - 8),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(210, 210, 210),
			1,
		)

		cv2.rectangle(frame_bgr, (x, bar_y0), (x + width, bar_y1), (180, 180, 180), 1)
		center_x = x + (width // 2)
		cv2.line(frame_bgr, (center_x, bar_y0), (center_x, bar_y1), (120, 120, 120), 1)

		steer = _clamp(steer, -1.0, 1.0)
		span = (width / 2.0) - 4.0
		marker_x = int(round(center_x + steer * span))
		marker_y = bar_y0 + (bar_h // 2)

		abs_steer = abs(steer)
		if abs_steer < 0.20:
			marker_color = (0, 200, 70)
		elif abs_steer < 0.50:
			marker_color = (0, 220, 255)
		else:
			marker_color = (0, 120, 255)

		cv2.circle(frame_bgr, (marker_x, marker_y), 5, marker_color, -1)
		cv2.putText(
			frame_bgr,
			f"{steer:+.3f}",
			(x + width + 10, marker_y + 5),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.50,
			marker_color,
			2,
		)


class RouteMapVisualizer:
	"""Render a lightweight 2D route map from planner waypoints."""

	COMMAND_LABELS = {
		0: "LANE_FOLLOW",
		1: "LEFT",
		2: "RIGHT",
		3: "STRAIGHT",
	}

	def __init__(self, window_name: str = "CARLA Route Map", canvas_size: int = 560) -> None:
		self.window_name = window_name
		self.canvas_size = max(320, int(canvas_size))
		self.enabled = cv2 is not None and np is not None

	def show(
		self,
		route_points: Optional[Iterable[Any]],
		current_location: Optional[Any],
		start_location: Optional[Any],
		destination_location: Optional[Any],
		heading_yaw_deg: Optional[float] = None,
		trajectory_points: Optional[Iterable[Any]] = None,
		command: Optional[int] = None,
	) -> None:
		if not self.enabled or cv2 is None or np is None:
			return

		canvas = np.full((self.canvas_size, self.canvas_size, 3), 20, dtype=np.uint8)
		padding = 28

		route_xy = self._collect_xy(route_points)
		trajectory_xy = self._collect_xy(trajectory_points)
		start_xy = self._to_xy(start_location)
		dest_xy = self._to_xy(destination_location)
		current_xy = self._to_xy(current_location)

		all_xy = list(route_xy)
		all_xy.extend(trajectory_xy)
		for pt in (start_xy, dest_xy, current_xy):
			if pt is not None:
				all_xy.append(pt)

		if not all_xy:
			cv2.putText(
				canvas,
				"Route map waiting for planner data...",
				(20, self.canvas_size // 2),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.62,
				(180, 180, 180),
				2,
			)
			cv2.imshow(self.window_name, canvas)
			cv2.waitKey(1)
			return

		bounds = self._compute_bounds(all_xy)
		proj = self._build_projector(bounds, self.canvas_size, padding)

		# Draw light grid for orientation.
		grid_color = (45, 45, 45)
		step = max(40, (self.canvas_size - 2 * padding) // 6)
		for offset in range(padding, self.canvas_size - padding + 1, step):
			cv2.line(canvas, (offset, padding), (offset, self.canvas_size - padding), grid_color, 1)
			cv2.line(canvas, (padding, offset), (self.canvas_size - padding, offset), grid_color, 1)

		if len(route_xy) >= 2:
			route_pixels = np.array([proj(pt) for pt in route_xy], dtype=np.int32).reshape((-1, 1, 2))
			cv2.polylines(canvas, [route_pixels], False, (0, 220, 255), 2)

		if len(trajectory_xy) >= 2:
			traj_pixels = np.array([proj(pt) for pt in trajectory_xy], dtype=np.int32).reshape((-1, 1, 2))
			cv2.polylines(canvas, [traj_pixels], False, (0, 140, 255), 2)

		if start_xy is not None:
			sx, sy = proj(start_xy)
			cv2.circle(canvas, (sx, sy), 6, (40, 220, 40), -1)
			cv2.putText(canvas, "S", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)

		if dest_xy is not None:
			dx, dy = proj(dest_xy)
			cv2.circle(canvas, (dx, dy), 6, (0, 0, 255), -1)
			cv2.putText(canvas, "D", (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

		if current_xy is not None:
			cx, cy = proj(current_xy)
			cv2.circle(canvas, (cx, cy), 7, (0, 255, 255), -1)
			cv2.circle(canvas, (cx, cy), 11, (0, 255, 255), 1)
			if heading_yaw_deg is not None:
				yaw_rad = math.radians(float(heading_yaw_deg))
				arrow_world = (current_xy[0] + 3.5 * math.cos(yaw_rad), current_xy[1] + 3.5 * math.sin(yaw_rad))
				ax, ay = proj(arrow_world)
				cv2.arrowedLine(canvas, (cx, cy), (ax, ay), (0, 255, 255), 2, tipLength=0.35)

		cv2.rectangle(canvas, (padding, padding), (self.canvas_size - padding, self.canvas_size - padding), (120, 120, 120), 1)

		cmd_text = "-"
		if command is not None:
			command_id = _to_int(command)
			cmd_text = f"{self.COMMAND_LABELS.get(command_id, f'CMD_{command_id}')} ({command_id})"

		legend = [
			"Route Map (Navigator vs Vehicle)",
			"Cyan: planner path | Orange: vehicle trail",
			f"Command: {cmd_text}",
		]
		ly = 24
		for text in legend:
			cv2.putText(canvas, text, (14, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
			ly += 20

		cv2.imshow(self.window_name, canvas)
		cv2.waitKey(1)

	def close(self) -> None:
		if not self.enabled or cv2 is None:
			return
		try:
			cv2.destroyWindow(self.window_name)
		except Exception:
			pass

	def _collect_xy(self, points: Optional[Iterable[Any]]) -> list[tuple[float, float]]:
		result: list[tuple[float, float]] = []
		if points is None:
			return result
		for point in points:
			xy = self._to_xy(point)
			if xy is not None:
				result.append(xy)
		return result

	def _to_xy(self, value: Optional[Any]) -> Optional[tuple[float, float]]:
		if value is None:
			return None
		try:
			return (float(value.x), float(value.y))
		except Exception:
			pass
		if isinstance(value, (tuple, list)) and len(value) >= 2:
			try:
				return (float(value[0]), float(value[1]))
			except (TypeError, ValueError):
				return None
		return None

	def _compute_bounds(self, points_xy: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
		x_values = [pt[0] for pt in points_xy]
		y_values = [pt[1] for pt in points_xy]
		min_x, max_x = min(x_values), max(x_values)
		min_y, max_y = min(y_values), max(y_values)

		span_x = max(1.0, max_x - min_x)
		span_y = max(1.0, max_y - min_y)
		margin_x = max(4.0, span_x * 0.15)
		margin_y = max(4.0, span_y * 0.15)
		return (min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y)

	def _build_projector(
		self,
		bounds: tuple[float, float, float, float],
		canvas_size: int,
		padding: int,
	):
		min_x, max_x, min_y, max_y = bounds
		span_x = max(1e-3, max_x - min_x)
		span_y = max(1e-3, max_y - min_y)
		usable = max(1, canvas_size - 2 * padding)
		scale = min(usable / span_x, usable / span_y)
		offset_x = (usable - span_x * scale) * 0.5
		offset_y = (usable - span_y * scale) * 0.5

		def project(pt: tuple[float, float]) -> tuple[int, int]:
			x, y = pt
			u = int(round(padding + offset_x + (x - min_x) * scale))
			v = int(round(canvas_size - (padding + offset_y + (y - min_y) * scale)))
			return (u, v)

		return project
