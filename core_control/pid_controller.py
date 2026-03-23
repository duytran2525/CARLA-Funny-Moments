"""
PID Controller – Phase 2
========================
Điều khiển PID cho tốc độ dọc (longitudinal) của xe CARLA.

Được dùng bởi ``run_hybrid_agent.py`` khi không có tín hiệu override từ ADAS.
``ADASModule`` trong ``adas_logic.py`` có thể ghi đè lệnh phanh khi TTC nguy hiểm.

Sử dụng
-------
::

    pid = PIDController(kp=0.5, ki=0.02, kd=0.1)
    throttle = pid.step(error=target_speed - current_speed, dt=0.05)
"""

from __future__ import annotations


class PIDController:
    """
    Bộ điều khiển PID đơn giản cho một trục điều khiển.

    Parameters
    ----------
    kp : float
        Hệ số tỉ lệ (Proportional gain).
    ki : float
        Hệ số tích phân (Integral gain).
    kd : float
        Hệ số đạo hàm (Derivative gain).
    output_min : float
        Giới hạn dưới của đầu ra. Mặc định 0.0.
    output_max : float
        Giới hạn trên của đầu ra. Mặc định 1.0.
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.02,
        kd: float = 0.1,
        output_min: float = 0.0,
        output_max: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        """
        Tính đầu ra PID cho một bước thời gian.

        Parameters
        ----------
        error : float
            Sai số hiện tại (target - current).
        dt : float
            Khoảng thời gian từ bước trước (giây). Phải > 0.

        Returns
        -------
        float
            Đầu ra đã clamp về ``[output_min, output_max]``.
        """
        if dt <= 0:
            dt = 1e-3

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))

    def reset(self) -> None:
        """Reset tích phân và sai số trước về 0 (dùng khi bắt đầu episode mới)."""
        self._integral = 0.0
        self._prev_error = 0.0
