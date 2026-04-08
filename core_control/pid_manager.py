"""
SpeedPIDController - Production-Ready Speed Control for CARLA Autonomous Driving
Version: 3.0 (Final with Deadband + PRE-CLEAN + Integral Decay)

Features:
- Separate throttle/brake PID tuning (smooth acceleration, responsive braking)
- Deadband hysteresis (prevents chattering/oscillation)
- PRE-CLEAN state management (prevents state contamination on mode switch)
- Integral decay (smooth transitions, eliminates jerkiness)
- Dual anti-windup (clamping + conditional integration)
- Derivative filtering (exponential smoothing for noise reduction)
"""

import numpy as np
from typing import Tuple


class SpeedPIDController:
    """
    Production-ready PID controller for speed regulation in autonomous driving.
    
    Handles both throttle (acceleration) and brake (deceleration) with separate
    tuning parameters. Implements advanced anti-windup and deadband hysteresis
    to ensure smooth, stable vehicle control.
    """
    
    def __init__(
        self,
        target_speed_kmh: float = 30.0,
        # Throttle PID parameters (smooth acceleration)
        kp_throttle: float = 0.5,
        ki_throttle: float = 0.05,
        kd_throttle: float = 0.1,
        # Brake PID parameters (responsive deceleration)
        kp_brake: float = 0.8,
        ki_brake: float = 0.01,
        kd_brake: float = 0.2,
        # Control limits
        max_throttle: float = 1.0,
        max_brake: float = 1.0,
        # Anti-windup (clamping)
        integral_limit_throttle: float = 2.0,
        integral_limit_brake: float = 3.0,
        # Deadband (hysteresis zone)
        deadband_kmh: float = 0.5,
        # Integral decay rate (for smooth transitions in deadband)
        integral_decay_rate: float = 0.95,
        # Derivative filter (exponential smoothing)
        derivative_filter_alpha: float = 0.7,
        # Frame time
        dt: float = 0.05,
    ):
        """
        Initialize SpeedPIDController with tuning parameters.
        
        Args:
            target_speed_kmh: Target speed in km/h
            kp_throttle: Proportional gain for throttle (smooth acceleration)
            ki_throttle: Integral gain for throttle
            kd_throttle: Derivative gain for throttle
            kp_brake: Proportional gain for brake (responsive braking)
            ki_brake: Integral gain for brake
            kd_brake: Derivative gain for brake
            max_throttle: Maximum throttle value [0.0, 1.0]
            max_brake: Maximum brake value [0.0, 1.0]
            integral_limit_throttle: Maximum integral accumulation for throttle (anti-windup)
            integral_limit_brake: Maximum integral accumulation for brake (anti-windup)
            deadband_kmh: Dead band zone width (prevents oscillation at setpoint)
            integral_decay_rate: Exponential decay rate for integral (smooths transitions)
            derivative_filter_alpha: Filter coefficient for derivative term [0.0, 1.0]
            dt: Time step between calls (seconds, typically 0.05 for 20 FPS)
        """
        self.target_speed_kmh = target_speed_kmh
        
        # Throttle PID gains
        self.kp_throttle = kp_throttle
        self.ki_throttle = ki_throttle
        self.kd_throttle = kd_throttle
        
        # Brake PID gains
        self.kp_brake = kp_brake
        self.ki_brake = ki_brake
        self.kd_brake = kd_brake
        
        # Control limits
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        
        # Anti-windup limits
        self.integral_limit_throttle = integral_limit_throttle
        self.integral_limit_brake = integral_limit_brake
        
        # Deadband parameters
        self.deadband_kmh = deadband_kmh
        
        # Integral decay (smooth out transitions)
        self.integral_decay_rate = integral_decay_rate
        
        # Derivative filter coefficient
        self.derivative_filter_alpha = derivative_filter_alpha
        
        # Time step
        self.dt = dt
        
        # State variables
        self.current_mode = 'idle'  # 'idle', 'throttle', or 'brake'
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.integral_t = 0.0  # Throttle integral accumulator
        self.integral_b = 0.0  # Brake integral accumulator
    
    def set_target_speed(self, target_speed_kmh: float) -> None:
        """Update target speed."""
        self.target_speed_kmh = target_speed_kmh
    
    def _determine_mode(self, error: float) -> str:
        """
        Determine control mode (throttle/brake) with deadband hysteresis.
        
        Deadband prevents constant switching when oscillating near setpoint:
        - Within deadband: maintain current mode
        - Above deadband: switch to throttle mode
        - Below deadband: switch to brake mode
        
        Args:
            error: Speed error (target - current) in km/h
            
        Returns:
            Mode: 'throttle', 'brake', or 'idle'
        """
        # Within deadband: maintain current mode (hysteresis)
        if abs(error) < self.deadband_kmh:
            return self.current_mode
        
        # Outside deadband: determine new mode
        if error > 0.0:
            return 'throttle'
        else:
            return 'brake'
    
    def _clean_throttle_state(self) -> None:
        """
        Clean throttle state when switching from throttle to brake.
        PRE-CLEAN: Called BEFORE mode change to prevent state contamination.
        
        NOTE: We preserve self.prev_error to maintain continuous velocity history.
        Resetting prev_error would cause a false spike in the derivative term
        when the mode switches, creating unnatural jerking behavior.
        """
        self.integral_t = 0.0
        self.prev_derivative = 0.0
    
    def _clean_brake_state(self) -> None:
        """
        Clean brake state when switching from brake to throttle.
        PRE-CLEAN: Called BEFORE mode change to prevent state contamination.
        
        NOTE: We preserve self.prev_error to maintain continuous velocity history.
        Resetting prev_error would cause a false spike in the derivative term
        when the mode switches, creating unnatural jerking behavior.
        """
        self.integral_b = 0.0
        self.prev_derivative = 0.0
    
    def _apply_integral_decay(self, error: float) -> None:
        """
        Apply exponential decay to integral terms when in deadband.
        
        Smooths transitions and eliminates "khựng nhẹ" (slight jerkiness)
        when vehicle enters deadband zone.
        
        Args:
            error: Current speed error in km/h
        """
        if abs(error) < self.deadband_kmh:
            # In deadband: apply decay to both integrals
            self.integral_t *= self.integral_decay_rate
            self.integral_b *= self.integral_decay_rate
    
    def compute(self, current_speed_kmh: float) -> Tuple[float, float]:
        """
        Compute throttle and brake values using PID control.
        
        Algorithm:
        1. Calculate error (target - current)
        2. Determine mode with deadband hysteresis
        3. PRE-CLEAN state if switching modes
        4. Compute PID terms (P, I, D)
        5. Apply anti-windup (clamping)
        6. Apply integral decay in deadband
        7. Return throttle and brake values
        
        Args:
            current_speed_kmh: Current speed in km/h
            
        Returns:
            Tuple[throttle, brake]: Control values in [0.0, 1.0]
        """
        # Calculate error
        error = self.target_speed_kmh - current_speed_kmh
        
        # Determine mode with deadband hysteresis
        new_mode = self._determine_mode(error)
        
        # PRE-CLEAN state BEFORE switching modes (prevents contamination)
        if new_mode != self.current_mode:
            if self.current_mode == 'throttle':
                self._clean_throttle_state()
            elif self.current_mode == 'brake':
                self._clean_brake_state()
            self.current_mode = new_mode
        
        # Initialize outputs
        throttle = 0.0
        brake = 0.0
        
        # ==================== THROTTLE MODE ====================
        if self.current_mode == 'throttle' and error > 0.0:
            # Proportional term
            p_term = self.kp_throttle * error
            
            # Integral term (anti-windup: only accumulate if not saturated)
            if abs(self.integral_t) < self.integral_limit_throttle:
                self.integral_t += error * self.dt
            # Clamp integral
            self.integral_t = np.clip(
                self.integral_t,
                -self.integral_limit_throttle,
                self.integral_limit_throttle
            )
            i_term = self.ki_throttle * self.integral_t
            
            # Derivative term (with filter to reduce noise)
            derivative = (error - self.prev_error) / self.dt
            self.prev_derivative = (
                self.derivative_filter_alpha * derivative +
                (1.0 - self.derivative_filter_alpha) * self.prev_derivative
            )
            d_term = self.kd_throttle * self.prev_derivative
            
            # Compute PID output
            throttle = p_term + i_term + d_term
            throttle = np.clip(throttle, 0.0, self.max_throttle)
            brake = 0.0
        
        # ==================== BRAKE MODE ====================
        elif self.current_mode == 'brake' and error < 0.0:
            # Proportional term (use absolute error)
            p_term = self.kp_brake * abs(error)
            
            # Integral term (anti-windup: only accumulate if not saturated)
            if abs(self.integral_b) < self.integral_limit_brake:
                self.integral_b += abs(error) * self.dt
            # Clamp integral
            self.integral_b = np.clip(
                self.integral_b,
                -self.integral_limit_brake,
                self.integral_limit_brake
            )
            i_term = self.ki_brake * self.integral_b
            
            # Derivative term (with filter)
            derivative = (abs(error) - abs(self.prev_error)) / self.dt
            self.prev_derivative = (
                self.derivative_filter_alpha * derivative +
                (1.0 - self.derivative_filter_alpha) * self.prev_derivative
            )
            d_term = self.kd_brake * self.prev_derivative
            
            # Compute PID output
            brake = p_term + i_term + d_term
            brake = np.clip(brake, 0.0, self.max_brake)
            throttle = 0.0
        
        # ==================== IDLE MODE ====================
        else:
            # No acceleration or braking
            throttle = 0.0
            brake = 0.0
        
        # Apply integral decay in deadband zone (smooth transitions)
        self._apply_integral_decay(error)
        
        # Update previous error for next iteration
        self.prev_error = error
        
        return throttle, brake
    
    def reset(self) -> None:
        """Reset controller state (use after vehicle stops or on restart)."""
        self.current_mode = 'idle'
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.integral_t = 0.0
        self.integral_b = 0.0
    
    def get_state(self) -> dict:
        """Return current controller state (for debugging/logging)."""
        return {
            'mode': self.current_mode,
            'integral_throttle': self.integral_t,
            'integral_brake': self.integral_b,
            'prev_error': self.prev_error,
            'prev_derivative': self.prev_derivative,
        }
