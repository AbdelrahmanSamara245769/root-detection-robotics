# pid.py
"""
Discrete PID controller – Task 10.

Design goals:
- Explicit P, I, D formulation
- Fixed dt for reproducibility
- Output saturation (actuator limits)
- Integral clamping to reduce windup
- Simple and easy to document

This implementation is intentionally different from common
wall-clock-based PIDs while remaining practical.
"""

class PID:
    def __init__(
        self,
        kp=0.0,
        ki=0.0,
        kd=0.0,
        dt=0.01,
        output_limit=None,
        integral_limit=None
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def step(self, error):
        """
        Compute PID output from error.

        Args:
            error (float): setpoint − measurement

        Returns:
            float: control output
        """
        # Proportional
        p = self.kp * error

        # Integral
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = max(
                min(self.integral, self.integral_limit),
                -self.integral_limit
            )
        i = self.ki * self.integral

        # Derivative
        if not self.initialized:
            d = 0.0
            self.initialized = True
        else:
            d = self.kd * (error - self.prev_error) / self.dt

        self.prev_error = error

        # PID output
        output = p + i + d

        # Output saturation
        if self.output_limit is not None:
            output = max(min(output, self.output_limit), -self.output_limit)

        return output