"""
6-DoF Flight Trajectory Generator for UAV and Autonomous Flight Simulation.
"""

import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class FlightState:
    time: float
    # True Euler angles [rad]
    roll: float
    pitch: float
    yaw: float
    # True Euler angles [deg]
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    # True Body Angular Velocity [rad/s]
    gyro: Tuple[float, float, float]
    # True Specific Force in Body Frame [m/s^2]
    accel: Tuple[float, float, float]
    # True Earth Magnetic Field in Body Frame
    mag: Tuple[float, float, float]


class TrajectoryGenerator:
    """Generates deterministic, continuous 6-DoF flight dynamics for a multirotor UAV."""

    def __init__(self, duration_sec: float = 30.0, rate_hz: float = 200.0):
        self.duration = duration_sec
        self.dt = 1.0 / rate_hz
        self.rate_hz = rate_hz
        self.gravity = 9.80665

    def generate(self):
        """Yields ground truth FlightState over the mission profile."""
        total_steps = int(self.duration * self.rate_hz)
        prev_roll = 0.0
        prev_pitch = 0.0
        prev_yaw = 0.0

        for i in range(total_steps):
            t = i * self.dt

            # Multi-phase flight profile
            if t < 2.0:
                roll = 0.0
                pitch = 0.0
                yaw = 0.0
            elif t < 6.0:
                phase = (t - 2.0) / 4.0
                roll = 0.05 * math.sin(2.0 * math.pi * phase)
                pitch = 0.10 * math.sin(math.pi * phase)
                yaw = 0.20 * phase
            elif t < 16.0:
                phase = (t - 6.0)
                roll = 0.35 * math.sin(0.8 * phase)  # 20 deg bank
                pitch = 0.15 * math.cos(1.6 * phase) # 8.5 deg pitch
                yaw = 0.40 * phase                   # Continuous turn
            elif t < 24.0:
                phase = (t - 16.0)
                roll = 0.45 * math.cos(1.8 * phase) * math.exp(-0.05 * phase)
                pitch = 0.20 * math.sin(1.2 * phase)
                yaw = 4.0 + 0.3 * phase
            else:
                phase = min(1.0, (t - 24.0) / 6.0)
                roll = (1.0 - phase) * 0.1 * math.sin(2.0 * phase)
                pitch = (1.0 - phase) * 0.05
                yaw = 6.4

            # Wrap yaw into [-pi, pi] for Euler reporting
            wrapped_yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi

            # Construct true quaternion
            cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
            cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
            cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            # Quaternion conjugate for Nav -> Body transformation
            # v_body = q^* * v_nav * q
            def unrotate(v):
                # q_conj = (qw, -qx, -qy, -qz)
                # v_body = v + 2*w*(q_v x v) + 2*(q_v x (q_v x v)) with q_v = (-qx, -qy, -qz)
                qx_c, qy_c, qz_c = -qx, -qy, -qz
                uv_x = qy_c * v[2] - qz_c * v[1]
                uv_y = qz_c * v[0] - qx_c * v[2]
                uv_z = qx_c * v[1] - qy_c * v[0]

                uuv_x = qy_c * uv_z - qz_c * uv_y
                uuv_y = qz_c * uv_x - qx_c * uv_z
                uuv_z = qx_c * uv_y - qy_c * uv_x

                return (
                    v[0] + 2.0 * (qw * uv_x + uuv_x),
                    v[1] + 2.0 * (qw * uv_y + uuv_y),
                    v[2] + 2.0 * (qw * uv_z + uuv_z),
                )

            # Rates
            roll_rate = (roll - prev_roll) / self.dt if i > 0 else 0.0
            pitch_rate = (pitch - prev_pitch) / self.dt if i > 0 else 0.0
            yaw_rate = (yaw - prev_yaw) / self.dt if i > 0 else 0.0

            prev_roll, prev_pitch, prev_yaw = roll, pitch, yaw

            cp_ang, sp_ang = math.cos(pitch), math.sin(pitch)
            cr_ang, sr_ang = math.cos(roll), math.sin(roll)

            p = roll_rate - sp_ang * yaw_rate
            q = cr_ang * pitch_rate + sr_ang * cp_ang * yaw_rate
            r = -sr_ang * pitch_rate + cr_ang * cp_ang * yaw_rate

            # True specific force measured by accelerometer in body frame
            # Upward reaction force against gravity: [0, 0, +g] in nav frame
            accel = unrotate((0.0, 0.0, self.gravity))

            # True magnetic vector in body frame: [1, 0, 0] in nav frame
            mag = unrotate((1.0, 0.0, 0.0))

            yield FlightState(
                time=t,
                roll=roll,
                pitch=pitch,
                yaw=wrapped_yaw,
                roll_deg=math.degrees(roll),
                pitch_deg=math.degrees(pitch),
                yaw_deg=math.degrees(wrapped_yaw),
                gyro=(p, q, r),
                accel=accel,
                mag=mag,
            )
