"""
Physical Sensor Error Modeling: White Noise, Bias Random Walk, Motor Harmonics.
"""

import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .trajectory import FlightState


@dataclass
class SensorPacket:
    time: float
    # True State
    true_roll_deg: float
    true_pitch_deg: float
    true_yaw_deg: float
    true_bias: Tuple[float, float, float]
    # Noisy Raw Measurements
    raw_gyro: Tuple[float, float, float]
    raw_accel: Tuple[float, float, float]
    raw_mag: Tuple[float, float, float]
    # Raw dead-reckoning un-fused attitude (shows gyro drift)
    raw_roll_deg: float
    raw_pitch_deg: float
    raw_yaw_deg: float


class ImuSensorModel:
    """Injects realistic aerospace sensor noise and environmental disturbances."""

    def __init__(
        self,
        gyro_noise_std: float = 0.008,          # 0.45 deg/s white noise
        gyro_bias_drift_std: float = 0.00015,   # Bias random walk
        initial_gyro_bias: Tuple[float, float, float] = (0.02, -0.015, 0.01), # 1 deg/s initial bias
        accel_noise_std: float = 0.35,          # Accelerometer noise (m/s^2)
        motor_vibe_freq: float = 55.0,          # Motor RPM harmonic (Hz)
        motor_vibe_amp: float = 0.5,            # Vibration amplitude
        mag_noise_std: float = 0.04,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_drift_std = gyro_bias_drift_std
        self.current_gyro_bias = np.array(initial_gyro_bias, dtype=np.float64)
        self.accel_noise_std = accel_noise_std
        self.motor_vibe_freq = motor_vibe_freq
        self.motor_vibe_amp = motor_vibe_amp
        self.mag_noise_std = mag_noise_std

        # Unassisted dead-reckoning accumulator
        self.dead_reckon_roll = 0.0
        self.dead_reckon_pitch = 0.0
        self.dead_reckon_yaw = 0.0

    def corrupt(self, state: FlightState, dt: float) -> SensorPacket:
        # 1. Update gyro bias random walk: b_{k+1} = b_k + N(0, sigma_w^2 * dt)
        bias_step = self.rng.normal(0.0, self.gyro_bias_drift_std * math.sqrt(dt), size=3)
        self.current_gyro_bias += bias_step

        # 2. Add gyro white noise and current bias
        gyro_noise = self.rng.normal(0.0, self.gyro_noise_std, size=3)
        measured_gyro = np.array(state.gyro) + self.current_gyro_bias + gyro_noise

        # 3. Add accelerometer noise + high-frequency motor vibration
        vibe_phase = 2.0 * math.pi * self.motor_vibe_freq * state.time
        vibe_vector = np.array([
            math.sin(vibe_phase),
            math.cos(vibe_phase * 1.05),
            math.sin(vibe_phase * 0.95),
        ]) * self.motor_vibe_amp

        accel_noise = self.rng.normal(0.0, self.accel_noise_std, size=3)
        measured_accel = np.array(state.accel) + vibe_vector + accel_noise

        # 4. Magnetometer noise
        mag_noise = self.rng.normal(0.0, self.mag_noise_std, size=3)
        measured_mag = np.array(state.mag) + mag_noise

        # 5. Raw unassisted dead-reckoning integration (for comparison plot)
        self.dead_reckon_roll += measured_gyro[0] * dt
        self.dead_reckon_pitch += measured_gyro[1] * dt
        self.dead_reckon_yaw += measured_gyro[2] * dt

        return SensorPacket(
            time=state.time,
            true_roll_deg=state.roll_deg,
            true_pitch_deg=state.pitch_deg,
            true_yaw_deg=state.yaw_deg,
            true_bias=(
                float(self.current_gyro_bias[0]),
                float(self.current_gyro_bias[1]),
                float(self.current_gyro_bias[2]),
            ),
            raw_gyro=(float(measured_gyro[0]), float(measured_gyro[1]), float(measured_gyro[2])),
            raw_accel=(float(measured_accel[0]), float(measured_accel[1]), float(measured_accel[2])),
            raw_mag=(float(measured_mag[0]), float(measured_mag[1]), float(measured_mag[2])),
            raw_roll_deg=math.degrees(self.dead_reckon_roll),
            raw_pitch_deg=math.degrees(self.dead_reckon_pitch),
            raw_yaw_deg=math.degrees(self.dead_reckon_yaw),
        )
