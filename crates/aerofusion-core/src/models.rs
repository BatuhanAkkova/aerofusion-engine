//! Aerospace sensor models, state representations, and noise parameters.

use crate::math::{EulerAngles, Quat, Vec3};

/// Raw or calibrated Inertial Measurement Unit (IMU) packet.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct ImuMeasurement {
    /// Linear acceleration in body frame [m/s^2]
    pub accel: Vec3,
    /// Angular velocity in body frame [rad/s]
    pub gyro: Vec3,
    /// Delta time since last sample [s]
    pub dt: f32,
}

/// Magnetometer packet in body frame (normalized or Gauss / mu T).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct MagMeasurement {
    pub mag: Vec3,
}

/// Tunable noise and uncertainty parameters for the EKF.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct EkfConfig {
    /// Gyroscope continuous white noise standard deviation (rad/s / sqrt(Hz))
    pub gyro_noise: f32,
    /// Gyroscope bias random walk standard deviation (rad/s^2 / sqrt(Hz))
    pub gyro_bias_noise: f32,
    /// Accelerometer measurement noise standard deviation (m/s^2)
    pub accel_noise: f32,
    /// Magnetometer measurement noise standard deviation
    pub mag_noise: f32,
    /// Initial uncertainty on attitude error (rad^2)
    pub init_att_cov: f32,
    /// Initial uncertainty on gyro bias ( (rad/s)^2 )
    pub init_bias_cov: f32,
    /// Gravity magnitude constant (default 9.80665 m/s^2)
    pub gravity_magnitude: f32,
}

impl Default for EkfConfig {
    fn default() -> Self {
        Self {
            gyro_noise: 0.005,       // 0.3 deg/s
            gyro_bias_noise: 0.0001, // slow drift
            accel_noise: 0.3,        // m/s^2
            mag_noise: 0.2,
            init_att_cov: 0.1,   // initial angle variance
            init_bias_cov: 0.01, // initial bias variance
            gravity_magnitude: 9.80665,
        }
    }
}

/// Comprehensive filter telemetry output for flight software consumers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct FilterOutput {
    /// Nominal orientation quaternion (Body to Navigation)
    pub q: Quat,
    /// Estimated gyroscope bias [rad/s]
    pub gyro_bias: Vec3,
    /// Euler attitude angles (Roll, Pitch, Yaw) [rad]
    pub euler: EulerAngles,
    /// Euler attitude angles in degrees for human readout
    pub euler_deg: EulerAngles,
    /// Estimated 1-sigma attitude uncertainty (Roll, Pitch, Yaw) [rad]
    pub attitude_std: Vec3,
    /// Estimated 1-sigma gyro bias uncertainty [rad/s]
    pub bias_std: Vec3,
}
