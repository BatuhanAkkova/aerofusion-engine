//! # aerofusion-core
//!
//! A safety-critical, high-integrity 6-DoF Error-State Extended Kalman Filter (ES-EKF)
//! for aerospace flight computers, UAV autopilots, and robotics.
//!


#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

pub mod ekf;
pub mod math;
pub mod models;

pub use ekf::EkfEngine;
pub use math::{EulerAngles, Mat3, Mat6, Quat, Vec3};
pub use models::{EkfConfig, FilterOutput, ImuMeasurement, MagMeasurement};

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_quaternion_euler_roundtrip() {
        let angles = [
            (0.0, 0.0, 0.0),
            (0.2, -0.3, 1.5),
            (-0.5, 0.4, -2.1),
            (PI * 0.4, -PI * 0.2, PI * 0.7),
        ];

        for (r, p, y) in angles {
            let q = Quat::from_euler(r, p, y);
            let recovered = q.to_euler();
            assert!((recovered.roll - r).abs() < 1e-4, "Roll mismatch: expected {}, got {}", r, recovered.roll);
            assert!((recovered.pitch - p).abs() < 1e-4, "Pitch mismatch: expected {}, got {}", p, recovered.pitch);
            assert!((recovered.yaw - y).abs() < 1e-4, "Yaw mismatch: expected {}, got {}", y, recovered.yaw);
        }
    }

    #[test]
    fn test_quaternion_vector_rotation() {
        // 90 deg rotation around Z axis
        let q = Quat::from_axis_angle_z(PI * 0.5);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vector(v);

        assert!((rotated.x).abs() < 1e-5);
        assert!((rotated.y - 1.0).abs() < 1e-5);
        assert!((rotated.z).abs() < 1e-5);

        let unrotated = q.unrotate_vector(rotated);
        assert!((unrotated.x - 1.0).abs() < 1e-5);
        assert!((unrotated.y).abs() < 1e-5);
    }

    impl Quat {
        pub fn from_axis_angle_z(angle: f32) -> Self {
            let half = angle * 0.5;
            Self::new(half.cos(), 0.0, 0.0, half.sin())
        }
    }

    #[test]
    fn test_ekf_stationary_leveling_and_mag() {
        let config = EkfConfig::default();
        let mut filter = EkfEngine::new(config);

        // Vehicle stationary at 10 deg pitch (0.1745 rad)
        let pitch_true = 0.1745f32;
        let q_true = Quat::from_euler(0.0, pitch_true, 0.0);
        
        // Sensor at rest sees gravity opposing nav z=[0,0,1]
        let g_nav = Vec3::new(0.0, 0.0, 9.80665);
        let a_body = q_true.unrotate_vector(g_nav);
        let m_nav = Vec3::new(1.0, 0.0, 0.0);
        let m_body = q_true.unrotate_vector(m_nav);

        // Run 200 steps at 100 Hz
        let dt = 0.01f32;
        for _ in 0..200 {
            let imu = ImuMeasurement {
                accel: a_body,
                gyro: Vec3::ZERO,
                dt,
            };
            let mag = MagMeasurement { mag: m_body };
            filter.step(imu, Some(mag));
        }

        let out = filter.get_telemetry();
        assert!(
            (out.euler.pitch - pitch_true).abs() < 0.02,
            "EKF failed to converge on pitch: expected {}, got {}",
            pitch_true,
            out.euler.pitch
        );
        assert!(out.euler.roll.abs() < 0.02, "Expected roll ~ 0, got {}", out.euler.roll);
        assert!(out.euler.yaw.abs() < 0.02, "Expected yaw ~ 0, got {}", out.euler.yaw);
    }

    #[test]
    fn test_ekf_gyro_bias_convergence() {
        let config = EkfConfig::default();
        let mut filter = EkfEngine::new(config);

        let g_nav = Vec3::new(0.0, 0.0, 9.80665);
        let true_bias = Vec3::new(0.03, -0.02, 0.01); // ~1-2 deg/s bias
        let dt = 0.01;

        // Vehicle is stationary, but gyro has a constant bias offset
        for _ in 0..1000 {
            let imu = ImuMeasurement {
                accel: g_nav,
                gyro: true_bias, // Noisy gyro output = 0 + true_bias
                dt,
            };
            let mag = MagMeasurement { mag: Vec3::new(1.0, 0.0, 0.0) };
            filter.step(imu, Some(mag));
        }

        let estimated_bias = filter.gyro_bias();
        assert!(
            (estimated_bias.x - true_bias.x).abs() < 0.01,
            "Failed to estimate bias X: expected {}, got {}",
            true_bias.x,
            estimated_bias.x
        );
        assert!(
            (estimated_bias.y - true_bias.y).abs() < 0.01,
            "Failed to estimate bias Y: expected {}, got {}",
            true_bias.y,
            estimated_bias.y
        );
    }
}
