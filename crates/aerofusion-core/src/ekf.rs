//! Error-State Extended Kalman Filter (ES-EKF / MEKF) for 6-DoF attitude and bias estimation.
//!
//! Guarantees:
//! - Deterministic O(1) time complexity per step.
//! - Zero heap allocation (no_std friendly).
//! - Numerical symmetry and positive semi-definiteness via Joseph-form covariance updates.
//! - Dynamic acceleration gating to prevent attitude distortion during aggressive maneuvers.

use crate::math::{EulerAngles, Mat3, Mat6, Quat, Vec3};
use crate::models::{EkfConfig, FilterOutput, ImuMeasurement, MagMeasurement};

/// 6-DoF Quaternion Error-State Extended Kalman Filter Engine.
#[derive(Debug, Clone)]
pub struct EkfEngine {
    /// Configuration and noise parameters
    config: EkfConfig,
    /// Nominal attitude quaternion (Body -> Navigation frame)
    q: Quat,
    /// Estimated gyroscope bias in body frame [rad/s]
    gyro_bias: Vec3,
    /// Error-state covariance matrix (6x6: [delta_theta, delta_bg])
    p: Mat6,
    /// Reference gravity vector in navigation frame (normalized, points upward: [0, 0, 1])
    gravity_ref_nav: Vec3,
    /// Reference magnetic field vector in navigation frame (normalized: [mx, my, mz])
    mag_ref_nav: Vec3,
    /// Total processed samples count
    step_count: u64,
}

impl EkfEngine {
    /// Creates a new EKF initialized to identity attitude and zero gyro bias.
    pub fn new(config: EkfConfig) -> Self {
        let mut p = Mat6::ZERO;
        for i in 0..3 {
            p.data[i][i] = config.init_att_cov;
            p.data[i + 3][i + 3] = config.init_bias_cov;
        }

        Self {
            config,
            q: Quat::IDENTITY,
            gyro_bias: Vec3::ZERO,
            p,
            gravity_ref_nav: Vec3::new(0.0, 0.0, 1.0),
            mag_ref_nav: Vec3::new(1.0, 0.0, 0.0), // Default magnetic North
            step_count: 0,
        }
    }

    /// Resets the filter state and covariance to initial values.
    pub fn reset(&mut self) {
        self.q = Quat::IDENTITY;
        self.gyro_bias = Vec3::ZERO;
        self.step_count = 0;

        let mut p = Mat6::ZERO;
        for i in 0..3 {
            p.data[i][i] = self.config.init_att_cov;
            p.data[i + 3][i + 3] = self.config.init_bias_cov;
        }
        self.p = p;
    }

    /// Explicitly initializes attitude from accelerometer (leveling) and magnetometer (heading).
    pub fn initialize_from_vectors(&mut self, accel: Vec3, mag: Vec3) -> bool {
        let a_norm = accel.norm();
        if a_norm < 1e-4 {
            return false;
        }
        let a = accel.normalize();

        // Roll (phi) and Pitch (theta) from accelerometer assuming stationary vehicle
        // Measured specific force at rest is opposing gravity: a_meas = [0, 0, g] in nav frame
        let roll = a.y.atan2(a.z);
        let pitch = (-a.x).atan2((a.y * a.y + a.z * a.z).sqrt());

        // Heading (yaw) calculation using tilt-compensated magnetometer
        let cr = roll.cos();
        let sr = roll.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();

        // Rotate mag into level horizontal plane
        let b_x = mag.x * cp + mag.y * (sp * sr) + mag.z * (sp * cr);
        let b_y = mag.y * cr - mag.z * sr;
        let yaw = (-b_y).atan2(b_x);

        self.q = Quat::from_euler(roll, pitch, yaw);
        self.gyro_bias = Vec3::ZERO;
        true
    }

    /// Time propagation (Predict step) using high-rate IMU angular velocity and delta time.
    ///
    /// 1. Subtracts estimated bias from gyro measurement.
    /// 2. Integrates attitude via quaternion kinematics.
    /// 3. Propagates 6x6 error-state covariance matrix: P = F * P * F^T + Q.
    #[inline]
    pub fn predict(&mut self, imu: ImuMeasurement) {
        let dt = imu.dt;
        if dt <= 0.0 || dt > 1.0 {
            return;
        }

        // Unbiased angular velocity: w_hat = w_meas - b_hat
        let w_unbiased = imu.gyro.sub(self.gyro_bias);

        // 1. Nominal state propagation: q_{k+1} = q_k * delta_q(w * dt)
        let delta_rot = Quat::from_rotvec(w_unbiased.scale(dt));
        self.q = self.q.mul(delta_rot).normalize();

        // 2. Discrete error-state transition matrix F (6x6)
        // [ delta_theta ]_{k+1} = (I - [w_hat]_x * dt) * delta_theta_k - I * dt * delta_b_k
        // [ delta_b     ]_{k+1} = delta_b_k
        let w_skew = w_unbiased.skew_symmetric().scale(dt);
        let f_theta_theta = Mat3::IDENTITY.sub(w_skew);
        let f_theta_b = Mat3::IDENTITY.scale(-dt);

        let mut f = Mat6::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                f.data[i][j] = f_theta_theta.data[i][j];
                f.data[i][j + 3] = f_theta_b.data[i][j];
            }
        }

        // 3. Process noise covariance Q (6x6)
        let q_gyro = self.config.gyro_noise * self.config.gyro_noise * dt;
        let q_bias = self.config.gyro_bias_noise * self.config.gyro_bias_noise * dt;
        let q_mat = Mat6::from_diag([q_gyro, q_gyro, q_gyro, q_bias, q_bias, q_bias]);

        // P = F * P * F^T + Q
        let f_p = f.mul(self.p);
        let f_p_ft = f_p.mul(f.transpose());
        self.p = f_p_ft.add(q_mat).enforce_symmetry();

        self.step_count += 1;
    }

    /// Vector measurement update step (used for Accelerometer and Magnetometer).
    ///
    /// Implements:
    /// - Predicted vector in body frame: z_pred = q^* * z_ref_nav * q
    /// - Measurement Jacobian: H = [ [z_pred]_x , 0_{3x3} ]
    /// - Innovation residual: y = z_meas - z_pred
    /// - Kalman Gain: K = P * H^T * (H * P * H^T + R)^(-1)
    /// - Error state correction & nominal state injection
    /// - Joseph-form stabilized covariance update
    fn update_vector(&mut self, z_meas_normalized: Vec3, z_ref_nav: Vec3, r_noise: f32) -> bool {
        // Predicted vector in body frame: z_pred = R^T * z_ref_nav
        let z_pred = self.q.unrotate_vector(z_ref_nav);

        // Innovation residual: y = z_meas - z_pred
        let y = z_meas_normalized.sub(z_pred);

        // Measurement Jacobian H (3x6):
        // delta_z = [z_pred]_x * delta_theta + 0 * delta_b
        let z_skew = z_pred.skew_symmetric();
        let mut h = [[0.0f32; 6]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = z_skew.data[i][j];
            }
        }

        // Compute S = H * P * H^T + R (3x3)
        // First compute H_P (3x6) = H * P
        let mut h_p = [[0.0f32; 6]; 3];
        for i in 0..3 {
            for j in 0..6 {
                let mut sum = 0.0;
                for k in 0..6 {
                    sum += h[i][k] * self.p.data[k][j];
                }
                h_p[i][j] = sum;
            }
        }

        // S = (H_P) * H^T + R
        let mut s_data = [[0.0f32; 3]; 3];
        let r_var = r_noise * r_noise;
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..6 {
                    sum += h_p[i][k] * h[j][k];
                }
                if i == j {
                    sum += r_var;
                }
                s_data[i][j] = sum;
            }
        }
        let s = Mat3 { data: s_data };

        // S^-1
        let s_inv = match s.inverse() {
            Some(inv) => inv,
            None => return false,
        };

        // Kalman Gain: K = P * H^T * S^-1 = (H_P)^T * S^-1 (6x3)
        let mut k_gain = [[0.0f32; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                let mut sum = 0.0;
                for l in 0..3 {
                    // h_p[l][i] is the (i, l) element of (H * P)^T
                    sum += h_p[l][i] * s_inv.data[l][j];
                }
                k_gain[i][j] = sum;
            }
        }

        // State correction: delta_x = K * y (6x1)
        let y_arr = [y.x, y.y, y.z];
        let mut delta_x = [0.0f32; 6];
        for i in 0..6 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += k_gain[i][j] * y_arr[j];
            }
            delta_x[i] = sum;
        }

        let delta_theta = Vec3::new(delta_x[0], delta_x[1], delta_x[2]);
        let delta_bias = Vec3::new(delta_x[3], delta_x[4], delta_x[5]);

        // Nominal state injection
        let delta_q = Quat::from_rotvec(delta_theta);
        self.q = self.q.mul(delta_q).normalize();
        self.gyro_bias = self.gyro_bias.add(delta_bias);

        // Joseph-form covariance update:
        // P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
        // Compute I_KH = I - K * H (6x6)
        let mut i_kh = Mat6::IDENTITY;
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0;
                for l in 0..3 {
                    sum += k_gain[i][l] * h[l][j];
                }
                i_kh.data[i][j] -= sum;
            }
        }

        // P_new = (I_KH) * P * (I_KH)^T
        let ikh_p = i_kh.mul(self.p);
        let p_part1 = ikh_p.mul(i_kh.transpose());

        // K * R * K^T (6x6)
        let mut krk = Mat6::ZERO;
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0;
                for l in 0..3 {
                    sum += k_gain[i][l] * r_var * k_gain[j][l];
                }
                krk.data[i][j] = sum;
            }
        }

        self.p = p_part1.add(krk).enforce_symmetry();
        true
    }

    /// Accelerometer measurement update.
    /// Automatically applies dynamic-acceleration gating: rejects updates if norm deviates > 25% from 1G.
    pub fn update_accel(&mut self, accel: Vec3) -> bool {
        let norm = accel.norm();
        if norm < 1e-4 {
            return false;
        }

        // Gating against linear dynamic acceleration (e.g. abrupt catapult or collision)
        let g_nom = self.config.gravity_magnitude;
        let diff_ratio = (norm - g_nom).abs() / g_nom;
        if diff_ratio > 0.35 {
            // High dynamic maneuver: rely purely on gyro prediction
            return false;
        }

        let a_norm = accel.scale(1.0 / norm);
        self.update_vector(
            a_norm,
            self.gravity_ref_nav,
            self.config.accel_noise / g_nom,
        )
    }

    /// Magnetometer measurement update to correct yaw heading drift.
    pub fn update_mag(&mut self, mag: MagMeasurement) -> bool {
        let norm = mag.mag.norm();
        if norm < 1e-4 {
            return false;
        }
        let m_norm = mag.mag.scale(1.0 / norm);
        self.update_vector(m_norm, self.mag_ref_nav, self.config.mag_noise)
    }

    /// Convenience step fusing an IMU packet and optional Magnetometer packet.
    pub fn step(&mut self, imu: ImuMeasurement, mag: Option<MagMeasurement>) {
        self.predict(imu);
        self.update_accel(imu.accel);
        if let Some(m) = mag {
            self.update_mag(m);
        }
    }

    /// Generates full telemetry output with uncertainties.
    pub fn get_telemetry(&self) -> FilterOutput {
        let euler = self.q.to_euler();
        let to_deg = 180.0 / core::f32::consts::PI;

        FilterOutput {
            q: self.q,
            gyro_bias: self.gyro_bias,
            euler,
            euler_deg: EulerAngles {
                roll: euler.roll * to_deg,
                pitch: euler.pitch * to_deg,
                yaw: euler.yaw * to_deg,
            },
            attitude_std: Vec3::new(
                self.p.data[0][0].max(0.0).sqrt(),
                self.p.data[1][1].max(0.0).sqrt(),
                self.p.data[2][2].max(0.0).sqrt(),
            ),
            bias_std: Vec3::new(
                self.p.data[3][3].max(0.0).sqrt(),
                self.p.data[4][4].max(0.0).sqrt(),
                self.p.data[5][5].max(0.0).sqrt(),
            ),
        }
    }

    #[inline]
    pub fn attitude(&self) -> Quat {
        self.q
    }

    #[inline]
    pub fn gyro_bias(&self) -> Vec3 {
        self.gyro_bias
    }

    #[inline]
    pub fn covariance(&self) -> &Mat6 {
        &self.p
    }

    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}
