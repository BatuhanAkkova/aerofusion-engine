//! High-integrity vector, matrix, and quaternion mathematics for aerospace kinematics.
//!
//! Designed for no_std deterministic execution with zero dynamic memory allocation.

#![allow(dead_code)]

/// 3D Vector representation with fundamental vector algebra.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn norm_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f32 {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n > 1e-8 {
            Self {
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        } else {
            Self::ZERO
        }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Generates the skew-symmetric cross-product matrix [v]x such that [v]x * u = v x u.
    #[inline]
    pub fn skew_symmetric(self) -> Mat3 {
        Mat3::from_rows(
            [0.0, -self.z, self.y],
            [self.z, 0.0, -self.x],
            [-self.y, self.x, 0.0],
        )
    }
}

/// Unit quaternion representing 3D spatial rotation: q = [w, x, y, z]^T.
/// Follows Hamilton convention (w is real scalar, [x, y, z] is imaginary vector).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    #[inline]
    pub fn norm_squared(self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn norm(self) -> f32 {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n > 1e-8 {
            Self {
                w: self.w / n,
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        } else {
            Self::IDENTITY
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product: q1 * q2
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Rotates a 3D vector v from Body frame to Navigation/World frame: v_nav = q * v_body * q^*
    #[inline]
    pub fn rotate_vector(self, v: Vec3) -> Vec3 {
        let q_vec = Vec3::new(self.x, self.y, self.z);
        let uv = q_vec.cross(v);
        let uuv = q_vec.cross(uv);
        v.add(uv.scale(2.0 * self.w)).add(uuv.scale(2.0))
    }

    /// Rotates a 3D vector v from Navigation/World frame to Body frame: v_body = q^* * v_nav * q
    #[inline]
    pub fn unrotate_vector(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate_vector(v)
    }

    /// Constructs rotation quaternion from small rotation vector (axis * angle).
    #[inline]
    pub fn from_rotvec(v: Vec3) -> Self {
        let angle = v.norm();
        if angle < 1e-6 {
            Self {
                w: 1.0,
                x: 0.5 * v.x,
                y: 0.5 * v.y,
                z: 0.5 * v.z,
            }
            .normalize()
        } else {
            let half = 0.5 * angle;
            let s = half.sin() / angle;
            Self {
                w: half.cos(),
                x: v.x * s,
                y: v.y * s,
                z: v.z * s,
            }
        }
    }

    /// Converts Euler angles (Roll, Pitch, Yaw) in radians to Quaternion.
    /// Rotation sequence: Z (Yaw) -> Y (Pitch) -> X (Roll) (Tait-Bryan aerospace convention).
    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();

        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
        .normalize()
    }

    /// Converts Quaternion to Euler angles (Roll, Pitch, Yaw) in radians.
    pub fn to_euler(self) -> EulerAngles {
        let q = self.normalize();

        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation) with gimbal-lock safety clamp
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        let pitch = if sinp.abs() >= 1.0 {
            core::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        EulerAngles { roll, pitch, yaw }
    }

    /// Computes Direction Cosine Matrix (DCM) mapping Body coordinates to Navigation coordinates: R_nb.
    pub fn to_dcm(self) -> Mat3 {
        let q = self.normalize();
        let w2 = q.w * q.w;
        let x2 = q.x * q.x;
        let y2 = q.y * q.y;
        let z2 = q.z * q.z;

        Mat3::from_rows(
            [
                w2 + x2 - y2 - z2,
                2.0 * (q.x * q.y - q.w * q.z),
                2.0 * (q.x * q.z + q.w * q.y),
            ],
            [
                2.0 * (q.x * q.y + q.w * q.z),
                w2 - x2 + y2 - z2,
                2.0 * (q.y * q.z - q.w * q.x),
            ],
            [
                2.0 * (q.x * q.z - q.w * q.y),
                2.0 * (q.y * q.z + q.w * q.x),
                w2 - x2 - y2 + z2,
            ],
        )
    }
}

/// Euler angles in aerospace convention (Roll: X, Pitch: Y, Yaw: Z).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct EulerAngles {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
}

/// 3x3 Matrix for rigid-body transformations and Kalman innovation covariances.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    pub data: [[f32; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Self = Self {
        data: [[0.0; 3]; 3],
    };

    pub const IDENTITY: Self = Self {
        data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    #[inline]
    pub const fn from_rows(r0: [f32; 3], r1: [f32; 3], r2: [f32; 3]) -> Self {
        Self { data: [r0, r1, r2] }
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            data: [
                [self.data[0][0], self.data[1][0], self.data[2][0]],
                [self.data[0][1], self.data[1][1], self.data[2][1]],
                [self.data[0][2], self.data[1][2], self.data[2][2]],
            ],
        }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][0] * rhs.data[0][j]
                    + self.data[i][1] * rhs.data[1][j]
                    + self.data[i][2] * rhs.data[2][j];
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.data[0][0] * v.x + self.data[0][1] * v.y + self.data[0][2] * v.z,
            y: self.data[1][0] * v.x + self.data[1][1] * v.y + self.data[1][2] * v.z,
            z: self.data[2][0] * v.x + self.data[2][1] * v.y + self.data[2][2] * v.z,
        }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] * s;
            }
        }
        Self { data: out }
    }

    /// Analytical inversion of 3x3 matrix via cofactor expansion.
    /// Returns None if matrix is singular (det ~ 0).
    pub fn inverse(self) -> Option<Self> {
        let m = self.data;
        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;

        let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
        let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
        let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

        let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
        let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

        // Transposed cofactor matrix * inv_det
        Some(Self {
            data: [
                [c00 * inv_det, c10 * inv_det, c20 * inv_det],
                [c01 * inv_det, c11 * inv_det, c21 * inv_det],
                [c02 * inv_det, c12 * inv_det, c22 * inv_det],
            ],
        })
    }
}

/// 6x6 Matrix for Error-State Covariance (attitude error: 3, gyro bias error: 3).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat6 {
    pub data: [[f32; 6]; 6],
}

impl Mat6 {
    pub const ZERO: Self = Self {
        data: [[0.0; 6]; 6],
    };

    pub const IDENTITY: Self = Self {
        data: [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    };

    #[inline]
    pub fn from_diag(diag: [f32; 6]) -> Self {
        let mut m = Self::ZERO;
        for i in 0..6 {
            m.data[i][i] = diag[i];
        }
        m
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn transpose(self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.data[j][i];
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for k in 0..6 {
                let r = self.data[i][k];
                for j in 0..6 {
                    out[i][j] += r * rhs.data[k][j];
                }
            }
        }
        Self { data: out }
    }

    #[inline]
    pub fn mul_vec(self, v: [f32; 6]) -> [f32; 6] {
        let mut out = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i] += self.data[i][j] * v[j];
            }
        }
        out
    }

    /// Enforces numerical symmetry: P = 0.5 * (P + P^T)
    #[inline]
    pub fn enforce_symmetry(self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let v = 0.5 * (self.data[i][j] + self.data[j][i]);
                out[i][j] = v;
            }
        }
        Self { data: out }
    }
}
