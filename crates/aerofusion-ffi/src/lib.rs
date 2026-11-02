//! C-compatible ABI interface for the Aerofusion sensor fusion engine.
//!
//! Exposes stable `extern "C"` functions for embedding into C, C++, and RTOS environments.

use aerofusion_core::{EkfConfig, EkfEngine, ImuMeasurement, MagMeasurement, Vec3};
use std::ffi::c_char;
use std::panic::catch_unwind;

pub type AerofusionEngine = EkfEngine;

/// C-compatible representation of filter configuration parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AerofusionConfigC {
    pub gyro_noise: f32,
    pub gyro_bias_noise: f32,
    pub accel_noise: f32,
    pub mag_noise: f32,
    pub init_att_cov: f32,
    pub init_bias_cov: f32,
    pub gravity_magnitude: f32,
}

impl From<AerofusionConfigC> for EkfConfig {
    fn from(c: AerofusionConfigC) -> Self {
        Self {
            gyro_noise: c.gyro_noise,
            gyro_bias_noise: c.gyro_bias_noise,
            accel_noise: c.accel_noise,
            mag_noise: c.mag_noise,
            init_att_cov: c.init_att_cov,
            init_bias_cov: c.init_bias_cov,
            gravity_magnitude: c.gravity_magnitude,
        }
    }
}

/// C-compatible telemetry output payload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AerofusionTelemetryC {
    pub qw: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub roll_rad: f32,
    pub pitch_rad: f32,
    pub yaw_rad: f32,
    pub roll_deg: f32,
    pub pitch_deg: f32,
    pub yaw_deg: f32,
    pub bias_x: f32,
    pub bias_y: f32,
    pub bias_z: f32,
    pub att_std_x: f32,
    pub att_std_y: f32,
    pub att_std_z: f32,
    pub bias_std_x: f32,
    pub bias_std_y: f32,
    pub bias_std_z: f32,
    pub step_count: u64,
}

pub const AEROFUSION_OK: i32 = 0;
pub const AEROFUSION_ERR_NULL_PTR: i32 = -1;
pub const AEROFUSION_ERR_PANIC: i32 = -2;
pub const AEROFUSION_ERR_INVALID_PARAM: i32 = -3;

#[no_mangle]
pub extern "C" fn aerofusion_version() -> *const c_char {
    static VERSION: &[u8] = b"aerofusion-v0.1.0-safety-critical\0";
    VERSION.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn aerofusion_create() -> *mut AerofusionEngine {
    let result = catch_unwind(|| {
        let engine = EkfEngine::new(EkfConfig::default());
        Box::into_raw(Box::new(engine))
    });
    result.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn aerofusion_create_with_config(config: *const AerofusionConfigC) -> *mut AerofusionEngine {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(|| {
        let rust_config = unsafe { (*config).into() };
        let engine = EkfEngine::new(rust_config);
        Box::into_raw(Box::new(engine))
    });
    result.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn aerofusion_destroy(engine: *mut AerofusionEngine) {
    if engine.is_null() {
        return;
    }
    let _ = catch_unwind(|| {
        unsafe {
            drop(Box::from_raw(engine));
        }
    });
}

#[no_mangle]
pub extern "C" fn aerofusion_reset(engine: *mut AerofusionEngine) -> i32 {
    if engine.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        (*engine).reset();
        AEROFUSION_OK
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_predict(
    engine: *mut AerofusionEngine,
    gx: f32,
    gy: f32,
    gz: f32,
    dt: f32,
) -> i32 {
    if engine.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    if dt <= 0.0 {
        return AEROFUSION_ERR_INVALID_PARAM;
    }
    let res = catch_unwind(|| unsafe {
        let imu = ImuMeasurement {
            accel: Vec3::ZERO,
            gyro: Vec3::new(gx, gy, gz),
            dt,
        };
        (*engine).predict(imu);
        AEROFUSION_OK
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_update_accel(
    engine: *mut AerofusionEngine,
    ax: f32,
    ay: f32,
    az: f32,
) -> i32 {
    if engine.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        let updated = (*engine).update_accel(Vec3::new(ax, ay, az));
        if updated { 1 } else { 0 }
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_update_mag(
    engine: *mut AerofusionEngine,
    mx: f32,
    my: f32,
    mz: f32,
) -> i32 {
    if engine.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        let updated = (*engine).update_mag(MagMeasurement {
            mag: Vec3::new(mx, my, mz),
        });
        if updated { 1 } else { 0 }
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_step(
    engine: *mut AerofusionEngine,
    gx: f32,
    gy: f32,
    gz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    dt: f32,
) -> i32 {
    if engine.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        let imu = ImuMeasurement {
            accel: Vec3::new(ax, ay, az),
            gyro: Vec3::new(gx, gy, gz),
            dt,
        };
        (*engine).step(imu, None);
        AEROFUSION_OK
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_step_with_mag(
    engine: *mut AerofusionEngine,
    gx: f32,
    gy: f32,
    gz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    mx: f32,
    my: f32,
    mz: f32,
    dt: f32,
) -> i32 {
    if engine.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        let imu = ImuMeasurement {
            accel: Vec3::new(ax, ay, az),
            gyro: Vec3::new(gx, gy, gz),
            dt,
        };
        let mag = MagMeasurement {
            mag: Vec3::new(mx, my, mz),
        };
        (*engine).step(imu, Some(mag));
        AEROFUSION_OK
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_get_telemetry(
    engine: *const AerofusionEngine,
    out: *mut AerofusionTelemetryC,
) -> i32 {
    if engine.is_null() || out.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        let tel = (*engine).get_telemetry();
        *out = AerofusionTelemetryC {
            qw: tel.q.w,
            qx: tel.q.x,
            qy: tel.q.y,
            qz: tel.q.z,
            roll_rad: tel.euler.roll,
            pitch_rad: tel.euler.pitch,
            yaw_rad: tel.euler.yaw,
            roll_deg: tel.euler_deg.roll,
            pitch_deg: tel.euler_deg.pitch,
            yaw_deg: tel.euler_deg.yaw,
            bias_x: tel.gyro_bias.x,
            bias_y: tel.gyro_bias.y,
            bias_z: tel.gyro_bias.z,
            att_std_x: tel.attitude_std.x,
            att_std_y: tel.attitude_std.y,
            att_std_z: tel.attitude_std.z,
            bias_std_x: tel.bias_std.x,
            bias_std_y: tel.bias_std.y,
            bias_std_z: tel.bias_std.z,
            step_count: (*engine).step_count(),
        };
        AEROFUSION_OK
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn aerofusion_get_euler_deg(
    engine: *const AerofusionEngine,
    roll: *mut f32,
    pitch: *mut f32,
    yaw: *mut f32,
) -> i32 {
    if engine.is_null() || roll.is_null() || pitch.is_null() || yaw.is_null() {
        return AEROFUSION_ERR_NULL_PTR;
    }
    let res = catch_unwind(|| unsafe {
        let tel = (*engine).get_telemetry();
        *roll = tel.euler_deg.roll;
        *pitch = tel.euler_deg.pitch;
        *yaw = tel.euler_deg.yaw;
        AEROFUSION_OK
    });
    res.unwrap_or(AEROFUSION_ERR_PANIC)
}

#[cfg(test)]
mod ffi_tests {
    use super::*;

    #[test]
    fn test_ffi_lifecycle() {
        unsafe {
            let engine = aerofusion_create();
            assert!(!engine.is_null());

            // Predict step with 100 Hz dt
            let res = aerofusion_predict(engine, 0.01, -0.02, 0.05, 0.01);
            assert_eq!(res, AEROFUSION_OK);

            // Step with IMU
            let res2 = aerofusion_step(engine, 0.0, 0.0, 0.0, 0.0, 0.0, 9.80665, 0.01);
            assert_eq!(res2, AEROFUSION_OK);

            // Fetch telemetry
            let mut tel = std::mem::zeroed::<AerofusionTelemetryC>();
            let res3 = aerofusion_get_telemetry(engine, &mut tel);
            assert_eq!(res3, AEROFUSION_OK);
            assert_eq!(tel.step_count, 2);
            assert!((tel.qw - 1.0).abs() < 0.1);

            let res4 = aerofusion_reset(engine);
            assert_eq!(res4, AEROFUSION_OK);

            aerofusion_destroy(engine);
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        let null_engine: *mut AerofusionEngine = std::ptr::null_mut();
        assert_eq!(aerofusion_reset(null_engine), AEROFUSION_ERR_NULL_PTR);
        assert_eq!(aerofusion_predict(null_engine, 0.0, 0.0, 0.0, 0.01), AEROFUSION_ERR_NULL_PTR);
        
        let mut tel = unsafe { std::mem::zeroed::<AerofusionTelemetryC>() };
        assert_eq!(aerofusion_get_telemetry(null_engine, &mut tel), AEROFUSION_ERR_NULL_PTR);
        
        // Destroying null pointer must not crash
        aerofusion_destroy(null_engine);
    }
}
