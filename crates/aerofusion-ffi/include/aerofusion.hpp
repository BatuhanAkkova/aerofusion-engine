/**
 * @file aerofusion.hpp
 * @brief Modern C++20 / C++17 RAII wrapper for the Aerofusion Sensor Fusion Engine.
 *
 * Provides type-safe, zero-cost abstractions for integration into C++ flight stacks
 * (e.g. ROS 2, PX4, ArduPilot, or custom embedded flight controllers).
 */

#ifndef AEROFUSION_HPP
#define AEROFUSION_HPP

#include "aerofusion.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace aerofusion {

struct Vec3 {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    constexpr Vec3(const std::array<float, 3>& arr) : x(arr[0]), y(arr[1]), z(arr[2]) {}
};

struct Quaternion {
    float w{1.0f};
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
};

struct EulerAngles {
    float roll{0.0f};
    float pitch{0.0f};
    float yaw{0.0f};
};

struct Telemetry {
    Quaternion attitude;
    EulerAngles euler_rad;
    EulerAngles euler_deg;
    Vec3 gyro_bias;
    Vec3 attitude_std;
    Vec3 bias_std;
    uint64_t step_count{0};
};

/**
 * @brief High-integrity C++ RAII wrapper managing an underlying AerofusionEngine instance.
 */
class SensorFusionEngine {
public:
    /// Default constructor initializing engine with default aerospace tuning parameters
    SensorFusionEngine() : handle_(aerofusion_create()) {
        if (!handle_) {
            throw std::runtime_error("Failed to allocate AerofusionEngine in Rust core");
        }
    }

    /// Parameterized constructor with custom sensor noise profiles
    explicit SensorFusionEngine(const AerofusionConfigC& config)
        : handle_(aerofusion_create_with_config(&config)) {
        if (!handle_) {
            throw std::runtime_error("Failed to initialize AerofusionEngine with custom config");
        }
    }

    /// Destructor enforcing RAII cleanup across the FFI boundary
    ~SensorFusionEngine() noexcept {
        if (handle_) {
            aerofusion_destroy(handle_);
            handle_ = nullptr;
        }
    }

    // Disable copy semantics to ensure exclusive ownership of the underlying filter state
    SensorFusionEngine(const SensorFusionEngine&) = delete;
    SensorFusionEngine& operator=(const SensorFusionEngine&) = delete;

    // Enable move semantics
    SensorFusionEngine(SensorFusionEngine&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    SensorFusionEngine& operator=(SensorFusionEngine&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                aerofusion_destroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /// Resets the internal state and covariance
    void reset() {
        if (aerofusion_reset(handle_) != AEROFUSION_OK) {
            throw std::runtime_error("aerofusion_reset failed");
        }
    }

    /// Predicts attitude state forward in time using gyroscope angular velocities
    void predict(const Vec3& gyro, float dt) {
        if (aerofusion_predict(handle_, gyro.x, gyro.y, gyro.z, dt) != AEROFUSION_OK) {
            throw std::runtime_error("aerofusion_predict failed");
        }
    }

    /// Accelerometer measurement update with gravity reference
    bool update_accel(const Vec3& accel) {
        int32_t ret = aerofusion_update_accel(handle_, accel.x, accel.y, accel.z);
        return ret == 1;
    }

    /// Magnetometer measurement update with magnetic north reference
    bool update_mag(const Vec3& mag) {
        int32_t ret = aerofusion_update_mag(handle_, mag.x, mag.y, mag.z);
        return ret == 1;
    }

    /// High-rate integrated filter step
    void step(const Vec3& gyro, const Vec3& accel, float dt) {
        if (aerofusion_step(handle_, gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z, dt) != AEROFUSION_OK) {
            throw std::runtime_error("aerofusion_step failed");
        }
    }

    /// Combined step with 9-DoF (gyro + accel + mag)
    void step(const Vec3& gyro, const Vec3& accel, const Vec3& mag, float dt) {
        if (aerofusion_step_with_mag(handle_, gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z,
                                      mag.x, mag.y, mag.z, dt) != AEROFUSION_OK) {
            throw std::runtime_error("aerofusion_step_with_mag failed");
        }
    }

    /// Retrieves a complete telemetry snapshot
    [[nodiscard]] Telemetry get_telemetry() const {
        AerofusionTelemetryC raw{};
        if (aerofusion_get_telemetry(handle_, &raw) != AEROFUSION_OK) {
            throw std::runtime_error("aerofusion_get_telemetry failed");
        }

        Telemetry tel;
        tel.attitude = {raw.qw, raw.qx, raw.qy, raw.qz};
        tel.euler_rad = {raw.roll_rad, raw.pitch_rad, raw.yaw_rad};
        tel.euler_deg = {raw.roll_deg, raw.pitch_deg, raw.yaw_deg};
        tel.gyro_bias = {raw.bias_x, raw.bias_y, raw.bias_z};
        tel.attitude_std = {raw.att_std_x, raw.att_std_y, raw.att_std_z};
        tel.bias_std = {raw.bias_std_x, raw.bias_std_y, raw.bias_std_z};
        tel.step_count = raw.step_count;
        return tel;
    }

    /// Quick access to Euler attitude angles in degrees
    [[nodiscard]] EulerAngles get_euler_deg() const {
        float r = 0.0f, p = 0.0f, y = 0.0f;
        if (aerofusion_get_euler_deg(handle_, &r, &p, &y) != AEROFUSION_OK) {
            throw std::runtime_error("aerofusion_get_euler_deg failed");
        }
        return {r, p, y};
    }

    /// Engine build version string
    static std::string_view version() noexcept {
        return aerofusion_version();
    }

private:
    AerofusionEngine* handle_{nullptr};
};

} // namespace aerofusion

#endif /* AEROFUSION_HPP */
