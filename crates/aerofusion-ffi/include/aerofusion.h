/**
 * @file aerofusion.h
 * @brief Safety-Critical 6-DoF Sensor Fusion Engine C ABI Interface.
 *
 * Designed for flight computers, autopilots, and hard real-time embedded systems.
 */

#ifndef AEROFUSION_H
#define AEROFUSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define AEROFUSION_OK                 0
#define AEROFUSION_ERR_NULL_PTR      -1
#define AEROFUSION_ERR_PANIC         -2
#define AEROFUSION_ERR_INVALID_PARAM -3

/**
 * @brief Opaque handle representing an instance of the Aerofusion EKF engine.
 */
typedef struct AerofusionEngine AerofusionEngine;

/**
 * @brief Configuration parameters for filter tuning and sensor noise models.
 */
typedef struct {
    float gyro_noise;        /**< Continuous gyro noise density (rad/s / sqrt(Hz)) */
    float gyro_bias_noise;   /**< Gyro bias random-walk density (rad/s^2 / sqrt(Hz)) */
    float accel_noise;       /**< Accelerometer measurement noise (m/s^2) */
    float mag_noise;         /**< Magnetometer measurement noise */
    float init_att_cov;      /**< Initial attitude error variance (rad^2) */
    float init_bias_cov;     /**< Initial gyro bias variance ((rad/s)^2) */
    float gravity_magnitude; /**< Local gravity magnitude (typically 9.80665) */
} AerofusionConfigC;

/**
 * @brief Telemetry output payload with full attitude and 1-sigma uncertainty bounds.
 */
typedef struct {
    float qw;         /**< Orientation quaternion W */
    float qx;         /**< Orientation quaternion X */
    float qy;         /**< Orientation quaternion Y */
    float qz;         /**< Orientation quaternion Z */
    float roll_rad;   /**< Roll angle in radians */
    float pitch_rad;  /**< Pitch angle in radians */
    float yaw_rad;    /**< Yaw angle in radians */
    float roll_deg;   /**< Roll angle in degrees */
    float pitch_deg;  /**< Pitch angle in degrees */
    float yaw_deg;    /**< Yaw angle in degrees */
    float bias_x;     /**< Gyroscope bias X [rad/s] */
    float bias_y;     /**< Gyroscope bias Y [rad/s] */
    float bias_z;     /**< Gyroscope bias Z [rad/s] */
    float att_std_x;  /**< 1-sigma Roll attitude uncertainty [rad] */
    float att_std_y;  /**< 1-sigma Pitch attitude uncertainty [rad] */
    float att_std_z;  /**< 1-sigma Yaw attitude uncertainty [rad] */
    float bias_std_x; /**< 1-sigma Gyro bias X uncertainty [rad/s] */
    float bias_std_y; /**< 1-sigma Gyro bias Y uncertainty [rad/s] */
    float bias_std_z; /**< 1-sigma Gyro bias Z uncertainty [rad/s] */
    uint64_t step_count; /**< Total filtered step counter */
} AerofusionTelemetryC;

/**
 * @brief Returns engine version descriptor string.
 */
const char* aerofusion_version(void);

/**
 * @brief Allocates and initializes a new EKF filter engine instance with default parameters.
 * @return Pointer to opaque AerofusionEngine instance, or NULL on allocation error.
 */
AerofusionEngine* aerofusion_create(void);

/**
 * @brief Allocates and initializes a new EKF filter engine with custom noise parameters.
 */
AerofusionEngine* aerofusion_create_with_config(const AerofusionConfigC* config);

/**
 * @brief Safely destroys the engine and reclaims all allocated resources.
 */
void aerofusion_destroy(AerofusionEngine* engine);

/**
 * @brief Resets filter states, bias estimates, and covariances to initial conditions.
 */
int32_t aerofusion_reset(AerofusionEngine* engine);

/**
 * @brief Executes time propagation (predict step) using angular rates from the gyroscope.
 * @param engine Pointer to engine instance.
 * @param gx Body-frame angular velocity X [rad/s]
 * @param gy Body-frame angular velocity Y [rad/s]
 * @param gz Body-frame angular velocity Z [rad/s]
 * @param dt Delta time since previous step [s]
 */
int32_t aerofusion_predict(AerofusionEngine* engine, float gx, float gy, float gz, float dt);

/**
 * @brief Executes measurement update with 3-axis accelerometer.
 */
int32_t aerofusion_update_accel(AerofusionEngine* engine, float ax, float ay, float az);

/**
 * @brief Executes measurement update with 3-axis magnetometer.
 */
int32_t aerofusion_update_mag(AerofusionEngine* engine, float mx, float my, float mz);

/**
 * @brief Convenience combined step: prediction and accelerometer update.
 */
int32_t aerofusion_step(AerofusionEngine* engine, float gx, float gy, float gz, float ax, float ay, float az, float dt);

/**
 * @brief Combined step: prediction, accelerometer update, and magnetometer update.
 */
int32_t aerofusion_step_with_mag(AerofusionEngine* engine, float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz, float dt);

/**
 * @brief Retrieves latest telemetry snapshot and covariance bounds.
 */
int32_t aerofusion_get_telemetry(const AerofusionEngine* engine, AerofusionTelemetryC* out);

/**
 * @brief Retrieves latest Euler angles in degrees.
 */
int32_t aerofusion_get_euler_deg(const AerofusionEngine* engine, float* roll, float* pitch, float* yaw);

#ifdef __cplusplus
}
#endif

#endif /* AEROFUSION_H */
