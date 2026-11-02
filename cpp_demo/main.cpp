#include "aerofusion.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  Aerofusion Sensor Fusion Engine - C++ Autopilot Harness \n";
    std::cout << "=========================================================\n";
    std::cout << "[INFO] Engine Version: " << aerofusion::SensorFusionEngine::version() << "\n\n";

    try {
        // Initialize engine with RAII wrapper
        aerofusion::SensorFusionEngine filter;

        std::cout << "[INIT] Initialized 6-DoF Error-State EKF.\n";

        // Simulated flight scenario: Quadcopter performing coordinated turn
        // 400 Hz IMU update rate (dt = 0.0025 s)
        constexpr float dt = 0.0025f;
        constexpr int kTotalSteps = 10000; // 25 seconds of simulated flight

        std::cout << "[SIM] Streaming " << kTotalSteps << " IMU packets at 400 Hz (dt = 2.5 ms)...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < kTotalSteps; ++i) {
            float t = i * dt;

            // Simulated vehicle dynamics: roll oscillates smoothly, pitch hovers near 5 deg
            float true_roll_rate = 0.25f * std::sin(0.5f * t);
            float true_pitch_rate = 0.05f * std::cos(0.2f * t);
            float true_yaw_rate = 0.10f; // constant slow orbit

            // Injected sensor imperfections: gyro bias (1 deg/s) + high-frequency vibration
            float bias_x = 0.0174f; 
            float noise = 0.002f * std::sin(120.0f * t);

            aerofusion::Vec3 gyro(
                true_roll_rate + bias_x + noise,
                true_pitch_rate + noise,
                true_yaw_rate + noise
            );

            // Gravity vector in body frame (simplified for hover / gentle bank)
            aerofusion::Vec3 accel(
                -9.80665f * std::sin(0.1f * std::sin(0.5f * t)),
                9.80665f * std::sin(0.05f),
                9.80665f * std::cos(0.05f)
            );

            // High-rate step
            filter.step(gyro, accel, dt);

            if (i % 2000 == 0) {
                auto telem = filter.get_telemetry();
                std::cout << "  [Step " << std::setw(5) << i << "] "
                          << "Roll: " << std::fixed << std::setprecision(2) << std::setw(6) << telem.euler_deg.roll << " deg | "
                          << "Pitch: " << std::setw(6) << telem.euler_deg.pitch << " deg | "
                          << "Yaw: " << std::setw(6) << telem.euler_deg.yaw << " deg | "
                          << "Est Bias X: " << std::setprecision(4) << telem.gyro_bias.x << " rad/s\n";
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> total_micros = end_time - start_time;
        double avg_step_us = total_micros.count() / kTotalSteps;

        auto final_telem = filter.get_telemetry();

        std::cout << "\n---------------- Benchmark & State Report ----------------\n";
        std::cout << "Total Steps Processed : " << final_telem.step_count << "\n";
        std::cout << "Total Execution Time  : " << std::fixed << std::setprecision(2) << total_micros.count() / 1000.0 << " ms\n";
        std::cout << "Average Latency / Step: " << std::setprecision(3) << avg_step_us << " microseconds\n";
        std::cout << "Max Supported Rate    : " << static_cast<int>(1000000.0 / avg_step_us) << " Hz\n";
        std::cout << "Attitude 1-Sigma Unc. : [" 
                  << final_telem.attitude_std.x << ", " 
                  << final_telem.attitude_std.y << ", " 
                  << final_telem.attitude_std.z << "] rad\n";
        std::cout << "Gyro Bias 1-Sigma Unc.: [" 
                  << final_telem.bias_std.x << ", " 
                  << final_telem.bias_std.y << ", " 
                  << final_telem.bias_std.z << "] rad/s\n";
        std::cout << "----------------------------------------------------------\n";
        std::cout << "[SUCCESS] C++ Flight Loop Completed without fault.\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught in flight harness: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
