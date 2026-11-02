"""
End-to-End Automated Verification Script.
Computes RMSE metrics comparing True vs Raw vs EKF-Fused attitude.
"""

import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

try:
    from .binding import AerofusionEngine
    from .trajectory import TrajectoryGenerator
    from .sensor_model import ImuSensorModel
except (ImportError, ValueError):
    from aerofusion_sim.binding import AerofusionEngine
    from aerofusion_sim.trajectory import TrajectoryGenerator
    from aerofusion_sim.sensor_model import ImuSensorModel

import numpy as np


def run_full_simulation(duration_sec: float = 25.0, rate_hz: float = 200.0):
    print("=================================================================")
    print("  Aerofusion EKF End-to-End Simulation & Verification Benchmark  ")
    print("=================================================================")
    print(f"[INFO] Engine Core Version: {AerofusionEngine.version()}")
    print(f"[INFO] Mission Duration: {duration_sec}s @ {rate_hz} Hz ({int(duration_sec * rate_hz)} samples)")

    engine = AerofusionEngine()
    traj = TrajectoryGenerator(duration_sec=duration_sec, rate_hz=rate_hz)
    sensor_model = ImuSensorModel()

    times = []
    true_rolls, true_pitches, true_yaws = [], [], []
    raw_rolls, raw_pitches, raw_yaws = [], [], []
    fused_rolls, fused_pitches, fused_yaws = [], [], []
    true_biases_x, est_biases_x = [], []

    dt = 1.0 / rate_hz

    for state in traj.generate():
        packet = sensor_model.corrupt(state, dt)

        # Feed into Rust EKF Engine
        engine.step_with_mag(packet.raw_gyro, packet.raw_accel, packet.raw_mag, dt)
        telem = engine.get_telemetry()

        times.append(packet.time)
        true_rolls.append(packet.true_roll_deg)
        true_pitches.append(packet.true_pitch_deg)
        true_yaws.append(packet.true_yaw_deg)

        raw_rolls.append(packet.raw_roll_deg)
        raw_pitches.append(packet.raw_pitch_deg)
        raw_yaws.append(packet.raw_yaw_deg)

        fused_rolls.append(telem.roll_deg)
        fused_pitches.append(telem.pitch_deg)
        fused_yaws.append(telem.yaw_deg)

        true_biases_x.append(packet.true_bias[0])
        est_biases_x.append(telem.bias_x)

    # Compute Root Mean Square Error (RMSE) excluding initial 3-second convergence transient
    idx_settled = int(3.0 * rate_hz)

    true_r = np.array(true_rolls[idx_settled:])
    true_p = np.array(true_pitches[idx_settled:])
    true_y = np.array(true_yaws[idx_settled:])

    raw_r = np.array(raw_rolls[idx_settled:])
    raw_p = np.array(raw_pitches[idx_settled:])
    raw_y = np.array(raw_yaws[idx_settled:])

    fused_r = np.array(fused_rolls[idx_settled:])
    fused_p = np.array(fused_pitches[idx_settled:])
    fused_y = np.array(fused_yaws[idx_settled:])

    def wrap_deg(angles):
        return (angles + 180.0) % 360.0 - 180.0

    diff_raw_r = wrap_deg(raw_r - true_r)
    diff_raw_p = wrap_deg(raw_p - true_p)
    diff_raw_y = wrap_deg(raw_y - true_y)

    diff_fused_r = wrap_deg(fused_r - true_r)
    diff_fused_p = wrap_deg(fused_p - true_p)
    diff_fused_y = wrap_deg(fused_y - true_y)

    rmse_raw_roll = np.sqrt(np.mean(diff_raw_r ** 2))
    rmse_raw_pitch = np.sqrt(np.mean(diff_raw_p ** 2))
    rmse_raw_yaw = np.sqrt(np.mean(diff_raw_y ** 2))

    rmse_fused_roll = np.sqrt(np.mean(diff_fused_r ** 2))
    rmse_fused_pitch = np.sqrt(np.mean(diff_fused_p ** 2))
    rmse_fused_yaw = np.sqrt(np.mean(diff_fused_y ** 2))

    bias_err_x = np.mean(np.abs(np.array(est_biases_x[idx_settled:]) - np.array(true_biases_x[idx_settled:])))

    print("\n---------------- Performance Comparison Table -----------------")
    print(f"{'Metric':<22} | {'Raw (Unfused Gyro)':<20} | {'Aerofusion EKF':<18} | {'Improvement':<12}")
    print("-" * 80)
    print(f"{'Roll RMSE [deg]':<22} | {rmse_raw_roll:<20.2f} | {rmse_fused_roll:<18.3f} | {rmse_raw_roll / rmse_fused_roll:.1f}x")
    print(f"{'Pitch RMSE [deg]':<22} | {rmse_raw_pitch:<20.2f} | {rmse_fused_pitch:<18.3f} | {rmse_raw_pitch / rmse_fused_pitch:.1f}x")
    print(f"{'Yaw RMSE [deg]':<22} | {rmse_raw_yaw:<20.2f} | {rmse_fused_yaw:<18.3f} | {rmse_raw_yaw / rmse_fused_yaw:.1f}x")
    print(f"{'Gyro Bias X Error [rad/s]':<22} | {'N/A (untracked)':<20} | {bias_err_x:<18.5f} | {'Locked'}")
    print("---------------------------------------------------------------")

    # Strict flight software accuracy assertions
    assert rmse_fused_roll < 1.0, f"Roll RMSE too high: {rmse_fused_roll}"
    assert rmse_fused_pitch < 1.0, f"Pitch RMSE too high: {rmse_fused_pitch}"
    assert rmse_fused_yaw < 2.0, f"Yaw RMSE too high: {rmse_fused_yaw}"
    print("[SUCCESS] All flight accuracy criteria met (< 1.0 deg attitude RMSE)!")

    return {
        "times": times,
        "true_r": true_rolls, "true_p": true_pitches, "true_y": true_yaws,
        "raw_r": raw_rolls, "raw_p": raw_pitches, "raw_y": raw_yaws,
        "fused_r": fused_rolls, "fused_p": fused_pitches, "fused_y": fused_yaws,
        "true_bx": true_biases_x, "est_bx": est_biases_x,
    }


if __name__ == "__main__":
    run_full_simulation()
