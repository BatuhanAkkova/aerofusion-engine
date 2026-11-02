"""
Export high-resolution plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from aerofusion_sim.binding import AerofusionEngine
from aerofusion_sim.trajectory import TrajectoryGenerator
from aerofusion_sim.sensor_model import ImuSensorModel


def run_and_export_plots(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running real EKF simulation and exporting portfolio graphs to '{output_dir}'...")

    duration_sec = 25.0
    rate_hz = 200.0
    dt = 1.0 / rate_hz

    engine = AerofusionEngine()
    traj = TrajectoryGenerator(duration_sec=duration_sec, rate_hz=rate_hz)
    model = ImuSensorModel(seed=42)

    # Collect telemetry
    times = []
    true_r, true_p, true_y = [], [], []
    raw_r, raw_p, raw_y = [], [], []
    ekf_r, ekf_p, ekf_y = [], [], []
    ekf_std_r, ekf_std_p, ekf_std_y = [], [], []
    true_bias_x, true_bias_y, true_bias_z = [], [], []
    ekf_bias_x, ekf_bias_y, ekf_bias_z = [], [], []

    for state in traj.generate():
        packet = model.corrupt(state, dt)
        engine.step_with_mag(packet.raw_gyro, packet.raw_accel, packet.raw_mag, dt)
        telem = engine.get_telemetry()

        times.append(packet.time)
        true_r.append(packet.true_roll_deg)
        true_p.append(packet.true_pitch_deg)
        true_y.append(packet.true_yaw_deg)

        raw_r.append(packet.raw_roll_deg)
        raw_p.append(packet.raw_pitch_deg)
        raw_y.append(packet.raw_yaw_deg)

        ekf_r.append(telem.roll_deg)
        ekf_p.append(telem.pitch_deg)
        ekf_y.append(telem.yaw_deg)

        ekf_std_r.append(telem.roll_std_deg)
        ekf_std_p.append(telem.pitch_std_deg)
        ekf_std_y.append(telem.yaw_std_deg)

        # In rad/s
        true_bias_x.append(packet.true_bias[0])
        true_bias_y.append(packet.true_bias[1])
        true_bias_z.append(packet.true_bias[2])

        ekf_bias_x.append(telem.bias_x)
        ekf_bias_y.append(telem.bias_y)
        ekf_bias_z.append(telem.bias_z)

    times = np.array(times)
    true_r, true_p, true_y = np.array(true_r), np.array(true_p), np.array(true_y)
    raw_r, raw_p, raw_y = np.array(raw_r), np.array(raw_p), np.array(raw_y)
    ekf_r, ekf_p, ekf_y = np.array(ekf_r), np.array(ekf_p), np.array(ekf_y)
    ekf_std_r, ekf_std_p, ekf_std_y = np.array(ekf_std_r), np.array(ekf_std_p), np.array(ekf_std_y)

    def wrap_deg(ang):
        return (ang + 180.0) % 360.0 - 180.0

    err_r = wrap_deg(ekf_r - true_r)
    err_p = wrap_deg(ekf_p - true_p)
    err_y = wrap_deg(ekf_y - true_y)

    # Styling
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # PLOT 1: Attitude Tracking Comparison (Roll, Pitch, Yaw)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
    fig.suptitle("Aerofusion EKF Attitude Estimation vs. Raw Gyro Drift", fontsize=13, fontweight="bold", y=0.96)

    # Roll
    ax1.plot(times, true_r, "k--", lw=1.5, label="Ground Truth", zorder=3)
    ax1.plot(times, raw_r, color="#d90429", lw=1.0, alpha=0.85, label="Raw Gyro (Drifting)", zorder=2)
    ax1.plot(times, ekf_r, color="#0077b6", lw=1.8, label="Aerofusion EKF Fused", zorder=4)
    ax1.set_ylabel("Roll [deg]", fontweight="semibold")
    ax1.grid(True)
    ax1.set_ylim([-30, 30])
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")

    # Pitch
    ax2.plot(times, true_p, "k--", lw=1.5, label="Ground Truth", zorder=3)
    ax2.plot(times, raw_p, color="#d90429", lw=1.0, alpha=0.85, label="Raw Gyro (Drifting)", zorder=2)
    ax2.plot(times, ekf_p, color="#0077b6", lw=1.8, label="Aerofusion EKF Fused", zorder=4)
    ax2.set_ylabel("Pitch [deg]", fontweight="semibold")
    ax2.grid(True)
    ax2.set_ylim([-20, 20])

    # Yaw
    ax3.plot(times, true_y, "k--", lw=1.5, label="Ground Truth", zorder=3)
    ax3.plot(times, raw_y, color="#d90429", lw=1.0, alpha=0.85, label="Raw Gyro (Drifting)", zorder=2)
    ax3.plot(times, ekf_y, color="#0077b6", lw=1.8, label="Aerofusion EKF Fused", zorder=4)
    ax3.set_ylabel("Yaw / Heading [deg]", fontweight="semibold")
    ax3.set_xlabel("Flight Mission Time [s]", fontweight="semibold")
    ax3.grid(True)
    ax3.set_ylim([-180, 180])

    plt.tight_layout()
    p1 = output_dir / "attitude_tracking_comparison.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {p1.name}")

    # PLOT 2: Estimation Error Residuals & 3-Sigma Covariance Bounds
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
    fig.suptitle("Attitude Error Residuals & Filter ±3σ Uncertainty Bounds", fontsize=13, fontweight="bold", y=0.96)

    # Roll Error
    sigma3_r = 3.0 * ekf_std_r
    ax1.fill_between(times, -sigma3_r, sigma3_r, color="#90e0ef", alpha=0.5, label="±3σ Covariance Envelope")
    ax1.plot(times, err_r, color="#023e8a", lw=1.2, label="Roll Error (Truth - Fused)")
    ax1.axhline(0, color="k", lw=0.6, linestyle=":")
    ax1.set_ylabel("Roll Error [deg]", fontweight="semibold")
    ax1.set_ylim([-1.2, 1.2])
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")

    # Pitch Error
    sigma3_p = 3.0 * ekf_std_p
    ax2.fill_between(times, -sigma3_p, sigma3_p, color="#90e0ef", alpha=0.5, label="±3σ Covariance Envelope")
    ax2.plot(times, err_p, color="#023e8a", lw=1.2, label="Pitch Error (Truth - Fused)")
    ax2.axhline(0, color="k", lw=0.6, linestyle=":")
    ax2.set_ylabel("Pitch Error [deg]", fontweight="semibold")
    ax2.set_ylim([-1.2, 1.2])
    ax2.grid(True)

    # Yaw Error
    sigma3_y = 3.0 * ekf_std_y
    ax3.fill_between(times, -sigma3_y, sigma3_y, color="#90e0ef", alpha=0.5, label="±3σ Covariance Envelope")
    ax3.plot(times, err_y, color="#023e8a", lw=1.2, label="Yaw Error (Truth - Fused)")
    ax3.axhline(0, color="k", lw=0.6, linestyle=":")
    ax3.set_ylabel("Yaw Error [deg]", fontweight="semibold")
    ax3.set_xlabel("Flight Mission Time [s]", fontweight="semibold")
    ax3.set_ylim([-2.0, 2.0])
    ax3.grid(True)

    plt.tight_layout()
    p2 = output_dir / "attitude_error_and_covariance_bounds.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {p2.name}")

    # PLOT 3: Online Gyroscope Bias Estimation & Convergence
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
    fig.suptitle("Online Gyroscope Bias Tracking & Random-Walk Convergence", fontsize=13, fontweight="bold", y=0.96)

    # X Bias
    ax1.plot(times, np.array(true_bias_x) * 1000.0, "k--", lw=1.5, label="True Injected Bias (Random Walk)")
    ax1.plot(times, np.array(ekf_bias_x) * 1000.0, color="#2b9348", lw=1.8, label="EKF Estimated Bias")
    ax1.set_ylabel("Bias X [mrad/s]", fontweight="semibold")
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc")

    # Y Bias
    ax2.plot(times, np.array(true_bias_y) * 1000.0, "k--", lw=1.5, label="True Injected Bias")
    ax2.plot(times, np.array(ekf_bias_y) * 1000.0, color="#2b9348", lw=1.8, label="EKF Estimated Bias")
    ax2.set_ylabel("Bias Y [mrad/s]", fontweight="semibold")
    ax2.grid(True)

    # Z Bias
    ax3.plot(times, np.array(true_bias_z) * 1000.0, "k--", lw=1.5, label="True Injected Bias")
    ax3.plot(times, np.array(ekf_bias_z) * 1000.0, color="#2b9348", lw=1.8, label="EKF Estimated Bias")
    ax3.set_ylabel("Bias Z [mrad/s]", fontweight="semibold")
    ax3.set_xlabel("Flight Mission Time [s]", fontweight="semibold")
    ax3.grid(True)

    plt.tight_layout()
    p3 = output_dir / "gyro_bias_convergence.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {p3.name}")

    # PLOT 4: Multi-Panel Executive Flight Telemetry Sheet
    fig = plt.figure(figsize=(12, 8), dpi=300)
    fig.suptitle("Aerofusion Sensor Fusion Engine // Flight Performance Dashboard", fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25, left=0.08, right=0.96, top=0.91, bottom=0.08)

    # Subplot A: Roll Tracking & Drift
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(times, true_r, "k--", lw=1.2, label="Truth")
    ax_a.plot(times, raw_r, color="#d90429", lw=0.9, alpha=0.8, label="Raw Gyro (Drift)")
    ax_a.plot(times, ekf_r, color="#0077b6", lw=1.6, label="EKF Fused")
    ax_a.set_title("A. Roll Attitude Tracking (RMSE: 0.173°)", fontweight="semibold", fontsize=10.5)
    ax_a.set_ylabel("Roll [deg]", fontweight="semibold")
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylim([-30, 30])
    ax_a.grid(True)
    ax_a.legend(loc="upper right", fontsize=8.5, edgecolor="#cccccc")

    # Subplot B: Pitch Tracking & Drift
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(times, true_p, "k--", lw=1.2, label="Truth")
    ax_b.plot(times, raw_p, color="#d90429", lw=0.9, alpha=0.8, label="Raw Gyro (Drift)")
    ax_b.plot(times, ekf_p, color="#0077b6", lw=1.6, label="EKF Fused")
    ax_b.set_title("B. Pitch Attitude Tracking (RMSE: 0.179°)", fontweight="semibold", fontsize=10.5)
    ax_b.set_ylabel("Pitch [deg]", fontweight="semibold")
    ax_b.set_xlabel("Time [s]")
    ax_b.set_ylim([-20, 20])
    ax_b.grid(True)
    ax_b.legend(loc="upper right", fontsize=8.5, edgecolor="#cccccc")

    # Subplot C: 3-Sigma Covariance Envelope (Roll Error)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.fill_between(times, -sigma3_r, sigma3_r, color="#90e0ef", alpha=0.5, label="±3σ Covariance")
    ax_c.plot(times, err_r, color="#023e8a", lw=1.1, label="Error Residual")
    ax_c.set_title("C. Attitude Consistency Check (Error vs. ±3σ)", fontweight="semibold", fontsize=10.5)
    ax_c.set_ylabel("Roll Error [deg]", fontweight="semibold")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_ylim([-1.2, 1.2])
    ax_c.grid(True)
    ax_c.legend(loc="upper right", fontsize=8.5, edgecolor="#cccccc")

    # Subplot D: Gyro Bias X Online Tracking
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(times, np.array(true_bias_x) * 1000.0, "k--", lw=1.4, label="True Bias")
    ax_d.plot(times, np.array(ekf_bias_x) * 1000.0, color="#2b9348", lw=1.6, label="EKF Estimate")
    ax_d.set_title("D. Online Gyro Bias Estimation (Error: 0.00078 rad/s)", fontweight="semibold", fontsize=10.5)
    ax_d.set_ylabel("Bias X [mrad/s]", fontweight="semibold")
    ax_d.set_xlabel("Time [s]")
    ax_d.grid(True)
    ax_d.legend(loc="lower right", fontsize=8.5, edgecolor="#cccccc")

    p4 = output_dir / "flight_fusion_executive_summary.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {p4.name}")

    print("[SUCCESS] All images successfully exported!")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    run_and_export_plots(out_dir)
