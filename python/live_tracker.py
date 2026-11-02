"""
Native Desktop 3D Drone Attitude Tracker & Telemetry Animator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from aerofusion_sim.binding import AerofusionEngine
from aerofusion_sim.trajectory import TrajectoryGenerator
from aerofusion_sim.sensor_model import ImuSensorModel


def run_live_tracker(duration_sec=6.0, fps=20, output_path=None):
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent.parent / "assets" / "aerofusion_flight_demo.gif")
    print(f"[INFO] Initializing Aerofusion EKF live tracker ({duration_sec}s @ {fps} FPS)...")
    rate_hz = 100.0
    dt = 1.0 / rate_hz

    engine = AerofusionEngine()
    traj = TrajectoryGenerator(duration_sec=duration_sec + 4.0, rate_hz=rate_hz)
    model = ImuSensorModel(seed=42)

    # Pre-simulate data: capture dynamic banking maneuvers
    times, true_r, true_p, true_y = [], [], [], []
    raw_r, raw_p, raw_y = [], [], []
    ekf_r, ekf_p, ekf_y = [], [], []
    quats = []

    for state in traj.generate():
        packet = model.corrupt(state, dt)
        engine.step_with_mag(packet.raw_gyro, packet.raw_accel, packet.raw_mag, dt)
        telem = engine.get_telemetry()

        if packet.time >= 4.0 and len(times) < int(duration_sec * rate_hz):
            rel_time = packet.time - 4.0
            times.append(rel_time)
            true_r.append(packet.true_roll_deg)
            true_p.append(packet.true_pitch_deg)
            true_y.append(packet.true_yaw_deg)

            raw_r.append(packet.raw_roll_deg)
            raw_p.append(packet.raw_pitch_deg)
            raw_y.append(packet.raw_yaw_deg)

            ekf_r.append(telem.roll_deg)
            ekf_p.append(telem.pitch_deg)
            ekf_y.append(telem.yaw_deg)
            quats.append((telem.qw, telem.qx, telem.qy, telem.qz))

    total_frames = len(times)
    frame_step = max(1, int(rate_hz / fps))

    # Styling
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(13, 7.5), facecolor="white", dpi=100)
    fig.canvas.manager.set_window_title("Aerofusion Engine - Live EKF 3D Attitude Tracker")

    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.36, wspace=0.22,
                           left=0.06, right=0.96, top=0.91, bottom=0.08)

    # Standard master title
    fig.suptitle("Aerofusion EKF // Real-Time 3D Attitude Estimation",
                 fontsize=14, fontweight="bold", color="#111827", y=0.96)

    # 3D Drone View Subplot
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="white")

    # 2D Strip Charts
    ax_roll = fig.add_subplot(gs[0, 1], facecolor="white")
    ax_roll.set_title("Roll Attitude [deg] // Truth vs Raw Drifting vs EKF", color="#1f2937", fontsize=9.5, fontweight="semibold")
    ax_roll.grid(True, linestyle=":", alpha=0.7, color="#d1d5db")

    ax_pitch = fig.add_subplot(gs[1, 1], facecolor="white")
    ax_pitch.set_title("Pitch Attitude [deg] // Dynamic Banking Tracking", color="#1f2937", fontsize=9.5, fontweight="semibold")
    ax_pitch.grid(True, linestyle=":", alpha=0.7, color="#d1d5db")

    ax_yaw = fig.add_subplot(gs[2, 1], facecolor="white")
    ax_yaw.set_title("Yaw / Heading [deg] // Magnetometer Fusion", color="#1f2937", fontsize=9.5, fontweight="semibold")
    ax_yaw.grid(True, linestyle=":", alpha=0.7, color="#d1d5db")
    ax_yaw.set_xlabel("Elapsed Flight Time [s]", color="#4b5563", fontsize=9, fontweight="semibold")

    # Plot lines
    (line_r_true,) = ax_roll.plot([], [], "--", color="#111827", lw=1.5, label="Ground Truth")
    (line_r_raw,) = ax_roll.plot([], [], "-", color="#dc2626", lw=1.1, alpha=0.85, label="Raw Gyro (Drifting)")
    (line_r_ekf,) = ax_roll.plot([], [], "-", color="#0284c7", lw=2.0, label="Rust EKF Fused")
    ax_roll.legend(loc="upper right", fontsize=8.5, facecolor="white", edgecolor="#d1d5db", framealpha=0.95)

    (line_p_true,) = ax_pitch.plot([], [], "--", color="#111827", lw=1.5)
    (line_p_raw,) = ax_pitch.plot([], [], "-", color="#dc2626", lw=1.1, alpha=0.85)
    (line_p_ekf,) = ax_pitch.plot([], [], "-", color="#0284c7", lw=2.0)

    (line_y_true,) = ax_yaw.plot([], [], "--", color="#111827", lw=1.5)
    (line_y_raw,) = ax_yaw.plot([], [], "-", color="#dc2626", lw=1.1, alpha=0.85)
    (line_y_ekf,) = ax_yaw.plot([], [], "-", color="#0284c7", lw=2.0)

    for ax in [ax_roll, ax_pitch, ax_yaw]:
        ax.tick_params(colors="#374151", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

    def rotate_pt(q, p):
        qw, qx, qy, qz = q
        uv_x = qy * p[2] - qz * p[1]
        uv_y = qz * p[0] - qx * p[2]
        uv_z = qx * p[1] - qy * p[0]
        uuv_x = qy * uv_z - qz * uv_y
        uuv_y = qz * uv_x - qx * uv_z
        uuv_z = qx * uv_y - qy * uv_x
        return np.array([
            p[0] + 2.0 * (qw * uv_x + uuv_x),
            p[1] + 2.0 * (qw * uv_y + uuv_y),
            p[2] + 2.0 * (qw * uv_z + uuv_z),
        ])

    def update(frame_idx):
        idx = min(frame_idx * frame_step, total_frames - 1)
        q = quats[idx]

        # 1. Update 3D drone scene
        ax_3d.cla()
        ax_3d.set_facecolor("white")
        ax_3d.set_xlim([-1.3, 1.3])
        ax_3d.set_ylim([-1.3, 1.3])
        ax_3d.set_zlim([-1.1, 1.1])
        ax_3d.set_axis_off()

        # Airframe arms (X configuration)
        arm_d = 0.85
        p1 = rotate_pt(q, [arm_d * 0.707, arm_d * 0.707, 0.0])
        p2 = rotate_pt(q, [-arm_d * 0.707, -arm_d * 0.707, 0.0])
        p3 = rotate_pt(q, [-arm_d * 0.707, arm_d * 0.707, 0.0])
        p4 = rotate_pt(q, [arm_d * 0.707, -arm_d * 0.707, 0.0])

        # Solid slate structural arms
        ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#334155", lw=5, solid_capstyle="round")
        ax_3d.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color="#334155", lw=5, solid_capstyle="round")

        # 4 Rotor Discs with spinning effect
        r_disc = 0.28
        th = np.linspace(0, 2 * np.pi, 24)
        rot_angle = frame_idx * 1.2
        for motor_pos in [p1, p2, p3, p4]:
            disc_pts = np.array([rotate_pt(q, [motor_pos[0] + r_disc * np.cos(t),
                                               motor_pos[1] + r_disc * np.sin(t),
                                               motor_pos[2]]) for t in th])
            ax_3d.plot(disc_pts[:, 0], disc_pts[:, 1], disc_pts[:, 2], color="#059669", lw=1.6, alpha=0.9)
            # Propeller blade
            b1 = rotate_pt(q, [motor_pos[0] + r_disc * np.cos(rot_angle), motor_pos[1] + r_disc * np.sin(rot_angle), motor_pos[2]])
            b2 = rotate_pt(q, [motor_pos[0] - r_disc * np.cos(rot_angle), motor_pos[1] - r_disc * np.sin(rot_angle), motor_pos[2]])
            ax_3d.plot([b1[0], b2[0]], [b1[1], b2[1]], [b1[2], b2[2]], color="#64748b", lw=2)

        # Center Body Avionics Hub
        center = rotate_pt(q, [0, 0, 0])
        ax_3d.scatter([center[0]], [center[1]], [center[2]], color="#0f172a", s=110, edgecolors="#38bdf8", lw=2, zorder=10)

        # Body Frame Coordinate Arrows (+X Red: Forward, +Y Green: Right, +Z Blue: Up)
        axes_scale = 0.65
        x_axis = rotate_pt(q, [axes_scale, 0, 0])
        y_axis = rotate_pt(q, [0, axes_scale, 0])
        z_axis = rotate_pt(q, [0, 0, axes_scale])

        ax_3d.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], color="#dc2626", lw=3.2)
        ax_3d.text(x_axis[0] * 1.1, x_axis[1] * 1.1, x_axis[2] * 1.1, "+X (Nose)", color="#dc2626", fontsize=8.5, fontweight="bold")

        ax_3d.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], color="#16a34a", lw=2)
        ax_3d.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], color="#0284c7", lw=2)

        # Live Telemetry HUD text
        r_deg = ekf_r[idx]
        p_deg = ekf_p[idx]
        y_deg = ekf_y[idx]
        t_sec = times[idx]
        hud_str = (
            f"TELEMETRY HUD\n"
            f"TIME : {t_sec:5.2f} s\n"
            f"ROLL : {r_deg:+6.2f}°\n"
            f"PITCH: {p_deg:+6.2f}°\n"
            f"YAW  : {y_deg:+6.2f}°\n"
        )
        ax_3d.text2D(0.04, 0.82, hud_str, transform=ax_3d.transAxes,
                    color="#111827", fontsize=8.5, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#d1d5db", alpha=0.95))

        # 2. Update Strip Charts
        w_start = 0
        t_w = times[w_start:idx + 1]

        line_r_true.set_data(t_w, true_r[w_start:idx + 1])
        line_r_raw.set_data(t_w, raw_r[w_start:idx + 1])
        line_r_ekf.set_data(t_w, ekf_r[w_start:idx + 1])
        ax_roll.set_xlim([0, duration_sec])
        ax_roll.set_ylim([-30, 30])

        line_p_true.set_data(t_w, true_p[w_start:idx + 1])
        line_p_raw.set_data(t_w, raw_p[w_start:idx + 1])
        line_p_ekf.set_data(t_w, ekf_p[w_start:idx + 1])
        ax_pitch.set_xlim([0, duration_sec])
        ax_pitch.set_ylim([-20, 20])

        line_y_true.set_data(t_w, true_y[w_start:idx + 1])
        line_y_raw.set_data(t_w, raw_y[w_start:idx + 1])
        line_y_ekf.set_data(t_w, ekf_y[w_start:idx + 1])
        ax_yaw.set_xlim([0, duration_sec])
        ax_yaw.set_ylim([-180, 180])

        return []

    anim_frames = int(total_frames / frame_step)
    ani = FuncAnimation(fig, update, frames=anim_frames, interval=1000.0 / fps, blit=False)

    if "--save-gif" in sys.argv:
        print(f"[INFO] Rendering {anim_frames} frames directly to '{output_path}'...")
        ani.save(output_path, writer="pillow", fps=fps)
        print(f"[SUCCESS] Animation saved to '{output_path}'!")
    else:
        print("[INFO] Displaying live interactive tracker window. Close the window to exit.")
        plt.show()


if __name__ == "__main__":
    run_live_tracker()
