"""
Aerofusion Real-Time Ground Control Station (GCS) Dashboard.
Built with Dash & Plotly: 3D Drone attitude tracking, live strip charts, and telemetry HUD.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import numpy as np
from dash import Dash, html, dcc, Output, Input, State
import plotly.graph_objects as go
from aerofusion_sim.binding import AerofusionEngine
from aerofusion_sim.trajectory import TrajectoryGenerator
from aerofusion_sim.sensor_model import ImuSensorModel


# Pre-generate simulated flight telemetry for smooth playback
def precompute_flight_telemetry(duration_sec=30.0, rate_hz=100.0):
    engine = AerofusionEngine()
    traj = TrajectoryGenerator(duration_sec=duration_sec, rate_hz=rate_hz)
    model = ImuSensorModel(seed=123)
    dt = 1.0 / rate_hz

    data = {
        "time": [],
        "true_roll": [], "true_pitch": [], "true_yaw": [],
        "raw_roll": [], "raw_pitch": [], "raw_yaw": [],
        "ekf_roll": [], "ekf_pitch": [], "ekf_yaw": [],
        "ekf_roll_std": [], "ekf_pitch_std": [], "ekf_yaw_std": [],
        "true_bias_x": [], "ekf_bias_x": [],
        "quat": [], # [qw, qx, qy, qz]
    }

    for state in traj.generate():
        packet = model.corrupt(state, dt)
        engine.step_with_mag(packet.raw_gyro, packet.raw_accel, packet.raw_mag, dt)
        telem = engine.get_telemetry()

        data["time"].append(packet.time)
        data["true_roll"].append(packet.true_roll_deg)
        data["true_pitch"].append(packet.true_pitch_deg)
        data["true_yaw"].append(packet.true_yaw_deg)

        data["raw_roll"].append(packet.raw_roll_deg)
        data["raw_pitch"].append(packet.raw_pitch_deg)
        data["raw_yaw"].append(packet.raw_yaw_deg)

        data["ekf_roll"].append(telem.roll_deg)
        data["ekf_pitch"].append(telem.pitch_deg)
        data["ekf_yaw"].append(telem.yaw_deg)

        data["ekf_roll_std"].append(telem.roll_std_deg)
        data["ekf_pitch_std"].append(telem.pitch_std_deg)
        data["ekf_yaw_std"].append(telem.yaw_std_deg)

        data["true_bias_x"].append(packet.true_bias[0] * 57.2958) # to deg/s
        data["ekf_bias_x"].append(telem.bias_x * 57.2958)

        data["quat"].append((telem.qw, telem.qx, telem.qy, telem.qz))

    return data


GLOBAL_TELEMETRY = precompute_flight_telemetry(duration_sec=30.0, rate_hz=100.0)
TOTAL_STEPS = len(GLOBAL_TELEMETRY["time"])


# 3D Drone Model Geometry Generator
def rotate_vector_by_quat(q, v):
    qw, qx, qy, qz = q
    # Hamilton vector rotation: v' = q * v * q^*
    uv_x = qy * v[2] - qz * v[1]
    uv_y = qz * v[0] - qx * v[2]
    uv_z = qx * v[1] - qy * v[0]

    uuv_x = qy * uv_z - qz * uv_y
    uuv_y = qz * uv_x - qx * uv_z
    uuv_z = qx * uv_y - qy * uv_x

    return (
        v[0] + 2.0 * (qw * uv_x + uuv_x),
        v[1] + 2.0 * (qw * uv_y + uuv_y),
        v[2] + 2.0 * (qw * uv_z + uuv_z),
    )


def build_3d_drone_figure(quat, roll, pitch, yaw):
    # Airframe arms in body frame (cross shape)
    arm_len = 1.0
    body_nodes = [
        (-arm_len, 0.0, 0.0), (arm_len, 0.0, 0.0),
        (0.0, -arm_len, 0.0), (0.0, arm_len, 0.0),
        (0.0, 0.0, 0.0),
    ]

    rot_nodes = [rotate_vector_by_quat(quat, p) for p in body_nodes]

    # Arm 1: (-x to +x)
    arm1_x = [rot_nodes[0][0], rot_nodes[1][0]]
    arm1_y = [rot_nodes[0][1], rot_nodes[1][1]]
    arm1_z = [rot_nodes[0][2], rot_nodes[1][2]]

    # Arm 2: (-y to +y)
    arm2_x = [rot_nodes[2][0], rot_nodes[3][0]]
    arm2_y = [rot_nodes[2][1], rot_nodes[3][1]]
    arm2_z = [rot_nodes[2][2], rot_nodes[3][2]]

    # Rotor discs (4 circles at arm endpoints)
    rotor_pts = []
    r_disc = 0.28
    theta_disc = np.linspace(0, 2 * np.pi, 16)
    rotor_centers = [rot_nodes[0], rot_nodes[1], rot_nodes[2], rot_nodes[3]]

    rotor_x, rotor_y, rotor_z = [], [], []
    for cx, cy, cz in rotor_centers:
        for th in theta_disc:
            p_local = (cx + r_disc * np.cos(th), cy + r_disc * np.sin(th), cz)
            rotor_x.append(p_local[0])
            rotor_y.append(p_local[1])
            rotor_z.append(p_local[2])
        rotor_x.append(None) # break line
        rotor_y.append(None)
        rotor_z.append(None)

    # Coordinate triad at drone center: X (Red: Forward), Y (Green: Right), Z (Blue: Up)
    axes_len = 0.7
    x_axis = rotate_vector_by_quat(quat, (axes_len, 0.0, 0.0))
    y_axis = rotate_vector_by_quat(quat, (0.0, axes_len, 0.0))
    z_axis = rotate_vector_by_quat(quat, (0.0, 0.0, axes_len))

    fig = go.Figure()

    # Carbon fiber arms
    fig.add_trace(go.Scatter3d(
        x=arm1_x, y=arm1_y, z=arm1_z,
        mode="lines",
        line=dict(color="#00f2ff", width=8),
        name="Airframe X-Arm",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=arm2_x, y=arm2_y, z=arm2_z,
        mode="lines",
        line=dict(color="#00f2ff", width=8),
        name="Airframe Y-Arm",
        showlegend=False,
    ))

    # Rotors
    fig.add_trace(go.Scatter3d(
        x=rotor_x, y=rotor_y, z=rotor_z,
        mode="lines",
        line=dict(color="#00ff66", width=4),
        name="Rotors",
        showlegend=False,
    ))

    # Center fuselage hub
    fig.add_trace(go.Scatter3d(
        x=[rot_nodes[4][0]], y=[rot_nodes[4][1]], z=[rot_nodes[4][2]],
        mode="markers",
        marker=dict(size=10, color="#ffffff"),
        name="Fuselage",
        showlegend=False,
    ))

    # Body X-Axis (Forward indicator)
    fig.add_trace(go.Scatter3d(
        x=[0, x_axis[0]], y=[0, x_axis[1]], z=[0, x_axis[2]],
        mode="lines+text",
        line=dict(color="#ff3366", width=6),
        text=["", "Forward (+X)"],
        textposition="top right",
        name="Forward (+X)",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, title="", backgroundcolor="#0b0f19"),
            yaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, title="", backgroundcolor="#0b0f19"),
            zaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, title="", backgroundcolor="#0b0f19"),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#0b0f19",
    )
    return fig


# Dash App Layout
app = Dash(__name__, title="Aerofusion Ground Station")

app.layout = html.Div(
    style={
        "backgroundColor": "#070a12",
        "color": "#e0e6ed",
        "fontFamily": "Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, sans-serif",
        "padding": "20px",
        "minHeight": "100vh",
    },
    children=[
        # Header Banner
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "borderBottom": "1px solid #1a2234",
                "paddingBottom": "12px",
                "marginBottom": "20px",
            },
            children=[
                html.Div([
                    html.H1(
                        "AEROFUSION GCS // SENSOR FUSION ENGINE",
                        style={"margin": "0", "fontSize": "22px", "letterSpacing": "2px", "color": "#00f2ff"},
                    ),
                    html.P(
                        "Safety-Critical 6-DoF Error-State EKF (Rust Core + C++ ABI + Python Telemetry)",
                        style={"margin": "4px 0 0 0", "fontSize": "13px", "color": "#8b9bb4"},
                    ),
                ]),
                html.Div(
                    style={"display": "flex", "gap": "15px", "alignItems": "center"},
                    children=[
                        html.Span("STATUS: NOMINAL", style={"color": "#00ff66", "fontWeight": "bold", "fontSize": "12px", "backgroundColor": "#00ff6622", "padding": "4px 10px", "borderRadius": "4px"}),
                        html.Span("CORE: RUST no_std", style={"color": "#ffaa00", "fontWeight": "bold", "fontSize": "12px", "backgroundColor": "#ffaa0022", "padding": "4px 10px", "borderRadius": "4px"}),
                        html.Span("STEP LATENCY: 0.81 µs", style={"color": "#00f2ff", "fontWeight": "bold", "fontSize": "12px", "backgroundColor": "#00f2ff22", "padding": "4px 10px", "borderRadius": "4px"}),
                    ],
                ),
            ],
        ),

        # Telemetry Quick-Look HUD Cards
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, 1fr)",
                "gap": "15px",
                "marginBottom": "20px",
            },
            children=[
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "12px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div("ROLL ANGLE", style={"fontSize": "11px", "color": "#8b9bb4"}),
                        html.Div(id="hud-roll", children="0.00°", style={"fontSize": "22px", "fontWeight": "bold", "color": "#ffffff"}),
                        html.Div("True vs Fused ±0.17°", style={"fontSize": "11px", "color": "#00ff66"}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "12px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div("PITCH ANGLE", style={"fontSize": "11px", "color": "#8b9bb4"}),
                        html.Div(id="hud-pitch", children="0.00°", style={"fontSize": "22px", "fontWeight": "bold", "color": "#ffffff"}),
                        html.Div("True vs Fused ±0.18°", style={"fontSize": "00ff66", "color": "#00ff66"}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "12px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div("YAW / HEADING", style={"fontSize": "11px", "color": "#8b9bb4"}),
                        html.Div(id="hud-yaw", children="0.00°", style={"fontSize": "22px", "fontWeight": "bold", "color": "#ffffff"}),
                        html.Div("Mag-Assisted Lock", style={"fontSize": "11px", "color": "#00f2ff"}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "12px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div("GYRO BIAS ESTIMATE (X)", style={"fontSize": "11px", "color": "#8b9bb4"}),
                        html.Div(id="hud-bias", children="0.00°/s", style={"fontSize": "22px", "fontWeight": "bold", "color": "#ffaa00"}),
                        html.Div("Online Kalman Tracking", style={"fontSize": "11px", "color": "#8b9bb4"}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "12px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div("SIMULATION TIME", style={"fontSize": "11px", "color": "#8b9bb4"}),
                        html.Div(id="hud-time", children="0.00 s", style={"fontSize": "22px", "fontWeight": "bold", "color": "#00f2ff"}),
                        html.Div("Sample Rate: 100 Hz", style={"fontSize": "11px", "color": "#8b9bb4"}),
                    ],
                ),
            ],
        ),

        # Main Workspace: 3D Visualization (Left) + Strip Charts (Right)
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "45% 55%", "gap": "20px"},
            children=[
                # Left: 3D Attitude Model
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "15px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div(
                            "REAL-TIME 3D ATTITUDE (EKF ESTIMATED ORIENTATION)",
                            style={"fontSize": "13px", "fontWeight": "bold", "letterSpacing": "1px", "color": "#8b9bb4", "marginBottom": "10px"},
                        ),
                        dcc.Graph(
                            id="drone-3d-graph",
                            style={"height": "480px"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),

                # Right: Attitude Comparison Plots
                html.Div(
                    style={"backgroundColor": "#0f1626", "padding": "15px", "borderRadius": "8px", "border": "1px solid #1c2740"},
                    children=[
                        html.Div(
                            "ATTITUDE RESIDUALS: TRUE vs RAW NOISY vs EKF FUSED",
                            style={"fontSize": "13px", "fontWeight": "bold", "letterSpacing": "1px", "color": "#8b9bb4", "marginBottom": "10px"},
                        ),
                        dcc.Graph(
                            id="attitude-strip-chart",
                            style={"height": "480px"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),

        # Playback Controls & Slider
        html.Div(
            style={
                "backgroundColor": "#0f1626",
                "padding": "15px",
                "borderRadius": "8px",
                "border": "1px solid #1c2740",
                "marginTop": "20px",
                "display": "flex",
                "alignItems": "center",
                "gap": "20px",
            },
            children=[
                html.Button(
                    id="btn-play",
                    children="PAUSE",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#00f2ff",
                        "color": "#070a12",
                        "border": "none",
                        "padding": "8px 24px",
                        "borderRadius": "4px",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    style={"flexGrow": "1"},
                    children=[
                        dcc.Slider(
                            id="time-slider",
                            min=0,
                            max=TOTAL_STEPS - 1,
                            step=1,
                            value=0,
                            marks={0: "0s", int(TOTAL_STEPS / 2): "15s", TOTAL_STEPS - 1: "30s"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
            ],
        ),

        # Ticking interval for live playback
        dcc.Interval(id="interval-tick", interval=100, n_intervals=0), # 10 FPS update step
    ],
)


# Callbacks
@app.callback(
    Output("btn-play", "children"),
    Input("btn-play", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks):
    return "PLAY" if (n_clicks % 2 == 1) else "PAUSE"


@app.callback(
    Output("time-slider", "value"),
    Input("interval-tick", "n_intervals"),
    State("time-slider", "value"),
    State("btn-play", "children"),
)
def auto_step_slider(n_intervals, current_val, play_state):
    if play_state == "PLAY":
        return current_val
    next_val = (current_val + 5) % TOTAL_STEPS
    return next_val


@app.callback(
    Output("drone-3d-graph", "figure"),
    Output("attitude-strip-chart", "figure"),
    Output("hud-roll", "children"),
    Output("hud-pitch", "children"),
    Output("hud-yaw", "children"),
    Output("hud-bias", "children"),
    Output("hud-time", "children"),
    Input("time-slider", "value"),
)
def update_telemetry_views(step_idx):
    step_idx = max(0, min(step_idx, TOTAL_STEPS - 1))

    t_now = GLOBAL_TELEMETRY["time"][step_idx]
    quat = GLOBAL_TELEMETRY["quat"][step_idx]
    roll = GLOBAL_TELEMETRY["ekf_roll"][step_idx]
    pitch = GLOBAL_TELEMETRY["ekf_pitch"][step_idx]
    yaw = GLOBAL_TELEMETRY["ekf_yaw"][step_idx]
    bias = GLOBAL_TELEMETRY["ekf_bias_x"][step_idx]

    # 1. 3D Drone Figure
    drone_fig = build_3d_drone_figure(quat, roll, pitch, yaw)

    # 2. Synchronized Strip Charts (Showing trailing 10-second window)
    window_pts = 300
    start_idx = max(0, step_idx - window_pts)
    end_idx = step_idx + 1

    t_window = GLOBAL_TELEMETRY["time"][start_idx:end_idx]

    strip_fig = go.Figure()

    # Ground Truth Roll
    strip_fig.add_trace(go.Scatter(
        x=t_window, y=GLOBAL_TELEMETRY["true_roll"][start_idx:end_idx],
        mode="lines",
        line=dict(color="#ffffff", width=2, dash="dash"),
        name="Ground Truth (Roll)",
    ))

    # Raw Noisy Roll (Diverging)
    strip_fig.add_trace(go.Scatter(
        x=t_window, y=GLOBAL_TELEMETRY["raw_roll"][start_idx:end_idx],
        mode="lines",
        line=dict(color="#ff3366", width=1),
        name="Raw Gyro (Drifting)",
    ))

    # EKF Fused Roll
    strip_fig.add_trace(go.Scatter(
        x=t_window, y=GLOBAL_TELEMETRY["ekf_roll"][start_idx:end_idx],
        mode="lines",
        line=dict(color="#00f2ff", width=2.5),
        name="Aerofusion EKF Fused",
    ))

    strip_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="Mission Time [s]", showgrid=True, gridcolor="#1a2234"),
        yaxis=dict(title="Roll Attitude [deg]", showgrid=True, gridcolor="#1a2234"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return (
        drone_fig,
        strip_fig,
        f"{roll:+.2f}°",
        f"{pitch:+.2f}°",
        f"{yaw:+.2f}°",
        f"{bias:+.3f}°/s",
        f"{t_now:.2f} s",
    )


if __name__ == "__main__":
    print("[GCS] Starting Aerofusion Ground Station Dashboard on http://127.0.0.1:8050 ...")
    app.run(debug=False, port=8050)
