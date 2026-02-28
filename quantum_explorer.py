"""
Quantum Physics Explorer
========================
Interactive web app for exploring quantum computing concepts:
  - Bloch Sphere
  - Quantum Gates
  - Superposition & Interference
  - Measurement & Probability
  - Quantum Entanglement

Run with:  streamlit run quantum_explorer.py
Install:   pip install streamlit plotly numpy
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qutip import (sigmax, sigmay, sigmaz, basis, destroy,
                   mesolve, Qobj, wigner, coherent, fock,
                   thermal_dm, squeeze)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Physics Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0d0d1a; }
.concept-box {
    background: linear-gradient(135deg, #12122b 0%, #1a1a3e 100%);
    border-left: 4px solid #7b68ee;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 12px 0 20px 0;
    color: #d0d0f0;
    font-size: 0.97rem;
    line-height: 1.7;
}
.formula-box {
    background: #0b0b20;
    border: 1px solid #3a3a6e;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 10px 0;
    font-family: monospace;
    color: #a0d8ef;
    font-size: 1.05rem;
    text-align: center;
}
.state-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #7ec8e3;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: monospace;
    font-size: 1.1rem;
    margin: 2px;
}
.highlight { color: #f0c040; font-weight: bold; }
h1, h2, h3 { color: #c8b8ff; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# QUANTUM MATH HELPERS
# ═══════════════════════════════════════════════════════════════

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S  = np.array([[1, 0], [0, 1j]], dtype=complex)
T  = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

GATES = {"I (Identity)": I2, "X (NOT)": X, "Y": Y, "Z": Z,
         "H (Hadamard)": H, "S": S, "T": T}

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS  = (KET_0 + KET_1) / np.sqrt(2)
KET_MINUS = (KET_0 - KET_1) / np.sqrt(2)
KET_I     = (KET_0 + 1j * KET_1) / np.sqrt(2)
KET_MI    = (KET_0 - 1j * KET_1) / np.sqrt(2)

NAMED_STATES = {
    "|0⟩  (north pole)": KET_0,
    "|1⟩  (south pole)": KET_1,
    "|+⟩  (X+ equator)": KET_PLUS,
    "|-⟩  (X- equator)": KET_MINUS,
    "|i⟩  (Y+ equator)": KET_I,
    "|-i⟩ (Y- equator)": KET_MI,
}


def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """Qubit state from Bloch sphere angles."""
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def angles_from_state(psi: np.ndarray):
    """Extract Bloch sphere angles from a qubit state vector."""
    psi = psi / np.linalg.norm(psi)
    # Make global phase so that alpha is real+positive
    if abs(psi[0]) > 1e-9:
        phase = np.angle(psi[0])
        psi = psi * np.exp(-1j * phase)
    alpha, beta = psi[0], psi[1]
    theta = 2 * np.arccos(np.clip(abs(alpha), 0, 1))
    phi   = np.angle(beta) % (2 * np.pi)
    return float(theta), float(phi)


def bloch_xyz(theta: float, phi: float):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def rotation_gate(axis: str, angle: float) -> np.ndarray:
    """Rx, Ry, or Rz rotation gate."""
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    elif axis == "y":
        return np.array([[c, -s], [s, c]], dtype=complex)
    else:  # z
        return np.array([[np.exp(-1j * angle / 2), 0],
                         [0, np.exp(1j * angle / 2)]], dtype=complex)


# ═══════════════════════════════════════════════════════════════
# BLOCH SPHERE FIGURE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_bloch_sphere(vectors: list, labels: list = None,
                       colors: list = None, title: str = "Bloch Sphere") -> go.Figure:
    """
    Draw a Bloch sphere with one or more state vectors.
    vectors: list of (theta, phi) tuples
    """
    labels = labels or [f"ψ{i+1}" for i in range(len(vectors))]
    colors = colors or ["#ff4444", "#44aaff", "#44ff88", "#ffaa44"]

    fig = go.Figure()

    # ── Sphere surface ──────────────────────────────────────────
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(len(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.08,
        colorscale=[[0, "rgba(120,100,255,0.05)"],
                    [1, "rgba(180,160,255,0.15)"]],
        showscale=False, hoverinfo="skip",
    ))

    # ── Equator & meridians ─────────────────────────────────────
    t = np.linspace(0, 2 * np.pi, 200)
    for xe, ye, ze in [
        (np.cos(t), np.sin(t), np.zeros_like(t)),          # XY equator
        (np.cos(t), np.zeros_like(t), np.sin(t)),          # XZ meridian
        (np.zeros_like(t), np.cos(t), np.sin(t)),          # YZ meridian
    ]:
        fig.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze, mode="lines",
            line=dict(color="rgba(160,150,220,0.25)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # ── Axes ────────────────────────────────────────────────────
    ax_len = 1.35
    axis_cfg = [
        ("X", [0, ax_len], [0, 0], [0, 0]),
        ("Y", [0, 0], [0, ax_len], [0, 0]),
        ("Z", [0, 0], [0, 0], [0, ax_len]),
        ("-X", [0, -ax_len], [0, 0], [0, 0]),
        ("-Y", [0, 0], [0, -ax_len], [0, 0]),
        ("-Z", [0, 0], [0, 0], [0, -ax_len]),
    ]
    for _, ax, ay, az in axis_cfg:
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az, mode="lines",
            line=dict(color="rgba(200,200,255,0.4)", width=2),
            hoverinfo="skip", showlegend=False,
        ))

    # ── Special state labels ─────────────────────────────────────
    special = {
        "|0⟩": (0, 0, 1.18), "|1⟩": (0, 0, -1.18),
        "|+⟩": (1.18, 0, 0), "|-⟩": (-1.18, 0, 0),
        "|i⟩": (0, 1.18, 0), "|-i⟩": (0, -1.18, 0),
    }
    fig.add_trace(go.Scatter3d(
        x=[v[0] for v in special.values()],
        y=[v[1] for v in special.values()],
        z=[v[2] for v in special.values()],
        mode="text",
        text=list(special.keys()),
        textfont=dict(size=12, color="rgba(200,200,255,0.8)"),
        hoverinfo="skip", showlegend=False,
    ))

    # ── State vectors ────────────────────────────────────────────
    for i, (theta, phi) in enumerate(vectors):
        bx, by, bz = bloch_xyz(theta, phi)
        col = colors[i % len(colors)]

        # Dashed projection lines
        for px, py, pz in [(bx, by, 0), (bx, 0, bz), (0, by, bz)]:
            fig.add_trace(go.Scatter3d(
                x=[bx, px], y=[by, py], z=[bz, pz], mode="lines",
                line=dict(color=col, width=1, dash="dot"),
                hoverinfo="skip", showlegend=False, opacity=0.4,
            ))

        # Arrow shaft
        fig.add_trace(go.Scatter3d(
            x=[0, bx], y=[0, by], z=[0, bz],
            mode="lines+markers",
            line=dict(color=col, width=6),
            marker=dict(size=[0, 10], color=col,
                        symbol=["circle", "circle"]),
            name=labels[i],
        ))

    # ── Layout ───────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(color="#c8b8ff", size=16)),
        paper_bgcolor="rgba(10,10,26,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(10,10,30,0.95)",
            xaxis=dict(title=dict(text="X", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title=dict(text="Y", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title=dict(text="Z", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# PROBABILITY BAR CHART
# ═══════════════════════════════════════════════════════════════

def prob_chart(p0: float, p1: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["|0⟩", "|1⟩"],
        y=[p0, p1],
        marker_color=["#7b68ee", "#ff6b6b"],
        text=[f"{p0:.1%}", f"{p1:.1%}"],
        textposition="outside",
        textfont=dict(color="#fff", size=14),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,35,0.9)",
        yaxis=dict(range=[0, 1.15], tickformat=".0%",
                   gridcolor="rgba(100,100,150,0.2)", tickfont=dict(color="#aaa")),
        xaxis=dict(tickfont=dict(color="#fff", size=16)),
        margin=dict(l=20, r=20, t=20, b=20),
        height=200,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# SIMULATION: MEASUREMENT OUTCOMES
# ═══════════════════════════════════════════════════════════════

def simulate_measurements(p0: float, n: int) -> go.Figure:
    outcomes = np.random.choice([0, 1], size=n, p=[p0, 1 - p0])
    counts = [int(np.sum(outcomes == 0)), int(np.sum(outcomes == 1))]
    fig = go.Figure(go.Bar(
        x=["|0⟩ observed", "|1⟩ observed"],
        y=counts,
        marker_color=["#7b68ee", "#ff6b6b"],
        text=counts, textposition="outside",
        textfont=dict(color="#fff"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,35,0.9)",
        yaxis=dict(gridcolor="rgba(100,100,150,0.2)", tickfont=dict(color="#aaa")),
        xaxis=dict(tickfont=dict(color="#fff")),
        margin=dict(l=20, r=20, t=30, b=20),
        height=220,
        title=dict(text=f"{n} Simulated Measurements",
                   font=dict(color="#c8b8ff", size=13)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# BLOCH SPHERE WITH TRAJECTORY  (used by Rabi + Decoherence pages)
# ═══════════════════════════════════════════════════════════════

def bloch_with_traj(sx, sy, sz, label="ψ(t)",
                    color="#ff6b6b", end_color="#ffaa00",
                    title="Bloch Sphere Trajectory"):
    """
    Plotly Bloch sphere showing a full time-evolution trajectory.
    sx, sy, sz : 1-D arrays of ⟨X⟩, ⟨Y⟩, ⟨Z⟩ over time (from QuTiP mesolve).
    Blue dot = initial state, coloured dot = final state.
    """
    fig = go.Figure()

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 35)
    fig.add_trace(go.Surface(
        x=np.outer(np.cos(u), np.sin(v)),
        y=np.outer(np.sin(u), np.sin(v)),
        z=np.outer(np.ones(len(u)), np.cos(v)),
        opacity=0.08,
        colorscale=[[0, "rgba(120,100,255,0.05)"],
                    [1, "rgba(180,160,255,0.15)"]],
        showscale=False, hoverinfo="skip",
    ))

    t = np.linspace(0, 2 * np.pi, 200)
    for xe, ye, ze in [(np.cos(t), np.sin(t), np.zeros_like(t)),
                       (np.cos(t), np.zeros_like(t), np.sin(t)),
                       (np.zeros_like(t), np.cos(t), np.sin(t))]:
        fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
            line=dict(color="rgba(160,150,220,0.25)", width=1),
            hoverinfo="skip", showlegend=False))

    ax_len = 1.3
    for ax, ay, az in [([0,ax_len],[0,0],[0,0]), ([0,0],[0,ax_len],[0,0]),
                       ([0,0],[0,0],[0,ax_len]), ([0,-ax_len],[0,0],[0,0]),
                       ([0,0],[0,-ax_len],[0,0]), ([0,0],[0,0],[0,-ax_len])]:
        fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode="lines",
            line=dict(color="rgba(200,200,255,0.4)", width=2),
            hoverinfo="skip", showlegend=False))

    special = {"|0⟩":(0,0,1.18), "|1⟩":(0,0,-1.18),
               "|+⟩":(1.18,0,0), "|-⟩":(-1.18,0,0),
               "|i⟩":(0,1.18,0), "|-i⟩":(0,-1.18,0)}
    fig.add_trace(go.Scatter3d(
        x=[sv[0] for sv in special.values()],
        y=[sv[1] for sv in special.values()],
        z=[sv[2] for sv in special.values()],
        mode="text", text=list(special.keys()),
        textfont=dict(size=11, color="rgba(200,200,255,0.7)"),
        hoverinfo="skip", showlegend=False,
    ))

    # Colour trajectory by time (fade from dim to bright)
    n_pts = len(sx)
    for i in range(0, n_pts - 1, max(1, n_pts // 120)):
        alpha = 0.2 + 0.8 * i / n_pts
        fig.add_trace(go.Scatter3d(
            x=sx[i:i+2], y=sy[i:i+2], z=sz[i:i+2], mode="lines",
            line=dict(color=color, width=4),
            opacity=alpha, showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=[sx[0]], y=[sy[0]], z=[sz[0]], mode="markers",
        marker=dict(size=10, color="#44aaff"), name="Initial state",
    ))
    fig.add_trace(go.Scatter3d(
        x=[sx[-1]], y=[sy[-1]], z=[sz[-1]], mode="markers",
        marker=dict(size=12, color=end_color), name="Final state",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#c8b8ff", size=14)),
        paper_bgcolor="rgba(10,10,26,0)",
        scene=dict(
            bgcolor="rgba(10,10,30,0.95)",
            xaxis=dict(title=dict(text="X", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title=dict(text="Y", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title=dict(text="Z", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
        margin=dict(l=0, r=0, t=40, b=0), height=500,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════

st.sidebar.markdown("## ⚛️ Quantum Explorer")
st.sidebar.markdown("---")

PAGES = [
    "🏠  Introduction",
    "🔵  Bloch Sphere",
    "🔀  Quantum Gates",
    "〰️  Superposition & Interference",
    "📏  Measurement",
    "🔗  Entanglement",
    "🌡️  Laser Cooling",
    "⚡  Rydberg Atoms",
    "🔢  Two-Qubit Gates",
    "🌀  Rabi Oscillations",
    "📉  Decoherence",
    "🌊  Wigner Function",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small style='color:#666'>
Built with Streamlit + Plotly<br>
No physics background required!
</small>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL HELPERS  (Laser Cooling · Rydberg · Two-Qubit Gates)
# ═══════════════════════════════════════════════════════════════

kB_SI = 1.38064852e-23  # J/K

ATOMS = {
    "⁶Li  (Lithium-6)": {
        "m": 6.015e-3 / 6.022e23,
        "gamma": 2 * np.pi * 5.87e6,
        "lam": 671e-9,
        "T_D": 141e-6,
    },
    "¹³³Cs  (Cesium-133)": {
        "m": 132.905e-3 / 6.022e23,
        "gamma": 2 * np.pi * 5.23e6,
        "lam": 852e-9,
        "T_D": 125e-6,
    },
}


def mb_1d(v_arr, T, m):
    """1D Maxwell-Boltzmann speed distribution."""
    sigma = np.sqrt(kB_SI * T / m)
    return np.exp(-v_arr ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def doppler_force_norm(u_arr, delta_norm, s0=0.5):
    """
    Normalised Doppler cooling force (units: ℏkΓ/2).
    u = kv/Γ  |  delta_norm = δ/Γ  (negative = red-detuned = cooling)
    """
    F_p = s0 / (1 + s0 + 4 * (delta_norm - u_arr) ** 2)
    F_m = s0 / (1 + s0 + 4 * (delta_norm + u_arr) ** 2)
    return F_p - F_m


def build_bloch_xyz(vectors_xyz, labels=None, colors=None, title="Bloch Sphere"):
    """
    Bloch sphere figure where each vector is given as (rx, ry, rz).
    Vectors with |r| < 1 represent mixed states and sit inside the sphere.
    """
    labels = labels or [f"ψ{i+1}" for i in range(len(vectors_xyz))]
    colors = colors or ["#ff4444", "#44aaff", "#44ff88", "#ffaa44"]
    fig = go.Figure()

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 35)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(len(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs, opacity=0.08,
        colorscale=[[0, "rgba(120,100,255,0.05)"], [1, "rgba(180,160,255,0.15)"]],
        showscale=False, hoverinfo="skip",
    ))

    t = np.linspace(0, 2 * np.pi, 200)
    for xe, ye, ze in [(np.cos(t), np.sin(t), np.zeros_like(t)),
                       (np.cos(t), np.zeros_like(t), np.sin(t)),
                       (np.zeros_like(t), np.cos(t), np.sin(t))]:
        fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
            line=dict(color="rgba(160,150,220,0.25)", width=1),
            hoverinfo="skip", showlegend=False))

    ax_len = 1.3
    for ax, ay, az in [
        ([0, ax_len], [0, 0], [0, 0]), ([0, 0], [0, ax_len], [0, 0]),
        ([0, 0], [0, 0], [0, ax_len]), ([0, -ax_len], [0, 0], [0, 0]),
        ([0, 0], [0, -ax_len], [0, 0]), ([0, 0], [0, 0], [0, -ax_len]),
    ]:
        fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode="lines",
            line=dict(color="rgba(200,200,255,0.4)", width=2),
            hoverinfo="skip", showlegend=False))

    special = {"|0⟩": (0, 0, 1.18), "|1⟩": (0, 0, -1.18),
               "|+⟩": (1.18, 0, 0), "|-⟩": (-1.18, 0, 0),
               "|i⟩": (0, 1.18, 0), "|-i⟩": (0, -1.18, 0)}
    fig.add_trace(go.Scatter3d(
        x=[sv[0] for sv in special.values()],
        y=[sv[1] for sv in special.values()],
        z=[sv[2] for sv in special.values()],
        mode="text", text=list(special.keys()),
        textfont=dict(size=11, color="rgba(200,200,255,0.7)"),
        hoverinfo="skip", showlegend=False,
    ))

    for i, (rx, ry, rz) in enumerate(vectors_xyz):
        r = float(np.sqrt(rx ** 2 + ry ** 2 + rz ** 2))
        col = colors[i % len(colors)]
        for px, py, pz in [(rx, ry, 0), (rx, 0, rz), (0, ry, rz)]:
            fig.add_trace(go.Scatter3d(x=[rx, px], y=[ry, py], z=[rz, pz], mode="lines",
                line=dict(color=col, width=1, dash="dot"),
                hoverinfo="skip", showlegend=False, opacity=0.4))
        fig.add_trace(go.Scatter3d(
            x=[0, rx], y=[0, ry], z=[0, rz],
            mode="lines+markers", line=dict(color=col, width=6),
            marker=dict(size=[0, 8], color=col),
            name=f"{labels[i]}  (|r|={r:.2f})",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#c8b8ff", size=15)),
        paper_bgcolor="rgba(10,10,26,0)",
        scene=dict(
            bgcolor="rgba(10,10,30,0.95)",
            xaxis=dict(title=dict(text="X", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title=dict(text="Y", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title=dict(text="Z", font=dict(color="#aaa")),
                       showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )
    return fig


def reduced_bloch_vec(psi4, qubit):
    """Bloch vector (rx, ry, rz) of qubit A (0) or B (1) from a 2-qubit pure state."""
    C = psi4.reshape(2, 2)
    rho = C @ C.conj().T if qubit == 0 else C.conj().T @ C
    return (
        float(np.real(np.trace(rho @ X))),
        float(np.real(np.trace(rho @ Y))),
        float(np.real(np.trace(rho @ Z))),
    )


# Two-qubit gate matrices (basis: |00⟩, |01⟩, |10⟩, |11⟩)
CNOT  = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
CZ    = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
ISWAP = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
TWO_Q_GATES = {"CNOT": CNOT, "CZ": CZ, "iSWAP": ISWAP}


# ═══════════════════════════════════════════════════════════════
# PAGE: INTRODUCTION
# ═══════════════════════════════════════════════════════════════

if page == PAGES[0]:
    st.title("⚛️ Quantum Physics Explorer")
    st.markdown("#### An interactive guide to quantum computing fundamentals")

    st.markdown("""
<div class='concept-box'>
<b>Welcome!</b> This app lets you <em>see</em> and <em>feel</em> the core ideas behind quantum computing —
no maths degree required. Each section gives you a brief explanation and then hands you the
controls so you can explore the concept yourself in real time.
<br><br>
Quantum computers harness strange quantum phenomena — <b>superposition</b>, <b>entanglement</b>, and
<b>interference</b> — to solve certain problems exponentially faster than classical computers.
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🔵 Bloch Sphere**
The geometric picture of a single quantum bit (qubit).
Drag the angles and watch the state vector move in 3-D space.
""")
    with col2:
        st.markdown("""
**🔀 Quantum Gates**
Quantum gates are rotations on the Bloch sphere.
Apply real gate matrices and see exactly what they do.
""")
    with col3:
        st.markdown("""
**〰️ Superposition**
A qubit can be 0 *and* 1 at the same time.
Control the amplitudes and see how probabilities arise.
""")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
**📏 Measurement**
Looking at a qubit forces it to choose 0 or 1.
Simulate thousands of measurements and see the Born rule in action.
""")
    with col5:
        st.markdown("""
**🔗 Entanglement**
Two qubits can share a quantum connection across any distance.
Explore all four Bell states and their perfect correlations.
""")
    with col6:
        st.markdown("""
**Getting started**
Use the sidebar to jump between topics.
Every page has sliders and knobs — just play!
""")

    st.markdown("---")
    st.markdown("#### Key Vocabulary")

    terms = {
        "Qubit": "The quantum analogue of a classical bit. Unlike a bit (0 or 1), a qubit can exist in a *superposition* of both.",
        "State vector |ψ⟩": "A complex-valued vector that fully describes the quantum state of a system.",
        "Amplitude": "A complex number whose squared magnitude gives the probability of a measurement outcome.",
        "Unitary gate": "A reversible quantum operation — mathematically a unitary matrix U where U†U = I.",
        "Measurement": "The act of observing a qubit, which collapses its state to |0⟩ or |1⟩ with probabilities given by the Born rule.",
        "Entanglement": "A quantum correlation between two or more qubits that cannot be explained classically.",
    }
    for term, defn in terms.items():
        with st.expander(f"📖  {term}"):
            st.markdown(defn)

    st.markdown("---")
    with st.expander("📚 Key literature behind this app"):
        st.markdown("""
**Quantum Computing Foundations**
- Nielsen, M. A. & Chuang, I. L. (2000). *Quantum Computation and Quantum Information.* Cambridge University Press. — The standard textbook.
- Preskill, J. (2018). *Quantum Computing in the NISQ Era and Beyond.* Quantum 2, 79.

**Laser Cooling & Optical Tweezers**
- Chu, S. et al. (1985). *Three-dimensional viscous confinement and cooling of atoms by resonance radiation pressure.* PRL 55, 48. — Nobel-prize work.
- Cohen-Tannoudji, C. N. (1997). *Manipulating atoms with photons.* Nobel Lecture, Rev. Mod. Phys. 70, 707.
- Kaufman, A. M. & Ni, K.-K. (2021). *Quantum science with optical tweezer arrays of ultracold atoms and molecules.* Nature Physics 17, 1324.

**Rydberg Quantum Gates**
- Jaksch, D. et al. (2000). *Fast quantum gates for neutral atoms.* PRL 85, 2208. — Proposed the Rydberg blockade gate.
- Saffman, M., Walker, T. G. & Mølmer, K. (2010). *Quantum information with Rydberg atoms.* Rev. Mod. Phys. 82, 2313.
- Levine, H. et al. (2019). *Parallel implementation of high-fidelity multiqubit gates with neutral atoms.* PRL 123, 170503.

**Ultracold Molecules (Li-Cs)**
- Liu, L. R. et al. (2019). *Building one molecule from a reservoir of two atoms.* Science 360, 900.
- Burchesky, S. et al. (2021). *Rotational coherence times of polar molecules in optical tweezers.* PRL 127, 123202.

**Decoherence & Open Systems**
- Bloch, F. (1946). *Nuclear Induction.* Physical Review 70, 460. — Origin of T₁ & T₂.
- Breuer, H.-P. & Petruccione, F. (2002). *The Theory of Open Quantum Systems.* Oxford University Press.

**Wigner Function & Motional States**
- Wigner, E. P. (1932). *On the Quantum Correction For Thermodynamic Equilibrium.* Physical Review 40, 749.
- Leibfried, D. et al. (1996). *Experimental Determination of the Motional Quantum State of a Trapped Atom.* PRL 77, 4281.
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: BLOCH SPHERE
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    st.title("🔵 The Bloch Sphere")

    st.markdown("""
<div class='concept-box'>
<b>What is the Bloch sphere?</b><br><br>
A qubit's state can always be written as:<br><br>
<span class='formula-box'>|ψ⟩ = cos(θ/2)|0⟩ + e<sup>iφ</sup> sin(θ/2)|1⟩</span><br>
Two real numbers — a <b>polar angle θ</b> (0 → π) and an <b>azimuthal angle φ</b> (0 → 2π) —
completely specify any pure single-qubit state.
This means every qubit state maps to a unique point on the surface of a unit sphere,
called the <b>Bloch sphere</b>.<br><br>
• <b>North pole</b> = |0⟩ &nbsp;&nbsp; <b>South pole</b> = |1⟩<br>
• <b>Equator</b> = equal superpositions (e.g. |+⟩, |-⟩, |i⟩, |-i⟩)<br>
• Any rotation on the sphere corresponds to a quantum gate.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🎛️ Try It — Move the State Vector")
    col_ctrl, col_sphere = st.columns([1, 2])

    with col_ctrl:
        st.markdown("**Quick presets:**")
        preset = st.selectbox("Start from named state", list(NAMED_STATES.keys()))
        psi_preset = NAMED_STATES[preset]
        init_theta, init_phi = angles_from_state(psi_preset)

        st.markdown("---")
        st.markdown("**Or set angles manually:**")
        theta = st.slider("θ  (polar — 0=north, π=south)",
                          0.0, float(np.pi), float(init_theta),
                          step=0.01, format="%.2f rad")
        phi = st.slider("φ  (azimuthal — around equator)",
                        0.0, 2 * float(np.pi), float(init_phi),
                        step=0.01, format="%.2f rad")

        psi = state_from_angles(theta, phi)
        p0 = float(abs(psi[0]) ** 2)
        p1 = float(abs(psi[1]) ** 2)

        st.markdown("---")
        st.markdown("**Current state:**")
        a_re, a_im = psi[0].real, psi[0].imag
        b_re, b_im = psi[1].real, psi[1].imag
        a_str = f"{a_re:+.3f}" if abs(a_im) < 1e-6 else f"({a_re:+.3f}{a_im:+.3f}i)"
        b_str = f"{b_re:+.3f}" if abs(b_im) < 1e-6 else f"({b_re:+.3f}{b_im:+.3f}i)"
        st.markdown(f"<div class='formula-box'>|ψ⟩ = {a_str}|0⟩ {b_str[0]} {b_str[1:]}|1⟩</div>",
                    unsafe_allow_html=True)

        st.markdown("**Bloch coordinates:**")
        bx, by, bz = bloch_xyz(theta, phi)
        st.markdown(f"x = {bx:+.3f} &nbsp; y = {by:+.3f} &nbsp; z = {bz:+.3f}",
                    unsafe_allow_html=True)

        st.markdown("**Measurement probabilities:**")
        st.plotly_chart(prob_chart(p0, p1), use_container_width=True, key="bs_prob")

    with col_sphere:
        fig = build_bloch_sphere([(theta, phi)], labels=["ψ"])
        st.plotly_chart(fig, use_container_width=True, key="bloch_main")

    with st.expander("📐 The maths in detail"):
        st.markdown(r"""
The Bloch sphere coordinates are connected to measurement probabilities:

| Quantity | Formula | Meaning |
|---|---|---|
| P(0) | cos²(θ/2) | Probability of measuring |0⟩ |
| P(1) | sin²(θ/2) | Probability of measuring |1⟩ |
| ⟨Z⟩ | cos θ | Expected value of Z measurement |
| ⟨X⟩ | sin θ cos φ | Expected value of X measurement |
| ⟨Y⟩ | sin θ sin φ | Expected value of Y measurement |

The **global phase** e^(iγ) is unobservable — only the **relative phase** φ between
|0⟩ and |1⟩ components matters physically.
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: QUANTUM GATES
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[2]:
    st.title("🔀 Quantum Gates")

    st.markdown("""
<div class='concept-box'>
<b>What is a quantum gate?</b><br><br>
Classical logic gates (AND, OR, NOT) transform bits.
Quantum gates transform qubits.  Every quantum gate is a <b>unitary matrix</b> — it's
reversible and preserves the total probability.  On the Bloch sphere, every single-qubit
gate is simply a <b>rotation</b>.<br><br>
Experiment below: pick a starting state, pick a gate (or set a rotation angle), and watch
the state vector rotate on the sphere.
</div>
""", unsafe_allow_html=True)

    col_in, col_mid, col_out = st.columns([1, 1.4, 1])

    with col_in:
        st.markdown("#### 1. Input State")
        in_preset = st.selectbox("Named state", list(NAMED_STATES.keys()), key="gate_in")
        psi_in = NAMED_STATES[in_preset].copy()
        t_in, p_in = angles_from_state(psi_in)

        st.markdown("**or custom angles:**")
        t_in = st.slider("θ in", 0.0, float(np.pi), float(t_in), 0.01, format="%.2f", key="gate_tin")
        p_in = st.slider("φ in", 0.0, 2*float(np.pi), float(p_in), 0.01, format="%.2f", key="gate_pin")
        psi_in = state_from_angles(t_in, p_in)

    with col_mid:
        st.markdown("#### 2. Choose Gate")

        gate_tab, rot_tab = st.tabs(["Standard Gates", "Rotation Gates Rx/Ry/Rz"])

        with gate_tab:
            gate_name = st.radio("Gate", list(GATES.keys()), horizontal=False)
            U = GATES[gate_name]

            descriptions = {
                "I (Identity)": "Does nothing. State is unchanged.",
                "X (NOT)": "Bit-flip gate. Rotates 180° around X axis. Maps |0⟩↔|1⟩.",
                "Y": "Rotates 180° around Y axis. Adds an imaginary phase when flipping.",
                "Z": "Phase-flip gate. 180° around Z axis. Leaves |0⟩ unchanged, flips sign of |1⟩.",
                "H (Hadamard)": "Creates superposition. Maps |0⟩→|+⟩ and |1⟩→|-⟩. 180° around X+Z diagonal.",
                "S": "Phase gate: adds 90° phase to |1⟩. Quarter-turn around Z axis.",
                "T": "T-gate: adds 45° phase to |1⟩. Eighth-turn around Z axis.",
            }
            st.info(descriptions[gate_name])

            st.markdown("**Matrix:**")
            m = U
            rows = []
            for r in range(2):
                row = []
                for c in range(2):
                    v = m[r, c]
                    if abs(v.imag) < 1e-9:
                        row.append(f"{v.real:+.4f}")
                    elif abs(v.real) < 1e-9:
                        row.append(f"{v.imag:+.4f}i")
                    else:
                        row.append(f"{v.real:+.4f}{v.imag:+.4f}i")
                rows.append(row)
            st.table({"row 0": rows[0], "row 1": rows[1]})

        with rot_tab:
            ax = st.radio("Rotation axis", ["x", "y", "z"], horizontal=True)
            ang = st.slider("Rotation angle (radians)",
                            -float(np.pi), float(np.pi), float(np.pi / 2), 0.01,
                            format="%.2f rad", key="rot_ang")
            U = rotation_gate(ax, ang)
            st.markdown(f"Rotation by **{ang:.2f} rad** around **{ax.upper()}** axis")

        psi_out = U @ psi_in
        t_out, p_out = angles_from_state(psi_out)

    with col_out:
        st.markdown("#### 3. Output State")
        p0_in  = float(abs(psi_in[0])  ** 2)
        p0_out = float(abs(psi_out[0]) ** 2)

        st.markdown("**Input probabilities:**")
        st.plotly_chart(prob_chart(p0_in, 1 - p0_in), use_container_width=True, key="g_p_in")
        st.markdown("**Output probabilities:**")
        st.plotly_chart(prob_chart(p0_out, 1 - p0_out), use_container_width=True, key="g_p_out")

    st.markdown("### Before → After on the Bloch Sphere")
    fig2 = build_bloch_sphere(
        [(t_in, p_in), (t_out, p_out)],
        labels=["Input |ψ_in⟩", "Output |ψ_out⟩"],
        colors=["#44aaff", "#ff4444"],
        title="Gate applied",
    )
    st.plotly_chart(fig2, use_container_width=True, key="gate_sphere")

    with st.expander("🔢 Gate Reference Table"):
        st.markdown("""
| Gate | Symbol | Effect on Bloch Sphere | Key action |
|---|---|---|---|
| Identity | I | No rotation | State unchanged |
| Pauli-X | X | 180° around X | Bit flip: |0⟩↔|1⟩ |
| Pauli-Y | Y | 180° around Y | Bit + phase flip |
| Pauli-Z | Z | 180° around Z | Phase flip on |1⟩ |
| Hadamard | H | 180° around X+Z | |0⟩→|+⟩, creates superposition |
| S | S | 90° around Z | Quarter-phase on |1⟩ |
| T | T | 45° around Z | Eighth-phase on |1⟩ |
| Rx(θ) | Rx | θ rotation around X | Generalised X |
| Ry(θ) | Ry | θ rotation around Y | Generalised Y |
| Rz(θ) | Rz | θ rotation around Z | Generalised phase |
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: SUPERPOSITION & INTERFERENCE
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[3]:
    st.title("〰️ Superposition & Interference")

    st.markdown("""
<div class='concept-box'>
<b>Superposition</b> means a qubit doesn't have to be definitely 0 or definitely 1 —
it can be in a weighted combination of both simultaneously:<br><br>
<span class='formula-box'>|ψ⟩ = α|0⟩ + β|1⟩</span><br>
where α and β are complex numbers with |α|² + |β|² = 1.<br><br>
<b>Interference</b> is when paths through a quantum circuit combine constructively (amplitudes
add up) or destructively (amplitudes cancel), just like waves in water.  This is the key
mechanism that makes quantum algorithms powerful.
</div>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎛️ Amplitude Explorer", "〰️ Interference Demo"])

    with tab1:
        st.markdown("### Control the amplitudes α and β directly")
        col_s, col_v = st.columns([1, 2])
        with col_s:
            st.markdown("#### Set amplitudes")
            theta_s = st.slider("θ  (controls |α|² and |β|²)",
                                0.0, float(np.pi), float(np.pi / 4), 0.01,
                                key="sup_theta")
            phi_s = st.slider("φ  (relative phase between |0⟩ and |1⟩)",
                              0.0, 2*float(np.pi), 0.0, 0.01, key="sup_phi")

            psi_s = state_from_angles(theta_s, phi_s)
            alpha, beta = psi_s
            p0s = float(abs(alpha)**2)
            p1s = float(abs(beta)**2)

            st.markdown("---")
            st.markdown(f"**|α|² = P(0) = {p0s:.3f}**")
            st.markdown(f"**|β|² = P(1) = {p1s:.3f}**")
            st.markdown(f"Phase φ = {phi_s:.2f} rad = {np.degrees(phi_s):.0f}°")
            st.progress(p0s)

            st.markdown("**Probabilities:**")
            st.plotly_chart(prob_chart(p0s, p1s), use_container_width=True, key="sup_prob")

        with col_v:
            fig_s = build_bloch_sphere([(theta_s, phi_s)],
                                       labels=["ψ"],
                                       title="Superposition state on Bloch Sphere")
            st.plotly_chart(fig_s, use_container_width=True, key="sup_sphere")

    with tab2:
        st.markdown("### Mach-Zehnder style interference")
        st.markdown("""
<div class='concept-box'>
Classic experiment: send |0⟩ through two Hadamard gates.
H then H brings it back to |0⟩ — <b>destructive interference</b> kills the |1⟩ component.
<br>Inject a Z (phase-flip) between the two H gates to see <b>constructive interference</b> into |1⟩.
</div>
""", unsafe_allow_html=True)

        mid_gate_name = st.radio(
            "Middle gate (between the two Hadamards):",
            ["I (Identity) — no phase", "Z — phase flip", "S — 90° phase", "T — 45° phase"],
            horizontal=True,
        )
        mid_map = {
            "I (Identity) — no phase": I2,
            "Z — phase flip": Z,
            "S — 90° phase": S,
            "T — 45° phase": T,
        }
        G_mid = mid_map[mid_gate_name]

        psi_0 = KET_0.copy()
        psi_1 = H @ psi_0
        psi_2 = G_mid @ psi_1
        psi_3 = H @ psi_2

        stages = [
            ("Start", psi_0),
            ("After H", psi_1),
            (f"After {mid_gate_name.split('—')[0].strip()}", psi_2),
            ("After H", psi_3),
        ]

        cols = st.columns(len(stages))
        for ci, (label, psi_stage) in enumerate(stages):
            t_st, p_st = angles_from_state(psi_stage)
            p0_st = float(abs(psi_stage[0])**2)
            with cols[ci]:
                st.markdown(f"**Step {ci+1}: {label}**")
                mini_fig = build_bloch_sphere(
                    [(t_st, p_st)], labels=["ψ"], title=label,
                )
                mini_fig.update_layout(height=280, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(mini_fig, use_container_width=True, key=f"interf_{ci}")
                st.plotly_chart(prob_chart(p0_st, 1 - p0_st),
                                use_container_width=True, key=f"interf_p_{ci}")


# ═══════════════════════════════════════════════════════════════
# PAGE: MEASUREMENT
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[4]:
    st.title("📏 Measurement")

    st.markdown("""
<div class='concept-box'>
<b>The measurement problem:</b> when you observe a qubit, you always get a definite answer —
either 0 or 1.  You <em>never</em> see the superposition directly.  The qubit's state
collapses instantly to the outcome you got.<br><br>
The <b>Born rule</b> tells us the probability of each outcome:<br>
<span class='formula-box'>P(0) = |α|² = cos²(θ/2) &nbsp;&nbsp; P(1) = |β|² = sin²(θ/2)</span><br>
The only way to learn these probabilities is to prepare the same state <em>many times</em>
and measure each copy.  Run the simulation below to see this in action.
</div>
""", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 2])

    with col_m1:
        st.markdown("### Set the qubit state")
        m_preset = st.selectbox("Start from", list(NAMED_STATES.keys()), key="meas_pre")
        psi_m = NAMED_STATES[m_preset].copy()
        tm, pm = angles_from_state(psi_m)

        tm = st.slider("θ", 0.0, float(np.pi), float(tm), 0.01, key="meas_t")
        pm = st.slider("φ", 0.0, 2*float(np.pi), float(pm), 0.01, key="meas_p")
        psi_m = state_from_angles(tm, pm)
        p0m = float(abs(psi_m[0])**2)

        st.markdown(f"**P(0) = {p0m:.4f}  =  {p0m:.1%}**")
        st.markdown(f"**P(1) = {1-p0m:.4f}  =  {1-p0m:.1%}**")
        st.plotly_chart(prob_chart(p0m, 1 - p0m), use_container_width=True, key="meas_prob")

        st.markdown("### Simulate measurements")
        n_shots = st.select_slider("Number of measurements",
                                   options=[10, 50, 100, 500, 1000, 5000, 10000], value=100)
        if st.button("▶  Run simulation"):
            st.session_state["meas_fig"] = simulate_measurements(p0m, n_shots)

    with col_m2:
        fig_m = build_bloch_sphere([(tm, pm)], title="State being measured")
        st.plotly_chart(fig_m, use_container_width=True, key="meas_sphere")

        if "meas_fig" in st.session_state:
            st.plotly_chart(st.session_state["meas_fig"], use_container_width=True, key="meas_sim")
            st.markdown("""
<small style='color:#888'>
Each run is independent — with enough repetitions the frequencies converge to the Born rule probabilities.
</small>""", unsafe_allow_html=True)

    with st.expander("📖 What is 'collapse'?"):
        st.markdown("""
After measuring |ψ⟩ = α|0⟩ + β|1⟩:

- If the outcome is **0**, the new state is **|0⟩** (north pole)
- If the outcome is **1**, the new state is **|1⟩** (south pole)

The superposition is gone — you've extracted one bit of classical information.
This is why you can't directly read out α or β.  State tomography (measuring many copies
in X, Y, and Z bases) is needed to reconstruct the full state.

**Bases:** The standard measurement is in the Z basis {|0⟩, |1⟩}.
You can also measure in the X basis {|+⟩, |-⟩} or Y basis {|i⟩, |-i⟩}
by first applying the right rotation gate.
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: ENTANGLEMENT
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[5]:
    st.title("🔗 Quantum Entanglement")

    st.markdown("""
<div class='concept-box'>
<b>Entanglement</b> is a quantum correlation that has no classical equivalent.
Two qubits are entangled when their joint state <em>cannot</em> be written as a product
of two individual qubit states.<br><br>
The four <b>Bell states</b> are the maximally-entangled two-qubit states.
Once prepared, measuring qubit A instantly determines the outcome of measuring qubit B —
even if they are light-years apart.  (No information is transmitted; the correlation was
built in at creation time.)
</div>
""", unsafe_allow_html=True)

    bell_states = {
        "Φ⁺  (|00⟩ + |11⟩)/√2": {
            "ket": "|Φ⁺⟩ = (|00⟩ + |11⟩) / √2",
            "probs": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
            "description": "Measure qubit A: get 0 → B is 0. Get 1 → B is 1. Always same.",
            "circuit": "Start |00⟩ → H on A → CNOT(A,B)",
        },
        "Φ⁻  (|00⟩ − |11⟩)/√2": {
            "ket": "|Φ⁻⟩ = (|00⟩ − |11⟩) / √2",
            "probs": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
            "description": "Same correlations as Φ⁺ but with a relative phase of −1.",
            "circuit": "Start |00⟩ → H on A → CNOT(A,B) → Z on A",
        },
        "Ψ⁺  (|01⟩ + |10⟩)/√2": {
            "ket": "|Ψ⁺⟩ = (|01⟩ + |10⟩) / √2",
            "probs": {"00": 0.0, "01": 0.5, "10": 0.5, "11": 0.0},
            "description": "Measure A: get 0 → B is 1. Get 1 → B is 0. Always opposite.",
            "circuit": "Start |00⟩ → H on A → CNOT(A,B) → X on B",
        },
        "Ψ⁻  (|01⟩ − |10⟩)/√2": {
            "ket": "|Ψ⁻⟩ = (|01⟩ − |10⟩) / √2",
            "probs": {"00": 0.0, "01": 0.5, "10": 0.5, "11": 0.0},
            "description": "Anti-correlated with extra phase. The 'singlet' state.",
            "circuit": "Start |00⟩ → H on A → CNOT(A,B) → X on B → Z on B",
        },
    }

    selected_bell = st.radio("Select a Bell state:", list(bell_states.keys()), horizontal=True)
    info = bell_states[selected_bell]

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.markdown(f"### {info['ket']}")
        st.markdown(f"**Circuit:** {info['circuit']}")
        st.markdown(f"**Correlation:** {info['description']}")

        st.markdown("#### Measurement outcome probabilities")
        outcomes = list(info["probs"].keys())
        probs    = list(info["probs"].values())
        # Bug fix 1: suppress "0%" labels on zero-height bars to avoid axis collisions
        fig_bell = go.Figure(go.Bar(
            x=outcomes, y=probs,
            marker_color=["#7b68ee", "#ff6b6b", "#44ccaa", "#ffaa44"],
            text=[f"{p:.0%}" if p > 0 else "" for p in probs],
            textposition="outside",
            textfont=dict(color="#fff", size=13),
        ))
        fig_bell.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.9)",
            yaxis=dict(range=[0, 0.7], tickformat=".0%",
                       gridcolor="rgba(100,100,150,0.2)", tickfont=dict(color="#aaa")),
            xaxis=dict(title="Two-qubit outcome |AB⟩",
                       tickfont=dict(color="#fff", size=14)),
            margin=dict(l=20, r=20, t=20, b=40),
            height=250,
        )
        st.plotly_chart(fig_bell, use_container_width=True, key="bell_bar")

        st.markdown("#### Simulate entangled measurements")
        n_bell = st.select_slider("Shots", [50, 100, 500, 1000, 5000], 200, key="bell_n")

        # Bug fix 2: store simulation in session_state so it survives reruns
        sim_key = f"bell_sim_{selected_bell}"
        if st.button("▶  Run Bell experiment"):
            non_zero = {k: v for k, v in info["probs"].items() if v > 0}
            sim_choices = list(non_zero.keys())
            pvec = np.array(list(non_zero.values()))
            pvec = pvec / pvec.sum()
            samp = np.random.choice(sim_choices, size=n_bell, p=pvec)
            counts = {k: int(np.sum(samp == k)) for k in outcomes}
            fig_sim = go.Figure(go.Bar(
                x=list(counts.keys()), y=list(counts.values()),
                marker_color=["#7b68ee", "#ff6b6b", "#44ccaa", "#ffaa44"],
                text=[str(v) if v > 0 else "" for v in counts.values()],
                textposition="outside",
                textfont=dict(color="#fff"),
            ))
            fig_sim.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                yaxis=dict(gridcolor="rgba(100,100,150,0.2)", tickfont=dict(color="#aaa")),
                xaxis=dict(tickfont=dict(color="#fff")),
                margin=dict(l=20, r=20, t=30, b=40),
                height=230,
                title=dict(text=f"{n_bell} shots — {selected_bell.split('(')[0].strip()}",
                           font=dict(color="#c8b8ff", size=13)),
            )
            st.session_state[sim_key] = fig_sim

        if sim_key in st.session_state:
            st.plotly_chart(st.session_state[sim_key], use_container_width=True, key="bell_sim")

    with col_e2:
        st.markdown("### Single-qubit views after entanglement")
        st.markdown("""
<div class='concept-box'>
When two qubits are maximally entangled, looking at <em>either one alone</em>
shows a perfectly mixed state — sitting exactly at the <b>centre</b> of the Bloch sphere
(not on the surface).  Only the joint two-qubit state contains the full information.
</div>
""", unsafe_allow_html=True)
        # Reduced state of each qubit = maximally mixed → centre of Bloch sphere
        # Represent with θ=π/2, but show as faded centre dot
        st.markdown("""
The individual qubit A or B, when entangled, has **no definite direction** on the Bloch
sphere — its reduced density matrix is I/2, the centre point.

The entanglement lives in the *correlations* between A and B, not in either one alone.
""")

        with st.expander("📖 Why can't you use entanglement for FTL communication?"):
            st.markdown("""
Measuring qubit A collapses qubit B's state — but you **cannot choose** what outcome
you get when you measure A.  The outcome is random (50/50 for Bell states).

For Alice to signal Bob, she would need to control which result she gets — but quantum
mechanics forbids that.  Bob's measurement statistics look perfectly random whether or
not Alice has measured her qubit.  Information transfer requires a classical channel
to compare results, which is limited to the speed of light.
""")

        with st.expander("📖 Real-world applications of entanglement"):
            st.markdown("""
- **Quantum cryptography (QKD):** Entanglement enables provably secure key distribution
- **Quantum teleportation:** Transfer a quantum state using entanglement + classical bits
- **Superdense coding:** Send 2 classical bits using 1 qubit + 1 ebit
- **Quantum error correction:** Entangle logical qubits to protect against noise
- **Bell test experiments:** Proved quantum mechanics is non-local (Nobel Prize 2022)
""")

# ═══════════════════════════════════════════════════════════════
# PAGE: LASER COOLING & OPTICAL TWEEZERS
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[6]:
    st.title("🌡️ Laser Cooling & Optical Tweezers")

    st.markdown("""
<div class='concept-box'>
<b>How do you stop an atom?</b> — with light.<br><br>
When an atom moving <em>toward</em> a laser absorbs a photon, momentum conservation gives it
a tiny kick <em>backward</em>.  With two counter-propagating beams tuned slightly
<b>below</b> the atomic resonance (red-detuned by δ), the Doppler shift makes moving atoms
preferentially absorb from the beam they are heading into — always being pushed back.
Repeat millions of times per second: atoms slow from ~600 m/s to centimetres per second.<br><br>
A tightly focused Gaussian laser beam creates a <b>dipole trap</b> — the intensity gradient
pulls atoms toward the focal point.  This is the <b>optical tweezer</b> used in the
<a href="https://hoodlab.physics.purdue.edu" style="color:#7b68ee">Hood Lab at Purdue</a>
to trap individual <sup>6</sup>Li and <sup>133</sup>Cs atoms — the exact atoms you can
explore below.
</div>
""", unsafe_allow_html=True)

    tab_cool, tab_tweezer = st.tabs(["🌡️ Doppler Cooling", "🔬 Optical Tweezer Trap"])

    # ── Doppler Cooling tab ──────────────────────────────────────
    with tab_cool:
        col_cc, col_cp = st.columns([1, 2])
        with col_cc:
            atom_name = st.selectbox("Atom", list(ATOMS.keys()))
            atom = ATOMS[atom_name]
            m, gamma, lam, T_D = atom["m"], atom["gamma"], atom["lam"], atom["T_D"]
            k_atom = 2 * np.pi / lam
            v_unit = gamma / k_atom          # velocity unit = Γ/k

            st.markdown(f"""
**{atom_name}**
- Cooling wavelength: **{lam*1e9:.0f} nm**
- Natural linewidth Γ/2π: **{gamma/(2*np.pi*1e6):.2f} MHz**
- Doppler temperature limit: **{T_D*1e6:.0f} μK**
- RMS speed at T_D: **{np.sqrt(kB_SI*T_D/m)*100:.1f} cm/s**
""")
            st.markdown("---")
            log_T = st.slider(
                "Temperature (powers of 10)",
                -6.0, 2.5, 2.3, step=0.05,
                help="Slide left to cool the atom cloud",
                format="10^%.2f K",
            )
            T = 10 ** log_T
            st.markdown(
                f"**T = {T:.3g} K** = {T*1e6:.2f} μK" if T < 1e-3
                else f"**T = {T:.3f} K**"
            )

            delta_norm = st.slider(
                "Laser detuning δ/Γ", -5.0, -0.1, -0.5, step=0.05,
                help="Optimal cooling near δ = −Γ/2",
            )
            s0 = st.slider(
                "Saturation s₀", 0.05, 2.0, 0.5, step=0.05,
                help="Beam intensity / saturation intensity",
            )

        with col_cp:
            # Maxwell-Boltzmann distribution
            v_max = min(5 * np.sqrt(kB_SI * 300 / m), 3000.0)
            v_arr = np.linspace(-v_max, v_max, 800)
            f_room = mb_1d(v_arr, 300.0, m)
            f_cool = mb_1d(v_arr, T, m)
            peak   = float(f_room.max())

            fig_mb = go.Figure()
            fig_mb.add_trace(go.Scatter(
                x=v_arr, y=f_room / peak,
                name="T = 300 K (room temp)",
                line=dict(color="#ff6b6b", width=2),
                fill="tozeroy", fillcolor="rgba(255,107,107,0.1)",
            ))
            fig_mb.add_trace(go.Scatter(
                x=v_arr, y=f_cool / peak,
                name=f"T = {T:.3g} K",
                line=dict(color="#44aaff", width=2.5),
                fill="tozeroy", fillcolor="rgba(68,170,255,0.15)",
            ))
            fig_mb.update_layout(
                title=dict(text="Velocity distribution (narrowing = cooling)",
                           font=dict(color="#c8b8ff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                xaxis=dict(title="Velocity (m/s)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                yaxis=dict(title="Normalised probability", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
                margin=dict(l=20, r=20, t=40, b=40), height=250,
            )
            st.plotly_chart(fig_mb, use_container_width=True, key="mb_dist")

            # Doppler force vs velocity
            u_arr  = np.linspace(-12, 12, 600)
            F_norm = doppler_force_norm(u_arr, delta_norm, s0)
            v_cap  = abs(delta_norm) * v_unit

            fig_force = go.Figure()
            fig_force.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", dash="dot"))
            fig_force.add_vline(x=0, line=dict(color="rgba(255,255,255,0.2)", dash="dot"))
            fig_force.add_trace(go.Scatter(
                x=u_arr * v_unit, y=F_norm,
                line=dict(color="#7b68ee", width=2.5),
                fill="tozeroy", fillcolor="rgba(123,104,238,0.1)",
                name="Cooling force",
            ))
            fig_force.add_vrect(
                x0=-v_cap, x1=v_cap,
                fillcolor="rgba(68,255,136,0.07)", line_width=0,
                annotation_text="Capture range", annotation_position="top",
                annotation_font_color="#44ff88",
            )
            fig_force.update_layout(
                title=dict(text=f"Doppler force (δ/Γ = {delta_norm:.2f}, s₀ = {s0:.2f})",
                           font=dict(color="#c8b8ff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                xaxis=dict(title="Velocity (m/s)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                yaxis=dict(title="Force  (ℏkΓ/2)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
                margin=dict(l=20, r=20, t=40, b=40), height=270,
            )
            st.plotly_chart(fig_force, use_container_width=True, key="doppler_force")

    # ── Optical Tweezer tab ──────────────────────────────────────
    with tab_tweezer:
        col_tw, col_tp = st.columns([1, 2])
        with col_tw:
            st.markdown("### Gaussian beam profile")
            st.markdown("""
The intensity of a Gaussian beam is:

**I(r, z) = I₀ · exp(−2r²/w(z)²)**

where **w(z) = w₀√(1 + (z/z_R)²)** and **z_R = πw₀²/λ** is the Rayleigh range.

Atoms are attracted to the intensity maximum at the focus.
With a sub-micron beam waist only a **single atom** fits in the trap.
""")
            w0_um = st.slider("Beam waist w₀ (μm)", 0.5, 5.0, 1.0, 0.1)
            lam_choice = st.selectbox("Trapping laser", ["1064 nm (typical IR)", "760 nm", "532 nm (green)"])
            lam_tw = {"1064 nm (typical IR)": 1064e-9,
                      "760 nm": 760e-9, "532 nm (green)": 532e-9}[lam_choice]
            w0 = w0_um * 1e-6
            zR = np.pi * w0 ** 2 / lam_tw

            st.markdown(f"""
- Beam waist w₀ = **{w0_um:.1f} μm**
- Rayleigh range z_R = **{zR*1e6:.1f} μm**
- Depth of focus = **{2*zR*1e6:.1f} μm**

*Hood Lab tweezers: w₀ ≈ 0.8–1.5 μm, λ = 1064 nm*
""")

        with col_tp:
            r_max = 4 * w0
            r_pts = np.linspace(-r_max, r_max, 300)
            z_pts = np.linspace(-3 * zR, 3 * zR, 300)
            R, Z  = np.meshgrid(r_pts, z_pts)
            W     = w0 * np.sqrt(1 + (Z / zR) ** 2)
            I     = np.exp(-2 * R ** 2 / W ** 2)

            fig_tw = go.Figure(go.Heatmap(
                x=r_pts * 1e6, y=z_pts * 1e6, z=I,
                colorscale="Inferno",
                colorbar=dict(title=dict(text="Intensity", font=dict(color="#ccc")),
                             tickfont=dict(color="#ccc")),
            ))
            z_line = np.linspace(-3 * zR, 3 * zR, 300)
            w_line = w0 * np.sqrt(1 + (z_line / zR) ** 2)
            for sign in [1, -1]:
                fig_tw.add_trace(go.Scatter(
                    x=sign * w_line * 1e6, y=z_line * 1e6,
                    mode="lines", line=dict(color="white", width=1.5, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))
            fig_tw.update_layout(
                title=dict(text="Optical tweezer intensity I(r, z)  — dashed = beam waist",
                           font=dict(color="#c8b8ff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                xaxis=dict(title="Radial r (μm)", tickfont=dict(color="#aaa")),
                yaxis=dict(title="Axial z (μm)", tickfont=dict(color="#aaa")),
                margin=dict(l=20, r=20, t=40, b=40), height=430,
            )
            st.plotly_chart(fig_tw, use_container_width=True, key="tweezer_heat")


# ═══════════════════════════════════════════════════════════════
# PAGE: RYDBERG ATOMS & BLOCKADE
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[7]:
    st.title("⚡ Rydberg Atoms & Quantum Gate")

    st.markdown("""
<div class='concept-box'>
<b>Rydberg atoms</b> are atoms excited to high principal quantum number <em>n</em>.
They are enormous — orbital radius scales as <b>n²</b> — and interact with each other
via van der Waals forces that scale as <b>n¹¹</b>.<br><br>
This extreme sensitivity gives rise to the <b>Rydberg blockade</b>: once one atom is
in a Rydberg state, the interaction shifts its neighbour's transition frequency so far
that the neighbour <em>cannot</em> also be excited.  Only <em>one</em> atom within the
blockade radius Rᵦ can occupy the Rydberg level at a time.<br><br>
This conditional excitation is the mechanism behind the <b>Rydberg CZ gate</b> —
the entangling gate the Hood Lab is building with trapped Li and Cs atoms.
</div>
""", unsafe_allow_html=True)

    tab_r1, tab_r2 = st.tabs(["⚛️ Scaling Laws", "🚧 Blockade & Gate"])

    with tab_r1:
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            n_val = st.slider("Principal quantum number n", 10, 100, 30, step=1)
            a0_nm = 0.0529          # Bohr radius in nm
            r_n   = n_val ** 2 * a0_nm
            E_n   = 13.6 / n_val ** 2 * 1000   # binding energy in meV
            tau_n = 1.6e-8 * (n_val / 10) ** 3  # lifetime in seconds (rough n³ scaling)
            C6_n  = (n_val / 30) ** 11           # C₆ normalised to n = 30

            st.markdown(f"""
**n = {n_val}**

| Property | Value |
|---|---|
| Orbital radius | {r_n:.1f} nm = {r_n/a0_nm:.0f} a₀ |
| Binding energy | {E_n:.2f} meV |
| Lifetime τ | {tau_n*1e6:.1f} μs |
| C₆ (relative to n=30) | ×{C6_n:.1f} |

Ground state Li orbital radius: **0.17 nm**.
At n = {n_val}, the Rydberg atom is **{r_n/0.17:.0f}× larger**.
""")

        with col_r2:
            n_range    = np.arange(10, 101)
            r_range    = n_range ** 2 * a0_nm
            E_range    = 13.6 / n_range ** 2 * 1000
            tau_range  = 1.6e-8 * (n_range / 10) ** 3 * 1e6
            C6_range   = (n_range / 30) ** 11

            fig_r = make_subplots(
                rows=2, cols=2,
                subplot_titles=["Orbital radius (nm)", "Binding energy (meV)",
                                "Lifetime (μs)", "C₆ coefficient (norm. n=30)"],
            )
            for (row, col, y_data, color) in [
                (1, 1, r_range,   "#44aaff"),
                (1, 2, E_range,   "#ff6b6b"),
                (2, 1, tau_range, "#44ff88"),
                (2, 2, C6_range,  "#ffaa44"),
            ]:
                fig_r.add_trace(go.Scatter(
                    x=n_range, y=y_data, mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                ), row=row, col=col)
                idx = n_val - 10
                fig_r.add_trace(go.Scatter(
                    x=[n_val], y=[y_data[idx]], mode="markers",
                    marker=dict(color=color, size=10), showlegend=False,
                ), row=row, col=col)
                fig_r.update_xaxes(title_text="n", tickfont=dict(color="#aaa"),
                                   gridcolor="rgba(100,100,150,0.2)", row=row, col=col)
                fig_r.update_yaxes(tickfont=dict(color="#aaa"),
                                   gridcolor="rgba(100,100,150,0.2)", row=row, col=col)

            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.9)",
                height=420, margin=dict(l=20, r=20, t=60, b=20),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_r, use_container_width=True, key="rydberg_scale")

    with tab_r2:
        st.markdown("""
<div class='concept-box'>
<b>Rydberg CZ gate — three steps:</b><br>
1. π-pulse on atom A → excites A to Rydberg |r⟩ if A = |1⟩<br>
2. 2π-pulse on atom B → would flip B, but <b>blockade prevents it if A is excited</b>,
   so |11⟩ picks up a phase of −1 only<br>
3. π-pulse on atom A → de-excites A back<br><br>
Result: |00⟩, |01⟩, |10⟩ unchanged; |11⟩ → −|11⟩ &nbsp;=&nbsp; <b>CZ gate!</b>
</div>
""", unsafe_allow_html=True)

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            n_block    = st.slider("Rydberg level n", 20, 80, 50, key="nb")
            omega_mhz  = st.slider("Rabi frequency Ω/2π (MHz)", 0.1, 5.0, 1.0, step=0.1)

            # C₆ scaling: reference C₆(n=50) ≈ 862 GHz·μm⁶ (Rb, illustrative)
            C6_ref  = 862.0           # GHz·μm⁶
            C6_n    = C6_ref * (n_block / 50) ** 11
            Omega   = omega_mhz * 1e-3  # GHz
            R_b     = (C6_n / Omega) ** (1 / 6)   # μm

            st.markdown(f"""
**Gate parameters**
- n = **{n_block}**
- C₆ ≈ **{C6_n:.0f}** GHz·μm⁶
- Ω/2π = **{omega_mhz:.1f}** MHz
- **Blockade radius Rᵦ ≈ {R_b:.1f} μm**

*Typical Hood Lab tweezer spacing: 3–10 μm*
""")

        with col_b2:
            d_um = np.linspace(0.5, 25, 500)
            U_GHz = C6_n / d_um ** 6

            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=d_um, y=U_GHz, mode="lines",
                line=dict(color="#ff6b6b", width=2.5),
                name="C₆/R⁶ interaction",
            ))
            fig_b.add_hline(y=Omega,
                line=dict(color="#44ff88", width=2, dash="dash"),
                annotation_text=f"Ω/2π = {omega_mhz} MHz",
                annotation_position="right",
                annotation_font_color="#44ff88",
            )
            fig_b.add_vline(x=R_b,
                line=dict(color="#ffaa44", width=2, dash="dot"),
                annotation_text=f"Rᵦ = {R_b:.1f} μm",
                annotation_position="top right",
                annotation_font_color="#ffaa44",
            )
            fig_b.add_vrect(x0=0, x1=R_b,
                fillcolor="rgba(255,100,100,0.08)", line_width=0,
                annotation_text="BLOCKED", annotation_position="inside top",
                annotation_font_color="#ff6b6b",
            )
            fig_b.update_layout(
                title=dict(text="Rydberg interaction vs atom separation",
                           font=dict(color="#c8b8ff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                xaxis=dict(title="Inter-atom distance R (μm)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                yaxis=dict(title="Interaction U (GHz)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)", type="log"),
                legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
                margin=dict(l=20, r=20, t=40, b=40), height=380,
            )
            st.plotly_chart(fig_b, use_container_width=True, key="blockade_plot")


# ═══════════════════════════════════════════════════════════════
# PAGE: TWO-QUBIT GATES
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[8]:
    st.title("🔢 Two-Qubit Gates")

    st.markdown("""
<div class='concept-box'>
Single-qubit gates are rotations on the Bloch sphere.  To build a <b>universal quantum
computer</b> you also need <b>entangling gates</b> that create correlations between two qubits.<br><br>
• <b>CNOT</b> — flips qubit B only when qubit A = |1⟩<br>
• <b>CZ</b> — adds phase −1 to |11⟩ only; natively produced by the Rydberg blockade<br>
• <b>iSWAP</b> — swaps |01⟩↔|10⟩ with a phase of i; native in superconducting qubits<br><br>
When a two-qubit gate acts on a superposition, the output state is often
<b>entangled</b> — the individual qubit Bloch vectors shrink <em>inside</em> the sphere,
showing there is no valid single-qubit description of each qubit alone.
</div>
""", unsafe_allow_html=True)

    g_name = st.radio("Gate", list(TWO_Q_GATES.keys()), horizontal=True)
    U2 = TWO_Q_GATES[g_name]

    g_desc = {
        "CNOT":  "Controlled-NOT. Flips B when A=|1⟩. Maps |+⟩|0⟩ → |Φ⁺⟩ (Bell state). "
                 "Implemented via CZ + single-qubit rotations.",
        "CZ":    "Controlled-Z. Adds −1 phase to |11⟩ only. Symmetric in A↔B. "
                 "Directly realised via the Rydberg blockade in the Hood Lab.",
        "iSWAP": "Swaps |01⟩↔|10⟩ with a factor of i. Two iSWAPs = SWAP. "
                 "Native gate in superconducting transmon qubits.",
    }
    st.info(g_desc[g_name])

    # Truth table
    st.markdown("### Truth table")
    basis_labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    tt_rows = ["| Input \\|AB⟩ | Output |", "|---|---|"]
    for j, lbl in enumerate(basis_labels):
        e_j = np.zeros(4, dtype=complex); e_j[j] = 1.0
        out = U2 @ e_j
        terms = []
        for k, c in enumerate(out):
            if abs(c) > 1e-9:
                if abs(c.imag) < 1e-9:
                    pre = "" if abs(c.real - 1) < 1e-9 else ("-" if abs(c.real + 1) < 1e-9 else f"{c.real:+.2f}")
                elif abs(c.real) < 1e-9:
                    pre = "i" if abs(c.imag - 1) < 1e-9 else f"{c.imag:+.2f}i"
                else:
                    pre = f"({c.real:+.2f}{c.imag:+.2f}i)"
                terms.append(f"{pre}{basis_labels[k]}")
        tt_rows.append(f"| {lbl} | {'  +  '.join(terms)} |")
    st.markdown("\n".join(tt_rows))

    st.markdown("---")
    st.markdown("### Apply the gate — watch the Bloch vectors")

    col_qa, col_qb, col_out = st.columns([1, 1, 2])

    with col_qa:
        st.markdown("**Qubit A (control)**")
        preset_a = st.selectbox("State A", list(NAMED_STATES.keys()), key="tq_a")
        psi_a    = NAMED_STATES[preset_a].copy()
        ta, pa   = angles_from_state(psi_a)
        ta = st.slider("θ_A", 0.0, float(np.pi), float(ta), 0.01, key="tq_ta")
        pa = st.slider("φ_A", 0.0, 2*float(np.pi), float(pa), 0.01, key="tq_pa")
        psi_a = state_from_angles(ta, pa)

    with col_qb:
        st.markdown("**Qubit B (target)**")
        preset_b = st.selectbox("State B", list(NAMED_STATES.keys()), key="tq_b")
        psi_b    = NAMED_STATES[preset_b].copy()
        tb, pb   = angles_from_state(psi_b)
        tb = st.slider("θ_B", 0.0, float(np.pi), float(tb), 0.01, key="tq_tb")
        pb = st.slider("φ_B", 0.0, 2*float(np.pi), float(pb), 0.01, key="tq_pb")
        psi_b = state_from_angles(tb, pb)

    psi_in_2q  = np.kron(psi_a, psi_b)
    psi_out_2q = U2 @ psi_in_2q

    rA_in  = reduced_bloch_vec(psi_in_2q,  0)
    rB_in  = reduced_bloch_vec(psi_in_2q,  1)
    rA_out = reduced_bloch_vec(psi_out_2q, 0)
    rB_out = reduced_bloch_vec(psi_out_2q, 1)

    with col_out:
        fig_2q = build_bloch_xyz(
            [rA_in, rB_in, rA_out, rB_out],
            labels=["A  input", "B  input", "A  output", "B  output"],
            colors=["#44aaff", "#aaddff", "#ff4444", "#ffaaaa"],
            title=f"{g_name} gate — input (blue) → output (red)",
        )
        st.plotly_chart(fig_2q, use_container_width=True, key="tq_sphere")

    r_out_A = float(np.linalg.norm(rA_out))
    r_out_B = float(np.linalg.norm(rB_out))
    if r_out_A < 0.99 or r_out_B < 0.99:
        st.success(
            f"Output is **entangled** — Bloch vectors are inside the sphere "
            f"(|r_A| = {r_out_A:.2f}, |r_B| = {r_out_B:.2f}). "
            "Neither qubit has a definite state on its own."
        )
    else:
        st.info(
            f"Output is a **product state** — both vectors on the surface "
            f"(|r_A| = {r_out_A:.2f}, |r_B| = {r_out_B:.2f})."
        )

    with st.expander("🔬 Try: CNOT creates a Bell state from |+⟩|0⟩"):
        st.markdown("""
1. Set **qubit A = |+⟩**, **qubit B = |0⟩**, gate = **CNOT**
2. Input: |+⟩|0⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
3. CNOT flips B when A=1: → **(|00⟩ + |11⟩)/√2 = |Φ⁺⟩**
4. Watch both output Bloch vectors **shrink to the centre** — the output is
   maximally entangled and neither qubit has a well-defined state alone.

This is the exact type of entanglement the Hood Lab aims to generate between
trapped Li and Cs atoms using the Rydberg blockade.
""")

    with st.expander("📖 Gate decompositions"):
        st.markdown("""
| Gate | Rydberg realisation | Superconducting realisation |
|---|---|---|
| CZ | Direct via blockade | Cross-resonance + calibration |
| CNOT | CZ + H on B (before & after) | CZ decomposition |
| iSWAP | Tunable coupling | Native capacitive coupling |
| Universal | CZ + single-qubit gates | iSWAP + single-qubit gates |

Any two-qubit entangling gate is sufficient for universality when combined with
arbitrary single-qubit rotations (already explored in the Gates page).
""")

# ═══════════════════════════════════════════════════════════════
# PAGE: RABI OSCILLATIONS  (QuTiP mesolve)
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[9]:
    st.title("🌀 Rabi Oscillations")

    st.markdown("""
<div class='concept-box'>
<b>What is a Rabi oscillation?</b><br><br>
When a resonant laser drives a two-level atom, its population oscillates coherently
between |0⟩ and |1⟩ at the <b>Rabi frequency Ω</b>.  On the Bloch sphere this is a
clean rotation around the X axis — the simplest quantum gate.<br><br>
If the laser is <b>detuned by Δ</b> from resonance, the effective Rabi frequency
increases to <b>Ω_eff = √(Ω² + Δ²)</b> but complete inversion is no longer possible —
the oscillation amplitude shrinks.  Watching this is the clearest way to see that a
qubit is not just a bit: the atom is simultaneously 0 <em>and</em> 1 during the pulse.<br><br>
This is how the Hood Lab calibrates every single-qubit gate on trapped Li and Cs atoms.
</div>
""", unsafe_allow_html=True)

    col_rb1, col_rb2 = st.columns([1, 2])

    with col_rb1:
        omega_mhz = st.slider("Rabi frequency Ω/2π (MHz)", 0.1, 10.0, 1.0, 0.1)
        delta_mhz = st.slider("Detuning Δ/2π (MHz)", -8.0, 8.0, 0.0, 0.1,
                              help="0 = on resonance → full inversion")
        n_periods = st.slider("Duration (Rabi periods)", 0.5, 10.0, 3.0, 0.5)
        psi0_r    = st.selectbox("Initial state", list(NAMED_STATES.keys()), key="rabi_s0")
        psi0_np   = NAMED_STATES[psi0_r]

        omega_eff = float(np.sqrt(omega_mhz**2 + delta_mhz**2))
        T_rabi    = 1.0 / omega_eff if omega_eff > 0 else np.inf
        p1_max    = (omega_mhz / omega_eff)**2 if omega_eff > 0 else 0.0

        st.markdown(f"""
**Generalized Rabi frequency**
Ω_eff = √(Ω²+Δ²) = **{omega_eff:.2f} MHz**

**Period** T = **{T_rabi*1000:.1f} ns**

**Max P(|1⟩)** = (Ω/Ω_eff)² = **{p1_max:.3f}**

{"✅ Full inversion — resonant drive" if abs(delta_mhz) < 0.05
 else "⚠️ Partial inversion — off-resonance"}
""")

    # QuTiP time evolution
    t_max  = float(n_periods) / omega_eff if omega_eff > 0 else 1.0
    tlist  = np.linspace(0, t_max, 500)
    H_rabi = float(delta_mhz) / 2 * sigmaz() + float(omega_mhz) / 2 * sigmax()
    psi0_q = Qobj(psi0_np.reshape(2, 1))
    res_r  = mesolve(H_rabi, psi0_q, tlist, [], [sigmax(), sigmay(), sigmaz()])
    sx_r, sy_r, sz_r = (np.array(res_r.expect[i]) for i in range(3))
    p0_t, p1_t = (1 + sz_r) / 2, (1 - sz_r) / 2

    with col_rb2:
        tab_pop, tab_traj = st.tabs(["📈 Population vs time", "🔵 Bloch sphere trajectory"])

        with tab_pop:
            p1_analytic = p1_max * np.sin(np.pi * omega_eff * tlist) ** 2
            fig_pop = go.Figure()
            fig_pop.add_trace(go.Scatter(x=tlist * 1000, y=p0_t,
                name="|0⟩", line=dict(color="#44aaff", width=2.5)))
            fig_pop.add_trace(go.Scatter(x=tlist * 1000, y=p1_t,
                name="|1⟩", line=dict(color="#ff6b6b", width=2.5)))
            fig_pop.add_trace(go.Scatter(x=tlist * 1000, y=p1_analytic,
                name="P(|1⟩) analytic",
                line=dict(color="#ffaa44", width=1.5, dash="dash")))
            fig_pop.update_layout(
                title=dict(text="Rabi oscillations (QuTiP mesolve)",
                           font=dict(color="#c8b8ff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                xaxis=dict(title="Time (ns)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                yaxis=dict(title="Population", range=[-0.05, 1.1],
                           tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
                margin=dict(l=20, r=20, t=40, b=40), height=400,
            )
            st.plotly_chart(fig_pop, use_container_width=True, key="rabi_pop")

        with tab_traj:
            fig_rt = bloch_with_traj(sx_r, sy_r, sz_r,
                title=f"Bloch trajectory — Ω={omega_mhz} MHz, Δ={delta_mhz} MHz")
            st.plotly_chart(fig_rt, use_container_width=True, key="rabi_traj")

    with st.expander("📐 The Rabi formula"):
        st.markdown("""
Starting from |0⟩, the excited-state probability at time t is:

**P(|1⟩, t) = (Ω / Ω_eff)² sin²(π Ω_eff t)**

| Pulse | Condition | Effect |
|---|---|---|
| π-pulse | t = 1/(2Ω), Δ=0 | |0⟩ → |1⟩ (bit flip) |
| π/2-pulse | t = 1/(4Ω), Δ=0 | |0⟩ → |+⟩ (superposition) |
| 2π-pulse | t = 1/Ω, Δ=0 | Returns to |0⟩ but gains phase −1 (Rydberg gate!) |
""")

    with st.expander("📚 References"):
        st.markdown("""
- **Rabi, I. I.** (1937). Space Quantization in a Gyrating Magnetic Field. *Physical Review* 51, 652.
- **Allen, L. & Eberly, J. H.** (1975). *Optical Resonance and Two-Level Atoms.* Wiley.
- **Foot, C. J.** (2005). *Atomic Physics.* Oxford University Press — Ch. 7.
- **Johansson, J. R. et al.** (2013). QuTiP 2: *A Python framework for the dynamics of open quantum systems.* Comp. Phys. Comm. 184, 1234.
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: DECOHERENCE  (QuTiP Lindblad master equation)
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[10]:
    st.title("📉 Decoherence — T₁ & T₂")

    st.markdown("""
<div class='concept-box'>
<b>Real qubits are never perfectly isolated.</b>  Coupling to the environment causes
two distinct types of error:<br><br>
• <b>T₁  (energy relaxation)</b> — the qubit decays from |1⟩ → |0⟩ spontaneously.
  The Bloch z-component relaxes exponentially back to +1 (ground state).<br><br>
• <b>T₂  (dephasing)</b> — quantum coherence (the x/y components) is destroyed by
  low-frequency noise — laser phase jitter, magnetic field fluctuations, etc.
  T₂ ≤ 2T₁ always.  The gap is <em>pure dephasing</em>.<br><br>
On the Bloch sphere, decoherence makes the state vector <b>spiral inward</b> —
shrinking from the surface (pure state, |r|=1) toward the centre (maximally mixed,
|r|=0).  This is the Lindblad master equation, solved here exactly by QuTiP.
</div>
""", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([1, 2])

    with col_d1:
        T1_us    = st.slider("T₁ (μs)", 1.0, 500.0, 100.0, 1.0,
                             help="Energy relaxation: |1⟩→|0⟩")
        T2_us    = st.slider("T₂ (μs)", 0.5, float(2 * 100.0), 30.0, 0.5,
                             help="Total dephasing time — must be ≤ 2T₁", key="T2_sl")
        T2_us    = min(T2_us, 2.0 * T1_us)
        omega_q  = st.slider("Precession ω_q/2π (MHz)", 0.0, 3.0, 0.5, 0.05,
                             help="Qubit Larmor frequency in the lab frame")
        psi0_d   = st.selectbox("Initial state", list(NAMED_STATES.keys()),
                                index=2, key="dec_s0")   # |+⟩ shows both effects

        gamma1   = 1.0 / T1_us
        gamma2   = 1.0 / T2_us
        gamma_phi = max(0.0, gamma2 - gamma1 / 2)
        T2_eff   = 1.0 / (gamma1 / 2 + gamma_phi) if (gamma1 / 2 + gamma_phi) > 0 else np.inf

        st.markdown(f"""
**Rates**
- γ₁ = 1/T₁ = **{gamma1:.4f} MHz**
- γ_φ (pure dephasing) = **{gamma_phi:.4f} MHz**
- **T₂_eff = {T2_eff:.1f} μs**
- T₂ / T₁ = **{T2_us/T1_us:.2f}** (maximum is 2.0)
""")

    # QuTiP Lindblad master equation
    psi0_np_d = NAMED_STATES[psi0_d]
    psi0_qd   = Qobj(psi0_np_d.reshape(2, 1))
    t_max_d   = 5.0 * max(T1_us, T2_us)
    tlist_d   = np.linspace(0, t_max_d, 400)
    H_d       = float(omega_q) / 2 * sigmaz()
    c_ops_d   = [np.sqrt(gamma1) * destroy(2)]
    if gamma_phi > 1e-9:
        c_ops_d.append(np.sqrt(2 * gamma_phi) * sigmaz() / 2)
    res_d = mesolve(H_d, psi0_qd, tlist_d, c_ops_d,
                    [sigmax(), sigmay(), sigmaz()])
    sx_d, sy_d, sz_d = (np.array(res_d.expect[i]) for i in range(3))

    with col_d2:
        tab_dc1, tab_dc2 = st.tabs(["📈 Bloch components vs time", "🔵 Trajectory"])

        with tab_dc1:
            sz0  = float(np.real(psi0_np_d.conj() @ np.diag([1, -1]) @ psi0_np_d))
            fig_dc = go.Figure()
            fig_dc.add_trace(go.Scatter(x=tlist_d, y=sx_d, name="⟨X⟩",
                line=dict(color="#ff6b6b", width=2)))
            fig_dc.add_trace(go.Scatter(x=tlist_d, y=sy_d, name="⟨Y⟩",
                line=dict(color="#44ff88", width=2)))
            fig_dc.add_trace(go.Scatter(x=tlist_d, y=sz_d, name="⟨Z⟩",
                line=dict(color="#44aaff", width=2.5)))
            fig_dc.add_trace(go.Scatter(
                x=tlist_d,
                y=1.0 - (1.0 - sz0) * np.exp(-gamma1 * tlist_d),
                name="T₁ envelope", line=dict(color="#44aaff", width=1.5, dash="dash")))
            fig_dc.update_layout(
                title=dict(text="Bloch vector components (QuTiP Lindblad)",
                           font=dict(color="#c8b8ff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                xaxis=dict(title="Time (μs)", tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                yaxis=dict(title="Expectation value", range=[-1.1, 1.1],
                           tickfont=dict(color="#aaa"),
                           gridcolor="rgba(100,100,150,0.2)"),
                legend=dict(font=dict(color="#ccc"), bgcolor="rgba(20,20,40,0.8)"),
                margin=dict(l=20, r=20, t=40, b=40), height=400,
            )
            st.plotly_chart(fig_dc, use_container_width=True, key="dec_comp")

        with tab_dc2:
            fig_dt = bloch_with_traj(sx_d, sy_d, sz_d,
                color="#ff6b6b", end_color="#666688",
                title="Qubit spiralling toward thermal equilibrium at centre")
            st.plotly_chart(fig_dt, use_container_width=True, key="dec_traj")

    r_final = float(np.sqrt(sx_d[-1]**2 + sy_d[-1]**2 + sz_d[-1]**2))
    st.info(f"Final Bloch vector length |r| = **{r_final:.4f}** "
            f"(1.0 = pure, 0.0 = maximally mixed)")

    with st.expander("📐 Optical Bloch equations"):
        st.markdown(r"""
The equations of motion for the Bloch vector with relaxation:

d⟨X⟩/dt = −ω_q ⟨Y⟩ − ⟨X⟩ / T₂

d⟨Y⟩/dt =  ω_q ⟨X⟩ − ⟨Y⟩ / T₂

d⟨Z⟩/dt = −(⟨Z⟩ − Z_eq) / T₁

The Bloch vector **shrinks inward** — a pure state (on the surface) becomes a mixed
state (inside the sphere) as the qubit entangles with its environment.
""")

    with st.expander("📚 References"):
        st.markdown("""
- **Bloch, F.** (1946). Nuclear Induction. *Physical Review* 70, 460.
- **Krantz, P. et al.** (2019). A quantum engineer's guide to superconducting qubits. *Applied Physics Reviews* 6, 021318.
- **Saffman, M. et al.** (2016). Quantum computing with atomic qubits and Rydberg interactions. *Journal of Physics B* 49, 202001.
- **Johansson, J. R. et al.** (2013). QuTiP 2. *Computer Physics Communications* 184, 1234.
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: WIGNER FUNCTION  (QuTiP wigner + motional states)
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[11]:
    st.title("🌊 Wigner Function & Motional States")

    st.markdown("""
<div class='concept-box'>
<b>Quantised motion inside an optical tweezer.</b><br><br>
A trapped atom doesn't just have an internal qubit — its <em>motion</em> in the tweezer
potential is quantised, forming a <b>quantum harmonic oscillator</b> with Fock states
|0⟩, |1⟩, |2⟩, … (phonons / motional quanta).  Before running a gate the atom must
be cooled to the motional ground state |0⟩ via <b>sideband cooling</b>.<br><br>
The <b>Wigner function W(x, p)</b> maps a quantum state onto phase space.  Unlike a
classical probability distribution it can be <b>negative</b> — a smoking-gun signature
of quantum behaviour.  Negative regions appear for Fock states and Schrödinger cat
states but not for thermal or coherent states.
</div>
""", unsafe_allow_html=True)

    N    = 35
    xvec = np.linspace(-6, 6, 250)

    col_w1, col_w2 = st.columns([1, 2])

    with col_w1:
        state_choice = st.selectbox("State of the motional oscillator", [
            "Fock |n⟩  — number state",
            "Coherent |α⟩  — classical-like",
            "Thermal ρ_th  — mixed",
            "Cat state  — Schrödinger's cat",
            "Squeezed vacuum",
        ])

        if state_choice.startswith("Fock"):
            n_fock = st.slider("Fock number n", 0, 10, 0)
            state  = fock(N, n_fock)
            desc   = (f"|{n_fock}⟩ has exactly {n_fock} phonons. "
                      f"{'Ground state — no negative regions.' if n_fock == 0 else 'Negative Wigner regions appear — genuine quantum state!'}")

        elif state_choice.startswith("Coherent"):
            a_re  = st.slider("Re(α)", -3.0, 3.0, 1.5, 0.1)
            a_im  = st.slider("Im(α)", -3.0, 3.0, 0.0, 0.1)
            alpha = complex(a_re, a_im)
            state = coherent(N, alpha)
            desc  = (f"Coherent |α={alpha:.2f}⟩. Mean phonon number ⟨n⟩ = |α|² = {abs(alpha)**2:.2f}. "
                     "Minimum-uncertainty Gaussian — the most classical-like quantum state.")

        elif state_choice.startswith("Thermal"):
            n_th  = st.slider("Mean phonon number ⟨n⟩", 0.0, 5.0, 1.0, 0.1)
            state = thermal_dm(N, n_th)
            desc  = (f"Thermal state, ⟨n⟩ = {n_th:.1f}. "
                     "Broader Gaussian, always positive — fully classical phase-space description.")

        elif state_choice.startswith("Cat"):
            a_cat    = st.slider("|α| (separation / 2)", 0.5, 3.0, 2.0, 0.1)
            cat_type = st.radio("Type", ["Even  +", "Odd  −"], horizontal=True)
            sign     = 1.0 if "Even" in cat_type else -1.0
            state    = (coherent(N, a_cat) + sign * coherent(N, -a_cat)).unit()
            desc     = (f"Schrödinger cat: superposition of |+{a_cat:.1f}⟩ and |−{a_cat:.1f}⟩. "
                        "The interference fringes between the two blobs are quantum coherence — "
                        "they disappear the instant the state decoheres.")

        else:
            r_sq  = st.slider("Squeezing r", 0.0, 2.0, 0.8, 0.05)
            phi_sq = st.slider("Squeezing angle φ", 0.0, float(np.pi), 0.0, 0.05)
            state  = squeeze(N, r_sq * np.exp(1j * phi_sq)) * basis(N, 0)
            desc   = (f"Squeezed vacuum, r = {r_sq:.2f}. "
                      "Noise is reduced in one quadrature below the vacuum level, "
                      "increased in the conjugate — Heisenberg uncertainty still satisfied.")

        st.markdown(f"*{desc}*")

        # Phonon number distribution
        if state.type == "ket":
            probs = np.abs(state.full().flatten()[:15]) ** 2
        else:
            probs = np.real(np.diag(state.full()))[:15]

        fig_pn = go.Figure(go.Bar(
            x=list(range(len(probs))), y=probs,
            marker_color="#7b68ee",
            text=[f"{p:.3f}" if p > 0.005 else "" for p in probs],
            textposition="outside", textfont=dict(color="#fff", size=10),
        ))
        fig_pn.update_layout(
            title=dict(text="Phonon number distribution P(n)",
                       font=dict(color="#c8b8ff", size=13)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.9)",
            xaxis=dict(title="n", tickfont=dict(color="#aaa")),
            yaxis=dict(title="P(n)", tickfont=dict(color="#aaa"),
                       gridcolor="rgba(100,100,150,0.2)"),
            margin=dict(l=20, r=20, t=35, b=30), height=210,
        )
        st.plotly_chart(fig_pn, use_container_width=True, key="wigner_pn")

    with col_w2:
        W      = wigner(state, xvec, xvec)
        w_abs  = float(np.max(np.abs(W))) or 1.0
        vol_neg = float(np.sum(W[W < 0]) * (xvec[1] - xvec[0]) ** 2)

        fig_wg = go.Figure(go.Heatmap(
            x=xvec, y=xvec, z=W,
            colorscale=[
                [0.0,  "rgb(200, 30,  30)"],
                [0.38, "rgb(80,  0,   0)"],
                [0.5,  "rgb(15,  15,  35)"],
                [0.62, "rgb(0,   0,   100)"],
                [1.0,  "rgb(50,  130, 255)"],
            ],
            zmin=-w_abs, zmax=w_abs,
            colorbar=dict(title=dict(text="W(x,p)", font=dict(color="#ccc")),
                          tickfont=dict(color="#ccc")),
        ))
        fig_wg.add_trace(go.Contour(
            x=xvec, y=xvec, z=W,
            showscale=False,
            contours=dict(coloring="none", showlines=True,
                          start=-w_abs, end=w_abs, size=w_abs / 6),
            line=dict(color="rgba(255,255,255,0.2)", width=0.8),
        ))
        fig_wg.update_layout(
            title=dict(
                text="Wigner function W(x, p)  —  🔴 red = negative = non-classical",
                font=dict(color="#c8b8ff")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.9)",
            xaxis=dict(title="Position quadrature  x", tickfont=dict(color="#aaa")),
            yaxis=dict(title="Momentum quadrature  p", tickfont=dict(color="#aaa")),
            margin=dict(l=20, r=20, t=45, b=40), height=490,
        )
        st.plotly_chart(fig_wg, use_container_width=True, key="wigner_map")

        if abs(vol_neg) > 1e-4:
            st.error(f"Negative Wigner volume = **{vol_neg:.4f}** — genuinely non-classical state!")
        else:
            st.success("Wigner function ≥ 0 everywhere — this state has a classical phase-space description.")

    with st.expander("📐 What is the Wigner function?"):
        st.markdown(r"""
The Wigner function is a **quasi-probability distribution** on phase space (x, p):

W(x, p) = (1/π) ∫ ⟨x+y | ρ | x−y⟩ e^(2ipy) dy

Key facts:
- Normalised: ∫∫ W dx dp = 1
- **Marginals** are real probabilities: ∫ W dp = |ψ(x)|², ∫ W dx = |φ(p)|²
- **Can be negative** — the hallmark of a quantum state with no classical analogue
- Measured experimentally via **homodyne tomography** (or ion-trap reconstructions)

In optical tweezer experiments, the motional Wigner function is accessed by mapping
motional state populations onto internal states via sideband pulses.
""")

    with st.expander("📚 References"):
        st.markdown("""
- **Wigner, E. P.** (1932). On the Quantum Correction For Thermodynamic Equilibrium. *Physical Review* 40, 749.
- **Leibfried, D. et al.** (1996). Experimental Determination of the Motional Quantum State of a Trapped Atom. *PRL* 77, 4281. ← First measurement of a trapped-atom Wigner function!
- **Lvovsky, A. I. & Raymer, M. G.** (2009). Continuous-variable optical quantum-state tomography. *Rev. Mod. Phys.* 81, 299.
- **de Léséleuc, S. et al.** (2019). Observation of a symmetry-protected topological phase of interacting bosons with Rydberg atoms. *Science* 365, 775.
""")
