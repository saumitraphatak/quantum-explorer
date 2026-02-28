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
    for name, ax, ay, az in axis_cfg:
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
            xaxis=dict(title="X", showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Y", showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa"),
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Z", showgrid=False, zeroline=False,
                       tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa"),
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
        fig_bell = go.Figure(go.Bar(
            x=outcomes, y=probs,
            marker_color=["#7b68ee", "#ff6b6b", "#44ccaa", "#ffaa44"],
            text=[f"{p:.0%}" for p in probs],
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
        if st.button("▶  Run Bell experiment"):
            non_zero = {k: v for k, v in info["probs"].items() if v > 0}
            keys = list(non_zero.keys())
            pvec = np.array(list(non_zero.values()))
            pvec = pvec / pvec.sum()
            samp = np.random.choice(keys, size=n_bell, p=pvec)
            counts = {k: int(np.sum(samp == k)) for k in outcomes}
            fig_sim = go.Figure(go.Bar(
                x=list(counts.keys()), y=list(counts.values()),
                marker_color=["#7b68ee", "#ff6b6b", "#44ccaa", "#ffaa44"],
                text=list(counts.values()), textposition="outside",
                textfont=dict(color="#fff"),
            ))
            fig_sim.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.9)",
                yaxis=dict(gridcolor="rgba(100,100,150,0.2)", tickfont=dict(color="#aaa")),
                xaxis=dict(tickfont=dict(color="#fff")),
                margin=dict(l=20, r=20, t=30, b=40),
                height=230,
                title=dict(text=f"{n_bell} shots", font=dict(color="#c8b8ff", size=13)),
            )
            st.plotly_chart(fig_sim, use_container_width=True, key="bell_sim")

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
