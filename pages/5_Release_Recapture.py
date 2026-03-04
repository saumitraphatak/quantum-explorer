"""
Release–Recapture Thermometry Simulator
========================================
Monte Carlo simulation of the fraction of atoms recaptured by an optical
tweezer after being released for a variable free-flight time Δt.

Physics
-------
An atom in a focused Gaussian beam experiences the optical dipole potential

    U(r) = -½ α(ω) |E_rms(r)|²

At each release time Δt, 1000 atoms are drawn from Maxwell–Boltzmann
at temperature T, propagated ballistically (with gravity), then checked
against the recapture condition  KE + U(r_final) < U₀.

Fitting P_cap(Δt) to measured survival fractions yields the atom temperature.
"""

import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Release–Recapture Simulator",
    page_icon="🪤",
    layout="wide",
)

# ─── CSS (matches existing app style) ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family:'Inter',sans-serif; }
  .main  { background:#0a0a0f; color:#e2e8f0; }
  .block-container { padding:2rem 3rem; max-width:1400px; }
  h1  { font-size:2.2rem; font-weight:700; color:#f0f4ff;
        border-bottom:2px solid #0ea5e9; padding-bottom:.5rem; }
  h2  { color:#7dd3fc; font-size:1.45rem; }
  h3  { color:#38bdf8; font-size:1.1rem; }

  .theory-box {
    background:linear-gradient(135deg,#0c1e3a 0%,#0f2a4a 100%);
    border-left:4px solid #0ea5e9; border-radius:8px;
    padding:1.1rem 1.4rem; margin:.8rem 0;
  }
  .theory-box h4 { color:#7dd3fc; margin:0 0 .4rem 0; font-size:1rem; }
  .theory-box p  { color:#bae6fd; margin:0; font-size:.9rem; line-height:1.6; }

  .formula-box {
    background:#0a1f0a; border:1px solid #166534;
    border-radius:8px; padding:.9rem 1.2rem; margin:.7rem 0;
    font-family:'Courier New',monospace; color:#86efac; font-size:.87rem;
    line-height:1.7;
  }

  .result-card {
    background:#052e16; border:1px solid #15803d;
    border-radius:8px; padding:1rem 1.4rem; margin:.5rem 0;
    color:#bbf7d0; font-size:.95rem; text-align:center;
  }
  .result-card .val { font-size:1.6rem; font-weight:700; color:#4ade80; }
  .result-card .lbl { font-size:.8rem; color:#86efac; }

  .warn-box {
    background:#1c1007; border:1px solid #92400e;
    border-radius:8px; padding:.9rem 1.2rem; margin:.7rem 0;
    color:#fde68a; font-size:.9rem;
  }

  .param-section {
    background:#0f172a; border:1px solid #1e293b;
    border-radius:10px; padding:1.2rem 1.5rem; margin:.8rem 0;
  }

  hr { border-color:#1e293b; margin:2rem 0; }

  .stButton>button {
    background:linear-gradient(135deg,#0369a1,#0284c7);
    color:white; border:none; border-radius:8px;
    padding:.6rem 2rem; font-weight:600; font-size:1rem;
    width:100%; transition:all .2s;
  }
  .stButton>button:hover { background:linear-gradient(135deg,#0284c7,#38bdf8); }
</style>
""", unsafe_allow_html=True)


# ─── Physical constants ───────────────────────────────────────────────────────
kB   = 1.380649e-23
hbar = 1.054571817e-34
c    = 299_792_458
eps0 = 8.8541878128e-12
e    = 1.602176634e-19
a0   = 5.29177210903e-11
Eh   = e**2 / (4 * math.pi * eps0 * a0)
amu  = 1.66053906660e-27
g_n  = 9.81

ALPHA_AU_TO_SI = 4 * math.pi * eps0 * a0**3   # 1 a.u. → J·m²/V²

# ─── Atomic data ─────────────────────────────────────────────────────────────
SPECIES = {
    "¹³³Cs": dict(dE_D1=0.050932, d_D1=4.4890, dE_D2=0.053456, d_D2=6.3238, alpha_core=17.35, A=133),
    "⁸⁷Rb":  dict(dE_D1=0.057314, d_D1=4.231,  dE_D2=0.058396, d_D2=5.977,  alpha_core=10.54, A=87),
    "²³Na":  dict(dE_D1=0.077258, d_D1=3.5246, dE_D2=0.077336, d_D2=4.9838, alpha_core=1.86,  A=23),
    "⁶Li":   dict(dE_D1=0.067906, d_D1=3.317,  dE_D2=0.067907, d_D2=4.689,  alpha_core=2.04,  A=6),
}


# ─── Physics functions ────────────────────────────────────────────────────────

def dynamic_polarizability(wavelength_m: float, species_key: str) -> float:
    """
    Ground-state scalar dynamic polarizability α [J·m²/V²].
    Uses a two-transition (D1 + D2) sum-over-states formula
    with a static core correction (all in atomic units):

        α_au = Σ_j  ΔE_j |d_j|² / [3(ΔE_j² − (ħω/E_h)²)]  + α_core
    """
    s = SPECIES[species_key]
    xi = hbar * (2 * math.pi * c / wavelength_m) / Eh   # ħω / E_h  [dimensionless]

    def term(dE, d):
        return (dE * d**2) / (3.0 * (dE**2 - xi**2))

    alpha_au = term(s["dE_D1"], s["d_D1"]) + term(s["dE_D2"], s["d_D2"]) + s["alpha_core"]
    return alpha_au * ALPHA_AU_TO_SI   # [J·m²/V²]


def trap_depth_J(power_W: float, w0_m: float, wavelength_m: float,
                 species_key: str, strehl: float) -> float:
    """Peak trap depth U₀ [J] = ½ α(ω) · (S · 2P / π w₀²) / (c ε₀)"""
    alpha = dynamic_polarizability(wavelength_m, species_key)
    I0    = strehl * 2.0 * power_W / (math.pi * w0_m**2)    # peak intensity [W/m²]
    return 0.5 * alpha * I0 / (c * eps0)                     # [J]


def run_simulation(
    temperatures_K: list,
    t_arr_s: np.ndarray,
    power_W: float,
    w0_m: float,
    wavelength_m: float,
    species_key: str,
    strehl: float,
    N: int,
    sigma_r_m: float,
    sigma_z_m: float,
    rng: np.random.Generator,
) -> dict:
    """
    Vectorised Monte Carlo: compute recapture probability at every time in
    t_arr_s for each temperature in temperatures_K.

    Returns a dict  { T_K: np.ndarray of probabilities (same shape as t_arr_s) }
    """
    s     = SPECIES[species_key]
    mass  = s["A"] * amu
    U0    = trap_depth_J(power_W, w0_m, wavelength_m, species_key, strehl)
    zR    = math.pi * w0_m**2 / wavelength_m   # Rayleigh range

    results = {}

    for T_K in temperatures_K:
        sigma_v = np.sqrt(kB * T_K / mass)

        # Sample N atoms — shape (N,)
        x0 = rng.normal(0.0, sigma_r_m, N)
        y0 = rng.normal(0.0, sigma_r_m, N)
        z0 = rng.normal(0.0, sigma_z_m, N)
        vx = rng.normal(0.0, sigma_v,   N)
        vy = rng.normal(0.0, sigma_v,   N)
        vz = rng.normal(0.0, sigma_v,   N)

        # KE per atom — shape (N, 1) for broadcasting against times
        KE = (0.5 * mass * (vx**2 + vy**2 + vz**2)).reshape(N, 1)

        # Positions at each release time — shape (N, N_t)
        t = t_arr_s.reshape(1, -1)   # (1, N_t)
        x1 = x0.reshape(N, 1) + vx.reshape(N, 1) * t
        y1 = y0.reshape(N, 1) + vy.reshape(N, 1) * t - 0.5 * g_n * t**2
        z1 = z0.reshape(N, 1) + vz.reshape(N, 1) * t

        # Trap potential at new positions — shape (N, N_t)
        w  = w0_m * np.sqrt(1.0 + (z1 / zR)**2)          # local waist
        profile = (w0_m / w)**2 * np.exp(-2.0 * (x1**2 + y1**2) / w**2)
        PE = U0 * (1.0 - profile)

        # Recapture: total energy < U₀
        captured = (KE + PE) < U0       # bool (N, N_t)
        results[T_K] = captured.mean(axis=0)   # (N_t,)

    return results, U0


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🪤 Release–Recapture Thermometry")
st.markdown("""
<div style='color:#94a3b8; font-size:1rem; margin-bottom:1.5rem;'>
Monte Carlo simulation of atom recapture after free flight in an optical tweezer.
Set trap parameters and temperatures below — the simulator computes survival curves for each temperature.<br>
<span style='color:#38bdf8; font-size:.85rem;'>
Based on Phatak (2025) PhD Thesis (Purdue Hood Lab)
</span>
</div>
""", unsafe_allow_html=True)

# ─── Theory expander ──────────────────────────────────────────────────────────
with st.expander("📖 Physics & Method", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="theory-box">
<h4>Optical Dipole Potential</h4>
<p>A focused Gaussian beam creates the 3D trap potential:<br><br>
U(x,y,z) = U₀ · (w₀/w(z))² · exp(−2r²/w(z)²)<br><br>
where r² = x²+y², w(z) = w₀√(1+(z/z_R)²) is the local beam radius,
and U₀ = ½ α(ω)·E_rms² is the peak trap depth.
The polarizability α uses a two-transition D-line sum-over-states formula.</p>
</div>
<div class="formula-box">
α_au = Σⱼ  ΔEⱼ|dⱼ|² / [3(ΔEⱼ² − (ħω/Eₕ)²)]  +  α_core<br>
α_SI  = α_au × 4πε₀a₀³  [J·m²/V²]
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="theory-box">
<h4>Release–Recapture Monte Carlo</h4>
<p>At each release time Δt, N atoms are sampled from a Maxwell–Boltzmann
distribution at temperature T. Atoms propagate ballistically under gravity:</p>
</div>
<div class="formula-box">
x(Δt) = x₀ + vₓ Δt<br>
y(Δt) = y₀ + v_y Δt − ½g Δt²<br>
z(Δt) = z₀ + v_z Δt
</div>
<div class="theory-box">
<h4>Recapture Condition</h4>
<p>Atom is retained when its total energy at the new position is below U₀:</p>
</div>
<div class="formula-box">
½mv² + U(r(Δt)) &lt; U₀<br>
⟺  KE &lt; U₀ · profile(r(Δt))
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Parameters ───────────────────────────────────────────────────────────────
st.markdown("## ⚙️ Simulation Parameters")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("#### 🔬 Atom & Trap")
    species_label = st.selectbox(
        "Atom species",
        list(SPECIES.keys()),
        index=0,
        help="Sets atomic mass and D-line polarizability parameters.",
    )
    wavelength_nm = st.number_input(
        "Trapping wavelength (nm)",
        min_value=700.0, max_value=2000.0, value=1064.0, step=1.0,
        help="Common choices: 1064 nm (Nd:YAG), 852 nm (Cs resonance ±), 780 nm (Rb).",
    )
    strehl = st.slider(
        "Strehl ratio",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Beam quality factor (1 = perfect Gaussian). Scales peak intensity.",
    )

with col_b:
    st.markdown("#### 💡 Beam Parameters")
    power_mW = st.slider(
        "Trap power (mW)",
        min_value=0.5, max_value=50.0, value=5.4, step=0.1,
        help="Laser power entering the objective.",
    )
    w0_um = st.slider(
        "Beam waist w₀ (µm)",
        min_value=0.3, max_value=5.0, value=0.95, step=0.05,
        help="1/e² intensity radius at the focus.",
    )
    sigma_r_um = st.slider(
        "Initial radial spread σ_r (µm)",
        min_value=0.01, max_value=1.0, value=0.1, step=0.01,
        help="Standard deviation of initial atom positions in x and y.",
    )
    sigma_z_um = st.slider(
        "Initial axial spread σ_z (µm)",
        min_value=0.1, max_value=5.0, value=0.5, step=0.1,
        help="Standard deviation of initial atom positions along z (beam axis).",
    )

with col_c:
    st.markdown("#### 🌡️ Temperature & Timing")
    temp_str = st.text_input(
        "Temperatures to simulate (µK), comma-separated",
        value="5, 10, 20, 30, 100",
        help="E.g.  5, 10, 20, 50, 100",
    )
    t_max_us = st.slider(
        "Max release time (µs)",
        min_value=10, max_value=500, value=100, step=10,
    )
    t_pts = st.slider(
        "Number of time points",
        min_value=40, max_value=300, value=100, step=10,
        help="More points = smoother curve but slower.",
    )
    N_mc = st.select_slider(
        "Monte Carlo samples per point (N)",
        options=[200, 500, 1000, 2000, 5000],
        value=1000,
        help="More samples = less noise. The vectorised simulation handles all time points at once.",
    )

# ─── Parse temperatures ───────────────────────────────────────────────────────
try:
    temperatures_uK = [float(t.strip()) for t in temp_str.split(",") if t.strip()]
    if not temperatures_uK:
        raise ValueError
except ValueError:
    st.error("⚠️ Could not parse temperatures. Use comma-separated numbers, e.g.  5, 10, 20")
    st.stop()

temperatures_uK = sorted(temperatures_uK)

# ─── Quick trap depth preview ─────────────────────────────────────────────────
wavelength_m = wavelength_nm * 1e-9
power_W      = power_mW * 1e-3
w0_m         = w0_um * 1e-6

try:
    U0_preview = trap_depth_J(power_W, w0_m, wavelength_m, species_label, strehl)
    U0_mK  = U0_preview / (kB * 1e-3)
    U0_uK  = U0_preview / (kB * 1e-6)
    mass   = SPECIES[species_label]["A"] * amu
    omega_r = math.sqrt(4 * U0_preview / (mass * w0_m**2))   # radial trap frequency
    zR_m   = math.pi * w0_m**2 / wavelength_m
    omega_z = math.sqrt(2 * U0_preview / (mass * zR_m**2))   # axial trap frequency

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="result-card">
        <div class="lbl">Trap depth U₀</div>
        <div class="val">{U0_mK:.2f} mK</div>
        <div class="lbl">({U0_uK:.0f} µK)</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="result-card">
        <div class="lbl">Radial freq. ω_r / 2π</div>
        <div class="val">{omega_r / (2*math.pi) / 1e3:.1f} kHz</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="result-card">
        <div class="lbl">Axial freq. ω_z / 2π</div>
        <div class="val">{omega_z / (2*math.pi) / 1e3:.2f} kHz</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="result-card">
        <div class="lbl">Rayleigh range z_R</div>
        <div class="val">{zR_m * 1e6:.1f} µm</div>
    </div>""", unsafe_allow_html=True)
except Exception as ex:
    st.warning(f"Could not compute trap depth preview: {ex}")

# ─── Run button ───────────────────────────────────────────────────────────────
st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("▶ Run Simulation")

if not run:
    st.markdown("""
<div class="warn-box">
⚙️ Set parameters above and click <strong>Run Simulation</strong> to generate the recapture curves.
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─── Run ──────────────────────────────────────────────────────────────────────
with st.spinner("Running Monte Carlo simulation…"):
    t_arr_us = np.linspace(0, t_max_us, t_pts)
    t_arr_s  = t_arr_us * 1e-6

    temperatures_K = [T * 1e-6 for T in temperatures_uK]

    rng = np.random.default_rng(seed=42)

    results, U0_J = run_simulation(
        temperatures_K = temperatures_K,
        t_arr_s        = t_arr_s,
        power_W        = power_W,
        w0_m           = w0_m,
        wavelength_m   = wavelength_m,
        species_key    = species_label,
        strehl         = strehl,
        N              = N_mc,
        sigma_r_m      = sigma_r_um * 1e-6,
        sigma_z_m      = sigma_z_um * 1e-6,
        rng            = rng,
    )

# ─── Plot ─────────────────────────────────────────────────────────────────────
COLORS = [
    "#38bdf8", "#4ade80", "#f59e0b", "#f87171", "#a78bfa",
    "#fb923c", "#34d399", "#e879f9", "#fbbf24", "#60a5fa",
]

fig = go.Figure()

for idx, (T_K, T_uK) in enumerate(zip(temperatures_K, temperatures_uK)):
    color = COLORS[idx % len(COLORS)]
    prob  = results[T_K]
    fig.add_trace(go.Scatter(
        x=t_arr_us,
        y=prob,
        mode="lines",
        name=f"{T_uK:.0f} µK",
        line=dict(color=color, width=2),
        hovertemplate="Δt = %{x:.1f} µs<br>P_cap = %{y:.3f}<extra>%{fullData.name}</extra>",
    ))

fig.update_layout(
    title=dict(
        text=f"Release–Recapture: {species_label} in {wavelength_nm:.0f} nm tweezer "
             f"(P = {power_mW:.1f} mW, w₀ = {w0_um:.2f} µm, S = {strehl:.2f})",
        font=dict(color="#e2e8f0", size=15),
    ),
    xaxis=dict(
        title="Release time Δt (µs)",
        color="#94a3b8", gridcolor="#1e293b", zeroline=False,
        range=[0, t_max_us],
    ),
    yaxis=dict(
        title="Recapture probability",
        color="#94a3b8", gridcolor="#1e293b", zeroline=False,
        range=[0, 1.05],
        tickformat=".1f",
    ),
    plot_bgcolor="#0a0a0f",
    paper_bgcolor="#0f172a",
    font=dict(family="Inter", color="#e2e8f0"),
    legend=dict(
        title="Temperature",
        bgcolor="#0f172a",
        bordercolor="#1e293b",
        borderwidth=1,
        font=dict(color="#e2e8f0"),
    ),
    hovermode="x unified",
    height=520,
    margin=dict(l=60, r=30, t=60, b=60),
)

st.plotly_chart(fig, use_container_width=True)

# ─── Summary table ────────────────────────────────────────────────────────────
st.markdown("### 📊 Recapture at selected times")

# Pick a few representative times to show in a summary table
sample_idx = np.linspace(0, len(t_arr_us) - 1, min(8, len(t_arr_us)), dtype=int)
sample_times = t_arr_us[sample_idx]

import pandas as pd

table_data = {"Δt (µs)": [f"{t:.1f}" for t in sample_times]}
for T_K, T_uK in zip(temperatures_K, temperatures_uK):
    col_name = f"{T_uK:.0f} µK"
    table_data[col_name] = [f"{results[T_K][i]:.3f}" for i in sample_idx]

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# ─── Notes ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="theory-box">
<h4>📌 Notes & Assumptions</h4>
<p>
• <strong>Ballistic propagation:</strong> velocities are not updated for gravity during
  free flight (i.e. the KE at recapture uses initial velocities). This is a good
  approximation when gΔt ≪ σ_v  (e.g. at Δt = 100 µs, gΔt ≈ 1 mm/s vs σ_v ≈ 10 mm/s at 10 µK).<br><br>
• <strong>Classical model:</strong> the simulation treats atoms classically. Quantum tunnelling
  and zero-point motion are not included, so it is most accurate for T ≫ T_ground = ħω/k_B.<br><br>
• <strong>Polarizability:</strong> uses a two-transition (D1 + D2) sum-over-states formula
  with a static core correction. Tensor and vector light shifts are neglected.<br><br>
• <strong>Reproducibility:</strong> the RNG is seeded at 42. Re-running with the same parameters
  gives identical curves. Increase N for smoother results.
</p>
</div>
""", unsafe_allow_html=True)
