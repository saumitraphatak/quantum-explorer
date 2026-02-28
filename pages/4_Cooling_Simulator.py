"""
Atom Cooling: From Hot Clouds to the Quantum Ground State
curious96.com — interactive cooling physics guide
Based on Phatak (2025) PhD Thesis, Purdue University Hood Lab
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Atom Cooling Simulator",
    page_icon="🧊",
    layout="wide",
)

# ── physical constants ────────────────────────────────────────────────────────
hbar   = 1.0545718e-34   # J·s
kB     = 1.380649e-23    # J/K
amu    = 1.66054e-27     # kg
c      = 2.998e8         # m/s

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family:'Inter',sans-serif; }
  .main  { background:#0a0a0f; color:#e2e8f0; }
  .block-container { padding:2rem 3rem; max-width:1400px; }
  h1  { font-size:2.4rem; font-weight:700; color:#f0f4ff;
        border-bottom:2px solid #0ea5e9; padding-bottom:.5rem; }
  h2  { color:#7dd3fc; font-size:1.5rem; }
  h3  { color:#38bdf8; font-size:1.15rem; }

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

  .step-box {
    background:#0f172a; border:1px solid #1e293b;
    border-radius:8px; padding:.9rem 1.2rem; margin:.5rem 0;
  }
  .step-box h5 { color:#e2e8f0; margin:0 0 .3rem 0; font-size:.95rem; }
  .step-box p  { color:#94a3b8; margin:0; font-size:.86rem; line-height:1.55; }

  .warn-box {
    background:#1c1007; border:1px solid #92400e;
    border-radius:8px; padding:.9rem 1.2rem; margin:.7rem 0;
    color:#fde68a; font-size:.9rem;
  }
  .result-box {
    background:#052e16; border:1px solid #15803d;
    border-radius:8px; padding:.9rem 1.2rem; margin:.7rem 0;
    color:#bbf7d0; font-size:.9rem;
  }
  .thesis-quote {
    background:linear-gradient(90deg,#1e1b4b 0%,#0f172a 100%);
    border-left:4px solid #7c3aed; border-radius:0 8px 8px 0;
    padding:1rem 1.4rem; margin:1rem 0;
    color:#c4b5fd; font-style:italic; font-size:.92rem;
  }
  .code-card {
    background:#0f172a; border:1px solid #334155;
    border-radius:10px; padding:1rem 1.2rem; margin:.5rem 0;
  }
  .code-card h4 { color:#f1f5f9; margin:0 0 .4rem 0; font-size:1rem; }
  .code-card p  { color:#94a3b8; margin:0; font-size:.86rem; line-height:1.55; }

  .stTabs [data-baseweb="tab-list"] {
    background:#0f172a; border-radius:8px; padding:4px;
  }
  .stTabs [data-baseweb="tab"] { color:#94a3b8; font-weight:600; }
  .stTabs [aria-selected="true"] {
    color:#7dd3fc !important; background:#0c1e3a !important; border-radius:6px;
  }
  hr { border-color:#1e293b; margin:2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── helper functions ──────────────────────────────────────────────────────────
def doppler_force(v, delta_hz, gamma_hz, s, k):
    """Radiation pressure force in 1D optical molasses (two counter-propagating beams)."""
    F = np.zeros_like(v, dtype=float)
    for sign in [+1, -1]:
        delta_eff = 2 * np.pi * delta_hz - sign * k * v
        F += sign * (hbar * k * gamma_hz / 2 * s) / (1 + s + (2 * delta_eff / (2 * np.pi * gamma_hz))**2)
    return F

def doppler_temp(delta_hz, gamma_hz, s):
    """Equilibrium Doppler cooling temperature [µK]. delta < 0 for cooling."""
    if delta_hz >= 0:
        return np.nan
    d = 2 * np.pi * delta_hz   # convert to rad/s
    g = 2 * np.pi * gamma_hz
    return -hbar * g * (1 + s + 4 * d**2 / g**2) / (8 * kB * d) * 1e6  # µK

def recoil_temp(wavelength_m, mass_kg):
    """Single-photon recoil temperature [µK]: T_r = (ℏk)² / (2m k_B)."""
    k = 2 * np.pi / wavelength_m
    return (hbar * k)**2 / (2 * mass_kg * kB) * 1e6

def sideband_spectrum(freq_arr, omega_trap_hz, gamma_hz, n_mean, eta, Omega_hz=1.0):
    """
    Absorption spectrum near carrier in the Lamb-Dicke regime.
    Shows carrier (0), red sideband (-ω_trap), blue sideband (+ω_trap).
    """
    g = 2 * np.pi * gamma_hz  # FWHM half-width in rad/s
    w_t = 2 * np.pi * omega_trap_hz

    def lorentzian(f, f0, area):
        return area * (g / 2) / (np.pi * ((2 * np.pi * (f - f0))**2 + (g / 2)**2))

    # Carrier: strength ~ e^{-2η²} ≈ 1 for small η
    carrier = lorentzian(freq_arr, 0, np.exp(-2 * eta**2))
    # Red sideband: strength ~ η² · n_mean
    rsb = lorentzian(freq_arr, -omega_trap_hz, eta**2 * n_mean)
    # Blue sideband: strength ~ η² · (n_mean + 1)
    bsb = lorentzian(freq_arr, +omega_trap_hz, eta**2 * (n_mean + 1))
    # Second-order sidebands (small)
    rsb2 = lorentzian(freq_arr, -2 * omega_trap_hz, 0.5 * eta**4 * n_mean**2)
    bsb2 = lorentzian(freq_arr, +2 * omega_trap_hz, 0.5 * eta**4 * (n_mean + 1)**2)

    return carrier + rsb + bsb + rsb2 + bsb2, rsb, bsb, carrier

def sideband_cooling_evolution(n0, gamma_c_hz, n_min, t_max_ms, n_pts=300):
    """Exponential approach to steady state: ⟨n⟩(t) = (n0 - n_min)·exp(-Γ_c·t) + n_min"""
    t = np.linspace(0, t_max_ms * 1e-3, n_pts)
    n = (n0 - n_min) * np.exp(-2 * np.pi * gamma_c_hz * t) + n_min
    return t * 1e3, n  # return t in ms

def n_min_resolved_sideband(gamma_hz, omega_trap_hz):
    """Minimum ⟨n⟩ achievable (resolved RSB limit): (γ/2ω_trap)²"""
    return (gamma_hz / (2 * omega_trap_hz))**2

# ── atom database ─────────────────────────────────────────────────────────────
ATOMS = {
    "⁶Li  D2  (671 nm)":     {"gamma": 5.87e6, "mass": 6 * amu,   "wl": 671e-9, "Isat_mW_cm2": 2.54},
    "⁸⁷Rb D2  (780 nm)":     {"gamma": 6.065e6,"mass": 87 * amu,  "wl": 780e-9, "Isat_mW_cm2": 1.67},
    "¹³³Cs D2  (852 nm)":    {"gamma": 5.234e6,"mass": 133 * amu, "wl": 852e-9, "Isat_mW_cm2": 1.10},
    "²³Na  D2  (589 nm)":    {"gamma": 9.795e6,"mass": 23 * amu,  "wl": 589e-9, "Isat_mW_cm2": 6.26},
    "¹³³Cs 685 nm (quad.)":  {"gamma": 117.6e3,"mass": 133 * amu, "wl": 685e-9, "Isat_mW_cm2": 7.3e-5},
    "⁸⁸Sr  689 nm (narrow)": {"gamma": 7.5e3,  "mass": 88 * amu,  "wl": 689e-9, "Isat_mW_cm2": 3e-6},
    "¹⁷⁴Yb 556 nm (narrow)": {"gamma": 182e3,  "mass": 174 * amu, "wl": 556e-9, "Isat_mW_cm2": 0.14},
}

# ── header ────────────────────────────────────────────────────────────────────
st.title("🧊 Atom Cooling: From Hot Clouds to the Quantum Ground State")
st.markdown("""
<div style='color:#94a3b8; font-size:1rem; margin-bottom:1.5rem;'>
Interactive guide to laser cooling physics — Doppler, sub-Doppler, gray molasses, sideband cooling.
Simulations, step-by-step derivations, existing code libraries, and complete references.<br>
<span style='color:#38bdf8; font-size:.85rem;'>
Based on Phatak (2025) PhD Thesis (Purdue Hood Lab) · Chapters 2, 3, 4
</span>
</div>
""", unsafe_allow_html=True)

# ── temperature ladder ────────────────────────────────────────────────────────
st.markdown("## The Temperature Ladder")

ladder_temps  = [300e6, 1e6, 400, 100, 20, 2, 0.5, 0.1, 0.01]
ladder_labels = [
    "Room temperature (300 K)",
    "Thermal beam (Boltzmann tail, ~1000 K)",
    "Zeeman slower output (~1 mK)",
    "MOT (Doppler limit, ~100–300 µK)",
    "Sub-Doppler / gray molasses (~5–50 µK)",
    "Sideband cooling / EIT (~1–5 µK)",
    "Narrow-line Doppler (Sr 689nm, ~0.5 µK)",
    "Recoil limit (~0.1–0.4 µK)",
    "Motional ground state (⟨n⟩ < 0.1, eff. ~10 nK)",
]
ladder_colors = [
    "#ef4444","#f97316","#f59e0b","#eab308",
    "#84cc16","#22c55e","#14b8a6","#0ea5e9","#6366f1",
]

fig_ladder = go.Figure()
fig_ladder.add_trace(go.Bar(
    x=ladder_temps, y=list(range(len(ladder_temps))),
    orientation="h",
    marker_color=ladder_colors,
    text=ladder_labels,
    textposition="inside",
    insidetextanchor="start",
    textfont=dict(color="#0f172a", size=10, family="Inter"),
    hovertemplate="<b>%{text}</b><br>T ~ %{x:.1e} µK<extra></extra>",
))
fig_ladder.update_layout(
    xaxis=dict(type="log", title="Temperature (µK)", color="#94a3b8",
               gridcolor="#1e293b", range=[-3, 9]),
    yaxis=dict(showticklabels=False, gridcolor="#1e293b"),
    paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0"),
    margin=dict(t=20, b=40, l=10, r=10),
    height=300,
)
st.plotly_chart(fig_ladder, use_container_width=True)

st.markdown("""
<div class='theory-box'>
<h4>Why do we need such cold atoms?</h4>
<p>
Rydberg-gate fidelity in neutral-atom quantum processors scales as <em>F ≈ 1 − α⟨n⟩</em>,
where ⟨n⟩ is the mean motional quantum number.
At a trap frequency of ω<sub>trap</sub> = 2π×100 kHz, the thermal energy
k<sub>B</sub>T = ℏω<sub>trap</sub>⟨n⟩, so ⟨n⟩ = 1 corresponds to T ≈ 4.8 µK.
<strong>A Rydberg π-pulse at 20 µK introduces ~4× more motional error than at 5 µK.</strong>
Cooling to ⟨n⟩ &lt; 0.1 is therefore not academic — it is the margin that high-fidelity gates require.
</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① Doppler Cooling",
    "② Sub-Doppler & Gray Molasses",
    "③ Sideband Cooling",
    "④ Method Comparison",
    "⑤ Simulation Codes",
    "⑥ References",
])

# ══════════════════════════ TAB 1: DOPPLER ════════════════════════════════════
with tab1:
    st.markdown("## Doppler Cooling — Step by Step")

    st.markdown("""
    <div class='theory-box'>
    <h4>Core Idea</h4>
    <p>A moving atom preferentially absorbs photons from the laser beam it is moving toward
    (Doppler shift brings it closer to resonance), then re-emits randomly.
    The absorbed photon gives a kick opposing motion; the emitted photon averages to zero over many cycles.
    Net result: a velocity-dependent damping force — <em>optical molasses</em>.</p>
    </div>
    """, unsafe_allow_html=True)

    col_th, col_int = st.columns([1, 1])

    with col_th:
        st.markdown("### Derivation")
        st.markdown("""
        **Step 1 — Scattering rate of a single beam:**

        For a 2-level atom with linewidth γ (FWHM), saturation parameter s = I/I_sat,
        and detuning Δ = ω_laser − ω_atom:
        """)
        st.markdown("""
        <div class='formula-box'>
# Scattering rate (photons/s):
R_sc(Δ) = (γ/2) · s / [1 + s + (2Δ/γ)²]

# At saturation (s→∞):  R_sc → γ/2  (half the linewidth)
# On resonance (Δ=0):   R_sc = (γ/2) · s/(1+s)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Step 2 — Doppler shift:** An atom moving at velocity v sees an effective detuning:")
        st.markdown("""
        <div class='formula-box'>
Δ_eff = Δ − k·v   (co-propagating beam, k = 2π/λ)
Δ_eff = Δ + k·v   (counter-propagating beam)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Step 3 — Radiation pressure force** from two counter-propagating beams:")
        st.markdown("""
        <div class='formula-box'>
F(v) = ℏk·R_sc(Δ − kv) − ℏk·R_sc(Δ + kv)

# For |kv| ≪ γ (low velocity), Taylor expand:
F(v) ≈ −β · v   (viscous damping)

# Damping coefficient:
β = −4ℏk² s (2Δ/γ) / [1 + s + (2Δ/γ)²]²

# β > 0 (cooling) requires Δ < 0 (red detuning)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Step 4 — Equilibrium temperature** (balance damping vs. diffusion from random recoils):")
        st.markdown("""
        <div class='formula-box'>
# Momentum diffusion: D_p = ℏ²k² · R_sc (two beams)
# Equipartition: (1/2)k_B T = D_p / (2β)

T_Doppler = −ℏγ(1 + s + 4Δ²/γ²) / (8k_B Δ)

# Minimum at Δ = −γ/2  (optimal detuning):
T_D = ℏγ / (2k_B)   ← THE DOPPLER LIMIT

# Lower bound set by photon recoil (not Doppler):
T_recoil = (ℏk)² / (2m k_B)   ← RECOIL LIMIT
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Step 5 — Doppler limits for common atoms:**")
        rows = []
        for name, p in ATOMS.items():
            T_D = hbar * 2*np.pi*p["gamma"] / (2*kB) * 1e6
            T_r = recoil_temp(p["wl"], p["mass"])
            rows.append((name, f"{p['gamma']/1e6:.3f} MHz" if p["gamma"] > 1e5 else f"{p['gamma']/1e3:.1f} kHz",
                         f"{T_D:.1f} µK", f"{T_r:.3f} µK"))
        table_html = """<style>.ct{width:100%;border-collapse:collapse;font-size:.84rem;}
        .ct th{background:#0c1e3a;color:#7dd3fc;padding:.5rem .7rem;text-align:left;border-bottom:2px solid #1e40af;}
        .ct td{color:#e2e8f0;padding:.4rem .7rem;border-bottom:1px solid #1e293b;}</style>
        <table class='ct'><tr><th>Atom / Line</th><th>γ (linewidth)</th><th>T_Doppler</th><th>T_recoil</th></tr>"""
        for r in rows:
            table_html += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("T_Doppler = ℏγ/(2k_B). T_recoil = (ℏk)²/(2mk_B). Sub-Doppler methods can reach T_recoil.")

    with col_int:
        st.markdown("### Interactive: Force Curve F(v)")

        atom_choice = st.selectbox("Atom / Transition", list(ATOMS.keys()), key="dc_atom")
        p = ATOMS[atom_choice]
        gamma_hz = p["gamma"]
        mass_kg  = p["mass"]
        wl_m     = p["wl"]
        k_val    = 2 * np.pi / wl_m

        delta_over_gamma = st.slider(
            "Detuning Δ/γ (negative = red-detuned)", -3.0, -0.1, -0.5, 0.05, key="dc_dg"
        )
        s_val = st.slider("Saturation parameter s = I/I_sat", 0.05, 5.0, 1.0, 0.05, key="dc_s")

        delta_hz  = delta_over_gamma * gamma_hz
        v_max     = 5 * gamma_hz / k_val
        v_arr     = np.linspace(-v_max, v_max, 800)
        F_arr     = doppler_force(v_arr, delta_hz, gamma_hz, s_val, k_val)

        # Capture region velocity
        v_capture = abs(delta_hz) / k_val

        fig_F = go.Figure()
        fig_F.add_trace(go.Scatter(
            x=v_arr * 1e3, y=F_arr * 1e24,
            mode="lines", name="F(v)",
            line=dict(color="#38bdf8", width=2.5),
        ))
        fig_F.add_hline(y=0, line_color="#475569", line_width=1)
        fig_F.add_vline(x=0, line_color="#475569", line_width=1)
        fig_F.update_layout(
            xaxis=dict(title="Velocity (mm/s)", color="#94a3b8", gridcolor="#1e293b", zeroline=False),
            yaxis=dict(title="Force (10⁻²⁴ N)", color="#94a3b8", gridcolor="#1e293b"),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            margin=dict(t=20, b=40), height=260,
            showlegend=False,
            title=dict(text=f"Δ = {delta_over_gamma:.2f}γ  |  s = {s_val:.2f}",
                       font=dict(color="#7dd3fc", size=12)),
        )
        st.plotly_chart(fig_F, use_container_width=True)

        # Compute quantities
        T_eq     = doppler_temp(delta_hz, gamma_hz, s_val)
        T_D_min  = hbar * 2 * np.pi * gamma_hz / (2 * kB) * 1e6
        T_r      = recoil_temp(wl_m, mass_kg)
        beta     = 8 * hbar * k_val**2 * s_val * abs(delta_over_gamma) / \
                   (1 + s_val + 4 * delta_over_gamma**2)**2 * 2 * np.pi * gamma_hz
        tau_damp = mass_kg / beta * 1e3 if beta > 0 else np.inf

        st.markdown(f"""
        <div class='result-box'>
        T_equilibrium = <b>{T_eq:.1f} µK</b> &nbsp;|&nbsp;
        T_Doppler min = <b>{T_D_min:.1f} µK</b> &nbsp;|&nbsp;
        T_recoil = <b>{T_r:.3f} µK</b><br>
        Damping time τ = m/β ≈ <b>{tau_damp:.1f} ms</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### T_Doppler vs Detuning")
        dg_arr   = np.linspace(-3, -0.1, 300)
        T_arr    = [doppler_temp(d * gamma_hz, gamma_hz, s_val) for d in dg_arr]
        T_arr    = np.array(T_arr)

        fig_T = go.Figure()
        fig_T.add_trace(go.Scatter(
            x=dg_arr, y=T_arr,
            mode="lines", line=dict(color="#f59e0b", width=2.5), name="T(Δ)",
        ))
        fig_T.add_hline(y=T_D_min, line_dash="dot", line_color="#10b981",
                        annotation_text=f"T_D min = {T_D_min:.1f} µK",
                        annotation_font_color="#6ee7b7")
        fig_T.add_hline(y=T_r, line_dash="dot", line_color="#818cf8",
                        annotation_text=f"T_recoil = {T_r:.3f} µK",
                        annotation_font_color="#a5b4fc")
        fig_T.add_vline(x=delta_over_gamma, line_dash="dash", line_color="#38bdf8")
        fig_T.update_layout(
            xaxis=dict(title="Δ/γ", color="#94a3b8", gridcolor="#1e293b"),
            yaxis=dict(title="T (µK)", color="#94a3b8", gridcolor="#1e293b",
                       range=[0, min(800, np.nanmax(T_arr) * 1.1)]),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            margin=dict(t=20, b=40), height=240, showlegend=False,
        )
        st.plotly_chart(fig_T, use_container_width=True)

    st.markdown("""
    <div class='warn-box'>
    ⚠️ <strong>The Doppler limit is not the final barrier.</strong>
    For ⁶Li (γ = 2π × 5.87 MHz), T_D = 140 µK — but the recoil temperature is only 3.5 µK.
    Sub-Doppler techniques exploit internal atomic structure to reach T_recoil, and sideband
    cooling on narrow transitions gets you all the way to the motional ground state.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════ TAB 2: SUB-DOPPLER ════════════════════════════════
with tab2:
    st.markdown("## Sub-Doppler Cooling: Sisyphus & Gray Molasses")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Polarization Gradient Cooling (Sisyphus)")
        st.markdown("""
        <div class='step-box'>
        <h5>Step 1 — Spatially varying polarization</h5>
        <p>Two counter-propagating beams with <em>orthogonal linear polarizations</em> (lin⊥lin)
        create a standing wave where polarization rotates from σ⁺ → π → σ⁻ → π → σ⁺
        with period λ/2. The light shift of each magnetic sublevel m_F varies spatially
        because σ⁺ and σ⁻ couple differently to different m_F states.</p>
        </div>
        <div class='step-box'>
        <h5>Step 2 — Optical pumping between hills and valleys</h5>
        <p>An atom in state m_F = +1/2 climbs a potential hill toward a σ⁻ region where
        it is pumped to m_F = −1/2 — which sits at the <em>top</em> of its hill.
        It then climbs again. Every time it reaches the top, optical pumping resets it.
        Like the myth: the atom is always climbing uphill. → Kinetic energy is lost.</p>
        </div>
        <div class='step-box'>
        <h5>Step 3 — Limit</h5>
        <p>The minimum temperature is set by the light shift energy (U₀) at the trap top,
        of order several T_recoil. Sisyphus cooling typically reaches
        T ~ 10–100 × T_recoil, well below T_Doppler.</p>
        </div>
        """, unsafe_allow_html=True)

        # Sisyphus potential illustration
        x = np.linspace(0, 4 * np.pi, 300)
        V_plus  =  np.cos(x)
        V_minus = -np.cos(x)

        fig_sis = go.Figure()
        fig_sis.add_trace(go.Scatter(x=x, y=V_plus,  mode="lines",
            name="m_F = +½", line=dict(color="#38bdf8", width=2.5)))
        fig_sis.add_trace(go.Scatter(x=x, y=V_minus, mode="lines",
            name="m_F = −½", line=dict(color="#f59e0b", width=2.5)))
        # Atom trajectory (always climbing)
        x_atom = [0.2, np.pi/2-0.1, np.pi/2+0.1, 3*np.pi/2-0.1, 3*np.pi/2+0.1, 5*np.pi/2-0.1]
        y_atom = [np.cos(0.2), 1.0, -1.0, 1.0, -1.0, 1.0]
        fig_sis.add_trace(go.Scatter(x=x_atom, y=y_atom, mode="markers",
            marker=dict(size=10, color="#ef4444", symbol="circle"),
            name="atom (climbing)", showlegend=True))
        fig_sis.update_layout(
            xaxis=dict(title="Position (λ units)", tickvals=[0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi],
                       ticktext=["0", "λ/2", "λ", "3λ/2", "2λ"], color="#94a3b8", gridcolor="#1e293b"),
            yaxis=dict(title="Light-shift potential U₀", color="#94a3b8", gridcolor="#1e293b"),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            title=dict(text="Sisyphus (lin⊥lin) potential landscape",
                       font=dict(color="#7dd3fc", size=12)),
            legend=dict(bgcolor="#0f172a", bordercolor="#334155", borderwidth=1),
            margin=dict(t=40, b=40), height=280,
        )
        st.plotly_chart(fig_sis, use_container_width=True)

    with col2:
        st.markdown("### Λ-Enhanced Gray Molasses (for ⁶Li)")

        st.markdown("""
        <div class='theory-box'>
        <h4>Why is ⁶Li hard to cool?</h4>
        <p>
        ⁶Li is the lightest alkali atom (m = 6 amu) — its recoil energy
        E_r = ℏ²k²/(2m) is the highest of any alkali, giving T_recoil = 3.5 µK.
        More critically, the <em>excited-state hyperfine splitting is only 4.4 MHz</em>
        (compare: 500 MHz for Rb, 9.2 GHz for Cs). This means the excited-state hyperfine
        levels completely overlap at typical laser detunings — the <em>lin⊥lin</em>
        Sisyphus mechanism fails because there is no clean cycling transition and photons
        from different hyperfine levels interfere destructively.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='step-box'>
        <h5>The Gray Molasses Solution</h5>
        <p>Use the D1 line (2S₁/₂ → 2P₁/₂) with a <em>Λ configuration</em>: two
        frequency components couple |F=1/2⟩ and |F=3/2⟩ ground states to the same excited
        state. At the two-photon resonance condition δ₁ = δ₂, a coherent dark state forms:</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
|dark⟩ = (Ω₂|F=1/2⟩ − Ω₁|F=3/2⟩) / √(Ω₁² + Ω₂²)

# Atoms in |dark⟩ do NOT scatter photons (CPT — coherent pop. trapping)
# Only atoms in the bright state scatter → heated bright state atoms cool
# Dark state is spatially varying → creates a restoring force

Λ-GM temperature (⁶Li D1):  T ≈ 40–70 µK
# Well below Doppler limit (140 µK) despite unresolved HFS
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='result-box'>
        <strong>Thesis result (Chapter 3):</strong><br>
        Λ-enhanced gray molasses on ⁶Li D1 line achieves:<br>
        • Per-image survival: <b>99.950(2)%</b> over 2000 consecutive 15 ms images<br>
        • State-detection fidelity: <b>99.97(2)%</b><br>
        • Radial ground-state population (Λ-GM pre-cooled): <b>43%</b><br>
        • Temperature: ~70–100 µK after GM (used as pre-cooling for sideband)<br><br>
        <em>First single-atom imaging of ⁶Li in an optical tweezer — enabled entirely by
        the dark-state physics of Λ-GM.</em>
        </div>
        """, unsafe_allow_html=True)

        # Dark state CPT illustration
        angles = np.linspace(0, 2 * np.pi, 500)
        theta = np.linspace(0, 2 * np.pi, 300)

        fig_lambda = go.Figure()
        # Ground state 1
        fig_lambda.add_trace(go.Scatter(
            x=[-1.5, -0.5], y=[0, 0], mode="lines",
            line=dict(color="#38bdf8", width=3), name="|F=1/2⟩ (ground)",
        ))
        # Ground state 2
        fig_lambda.add_trace(go.Scatter(
            x=[0.5, 1.5], y=[0, 0], mode="lines",
            line=dict(color="#f59e0b", width=3), name="|F=3/2⟩ (ground)",
        ))
        # Excited state
        fig_lambda.add_trace(go.Scatter(
            x=[-0.5, 0.5], y=[2, 2], mode="lines",
            line=dict(color="#a78bfa", width=3), name="|e⟩ (excited, D1)",
        ))
        # Transitions
        fig_lambda.add_trace(go.Scatter(
            x=[-1, 0], y=[0, 2], mode="lines",
            line=dict(color="#38bdf8", width=1.5, dash="dash"), showlegend=False,
        ))
        fig_lambda.add_trace(go.Scatter(
            x=[1, 0], y=[0, 2], mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dash"), showlegend=False,
        ))
        fig_lambda.add_annotation(x=-0.55, y=1.05, text="ω₁", showarrow=False,
                                  font=dict(color="#7dd3fc", size=13))
        fig_lambda.add_annotation(x=0.55, y=1.05, text="ω₂", showarrow=False,
                                  font=dict(color="#fcd34d", size=13))
        fig_lambda.add_annotation(x=0, y=2.25, text="|dark⟩ = Ω₂|F=½⟩−Ω₁|F=³⁄₂⟩",
                                  showarrow=False, font=dict(color="#c4b5fd", size=11))
        fig_lambda.update_layout(
            xaxis=dict(visible=False, range=[-2, 2]),
            yaxis=dict(visible=False, range=[-0.5, 2.8]),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            title=dict(text="Λ system — dark state (CPT)", font=dict(color="#7dd3fc", size=12)),
            legend=dict(bgcolor="#0f172a", bordercolor="#334155", borderwidth=1,
                        font=dict(size=10)),
            margin=dict(t=40, b=20), height=260,
        )
        st.plotly_chart(fig_lambda, use_container_width=True)

    st.markdown("""
    <div class='thesis-quote'>
    "The imperfect dark state of ⁶Li that provides a useful fluorescence signal while suppressing
    recoil heating — is also now understood quantitatively through the master-equation model,
    enabling systematic optimisation." — Phatak (2025), §8.5.1
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════ TAB 3: SIDEBAND ═══════════════════════════════════
with tab3:
    st.markdown("## Sideband Cooling — Reaching the Motional Ground State")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### The Harmonic Trap and Motional Sidebands")
        st.markdown("""
        <div class='step-box'>
        <h5>Step 1 — Atom in a harmonic trap</h5>
        <p>An optical tweezer confines the atom in a nearly harmonic potential with trap
        frequency ω_trap. The motional energy is quantized: E_n = ℏω_trap(n + 1/2),
        where n = 0, 1, 2, … is the motional quantum number.
        The <em>motional ground state</em> n = 0 has residual energy ℏω_trap/2 (zero-point motion)
        with RMS position x₀ = √(ℏ/2mω_trap).</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
# Quantized energy levels:
E_n = ℏ·ω_trap·(n + 1/2)

# Zero-point width:
x₀ = √(ℏ / 2mω_trap)

# Lamb-Dicke parameter:
η = k · x₀ = k · √(ℏ / 2mω_trap)
  = (recoil energy / trap energy)^(1/2) / √2

# Lamb-Dicke regime: η ≪ 1
# (atom's position uncertainty ≪ 1/k = λ/2π)
# For Cs at ω_trap=2π×100kHz: η ≈ 0.08 ✓
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='step-box'>
        <h5>Step 2 — Motional sidebands</h5>
        <p>In the Lamb-Dicke regime, the absorption spectrum of the trapped atom shows
        three features: the <strong>carrier</strong> at ω_atom (Δn = 0),
        the <strong>red sideband (RSB)</strong> at ω_atom − ω_trap (Δn = −1),
        and the <strong>blue sideband (BSB)</strong> at ω_atom + ω_trap (Δn = +1).
        The ratio BSB/RSB = (⟨n⟩ + 1)/⟨n⟩ is a direct thermometer.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
# Resolved sideband condition: γ ≪ ω_trap
# (linewidth must be smaller than sideband spacing)

# Cs D2 (γ=2π×5.2 MHz): NOT resolved for ω_trap~100 kHz → Doppler only
# Cs 685 nm (γ=2π×117.6 kHz): MARGINALLY resolved at ω_trap~100 kHz ✓
# Sr 689 nm (γ=2π×7.5 kHz): WELL resolved at any reasonable ω_trap ✓
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='step-box'>
        <h5>Step 3 — Red sideband cooling cycle</h5>
        <p>1. Tune laser to RSB (ω_laser = ω_atom − ω_trap).<br>
        2. Atom absorbs: |n, g⟩ → |n−1, e⟩ (loses one phonon).<br>
        3. Atom decays: |n−1, e⟩ → |n−1, g⟩ (spontaneous emission, recoil adds ~η² phonon on average).<br>
        4. After many cycles: ⟨n⟩ → ⟨n⟩_min = (γ/2ω_trap)².</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
# Cooling rate (weak driving, Lamb-Dicke):
Γ_c = η² · Ω² / γ      [RSB scattering rate × η²]

# Minimum phonon number (resolved RSB limit):
⟨n⟩_min ≈ (γ / 2ω_trap)²

# Time constant: τ_cool = 1/Γ_c

# Example (Cs 685nm, Phatak 2025, Ch. 4):
# γ = 2π×117.6 kHz, ω_trap = 2π×100 kHz
# → ⟨n⟩_min ~ 0.35 (borderline)
# With pulse optimization and higher trap: ⟨n⟩ = 0.01 achieved ✓
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Interactive: Sideband Spectrum")

        sb_atom = st.selectbox("Atom / Transition", list(ATOMS.keys()), index=4, key="sb_atom")
        p_sb = ATOMS[sb_atom]
        gamma_sb = p_sb["gamma"]
        wl_sb    = p_sb["wl"]
        mass_sb  = p_sb["mass"]

        omega_trap_kHz = st.slider("Trap frequency ω_trap (kHz)", 10, 500, 100, 5, key="sb_trap")
        n_mean_sb      = st.slider("Mean phonon number ⟨n⟩", 0.01, 30.0, 5.0, 0.1, key="sb_n")

        omega_trap_hz  = omega_trap_kHz * 1e3
        k_sb           = 2 * np.pi / wl_sb
        x0_sb          = np.sqrt(hbar / (2 * mass_sb * 2 * np.pi * omega_trap_hz))
        eta_sb         = k_sb * x0_sb

        resolved = gamma_sb / omega_trap_hz
        resolved_str = f"{'✅ Resolved' if resolved < 1 else '⚠️ Unresolved'} (γ/ω_trap = {resolved:.2f})"

        # Frequency range for spectrum
        f_range = max(3 * omega_trap_kHz, 5 * gamma_sb / 1e3)
        freq_arr = np.linspace(-f_range * 1.1, f_range * 1.1, 2000)   # in kHz
        freq_hz  = freq_arr * 1e3

        S, rsb_only, bsb_only, carrier_only = sideband_spectrum(
            freq_hz, omega_trap_hz, gamma_sb, n_mean_sb, eta_sb
        )
        # Normalise
        S_norm = S / (np.max(S) + 1e-30)
        rsb_n  = rsb_only / (np.max(S) + 1e-30)
        bsb_n  = bsb_only / (np.max(S) + 1e-30)
        car_n  = carrier_only / (np.max(S) + 1e-30)

        fig_sb = go.Figure()
        fig_sb.add_trace(go.Scatter(x=freq_arr, y=rsb_n, mode="lines",
            fill="tozeroy", name="Red sideband (RSB, Δn=−1)",
            line=dict(color="#ef4444", width=1.5), fillcolor="rgba(239,68,68,0.2)"))
        fig_sb.add_trace(go.Scatter(x=freq_arr, y=car_n, mode="lines",
            fill="tozeroy", name="Carrier (Δn=0)",
            line=dict(color="#94a3b8", width=1.5), fillcolor="rgba(148,163,184,0.15)"))
        fig_sb.add_trace(go.Scatter(x=freq_arr, y=bsb_n, mode="lines",
            fill="tozeroy", name="Blue sideband (BSB, Δn=+1)",
            line=dict(color="#3b82f6", width=1.5), fillcolor="rgba(59,130,246,0.2)"))
        fig_sb.add_vline(x=-omega_trap_kHz, line_dash="dot", line_color="#ef4444",
                         annotation_text="RSB", annotation_font_color="#fca5a5")
        fig_sb.add_vline(x=+omega_trap_kHz, line_dash="dot", line_color="#3b82f6",
                         annotation_text="BSB", annotation_font_color="#93c5fd")
        fig_sb.update_layout(
            xaxis=dict(title="Frequency offset from carrier (kHz)",
                       color="#94a3b8", gridcolor="#1e293b", zeroline=True, zerolinecolor="#475569"),
            yaxis=dict(title="Absorption (norm.)", color="#94a3b8", gridcolor="#1e293b"),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            legend=dict(bgcolor="#0f172a", bordercolor="#334155", borderwidth=1, font=dict(size=10)),
            margin=dict(t=20, b=40), height=280,
        )
        st.plotly_chart(fig_sb, use_container_width=True)

        bsb_rsb_ratio = (n_mean_sb + 1) / max(n_mean_sb, 0.001)
        T_motional = hbar * 2 * np.pi * omega_trap_hz * n_mean_sb / kB * 1e6

        st.markdown(f"""
        <div class='result-box'>
        η (Lamb-Dicke) = <b>{eta_sb:.4f}</b> &nbsp;|&nbsp; {resolved_str}<br>
        BSB/RSB ratio = <b>{bsb_rsb_ratio:.2f}</b> → thermometry ⟨n⟩ = 1/(ratio−1) = {1/(bsb_rsb_ratio-1+1e-6):.2f}<br>
        Motional temperature = <b>{T_motional:.2f} µK</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Interactive: Cooling Evolution ⟨n⟩(t)")

        n0_cool     = st.slider("Initial ⟨n⟩₀ (before cooling)", 1.0, 50.0, 20.0, 0.5, key="cool_n0")
        Omega_kHz   = st.slider("Drive Rabi freq. Ω (kHz, RSB beam)", 1.0, 100.0, 20.0, 1.0, key="cool_omega")
        t_max_ms    = st.slider("Cooling duration (ms)", 1, 200, 50, 1, key="cool_tmax")

        Omega_hz_val = Omega_kHz * 1e3
        Gamma_c_hz   = eta_sb**2 * (2 * np.pi * Omega_hz_val)**2 / (2 * np.pi * gamma_sb)
        n_min_val    = n_min_resolved_sideband(gamma_sb, omega_trap_hz)

        t_cool, n_cool = sideband_cooling_evolution(n0_cool, Gamma_c_hz, n_min_val, t_max_ms)

        fig_cool = go.Figure()
        fig_cool.add_trace(go.Scatter(x=t_cool, y=n_cool, mode="lines",
            line=dict(color="#38bdf8", width=2.5), name="⟨n⟩(t)"))
        fig_cool.add_hline(y=n_min_val, line_dash="dot", line_color="#6ee7b7",
                           annotation_text=f"⟨n⟩_min = {n_min_val:.3f}",
                           annotation_font_color="#6ee7b7")
        fig_cool.add_hline(y=0.1, line_dash="dot", line_color="#fcd34d",
                           annotation_text="n=0.1 target",
                           annotation_font_color="#fcd34d")
        fig_cool.update_layout(
            xaxis=dict(title="Time (ms)", color="#94a3b8", gridcolor="#1e293b"),
            yaxis=dict(title="Mean phonon number ⟨n⟩",
                       color="#94a3b8", gridcolor="#1e293b"),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            showlegend=False,
            margin=dict(t=20, b=40), height=260,
        )
        st.plotly_chart(fig_cool, use_container_width=True)

        tau_cool_ms = 1 / (Gamma_c_hz + 1e-6) * 1e3
        st.markdown(f"""
        <div class='result-box'>
        Cooling rate Γ_c = η²Ω²/γ = <b>{Gamma_c_hz:.0f} Hz</b><br>
        Time constant τ = 1/Γ_c = <b>{tau_cool_ms:.1f} ms</b><br>
        Steady-state ⟨n⟩_min = (γ/2ω_trap)² = <b>{n_min_val:.4f}</b><br>
        Ground-state population (n=0): <b>{(1/(n_min_val+1)) * 100:.1f}%</b> at steady state
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='thesis-quote'>
    "Chapter 4 demonstrates exactly this for ¹³³Cs: single-photon sideband cooling on the
    6S₁/₂ → 5D₅/₂ electric-quadrupole transition cools a single atom to ⟨n⟩ ≈ 0.01
    (~99% ground-state population) in a 1.1 mK tweezer. This is achieved using a single beam
    and the intrinsic narrowness of the 685 nm transition (γ ≈ 2π×117.6 kHz)."
    — Phatak (2025), §8.5.2
    </div>
    """, unsafe_allow_html=True)

    # EIT section
    st.markdown("### EIT Cooling (Electromagnetically Induced Transparency)")
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("""
        <div class='step-box'>
        <h5>Concept</h5>
        <p>In a Λ system, a strong coupling laser creates a transparency window — a dark
        resonance — in the probe absorption profile. If the transparency window is
        positioned at the <em>carrier frequency</em> and the probe laser drives from
        the carrier to the red sideband, the atom only absorbs on the RSB (phonon removal)
        and cannot absorb on the carrier (no random recoil from carrier scattering).
        This makes EIT cooling more efficient than standard sideband cooling.</p>
        </div>
        <div class='formula-box'>
# EIT dark resonance position:
ω_dark = ω_coupling - δ_two_photon

# Cooling condition: place dark resonance at carrier
# → absorption maximum is at RSB

# Ground state population at EIT steady state:
⟨n⟩_ss^EIT < ⟨n⟩_ss^RSB  (can reach below η²/4)

# Used for: Ca⁺ ions (Roos et al. PRL 2000),
#           ⁸⁷Rb in tweezers, alkaline-earth atoms
        </div>
        """, unsafe_allow_html=True)
    with c2:
        # EIT transparency window sketch
        w = np.linspace(-3, 3, 600)
        Omega_EIT_norm = 0.8
        absorption_EIT = (w**2) / ((w**2 - Omega_EIT_norm**2/4)**2 + (0.3 * w)**2)
        absorption_EIT = absorption_EIT / np.max(absorption_EIT)

        fig_eit = go.Figure()
        fig_eit.add_trace(go.Scatter(x=w, y=absorption_EIT, mode="lines",
            fill="tozeroy", name="EIT absorption profile",
            line=dict(color="#a78bfa", width=2.5), fillcolor="rgba(167,139,250,0.15)"))
        fig_eit.add_vline(x=0, line_dash="dot", line_color="#6ee7b7",
                          annotation_text="Dark resonance\n(no absorption)", annotation_font_color="#6ee7b7")
        fig_eit.add_vline(x=-1, line_dash="dot", line_color="#ef4444",
                          annotation_text="RSB\n(max absorption)", annotation_font_color="#fca5a5")
        fig_eit.update_layout(
            xaxis=dict(title="Detuning (units of Ω_coupling)", color="#94a3b8", gridcolor="#1e293b"),
            yaxis=dict(title="Absorption (norm.)", color="#94a3b8", gridcolor="#1e293b"),
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), showlegend=False,
            title=dict(text="EIT absorption profile (schematic)", font=dict(color="#7dd3fc", size=12)),
            margin=dict(t=40, b=40), height=260,
        )
        st.plotly_chart(fig_eit, use_container_width=True)


# ══════════════════════════ TAB 4: COMPARISON ════════════════════════════════
with tab4:
    st.markdown("## All Cooling Methods — Side-by-Side")

    comparison_data = [
        ("Doppler (broad)",      "D-line (Rb, Cs, Li)", "> T_Doppler",  "~100–240 µK", "> 10",   "Simple, robust, no resolved HFS needed"),
        ("Doppler (narrow)",     "Sr 689nm, Yb 556nm",  "> T_Doppler",  "< 1 µK",      "1–10",   "Narrow intercombination lines; near recoil limit"),
        ("Sisyphus (lin⊥lin)",   "Multi-level atoms",   "few × T_recoil","1–20 µK",    "5–50",   "Requires resolved HFS; fails for ⁶Li on D2"),
        ("Gray Molasses (Λ-GM)", "⁶Li D1, ⁴⁰K D1",     "few × T_recoil","10–60 µK",   "3–20",   "Dark states; works for unresolved HFS (thesis Ch. 3)"),
        ("RSB cooling (broad)",  "Cs D2, Rb D2",        "η² ≪ 1 but γ/ω large","5–30 µK","1–5",  "Not resolved; limited to high trap freq."),
        ("RSB cooling (narrow)", "Cs 685nm, Sr 689nm",  "γ/2ω_trap ≪ 1","0.1–2 µK",  "0.01–0.5","Best for single-atom QIS (thesis Ch. 4)"),
        ("EIT cooling",          "Ca⁺, Rb in Λ",        "below η²/4",   "< 0.5 µK",   "< 0.1",  "Suppressed carrier absorption; near motional GS"),
        ("Raman sideband",       "Sr, Cs, Rb in lattice","η² limit",     "< 1 µK",     "~0.01",  "Two-photon RSB via Raman beams; lattice geometry"),
    ]

    table_h = """<style>.cmpt{width:100%;border-collapse:collapse;font-size:.83rem;}
    .cmpt th{background:#0c1e3a;color:#7dd3fc;padding:.5rem .7rem;text-align:left;border-bottom:2px solid #1e40af;}
    .cmpt td{color:#e2e8f0;padding:.4rem .7rem;border-bottom:1px solid #1e293b;}
    .cmpt tr:nth-child(6) td{color:#bbf7d0;font-weight:600;} /* RSB narrow = thesis */
    </style>
    <table class='cmpt'>
    <tr><th>Method</th><th>Atoms / Lines</th><th>Condition</th><th>T achievable</th><th>⟨n⟩_ss</th><th>Notes</th></tr>"""
    for row in comparison_data:
        table_h += f"<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    table_h += "</table>"
    st.markdown(table_h, unsafe_allow_html=True)
    st.markdown("Highlighted row (green) = method used for Cs in thesis Chapter 4.")

    st.markdown("### Temperature Comparison Chart")

    methods    = [r[0] for r in comparison_data]
    T_low      = [240, 1,   20,  60, 30,  2,   0.5, 1]
    T_high     = [500, 5,  100, 150, 80, 10,   2.0, 3]
    bar_colors = ["#ef4444","#f59e0b","#f59e0b","#22c55e",
                  "#22c55e","#38bdf8","#6366f1","#8b5cf6"]

    fig_comp = go.Figure()
    for i, (m, tl, th, col) in enumerate(zip(methods, T_low, T_high, bar_colors)):
        fig_comp.add_trace(go.Bar(
            x=[th - tl], y=[m], base=[tl],
            orientation="h",
            marker_color=col, marker_opacity=0.8,
            name=m, showlegend=False,
            hovertemplate=f"<b>{m}</b><br>T: {tl}–{th} µK<extra></extra>",
        ))
    fig_comp.update_layout(
        xaxis=dict(type="log", title="Temperature (µK)", color="#94a3b8",
                   gridcolor="#1e293b", range=[-2, 3.2]),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        barmode="overlay",
        margin=dict(t=10, b=40, l=200, r=10), height=380,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### Which Cooling Method for Which Atom?")
    atom_guide = {
        "⁶Li": ["MOT (D2) → Λ-GM (D1) → Raman RSB or axis-selective pulsed cooling",
                 "Gray molasses essential — D2 Sisyphus fails (unresolved HFS). Thesis Ch. 3."],
        "¹³³Cs": ["MOT (D2) → PGC → Narrow-line RSB on 685 nm quadrupole transition",
                  "Magic wavelength tweezer. 685nm gives γ = 2π×117.6kHz — just resolved at ω_trap~100kHz. Thesis Ch. 4."],
        "⁸⁷Rb": ["MOT (D2) → Sisyphus (lin⊥lin) → RSB (D2 Raman) or EIT",
                  "Best-studied. Raman RSB in lattice commonly reaches ⟨n⟩ < 0.1."],
        "⁸⁸Sr": ["MOT (461nm broad) → Gray molasses (689nm) → RSB on 689nm",
                  "689nm: γ=2π×7.5kHz, easily resolved. Near-ideal for sideband cooling."],
        "¹⁷¹Yb": ["MOT (399nm) → GM (556nm) → Clock RSB (578nm, γ=10mHz!)",
                   "Nuclear spin qubit. 556nm: γ=2π×182kHz. Clock line RSB: basically zero linewidth."],
    }
    cols_ag = st.columns(len(atom_guide))
    for col, (atom, (path, notes)) in zip(cols_ag, atom_guide.items()):
        with col:
            st.markdown(f"""
            <div class='step-box'>
            <h5>{atom}</h5>
            <p><strong>Path:</strong> {path}<br><br>{notes}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════ TAB 5: CODES ═════════════════════════════════════
with tab5:
    st.markdown("## Simulation Codes & Libraries for Laser Cooling")

    codes = [
        {
            "name": "pylcp — Python Laser Cooling Physics",
            "org": "Eckel Group, NIST (Steve Eckel, Eric Norrgard et al.)",
            "url": "https://github.com/JQIamo/pylcp",
            "docs": "https://python-laser-cooling-physics.readthedocs.io",
            "lang": "Python",
            "desc": """The most general open-source library for simulating laser cooling forces on atoms.
                Computes radiation pressure forces, equilibrium temperatures, velocity distributions,
                and force curves for arbitrary polarization configurations, multi-level atoms, and
                arbitrary laser geometries. Based on solving the optical Bloch equations (OBE) or
                rate equations for a full multi-level Hamiltonian.""",
            "features": [
                "Full hyperfine structure (arbitrary F, m_F)",
                "Arbitrary laser polarizations and geometries",
                "Doppler, Sisyphus, gray molasses regimes",
                "Force curves, velocity distributions, trap depths",
                "Rate-equation and OBE solvers",
                "Optimized for multi-level Li, Na, K, Rb, Cs",
            ],
            "example": """import pylcp
from pylcp.hamiltonians import Rb87_D2
from pylcp.fields import MagneticField, LaserBeams

# Build laser field (optical molasses along x)
beams = LaserBeams.molasses(delta=-1.0, s=1.0, axis='x')
B = MagneticField.uniform([0, 0, 1e-4])  # 0.1 mT bias

# Build Hamiltonian + OBE
H = Rb87_D2()
obe = pylcp.obe(H, beams, B)

# Calculate force vs velocity
v = np.linspace(-0.5, 0.5, 100)
F = obe.force(v)""",
            "color": "#38bdf8",
        },
        {
            "name": "QuTiP — Quantum Toolbox in Python",
            "org": "Open-source (Nathan Shammah, Shahnawaz Ahmed, et al.)",
            "url": "https://qutip.org",
            "docs": "https://qutip.readthedocs.io",
            "lang": "Python",
            "desc": """The standard framework for simulating open quantum systems.
                Solves the Lindblad master equation (mesolve) — the same formalism
                used throughout this thesis for the generalized cooling theory (Chapter 2 & Appendix B).
                Handles arbitrary Hamiltonians and collapse operators.""",
            "features": [
                "mesolve: Lindblad master equation (time evolution)",
                "steadystate: find ρ_ss without time evolution",
                "floquet: Floquet theory for periodic drives",
                "mcsolve: Monte Carlo quantum trajectories",
                "Wigner functions, Q-functions (phase space)",
                "Spectrum calculations via correlation functions",
            ],
            "example": """import qutip as qt

# 2-level atom + harmonic trap (JC-like model)
N = 20  # motional Fock states
a = qt.tensor(qt.destroy(N), qt.qeye(2))   # phonon annihilation
sm = qt.tensor(qt.qeye(N), qt.sigmam())     # atomic lowering

omega_trap = 2*np.pi * 100e3  # 100 kHz trap
gamma = 2*np.pi * 117.6e3     # Cs 685nm linewidth
eta = 0.08                     # Lamb-Dicke parameter

# RSB drive Hamiltonian (rotating wave approx.)
Omega = 2*np.pi * 20e3  # drive Rabi freq
H = omega_trap * a.dag()*a + Omega * eta * (a * sm.dag() + a.dag() * sm)

# Collapse operators: spontaneous emission
c_ops = [np.sqrt(gamma) * sm]

# Initial state: thermal at n_bar=10
psi0 = qt.tensor(qt.thermal_dm(N, 10), qt.fock_dm(2, 0))

# Time evolution
tlist = np.linspace(0, 50e-3, 500)  # 0–50 ms
result = qt.mesolve(H, psi0, tlist, c_ops, [a.dag()*a])
n_mean = result.expect[0]""",
            "color": "#6ee7b7",
        },
        {
            "name": "ARC — Alkali Rydberg Calculator",
            "org": "Nikola Šibalić, Durham University",
            "url": "https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator",
            "docs": "https://arc-alkali-rydberg-calculator.readthedocs.io",
            "lang": "Python",
            "desc": """Computes atomic properties for alkali and divalent atoms: energy levels,
                matrix elements, polarizabilities, C₃/C₆ van der Waals coefficients,
                Rydberg blockade radii, magic wavelengths, and AC Stark shifts.
                Essential companion for computing tweezer trap depths, light shifts,
                and Rydberg gate parameters.""",
            "features": [
                "All alkali atoms + Rb, Cs, Sr, Ca, Ba, Yb",
                "Electric dipole / quadrupole matrix elements",
                "Dynamic polarizabilities α(ω) for magic wavelength search",
                "Rydberg C₃, C₆ interaction coefficients",
                "Pair potential curves and Förster resonances",
                "Blockade radius r_b(n, Ω_Ryd)",
            ],
            "example": """from arc import Caesium

Cs = Caesium()

# Cs 685nm transition: 6S_1/2 -> 5D_5/2
n, l, j = 6, 0, 0.5
n2, l2, j2 = 5, 2, 2.5

# Transition dipole matrix element
dme = Cs.getRadialMatrixElement(n, l, j, n2, l2, j2)
print(f"DME = {dme:.4f} a₀·e")

# Magic wavelength search
alpha_g = Cs.getPolarizability(6, 0, 0.5, wavelength=1064e-9)
alpha_e = Cs.getPolarizability(6, 1, 1.5, wavelength=1064e-9)
print(f"Δα(1064nm) = {alpha_g - alpha_e:.2f} a₀³")

# Rydberg C6 coefficient
C6 = Cs.getC6term(60, 0, 0.5, 60, 0, 0.5)
print(f"C6(n=60) = {C6:.2e} GHz·µm⁶")""",
            "color": "#f59e0b",
        },
        {
            "name": "Steck Quantum & Atom Optics Notes",
            "org": "Daniel A. Steck, University of Oregon",
            "url": "http://steck.us/teaching",
            "docs": "http://steck.us/alkalidata",
            "lang": "PDF + data files",
            "desc": """The definitive reference for alkali atom data (Li, Na, K, Rb, Cs, Fr).
                Comprehensive tables of wavelengths, linewidths, hyperfine constants,
                Clebsch-Gordan coefficients, polarizabilities, and recoil temperatures.
                The textbook 'Quantum and Atom Optics' covers laser cooling theory from
                first principles through sideband cooling.""",
            "features": [
                "Li, Na, K, Rb-85/87, Cs, Fr data sheets (updated 2023)",
                "Hyperfine constants, Zeeman structure",
                "D1/D2 matrix elements and oscillator strengths",
                "Recoil velocities and temperatures",
                "Comprehensive Quantum & Atom Optics textbook (free PDF)",
                "Used directly in thesis (reference [43])",
            ],
            "example": """# Data from Steck Cs-133 D2 data sheet:
# γ(D2) = 2π × 5.2347 MHz
# T_Doppler = 125.0 µK
# T_recoil  = 198.3 nK (D2 line)
# I_sat     = 1.1023 mW/cm² (σ⁺ transition)
# Hyperfine F=4 → F'=5: cycling transition

# Steck Cs-133 quadrupole transition data:
# γ(6S1/2 → 5D5/2, 685nm) = 2π × 117.6 kHz [Phatak et al. 2024]
# I_sat(685nm) ≈ 7.3×10⁻⁵ mW/cm²""",
            "color": "#a78bfa",
        },
        {
            "name": "QuantumOptics.jl",
            "org": "David Plankensteiner, University of Innsbruck (Julia lang)",
            "url": "https://github.com/qojulia/QuantumOptics.jl",
            "docs": "https://docs.qojulia.org",
            "lang": "Julia",
            "desc": """Julia-language quantum optics framework — similar scope to QuTiP but
                significantly faster for large Hilbert spaces (important for large motional
                Fock state bases). Time-correlated emission, cavity QED, many-body systems.
                Actively developed and used in European AMO groups.""",
            "features": [
                "Master equation (Lindblad) solver",
                "Monte Carlo wave functions (MCWF)",
                "Stochastic Schrödinger equation",
                "Large Fock space (N up to ~1000)",
                "10–100× faster than QuTiP for large N",
                "Cavity QED, optomechanics, spin systems",
            ],
            "example": """using QuantumOptics

# Sideband cooling in Julia (fast for large N)
N = 50  # motional Fock space
basis_motional = FockBasis(N)
basis_atom = SpinBasis(1//2)
basis = tensor(basis_motional, basis_atom)

a  = tensor(destroy(basis_motional), one(basis_atom))
sm = tensor(one(basis_motional), sigmam(basis_atom))

omega_trap = 2π * 100e3  # Hz
gamma_cs   = 2π * 117.6e3
eta = 0.08; Omega = 2π * 20e3

H = omega_trap * dagger(a)*a + Omega*eta*(a*dagger(sm) + dagger(a)*sm)
J = [sqrt(gamma_cs) * sm]  # jump operators

psi0 = tensor(thermalstate(FockBasis(N), 10), dm(spindown(basis_atom)))
tspan = [0, 0.05]  # 0 to 50 ms
tout, rho_t = timeevolution.master(tspan, psi0, H, J)""",
            "color": "#ec4899",
        },
        {
            "name": "MOLSCAT / BOUND",
            "org": "Jeremy Hutson, Durham University",
            "url": "https://github.com/molscat/molscat",
            "docs": "https://molscat.readthedocs.io",
            "lang": "Fortran / Python interface",
            "desc": """The standard code for quantum scattering calculations in ultracold atomic physics.
                Computes molecular interaction potentials, scattering cross sections, and
                Feshbach resonance positions — essential for cold-collision physics and
                molecule assembly (LiCs in this thesis' future work, Ch. 7).""",
            "features": [
                "Close-coupling scattering calculations",
                "Feshbach resonance positions and widths",
                "Molecular bound states (BOUND module)",
                "Ultracold collision cross sections",
                "LiCs, RbCs, KRb, NaK potentials included",
                "Used for LiCs molecule assembly planning",
            ],
            "example": """# Example: LiCs Feshbach resonance search
# (future work in thesis Chapter 7)

# MOLSCAT input file excerpt (NAMELIST format):
# &INPUT
#   NSTEPS=1000, JTOTL=0, JTOTU=0
#   MZERO=.true., RMIN=3.0, RMAX=100.0
#   MASS1=6.015, MASS2=132.905
#   ITYPE=6   ! singlet+triplet LiCs potential
#   BFIELD=0.0 to 2000.0 Gauss
# /
# Output: resonance near B=889 G for LiCs""",
            "color": "#14b8a6",
        },
    ]

    for code in codes:
        with st.expander(f"**{code['name']}**  ({code['lang']})"):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"""
                <div class='code-card'>
                <h4><a href='{code["url"]}' target='_blank' style='color:{code["color"]};text-decoration:none;'>
                GitHub ↗</a>
                &nbsp;|&nbsp;
                <a href='{code["docs"]}' target='_blank' style='color:#94a3b8;text-decoration:none;'>
                Docs ↗</a>
                &nbsp;|&nbsp; {code["org"]}</h4>
                <p>{code["desc"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Key features:**")
                for feat in code["features"]:
                    st.markdown(f"- {feat}")
            with c2:
                st.code(code["example"], language="python")


# ══════════════════════════ TAB 6: REFERENCES ════════════════════════════════
with tab6:
    st.markdown("## References")

    refs = {
        "Foundational Papers": [
            ("[Chu et al. 1985] Observation of sub-millikelvin temperatures by laser cooling",
             "https://link.aps.org/doi/10.1103/PhysRevLett.55.48",
             "*PRL* 55, 48 — First laser cooling demonstration by Steve Chu's group. Nobel Prize 1997."),
            ("[Lett et al. 1988] Observation of atoms laser cooled below the Doppler limit",
             "https://link.aps.org/doi/10.1103/PhysRevLett.61.169",
             "*PRL* 61, 169 — Discovery of sub-Doppler temperatures, triggering explanation via Sisyphus effect."),
            ("[Dalibard & Cohen-Tannoudji 1989] Laser cooling below the Doppler limit by polarization gradients",
             "https://opg.optica.org/josab/abstract.cfm?URI=josab-6-11-2023",
             "*JOSA B* 6, 2023 — Theoretical explanation of Sisyphus cooling. Nobel Prize 1997."),
            ("[Wineland & Itano 1979] Laser cooling of atoms",
             "https://link.aps.org/doi/10.1103/PhysRevA.20.1521",
             "*PRA* 20, 1521 — First theoretical treatment of sideband cooling in traps."),
            ("[Monroe et al. 1995] Resolved-sideband Raman cooling to the 3D zero-point energy",
             "https://link.aps.org/doi/10.1103/PhysRevLett.75.4011",
             "*PRL* 75, 4011 — First ground-state sideband cooling demonstration (trapped ion)."),
            ("[Morigi et al. 2000] Ground state laser cooling using EIT",
             "https://link.aps.org/doi/10.1103/PhysRevLett.85.4458",
             "*PRL* 85, 4458 — EIT cooling theory."),
        ],
        "Gray Molasses & ⁶Li": [
            ("[Grier et al. 2013] Λ-enhanced sub-Doppler cooling of ⁶Li atoms in D1 gray molasses",
             "https://link.aps.org/doi/10.1103/PhysRevA.87.063411",
             "*PRA* 87, 063411 — First Λ-GM implementation for ⁶Li. Foundational for thesis Chapter 3."),
            ("[Blodgett, Phatak et al. 2023] Imaging a ⁶Li atom in an optical tweezer 2000 times",
             "https://link.aps.org/doi/10.1103/PhysRevLett.131.083001",
             "*PRL* 131, 083001 — **This thesis's first paper.** 99.95% per-image survival of single Li atom."),
            ("[Ang'ong'a et al. 2022] Gray molasses cooling of ³⁹K atoms in optical tweezers",
             "https://link.aps.org/doi/10.1103/PhysRevResearch.4.013240",
             "*PRR* 4, 013240 — GM in tweezers (K atoms), closely related technique."),
        ],
        "Narrow-Line & Sideband Cooling": [
            ("[Berto et al. 2021] Prospects for single-photon sideband cooling of optically trapped neutral atoms",
             "https://link.aps.org/doi/10.1103/PhysRevResearch.3.043106",
             "*PRR* 3, 043106 — Theory paper motivating the 685nm Cs approach in the thesis."),
            ("[Blodgett, Phatak et al. 2025] Narrow-line electric quadrupole cooling of a single Cs atom",
             "https://arxiv.org/abs/2505.10540",
             "*arXiv:2505.10540* — **This thesis's second paper.** ⟨n⟩ ≈ 0.01 for Cs at 685nm."),
            ("[Phatak et al. 2024] Generalized theory for optical cooling of a trapped atom with spin",
             "https://link.aps.org/doi/10.1103/PhysRevA.110.043116",
             "*PRA* 110, 043116 — **This thesis's theory paper.** Unified Lindblad cooling formalism."),
            ("[Norcia et al. 2018] Microscopic control of Sr in optical tweezer arrays",
             "https://link.aps.org/doi/10.1103/PhysRevX.8.041054",
             "*PRX* 8, 041054 — Sr tweezer array using 689nm narrow line."),
            ("[Urech et al. 2022] Narrow-line imaging of single Sr in shallow tweezers",
             "https://link.aps.org/doi/10.1103/PhysRevResearch.4.023245",
             "*PRR* 4, 023245 — Related narrow-line tweezer work."),
        ],
        "Textbooks & Reviews": [
            ("[Metcalf & van der Straten] Laser Cooling and Trapping (textbook, Springer 1999)",
             "https://link.springer.com/book/9780387987286",
             "The standard textbook. Chapters 9–11 on sub-Doppler and sideband cooling."),
            ("[Steck] Quantum and Atom Optics (free online, 2023)",
             "http://steck.us/teaching",
             "Comprehensive free textbook covering all cooling methods from first principles."),
            ("[Kaufman & Ni 2021] Quantum science with optical tweezer arrays (review)",
             "https://arxiv.org/abs/2109.10087",
             "*Annu. Rev. Phys. Chem.* 72 — The definitive tweezer array review article."),
            ("[Leibfried et al. 2003] Quantum dynamics of single trapped ions (review)",
             "https://link.aps.org/doi/10.1103/RevModPhys.75.281",
             "*Rev. Mod. Phys.* 75, 281 — Comprehensive trapped-ion cooling review. Directly applies to neutral atoms."),
        ],
        "Simulation Codes": [
            ("[Johansson et al. 2012] QuTiP: An open-source Python framework for dynamics of OQS",
             "https://www.sciencedirect.com/science/article/pii/S0010465512000835",
             "*Comp. Phys. Comm.* 183, 1760 — QuTiP paper (reference [46] in thesis)."),
            ("[Norrgard et al. 2021] pylcp: A Python package for simulating laser cooling physics",
             "https://github.com/JQIamo/pylcp",
             "GitHub + documentation at readthedocs.io. NIST/JQI group."),
            ("[Šibalić et al. 2017] ARC: An open-source library for calculating properties of alkali Rydberg atoms",
             "https://www.sciencedirect.com/science/article/pii/S0010465516303836",
             "*Comp. Phys. Comm.* 220, 319 — ARC paper."),
            ("[Plankensteiner et al. 2019] QuantumOptics.jl: A Julia framework for simulating open quantum systems",
             "https://www.sciencedirect.com/science/article/pii/S0010465518304284",
             "*Comp. Phys. Comm.* 237, 66."),
        ],
    }

    for section, entries in refs.items():
        st.markdown(f"### {section}")
        for (title, url, desc) in entries:
            st.markdown(f"""
            <div class='step-box'>
            <h5><a href='{url}' target='_blank' style='color:#7dd3fc;text-decoration:none;'>{title} ↗</a></h5>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='color:#475569; font-size:.82rem; text-align:center; padding:.5rem 0;'>
Built for <a href='https://curious96.com' style='color:#38bdf8;'>curious96.com</a> ·
Interactive cooling simulations based on Phatak (2025) PhD Thesis, Purdue University Hood Lab ·
Formulas: Dalibard & Cohen-Tannoudji (1989), Wineland & Itano (1979), Metcalf & van der Straten (1999)
</div>
""", unsafe_allow_html=True)
