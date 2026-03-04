"""
Laser Frequency Locking
========================
Tutorial on the three main laser frequency stabilisation techniques
in AMO experiments: saturated-absorption spectroscopy (SAS), beat-note
(offset) locking, and Pound–Drever–Hall (PDH) cavity locking.
"""

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Laser Frequency Locking",
    page_icon="🔒",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family:'Inter',sans-serif; }
  .main  { background:#0a0a0f; color:#e2e8f0; }
  .block-container { padding:2rem 3rem; max-width:1400px; }
  h1  { font-size:2.2rem; font-weight:700; color:#f0f4ff;
        border-bottom:2px solid #0ea5e9; padding-bottom:.5rem; }
  h2  { color:#7dd3fc; font-size:1.4rem; }
  h3  { color:#38bdf8; font-size:1.05rem; margin-top:1.4rem; }

  .result-row {
    background:#052e16; border:1px solid #15803d;
    border-radius:8px; padding:.7rem 1.1rem; margin-top:.6rem;
    display:flex; flex-wrap:wrap; gap:.6rem 2rem; align-items:baseline;
  }
  .result-val { color:#4ade80; font-size:1.2rem; font-weight:700; font-family:monospace; }
  .result-lbl { color:#86efac; font-size:.82rem; }

  .formula { background:#0a1f0a; border:1px solid #166534;
    border-radius:6px; padding:.55rem 1rem; margin:.6rem 0;
    font-family:'Courier New',monospace; color:#86efac; font-size:.83rem; line-height:1.8; }

  .theory-box { background:#0a0a1f; border:1px solid #1e3a5f;
    border-radius:8px; padding:1rem 1.4rem; margin:.8rem 0;
    color:#cbd5e1; font-size:.9rem; line-height:1.7; }

  .warn { background:#1c1007; border:1px solid #92400e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#fde68a; font-size:.85rem; }

  .good { background:#052e16; border:1px solid #15803d;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#86efac; font-size:.85rem; }

  .info-box { background:#0a1020; border:1px solid #1e3a6e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#93c5fd; font-size:.85rem; }

  .step-box { background:#12101a; border-left:3px solid #7c3aed;
    border-radius:0 6px 6px 0; padding:.7rem 1.1rem; margin:.5rem 0;
    color:#ddd6fe; font-size:.88rem; line-height:1.6; }

  .summary-table { width:100%; border-collapse:collapse; margin-top:1rem; }
  .summary-table th { background:#0f172a; color:#7dd3fc;
    padding:.6rem 1rem; text-align:left; border:1px solid #1e293b; }
  .summary-table td { background:#0a0a0f; color:#cbd5e1;
    padding:.55rem 1rem; border:1px solid #1e293b; font-size:.87rem; }
  .summary-table tr:hover td { background:#0f172a; }

  hr { border-color:#1e293b; margin:1.5rem 0; }
  .stTabs [data-baseweb="tab-list"]  { background:#0f172a; border-radius:8px; padding:4px; }
  .stTabs [data-baseweb="tab"]       { color:#94a3b8; font-weight:600; }
  .stTabs [aria-selected="true"]     { color:#7dd3fc !important; background:#0c1e3a !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
kB   = 1.380649e-23
c    = 299_792_458.0
h    = 6.62607015e-34
amu  = 1.66053906660e-27
ln2  = math.log(2)

# ─── Atom data ────────────────────────────────────────────────────────────────
ATOMS = {
    "Cs D₂  (852 nm)": dict(lam_nm=852.347, Gamma_MHz=5.234, A=133),
    "Rb D₂  (780 nm)": dict(lam_nm=780.241, Gamma_MHz=6.065, A=87),
    "Na D₂  (589 nm)": dict(lam_nm=589.000, Gamma_MHz=9.795, A=23),
    "⁶Li D₂ (671 nm)": dict(lam_nm=670.977, Gamma_MHz=5.874, A=6),
    "⁴⁰K D₂ (767 nm)": dict(lam_nm=766.700, Gamma_MHz=6.035, A=40),
}

# ─── Spacer materials ─────────────────────────────────────────────────────────
MATERIALS = {
    "ULE glass (near zero-crossing)": 5e-9,   # 5 ppb/K — typical operating point
    "ULE glass (room temp)":           2e-8,
    "Fused silica":                    5.5e-7,
    "Zerodur":                         5e-9,
    "Invar":                           1e-6,
}

# ─── Display helpers ──────────────────────────────────────────────────────────
def result_box(*items):
    pairs = "".join(
        f'<span class="result-lbl">{lbl}</span>&nbsp;<span class="result-val">{val}</span>'
        for lbl, val in items
    )
    st.markdown(f'<div class="result-row">{pairs}</div>', unsafe_allow_html=True)

def formula(s):
    st.markdown(f'<div class="formula">{s}</div>', unsafe_allow_html=True)

def theory_box(s):
    st.markdown(f'<div class="theory-box">{s}</div>', unsafe_allow_html=True)

def step_box(s):
    st.markdown(f'<div class="step-box">{s}</div>', unsafe_allow_html=True)

def good(s):
    st.markdown(f'<div class="good">✅ {s}</div>', unsafe_allow_html=True)

def warn(s):
    st.markdown(f'<div class="warn">⚠️ {s}</div>', unsafe_allow_html=True)

def info(s):
    st.markdown(f'<div class="info-box">ℹ️ {s}</div>', unsafe_allow_html=True)

# ─── Physics helpers ──────────────────────────────────────────────────────────
def doppler_fwhm_MHz(lam_nm, T_K, A):
    lam = lam_nm * 1e-9
    m   = A * amu
    return (1 / lam) * math.sqrt(8 * kB * T_K * ln2 / m) / 1e6

def fmt_freq(Hz):
    """Auto-format frequency in Hz → Hz/kHz/MHz/GHz."""
    if abs(Hz) >= 1e9:   return f"{Hz/1e9:.3f} GHz"
    if abs(Hz) >= 1e6:   return f"{Hz/1e6:.3f} MHz"
    if abs(Hz) >= 1e3:   return f"{Hz/1e3:.3f} kHz"
    return f"{Hz:.1f} Hz"

# ─── Plotly dark theme helper ─────────────────────────────────────────────────
PLOTLY_DARK = dict(
    plot_bgcolor="#0a0a0f",
    paper_bgcolor="#0a0a0f",
    font=dict(color="#cbd5e1", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
)

# ─── SAS signal simulation ────────────────────────────────────────────────────
def plot_sas(lam_nm, Gamma_MHz, T_K, A):
    lam = lam_nm * 1e-9
    Gamma = Gamma_MHz * 1e6
    dnu_D_Hz = doppler_fwhm_MHz(lam_nm, T_K, A) * 1e6
    sigma_D  = dnu_D_Hz / (2 * math.sqrt(2 * ln2))

    # Frequency axis: ±2 Doppler widths around line centre
    nu = np.linspace(-2 * dnu_D_Hz, 2 * dnu_D_Hz, 8000)

    # Doppler-broadened absorption (Gaussian)
    alpha_D = 0.45 * np.exp(-(nu**2) / (2 * sigma_D**2))

    # Lamb dip at line centre (Lorentzian, contrast ~ 30% of Doppler peak)
    kap = Gamma / 2
    alpha_lamb = 0.45 * 0.35 * kap**2 / (nu**2 + kap**2)

    S = 1.0 - alpha_D + alpha_lamb       # SAS transmission

    # Derivative (FM error signal, normalised)
    dS = np.gradient(S, nu)
    dS /= np.max(np.abs(dS))

    nu_MHz = nu / 1e6

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=["Saturated absorption signal", "FM error signal  (d𝑆/d𝜈)"],
    )
    fig.add_trace(go.Scatter(x=nu_MHz, y=S, name="SAS transmission",
                             line=dict(color="#0ea5e9", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=nu_MHz, y=dS, name="Error signal",
                             line=dict(color="#f59e0b", width=1.5)), row=2, col=1)

    for row in [1, 2]:
        fig.add_vline(x=0, line_dash="dash", line_color="#64748b",
                      annotation_text="line centre", row=row, col=1)

    fig.update_layout(
        **PLOTLY_DARK, height=420,
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text="Detuning from line centre (MHz)", row=2, col=1)
    fig.update_yaxes(title_text="Transmission", row=1, col=1)
    fig.update_yaxes(title_text="Error (norm.)", row=2, col=1)
    return fig

# ─── PDH error signal simulation ─────────────────────────────────────────────
def plot_pdh(fsr_GHz, finesse):
    kappa_Hz = fsr_GHz * 1e9 / finesse       # cavity FWHM linewidth
    kappa_half = kappa_Hz / 2                 # half-width

    delta = np.linspace(-6 * kappa_Hz, 6 * kappa_Hz, 3000)

    # PDH error signal: ∝ Im[r(δ)] → dispersive Lorentzian
    V_err = -(delta / kappa_half) / (1 + (delta / kappa_half)**2)
    V_err /= np.max(np.abs(V_err))

    # Cavity reflection power (Airy function, impedance-matched)
    # |r|² = 1 - 1/(1 + (2F/π × sin(πδ/FSR))²)  ≈ for small δ:
    V_refl = 1 - 1 / (1 + (delta / kappa_half)**2)

    delta_kHz = delta / 1e3

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=["Cavity reflection (Airy dip)", "PDH error signal"],
    )
    fig.add_trace(go.Scatter(x=delta_kHz, y=V_refl, name="Reflected power",
                             line=dict(color="#0ea5e9", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=delta_kHz, y=V_err, name="PDH error",
                             line=dict(color="#f59e0b", width=2)), row=2, col=1)

    fig.add_vline(x=0, line_dash="dash", line_color="#64748b",
                  annotation_text="resonance", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#475569", row=2, col=1)

    # Mark half-linewidth
    fig.add_vline(x=-kappa_Hz / 1e3, line_dash="dot", line_color="#7c3aed", row=2, col=1)
    fig.add_vline(x=+kappa_Hz / 1e3, line_dash="dot", line_color="#7c3aed", row=2, col=1)

    fig.update_layout(
        **PLOTLY_DARK, height=420,
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text=f"Detuning from resonance (kHz)", row=2, col=1)
    fig.update_yaxes(title_text="Reflected power", row=1, col=1)
    fig.update_yaxes(title_text="Error (norm.)", row=2, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔒 Laser Frequency Locking")
st.markdown(
    "Free-running diode lasers drift by **MHz on minute timescales** — "
    "comparable to or larger than the natural linewidths of the transitions they address. "
    "Three complementary locking techniques, each matched to a different need, solve this."
)

info(
    "Running example throughout: the Cs D₂ line at 852 nm, Γ/2π = 5.23 MHz. "
    "The Cs electric-quadrupole 685 nm line (Γ/2π ≈ 1/(2π×1.36 µs) ≈ 117 kHz) "
    "is used to motivate the need for PDH cavity locking."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Saturated-Absorption Lock",
    "🎵 Beat-Note (Offset) Lock",
    "🏛️ PDH Cavity Lock",
    "🗺️ Summary & Hierarchy",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — SAS locking
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Saturated-Absorption Spectroscopy (SAS) Locking")

    theory_box(
        "SAS locking provides an <b>absolute optical frequency reference</b> tied directly "
        "to an atomic transition — no external frequency standard needed. "
        "The technique exploits the narrow Lamb dip hidden beneath the broad Doppler-broadened "
        "absorption profile of a room-temperature vapour cell."
    )

    st.markdown("### How it works")

    col_a, col_b = st.columns(2)
    with col_a:
        step_box(
            "<b>Step 1 — Counter-propagating beams.</b><br>"
            "A strong <em>pump</em> beam and weak <em>probe</em> beam travel in opposite "
            "directions through the vapour cell. Because they are counter-propagating, "
            "an atom moving at velocity v along the beam axis sees the pump "
            "Doppler-shifted to ν₀ − v/λ and the probe to ν₀ + v/λ."
        )
        step_box(
            "<b>Step 2 — Velocity selectivity.</b><br>"
            "Only atoms with v ≈ 0 (zero-velocity class) are simultaneously resonant "
            "with <em>both</em> beams at the unshifted line centre ν₀. "
            "The pump saturates these atoms, burning a hole in the population inversion."
        )
    with col_b:
        step_box(
            "<b>Step 3 — Lamb dip.</b><br>"
            "Because the zero-velocity atoms are partially saturated by the pump, "
            "they absorb less of the probe at ν₀ → a narrow <em>Lamb dip</em> "
            "appears in the probe transmission, sitting on the broad Gaussian Doppler background."
        )
        step_box(
            "<b>Step 4 — FM error signal.</b><br>"
            "The laser frequency is weakly modulated (via current or EOM). "
            "Lock-in demodulation of the probe signal yields a dispersive "
            "derivative of the Lamb dip, zero-crossing at ν₀."
        )

    formula(
        "V_err  ∝  dS(ν)/dν      [first-harmonic demodulation]\n\n"
        "Lamb dip FWHM  ≈  Γ/2π  (power-broadened: × √(1 + I/Iₛₐₜ))\n"
        "Doppler FWHM   =  (ν₀/c) √(8 k_B T ln2 / m)"
    )

    st.markdown("---")
    st.markdown("### Calculator — Doppler vs. Lamb dip widths")

    col1, col2 = st.columns(2)
    with col1:
        species = st.selectbox("Species", list(ATOMS.keys()), key="sas_sp")
        T_K = st.slider("Vapour cell temperature (K)", 20, 500, 300, 10, key="sas_T")
    with col2:
        sat_factor = st.slider(
            "Saturation parameter  I / Iₛₐₜ  (power broadening)", 0.0, 10.0, 1.0, 0.1, key="sas_sat"
        )
        mod_freq = st.slider("FM modulation frequency (MHz)", 0.1, 50.0, 5.0, 0.1, key="sas_mod")

    a = ATOMS[species]
    dnu_D   = doppler_fwhm_MHz(a["lam_nm"], T_K, a["A"])
    dnu_nat = a["Gamma_MHz"]
    dnu_lamb = dnu_nat * math.sqrt(1 + sat_factor)   # power-broadened Lamb dip
    contrast = dnu_D / dnu_nat

    result_box(
        ("Doppler FWHM:", f"{dnu_D:.0f} MHz"),
        ("Natural linewidth Γ/2π:", f"{dnu_nat:.3f} MHz"),
        ("Lamb dip FWHM (power-broadened):", f"{dnu_lamb:.3f} MHz"),
        ("Contrast ratio (Doppler / natural):", f"{contrast:.0f} : 1"),
    )

    if mod_freq < 0.05 * dnu_lamb:
        good(f"Modulation frequency {mod_freq} MHz ≪ Lamb dip FWHM {dnu_lamb:.2f} MHz — pure FM regime, good error signal slope.")
    elif mod_freq < dnu_lamb:
        info(f"Modulation frequency {mod_freq} MHz < Lamb dip FWHM {dnu_lamb:.2f} MHz — within feature, signal slightly reduced.")
    else:
        warn(f"Modulation frequency {mod_freq} MHz > Lamb dip FWHM {dnu_lamb:.2f} MHz — sidebands outside the feature, reduced slope.")

    st.markdown("### Simulated SAS spectrum")
    st.plotly_chart(
        plot_sas(a["lam_nm"], a["Gamma_MHz"], T_K, a["A"]),
        use_container_width=True,
    )

    with st.expander("📋 Practical notes"):
        st.markdown("""
- **Vapour cell temperature**: room temperature (20–60 °C) gives good SNR for alkali D lines.
  Higher temperature → stronger signal but broader Doppler background and higher collisional broadening.
- **Pump power**: needs I ≳ Iₛₐₜ to produce a visible Lamb dip. Excess power broadens the dip.
- **Modulation frequency**: typically 1–50 MHz. Higher frequencies push technical noise below the shot-noise floor.
- **Cross-over resonances**: at ν = (ν₁ + ν₂)/2 between two transitions, both zero-velocity and ±v classes contribute — these appear as additional dips, often the sharpest features in a multi-level spectrum.
- **SAS is impractical** for very weak transitions (E2, M1) where I_sat is W/cm² — the Lamb dip signal is too small to lock to in a room-temperature cell.
""")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Beat-note locking
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Beat-Note (Offset) Locking")

    theory_box(
        "Many experiments need two lasers separated by a <b>precise and stable frequency offset</b> "
        "— for example, a cooling laser and a repumper, or a Raman beam pair. "
        "Rather than locking each laser independently to an atomic reference, "
        "beat-note locking stabilises the <em>difference frequency</em> between a "
        "well-stabilised <b>master</b> laser and a <b>slave</b> laser."
    )

    st.markdown("### Physics of the beat note")

    theory_box(
        "Overlapping two optical fields on a fast photodiode produces a "
        "photocurrent at their <b>difference frequency</b>. For fields:"
    )
    formula(
        "E₁(t) = E₀₁ cos(2π ν₁ t + φ₁)\n"
        "E₂(t) = E₀₂ cos(2π ν₂ t + φ₂)\n\n"
        "→  i_PD(t)  ∝  cos[2π(ν₂ − ν₁)t + (φ₂ − φ₁)]"
    )
    theory_box(
        "This RF beat signal carries <em>both</em> the frequency difference Δν = ν₂ − ν₁ "
        "and the relative optical phase. Comparing it to a stable RF reference ν_ref gives "
        "an error signal:"
    )
    formula(
        "V_err  ∝  (ν₂ − ν₁) − ν_ref        [frequency-discriminator implementation]\n\n"
        "In a full optical phase-locked loop (OPLL) the phase difference is locked:\n"
        "  d/dt [φ_beat − φ_ref] = 0  →  ν₂ − ν₁  ≡  ν_ref  exactly."
    )

    st.markdown("### Calculator — Beat-note frequency & detector requirements")

    col1, col2 = st.columns(2)
    with col1:
        master_lam = st.number_input(
            "Master laser wavelength (nm)", 400.0, 1200.0, 670.977, 0.001, key="bn_master"
        )
        offset_MHz = st.number_input(
            "Desired frequency offset (MHz)  [slave − master]",
            -5000.0, 5000.0, 228.2, 0.1, key="bn_offset",
        )
    with col2:
        st.markdown("&nbsp;")
        show_phase_lock = st.checkbox("OPLL mode (phase lock, not frequency lock)", value=False)

    nu_master = c / (master_lam * 1e-9)
    nu_slave  = nu_master + offset_MHz * 1e6
    lam_slave = c / nu_slave * 1e9

    result_box(
        ("Master frequency ν₁:", f"{nu_master/1e12:.6f} THz"),
        ("Slave frequency ν₂:", f"{nu_slave/1e12:.6f} THz"),
        ("Slave wavelength λ₂:", f"{lam_slave:.4f} nm"),
        ("Beat frequency |Δν|:", fmt_freq(abs(offset_MHz * 1e6))),
    )

    req_bw = abs(offset_MHz) * 1.5
    info(
        f"Photodetector bandwidth required: ≳ {req_bw:.0f} MHz to resolve the beat note. "
        f"{'Phase lock requires carrier tracking; servo bandwidth must exceed laser linewidth.' if show_phase_lock else 'Frequency lock is more forgiving — a few MHz servo bandwidth suffices for most diode lasers.'}"
    )

    if abs(offset_MHz) > 3000:
        warn("Offset > 3 GHz is large — may exceed the bandwidth of standard fast photodiodes. Consider using a wavemeter or Fabry–Pérot to first pre-stabilise both lasers.")
    elif abs(offset_MHz) < 10:
        warn("Very small offset (< 10 MHz) — the beat note may fall within 1/f noise of the detector. Try using a double-pass AOM arrangement to increase the offset.")

    st.markdown("---")
    st.markdown("### RF signal chain (typical)")
    formula(
        "Slave + Master  →  fast PD (1–3 GHz BW)  →  amplifier\n"
        "→  RF power splitter  →  mixer × local oscillator (RF synthesiser)\n"
        "→  low-pass filter  →  error signal  →  PID  →  slave laser current / PZT"
    )

    with st.expander("📋 Practical notes"):
        st.markdown("""
- **Master must already be well-locked** (SAS or PDH). The slave inherits the master's long-term stability.
- **RF synthesiser sets the offset**: changing ν_ref tunes the slave frequency without touching the master.
- **OPLL vs. frequency lock**: phase-locked loops are technically harder (require ~1 MHz servo bandwidth)
  but provide phase coherence — essential for Raman spectroscopy and atom interferometry.
- **Signal-to-noise**: beat SNR = 10 log₁₀(P₁ P₂ R² / (2 e I_dc B)) where R is detector responsivity.
  You typically need ≳ 20 dBm RF power at the detector to lock cleanly.
- **Vescent D2-125**: widely used electronics that handle both SAS and beat-note locking from a single unit.
- **Limitation**: requires one well-stabilised master. Cannot provide an absolute reference on its own.
""")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — PDH cavity locking
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Pound–Drever–Hall (PDH) Cavity Locking")

    theory_box(
        "When no convenient atomic reference exists — or when the required short-term linewidth "
        "is narrower than SAS can provide — the laser is locked to a <b>high-finesse "
        "Fabry–Pérot cavity</b>. The cavity acts as a passive frequency discriminator; "
        "the laser is forced to track one of its resonant modes. The PDH technique generates "
        "an error signal from the <em>reflected</em> field using RF phase modulation, "
        "achieving much higher signal slope than simple transmission locking."
    )

    st.markdown("### Cavity basics")
    formula(
        "Free spectral range:     ν_FSR = c / (2 n L)\n"
        "Finesse:                 𝒻 = π √R / (1 − R)  ≈  π / (1 − R)   [for R ≈ 1]\n"
        "Cavity linewidth (FWHM): δν_cav = ν_FSR / 𝒻\n"
        "Fractional freq. drift:  δν / ν  =  −δL / L"
    )

    st.markdown("### PDH error signal — how it works")

    step_box(
        "<b>Step 1 — Phase modulation.</b>  "
        "An EOM imprints weak sidebands at ±Ω on the carrier field:<br>"
        "E_in ≈ E₀ e^{iωt} + (β/2)E₀ e^{i(ω+Ω)t} − (β/2)E₀ e^{i(ω−Ω)t}   [β ≪ 1]"
    )
    step_box(
        "<b>Step 2 — Frequency placement.</b>  "
        "Choose Ω ≫ δν_cav so the sidebands sit <em>outside</em> the cavity resonance. "
        "They reflect unchanged from the input mirror. "
        "The carrier acquires a strongly frequency-dependent phase shift near resonance."
    )
    step_box(
        "<b>Step 3 — Reflection detection and demodulation.</b>  "
        "The reflected field is detected on a fast photodiode. "
        "The carrier beats against the sidebands at Ω. "
        "Demodulating at Ω and low-pass filtering yields a bipolar error signal:"
    )
    formula(
        "V_err  ∝  Im[r(ω)]      [proportional to imaginary part of reflection coefficient]\n\n"
        "Near resonance:  V_err ≈ K · δ      where  δ = ω − ω_cavity\n\n"
        "Discriminator slope:  K ∝ √(P_c · P_s) · 𝒻  /  δν_cav\n"
        "  (P_c = carrier power, P_s ≈ β²P_c/4 = sideband power)"
    )
    theory_box(
        "The error signal is <b>linear</b> near resonance and changes sign on either side "
        "— exactly what a servo needs. The slope K grows with finesse, so "
        "higher-finesse cavities give steeper error signals and tighter locks, "
        "at the cost of a smaller capture range (∝ δν_cav)."
    )

    st.markdown("---")
    st.markdown("### Calculator — cavity parameters")

    col1, col2 = st.columns(2)
    with col1:
        L_mm      = st.number_input("Cavity length L (mm)", 1.0, 500.0, 77.5, 0.5, key="pdh_L")
        finesse   = st.number_input("Finesse 𝒻", 100.0, 1e6, 1.5e4, 100.0, key="pdh_F",
                                    format="%.0f")
        lam_nm_pdh = st.number_input("Laser wavelength (nm)", 400.0, 1200.0, 685.0, 0.1, key="pdh_lam")
    with col2:
        Omega_MHz  = st.number_input("EOM modulation frequency Ω (MHz)", 1.0, 500.0, 27.0, 1.0, key="pdh_Om")
        target_lw  = st.number_input("Target laser linewidth (Hz)", 100.0, 1e6, 1000.0, 100.0, key="pdh_target")
        mat        = st.selectbox("Spacer material (thermal expansion)", list(MATERIALS.keys()), key="pdh_mat")

    L     = L_mm * 1e-3
    nu0   = c / (lam_nm_pdh * 1e-9)
    fsr   = c / (2 * L)
    dnu_c = fsr / finesse
    alpha = MATERIALS[mat]

    # Fractional stability needed
    frac_stab = target_lw / nu0
    delta_L_pm = frac_stab * L * 1e12              # pm
    # Drift: δν = ν × α × δT  →  δT for δν = target_lw
    dT_mK = target_lw / (nu0 * alpha) * 1e3        # mK

    result_box(
        ("FSR:", fmt_freq(fsr)),
        ("Cavity linewidth δν_cav:", fmt_freq(dnu_c)),
        ("Ω / δν_cav:", f"{Omega_MHz*1e6/dnu_c:.1f}"),
    )
    result_box(
        ("Required δL/L for target linewidth:", f"{frac_stab:.2e}"),
        ("Absolute length stability:", f"{delta_L_pm:.2f} pm"),
        ("Max ΔT for target linewidth:", f"{dT_mK:.2f} mK"),
    )

    if Omega_MHz * 1e6 < 5 * dnu_c:
        warn(f"Ω = {Omega_MHz} MHz is only {Omega_MHz*1e6/dnu_c:.1f} × δν_cav. "
             f"Sidebands are inside the cavity linewidth — error signal will be degraded. "
             f"Increase Ω to at least {5*dnu_c/1e6:.0f} MHz.")
    else:
        good(f"Ω / δν_cav = {Omega_MHz*1e6/dnu_c:.0f} ≫ 1 — sidebands are well outside the cavity. ✓")

    st.markdown(f"### PDH error signal shape (cavity linewidth = {fmt_freq(dnu_c)})")
    st.plotly_chart(
        plot_pdh(fsr / 1e9, finesse),
        use_container_width=True,
    )

    st.markdown("### Typical optical chain")
    formula(
        "diode laser  →  prism pair (astigmatism)  →  optical isolator (30–40 dB)\n"
        "→  EOM (phase mod. at Ω)  →  mode-matching telescope\n"
        "→  ULE cavity\n"
        "Reflected beam:  PBS + QWP  →  fast PD  →  RF demodulation at Ω\n"
        "Error signal:  fast path → laser current  (high BW, small range)\n"
        "              slow path → PZT             (low BW, large range)"
    )

    with st.expander("📋 Practical notes"):
        st.markdown(f"""
- **ULE zero-crossing temperature**: ULE has CTE ≈ 0 near a specific temperature (typically 5–25 °C
  depending on the blank). Operating at the zero-crossing dramatically reduces thermal drift.
- **Vacuum and vibration isolation**: the cavity must be in vacuum (P < 10⁻⁵ mbar) to avoid
  refractive-index fluctuations from air. Sit it on a vibration-isolated platform.
- **Two-stage servo**: fast path (current) compensates high-frequency noise; slow path (PZT)
  compensates slow drift and prevents integrator windup on the fast path.
- **Capture range** ≈ several × δν_cav ≈ {fmt_freq(5*dnu_c)}.
  Pre-lock the laser to within this range using a wavemeter or the Fabry–Pérot transmission.
- **Finesse measurement**: scan the laser across a resonance and fit to an Airy function;
  or measure decay time τ_c = 𝒻/(π × FSR) of the ring-down after rapidly switching off the input.
- **PDH does not provide absolute frequency**: the cavity resonance drifts.
  For absolute knowledge, beat the locked laser against a frequency comb or another absolutely-referenced laser.
""")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Summary: The Locking Hierarchy")

    theory_box(
        "In practice, these three techniques form a <b>hierarchy</b> that covers "
        "the full frequency-stabilisation needs of an AMO experiment. "
        "SAS provides the absolute anchor; beat-note locks propagate stability to other "
        "lasers at controllable offsets; PDH locks provide narrow-linewidth operation "
        "wherever no atomic reference is available."
    )

    st.markdown("""
<table class="summary-table">
  <thead>
    <tr>
      <th>Technique</th>
      <th>Absolute reference?</th>
      <th>Typical linewidth</th>
      <th>Tunable offset?</th>
      <th>Best for</th>
      <th>Limitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>SAS locking</b></td>
      <td>✅ Yes (atomic line)</td>
      <td>100 kHz – 1 MHz</td>
      <td>❌ Fixed to transition</td>
      <td>Primary absolute reference (D lines)</td>
      <td>Needs strong transition; only works for accessible vapour-cell lines</td>
    </tr>
    <tr>
      <td><b>Beat-note lock</b></td>
      <td>Via master laser</td>
      <td>Same as master</td>
      <td>✅ Yes (RF synthesiser)</td>
      <td>Cooling/repump pairs, Raman beams</td>
      <td>Needs a pre-stabilised master; requires fast PD + RF chain</td>
    </tr>
    <tr>
      <td><b>PDH cavity lock</b></td>
      <td>❌ (cavity drifts)</td>
      <td>1 Hz – 10 kHz</td>
      <td>Via AOM after lock</td>
      <td>Narrow-linewidth spectroscopy, weak transitions</td>
      <td>Cavity drifts thermally; expensive; needs vacuum + isolation</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown("### Typical laboratory hierarchy")
    formula(
        "Layer 0  (Absolute):  D-line laser  →  SAS lock  →  ν known to ~1 MHz\n\n"
        "Layer 1  (Derived):   Laser B  →  beat-note lock to Layer 0  →  ν₀ ± ν_RF\n"
        "                      Laser C  →  beat-note lock to Layer 0  →  ν₀ ± ν_RF'\n\n"
        "Layer 2  (Narrow):    Spectroscopy laser  →  PDH lock to ULE cavity  →  ~kHz linewidth\n"
        "                      (Long-term drift corrected by beating against a Layer 0 laser)"
    )

    st.markdown("### Key equations at a glance")
    formula(
        "SAS:        V_err  ∝  dS(ν)/dν,    Lamb dip FWHM ≈ Γ/2π × √(1 + I/Iₛₐₜ)\n\n"
        "Beat-note:  i_PD   ∝  cos[2π(ν₂−ν₁)t],   V_err ∝ (ν₂−ν₁) − ν_ref\n\n"
        "PDH:        V_err  ∝  Im[r(ω)]  ≈  K·δ,\n"
        "            δν_cav = c/(2LF),   δν/ν = −δL/L = −α δT"
    )

    with st.expander("📐 Error signal slopes compared"):
        st.markdown("""
| Technique | Slope K (typical) | Notes |
|---|---|---|
| SAS (direct lock-in) | ~0.1–1 mV/MHz | Limited by Doppler background contrast |
| SAS (modulation transfer) | ~1–10 mV/MHz | Better baseline; uses four-wave mixing |
| Beat-note (frequency disc.) | ~1–10 mV/MHz | Scales with RF power and mixer conversion gain |
| PDH (high finesse) | ~10–1000 mV/MHz | Scales as √(P_c P_s) × 𝒻 / δν_cav |

Larger K → tighter lock for a given servo gain, smaller in-loop noise.
""")

    with st.expander("🔬 Our lab example: Cs and Li system"):
        st.markdown("""
- **852 nm Cs D₂**: SAS-locked in Cs vapour cell → primary absolute reference.
- **685 nm Cs E₂ (6S→5D₅/₂)**: PDH-locked to ULE cavity (L = 77.5 mm, 𝒻 ≈ 1.5×10⁴).
  FSR ≈ 1.93 GHz, δν_cav ≈ 130 kHz, laser linewidth ≈ 1 kHz.
  Thermal drift ≈ 2.5 kHz per 10 mK.
- **671 nm Li D₁**: SAS-locked in Li vapour cell.
- **671 nm Li D₂**: Beat-note locked to Li D₁ (Vescent D2-125), offset set by RF synthesiser.
""")
