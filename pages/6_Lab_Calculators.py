"""
AMO Lab Calculators
===================
Quick-reference calculators for everyday experimental AMO physics.

Tabs
----
1. Optics & Beams    – Gaussian beam from fiber, telescope, NA, waist
2. Power & RF        – mW / dBm / dB, AOM deflection
3. Atomic Physics    – Recoil, Doppler, Zeeman, de Broglie, sat. intensity
4. Trap & Cavity     – Trap frequencies, cavity FSR/finesse, mode matching
"""

import math
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="AMO Lab Calculators",
    page_icon="🧮",
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

  .calc-card {
    background:#0f172a; border:1px solid #1e293b;
    border-radius:10px; padding:1.2rem 1.4rem; margin:.7rem 0;
  }
  .calc-card h4 { color:#f1f5f9; margin:0 0 .1rem 0; font-size:1rem; }
  .calc-card .sub { color:#64748b; font-size:.82rem; margin-bottom:.9rem; }

  .result-row {
    background:#052e16; border:1px solid #15803d;
    border-radius:8px; padding:.7rem 1.1rem; margin-top:.6rem;
    display:flex; flex-wrap:wrap; gap:.6rem 2rem; align-items:baseline;
  }
  .result-val { color:#4ade80; font-size:1.25rem; font-weight:700; font-family:monospace; }
  .result-lbl { color:#86efac; font-size:.82rem; }

  .formula { background:#0a1f0a; border:1px solid #166534;
    border-radius:6px; padding:.55rem 1rem; margin:.6rem 0;
    font-family:'Courier New',monospace; color:#86efac; font-size:.83rem; line-height:1.7; }

  .warn { background:#1c1007; border:1px solid #92400e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#fde68a; font-size:.85rem; }

  hr { border-color:#1e293b; margin:1.5rem 0; }

  .stTabs [data-baseweb="tab-list"]  { background:#0f172a; border-radius:8px; padding:4px; }
  .stTabs [data-baseweb="tab"]       { color:#94a3b8; font-weight:600; }
  .stTabs [aria-selected="true"]     { color:#7dd3fc !important; background:#0c1e3a !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
h    = 6.62607015e-34
hbar = 1.054571817e-34
c    = 299_792_458.0
kB   = 1.380649e-23
amu  = 1.66053906660e-27
muB  = 9.2740100783e-24   # Bohr magneton [J/T]
eps0 = 8.8541878128e-12

def nm(x):  return x * 1e-9
def um(x):  return x * 1e-6
def mm(x):  return x * 1e-3
def MHz(x): return x * 1e6

# ─── Reusable display helpers ─────────────────────────────────────────────────
def result_box(*items):
    """items = (label, value_str) pairs"""
    pairs = "".join(
        f'<span class="result-lbl">{lbl}</span>&nbsp;<span class="result-val">{val}</span>'
        for lbl, val in items
    )
    st.markdown(f'<div class="result-row">{pairs}</div>', unsafe_allow_html=True)

def formula(s):
    st.markdown(f'<div class="formula">{s}</div>', unsafe_allow_html=True)

def warn(s):
    st.markdown(f'<div class="warn">⚠️ {s}</div>', unsafe_allow_html=True)

def card(title, subtitle=""):
    st.markdown(f'<div class="calc-card"><h4>{title}</h4>'
                f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🧮 AMO Lab Calculators")
st.markdown("""
<div style='color:#94a3b8; font-size:1rem; margin-bottom:1.5rem;'>
Quick-reference calculators for everyday experimental AMO physics.
Results update instantly as you adjust inputs.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔭 Optics & Beams",
    "📡 Power & RF",
    "⚛️ Atomic Physics",
    "🪤 Trap & Cavity",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPTICS & BEAMS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Optics & Beam Calculations")

    col1, col2 = st.columns(2, gap="large")

    # ── 1a. Gaussian beam from fiber ──────────────────────────────────────────
    with col1:
        st.markdown("### Fiber → Collimated → Focused Spot")
        st.markdown("""
<div class="calc-card">
<h4>Beam waist from SMF + asphere</h4>
<div class="sub">Given the fiber MFD and collimating lens focal length, computes the
collimated beam radius and (optionally) the focused spot after a second lens.</div>
""", unsafe_allow_html=True)

        lam_fib = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="fib_lam")
        mfd     = st.number_input("Fiber MFD (µm)", 1.0, 50.0, 5.3, 0.1, key="fib_mfd",
                                   help="Mode Field Diameter from fiber spec sheet. MFR = MFD/2.")
        f1      = st.number_input("Collimating lens f₁ (mm)", 1.0, 200.0, 11.0, 0.5, key="fib_f1",
                                   help="Focal length of the asphere or collimating lens.")

        lam_m  = nm(lam_fib)
        mfr_m  = um(mfd) / 2.0          # Mode field radius
        f1_m   = mm(f1)
        # Gaussian beam optics: w_col = f × λ / (π × w_fiber)
        w_col  = f1_m * lam_m / (math.pi * mfr_m)

        formula("w_col = f₁ · λ / (π · MFR)")
        result_box(
            ("Collimated 1/e² radius", f"{w_col*1e3:.3f} mm"),
            ("Full 1/e² diameter", f"{2*w_col*1e3:.3f} mm"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Optional: focus with a second lens**")
        use_f2 = st.checkbox("Add focusing lens f₂", key="fib_use_f2")
        if use_f2:
            f2 = st.number_input("Focusing lens f₂ (mm)", 1.0, 1000.0, 200.0, 1.0, key="fib_f2")
            f2_m  = mm(f2)
            w_foc = f2_m * lam_m / (math.pi * w_col)
            zR    = math.pi * w_foc**2 / lam_m
            formula("w_foc = f₂ · λ / (π · w_col)")
            result_box(
                ("Focused 1/e² waist", f"{w_foc*1e6:.3f} µm"),
                ("Rayleigh range z_R", f"{zR*1e3:.2f} mm"),
            )

    # ── 1b. NA ↔ Beam waist ───────────────────────────────────────────────────
    with col2:
        st.markdown("### NA ↔ Gaussian Beam Waist")
        st.markdown("""
<div class="calc-card">
<h4>Diffraction-limited spot / NA conversion</h4>
<div class="sub">For a Gaussian beam, NA = λ/(πw₀). Also gives the Abbe resolution limit.</div>
""", unsafe_allow_html=True)

        lam_na = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="na_lam")
        mode   = st.radio("Specify:", ["Numerical aperture (NA)", "Beam waist w₀"], key="na_mode", horizontal=True)

        lam_m2 = nm(lam_na)
        if mode == "Numerical aperture (NA)":
            NA = st.number_input("NA", 0.01, 1.0, 0.5, 0.01, key="na_val")
            w0_dl = lam_m2 / (math.pi * NA)
            abbe  = lam_m2 / (2 * NA)
            formula("w₀ = λ / (π · NA)      (Gaussian)\n"
                    "d_Abbe = λ / (2 · NA)  (Abbe resolution)")
            result_box(
                ("Gaussian w₀", f"{w0_dl*1e6:.3f} µm"),
                ("Abbe resolution d", f"{abbe*1e6:.3f} µm"),
            )
        else:
            w0_in = st.number_input("Beam waist w₀ (µm)", 0.1, 1000.0, 0.95, 0.05, key="na_w0")
            w0_m  = um(w0_in)
            NA_out = lam_m2 / (math.pi * w0_m)
            theta  = math.degrees(math.atan(NA_out))
            formula("NA = λ / (π · w₀)\nθ_half = arctan(NA)  [degrees]")
            result_box(
                ("Numerical aperture", f"{NA_out:.4f}"),
                ("Half-angle θ", f"{theta:.3f}°"),
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Rayleigh Range & Divergence")
        st.markdown('<div class="calc-card"><h4>Gaussian beam propagation</h4><div class="sub">From waist w₀ and wavelength.</div>', unsafe_allow_html=True)
        lam_rr = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 1064.0, 1.0, key="rr_lam")
        w0_rr  = st.number_input("Beam waist w₀ (µm)", 0.1, 5000.0, 950.0, 1.0, key="rr_w0")
        lam_m3 = nm(lam_rr);  w0_m3 = um(w0_rr)
        zR3    = math.pi * w0_m3**2 / lam_m3
        div    = lam_m3 / (math.pi * w0_m3)   # half-angle divergence
        formula("z_R = π w₀² / λ\nθ_div = λ / (π w₀)  (far-field half-angle)")
        result_box(
            ("Rayleigh range z_R", f"{zR3*1e3:.2f} mm  ({zR3*1e6:.1f} µm)"),
            ("Far-field divergence", f"{math.degrees(div):.4f}°  =  {div*1e3:.3f} mrad"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2, gap="large")

    # ── 1c. Beam telescope ────────────────────────────────────────────────────
    with col3:
        st.markdown("### Beam Telescope Magnification")
        st.markdown('<div class="calc-card"><h4>2-lens Galilean / Keplerian expander</h4><div class="sub">Output beam waist and divergence for two thin lenses separated by f₁ + f₂ (Keplerian) or |f₁ − f₂| (Galilean).</div>', unsafe_allow_html=True)
        f_in  = st.number_input("Input lens f₁ (mm)", 1.0, 1000.0, 25.0, 1.0, key="tel_f1")
        f_out = st.number_input("Output lens f₂ (mm)", 1.0, 2000.0, 200.0, 1.0, key="tel_f2")
        w0_in_tel = st.number_input("Input beam waist w₀_in (mm)", 0.01, 50.0, 0.5, 0.01, key="tel_w")
        M     = f_out / f_in
        w0_out_tel = M * w0_in_tel
        formula("M = f₂/f₁\nw₀_out = M · w₀_in\nθ_out = θ_in / M")
        result_box(
            ("Magnification M", f"{M:.3f}×"),
            ("Output waist w₀_out", f"{w0_out_tel:.3f} mm"),
            ("Divergence ratio θ_out/θ_in", f"1/{M:.3f}"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 1d. f-number ──────────────────────────────────────────────────────────
    with col4:
        st.markdown("### f-number → Spot Size")
        st.markdown('<div class="calc-card"><h4>Focused spot from f/# and beam diameter</h4><div class="sub">For an overfilled lens: spot ≈ 2.44 λ f/# (Airy). For a Gaussian: w₀ ≈ λ f / (π w_in).</div>', unsafe_allow_html=True)
        lam_fn = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 1064.0, 1.0, key="fn_lam")
        f_fn   = st.number_input("Focal length f (mm)", 1.0, 1000.0, 100.0, 1.0, key="fn_f")
        d_fn   = st.number_input("Input beam 1/e² diameter (mm)", 0.1, 200.0, 10.0, 0.1, key="fn_d",
                                   help="Full 1/e² diameter of the beam entering the lens.")
        lam_m4 = nm(lam_fn);  f_m4 = mm(f_fn);  w_in = mm(d_fn)/2.0
        fnum   = f_m4 / (2 * w_in * 2)   # f/# = f / (full aperture diameter, ~2×beam diam for filling)
        w0_foc = lam_m4 * f_m4 / (math.pi * w_in)
        airy   = 2.44 * lam_m4 * (f_m4 / (2 * w_in))  # Airy first dark ring
        formula("w₀ = λ f / (π w_in)      (Gaussian beam)\n"
                "d_Airy = 2.44 λ (f/D)   (plane wave / overfilled)")
        result_box(
            ("Gaussian w₀", f"{w0_foc*1e6:.3f} µm"),
            ("Airy disk diameter", f"{airy*1e6:.3f} µm"),
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POWER & RF
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Power, RF & AOM Calculators")
    col1, col2 = st.columns(2, gap="large")

    # ── 2a. mW ↔ dBm ─────────────────────────────────────────────────────────
    with col1:
        st.markdown("### mW ↔ dBm Conversion")
        st.markdown('<div class="calc-card"><h4>Optical / RF power units</h4><div class="sub">1 mW = 0 dBm by definition. Used everywhere: AOM drivers, amplifiers, photodetectors, VCOs.</div>', unsafe_allow_html=True)
        direction = st.radio("Convert from:", ["mW → dBm", "dBm → mW"], horizontal=True, key="pw_dir")
        if direction == "mW → dBm":
            p_mW  = st.number_input("Power (mW)", 0.001, 100000.0, 1.0, 0.1, format="%.4f", key="pw_mw")
            p_dBm = 10 * math.log10(p_mW)
            p_W   = p_mW * 1e-3
            formula("P [dBm] = 10 · log₁₀(P [mW])")
            result_box(("dBm", f"{p_dBm:.3f}"), ("Watts", f"{p_W:.4g} W"))
        else:
            p_dBm2 = st.number_input("Power (dBm)", -60.0, 60.0, 0.0, 0.5, key="pw_dbm")
            p_mW2  = 10 ** (p_dBm2 / 10)
            formula("P [mW] = 10^(P [dBm] / 10)")
            result_box(("mW", f"{p_mW2:.4g} mW"), ("W", f"{p_mW2*1e-3:.4g} W"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### dB Gain / Loss Chain")
        st.markdown('<div class="calc-card"><h4>Cascaded gain stages</h4><div class="sub">Enter input power and gains/losses in dB (negative = loss). Typical: AOM −10 dB, amplifier +30 dB, coupler −3 dB.</div>', unsafe_allow_html=True)
        p_in_dBm = st.number_input("Input power (dBm)", -60.0, 60.0, -10.0, 1.0, key="gain_in")
        gains_str = st.text_input("Gains/losses (dB), comma-separated", "-10, +30, -3, -6", key="gain_list",
                                   help="Positive = gain, negative = loss. E.g. AOM efficiency, amplifier, splitter.")
        try:
            gains = [float(g.strip()) for g in gains_str.split(",")]
            total_dB = sum(gains)
            p_out_dBm = p_in_dBm + total_dB
            p_out_mW  = 10 ** (p_out_dBm / 10)
            formula("P_out [dBm] = P_in [dBm] + Σ gains [dB]")
            result_box(
                ("Total gain/loss", f"{total_dB:+.1f} dB"),
                ("Output power", f"{p_out_dBm:.2f} dBm"),
                ("Output power", f"{p_out_mW:.4g} mW"),
            )
        except Exception:
            warn("Could not parse gain list.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 2b. AOM / AOD ────────────────────────────────────────────────────────
    with col2:
        st.markdown("### AOM / AOD Deflection")
        st.markdown('<div class="calc-card"><h4>Acoustic-optic modulator / deflector</h4><div class="sub">Bragg diffraction angle and velocity of the addressed atomic velocity class.</div>', unsafe_allow_html=True)
        lam_aom  = st.number_input("Optical wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="aom_lam")
        f_aom    = st.number_input("AOM drive frequency f_AOM (MHz)", 1.0, 3000.0, 80.0, 1.0, key="aom_freq")
        vs_aom   = st.number_input("Sound velocity in medium v_s (m/s)", 100.0, 10000.0, 613.0, 1.0, key="aom_vs",
                                    help="TeO₂ ≈ 617 m/s (slow shear), SiO₂ ≈ 5960 m/s, PbMoO₄ ≈ 3630 m/s.")
        order    = st.selectbox("Diffraction order", [1, 2, -1], key="aom_ord")

        lam_opt  = nm(lam_aom)
        lam_ac   = vs_aom / (MHz(f_aom))          # acoustic wavelength [m]
        theta_B  = lam_opt / (2 * lam_ac)         # Bragg angle [rad]
        theta_def = abs(order) * lam_opt * MHz(f_aom) / vs_aom  # deflection angle [rad]

        formula("Λ = v_s / f_AOM          (acoustic wavelength)\n"
                "θ_B = λ_opt / (2Λ)       (Bragg angle)\n"
                "θ_def = n·λ·f_AOM / v_s  (n-th order deflection)")
        result_box(
            ("Acoustic wavelength Λ", f"{lam_ac*1e6:.2f} µm"),
            ("Bragg angle θ_B", f"{math.degrees(theta_B):.4f}°  =  {theta_B*1e3:.3f} mrad"),
            (f"Order {order:+d} deflection", f"{math.degrees(theta_def):.4f}°  =  {theta_def*1e3:.3f} mrad"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Shot Noise on Photodetector")
        st.markdown('<div class="calc-card"><h4>Photon shot noise (quantum noise floor)</h4><div class="sub">Sets the minimum detectable power / frequency noise for a given detection bandwidth.</div>', unsafe_allow_html=True)
        lam_sn = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="sn_lam")
        P_sn   = st.number_input("Detected power P (µW)", 0.001, 100000.0, 10.0, 0.1, key="sn_P")
        eta_sn = st.number_input("Quantum efficiency η", 0.01, 1.0, 0.85, 0.01, key="sn_eta",
                                   help="Typical Si: 0.7–0.9. InGaAs: 0.8–0.95.")
        BW_sn  = st.number_input("Detection bandwidth (Hz)", 1.0, 1e9, 1e6, 1e5, format="%.0f", key="sn_bw")

        P_W    = P_sn * 1e-6
        lam_m5 = nm(lam_sn)
        e_ch   = 1.602176634e-19
        I_dc   = eta_sn * e_ch * P_W / (h * c / lam_m5)   # photocurrent [A]
        i_shot = math.sqrt(2 * e_ch * I_dc * BW_sn)        # shot noise current [A_rms]
        # NEP
        R_det  = eta_sn * e_ch * lam_m5 / (h * c)         # responsivity [A/W]
        NEP    = i_shot / R_det if R_det > 0 else float("nan")

        formula("I_dc = η · e · P / (hν)\n"
                "i_shot = √(2 e I_dc BW)   [A_rms]\n"
                "NEP = i_shot / R           [W/√Hz · √BW]")
        result_box(
            ("Photocurrent I_dc", f"{I_dc*1e6:.3f} µA"),
            ("Shot noise i_shot", f"{i_shot*1e12:.3f} pA_rms"),
            ("NEP (noise equiv. power)", f"{NEP*1e12:.3f} pW"),
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ATOMIC PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Atomic Physics Calculators")
    col1, col2 = st.columns(2, gap="large")

    # ── 3a. Photon recoil ─────────────────────────────────────────────────────
    with col1:
        st.markdown("### Photon Recoil")
        st.markdown('<div class="calc-card"><h4>Recoil velocity & temperature from one photon kick</h4><div class="sub">Sets the fundamental energy scale for laser cooling. T_rec = (ħk)² / (m k_B).</div>', unsafe_allow_html=True)
        lam_rec = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="rec_lam")
        A_rec   = st.number_input("Atomic mass A (amu)", 1.0, 250.0, 133.0, 1.0, key="rec_A")
        lam_m_r = nm(lam_rec);  m_r = A_rec * amu
        k_r     = 2 * math.pi / lam_m_r
        v_rec   = hbar * k_r / m_r
        T_rec   = m_r * v_rec**2 / kB
        E_rec_kHz = hbar * k_r**2 / (2 * m_r) / h / 1e3   # recoil freq in kHz
        formula("v_rec = ħk / m = h / (mλ)\n"
                "T_rec = m·v_rec² / k_B = (ħk)² / (m·k_B)\n"
                "E_rec/h = ħk² / (4πm)  [kHz]")
        result_box(
            ("Recoil velocity v_rec", f"{v_rec*1e3:.4f} mm/s"),
            ("Recoil temperature T_rec", f"{T_rec*1e9:.4f} nK  =  {T_rec*1e6:.4f} µK"),
            ("Recoil frequency E_rec/h", f"{E_rec_kHz:.4f} kHz"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Doppler Shift")
        st.markdown('<div class="calc-card"><h4>Frequency shift from atom/mirror velocity</h4><div class="sub">Used to find which velocity class a detuned laser addresses.</div>', unsafe_allow_html=True)
        mode_dop = st.radio("Specify:", ["Velocity → Δf", "Detuning Δf → velocity"], horizontal=True, key="dop_mode")
        lam_dop  = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="dop_lam")
        if mode_dop == "Velocity → Δf":
            v_dop = st.number_input("Atom velocity v (m/s)", -1000.0, 1000.0, 1.0, 0.1, key="dop_v")
            df    = v_dop / nm(lam_dop)
            formula("Δf = v / λ  (non-relativistic, counter-propagating beam)")
            result_box(("Doppler shift Δf", f"{df/1e6:.4f} MHz"), ("", f"{df:.2f} Hz"))
        else:
            df_in = st.number_input("Laser detuning Δf (MHz)", -2000.0, 2000.0, 10.0, 1.0, key="dop_df")
            v_out = MHz(df_in) * nm(lam_dop)
            formula("v = Δf · λ")
            result_box(("Addressed velocity", f"{v_out:.4f} m/s"), ("", f"{v_out*100:.4f} cm/s"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### de Broglie Wavelength")
        st.markdown('<div class="calc-card"><h4>Thermal de Broglie wavelength</h4><div class="sub">λ_dB ≫ inter-particle spacing → quantum degeneracy. Also used for matter-wave interferometry.</div>', unsafe_allow_html=True)
        A_dB  = st.number_input("Atomic mass A (amu)", 1.0, 250.0, 133.0, 1.0, key="ddb_A")
        T_dB  = st.number_input("Temperature T (µK)", 0.01, 1e6, 100.0, 1.0, key="ddb_T")
        m_dB  = A_dB * amu
        T_dB_K = T_dB * 1e-6
        lam_dB = h / math.sqrt(2 * math.pi * m_dB * kB * T_dB_K)
        formula("λ_dB = h / √(2π m k_B T)")
        result_box(("de Broglie wavelength", f"{lam_dB*1e9:.4f} nm"), ("", f"{lam_dB*1e6:.6f} µm"))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 3b. Zeeman & saturation ───────────────────────────────────────────────
    with col2:
        st.markdown("### Zeeman Shift (linear regime)")
        st.markdown('<div class="calc-card"><h4>Energy shift in a magnetic field</h4><div class="sub">Valid for |B| where the Zeeman energy ≪ HF splitting (typically B ≪ few hundred Gauss for alkalis).</div>', unsafe_allow_html=True)
        gF_z  = st.number_input("Landé g_F factor", -5.0, 5.0, 0.25, 0.01, key="zee_gF",
                                  help="Cs F=4: g_F=+1/4. Cs F=3: g_F=−1/4. Rb F=2: g_F=+1/2. Rb F=1: g_F=−1/2.")
        mF_z  = st.number_input("Magnetic sublevel m_F", -10.0, 10.0, 4.0, 1.0, key="zee_mF")
        B_z   = st.number_input("Magnetic field B (Gauss)", 0.0, 10000.0, 10.0, 0.1, key="zee_B",
                                  help="1 Gauss = 1e-4 Tesla.")
        B_T   = B_z * 1e-4   # Gauss → Tesla
        dE_J  = gF_z * mF_z * muB * B_T
        dE_MHz = dE_J / h / 1e6
        formula("ΔE = g_F · m_F · μ_B · B\n"
                "Δf = ΔE / h  [MHz]")
        result_box(
            ("Zeeman shift Δf", f"{dE_MHz:.4f} MHz"),
            ("Shift per Gauss", f"{dE_MHz/B_z:.4f} MHz/G" if B_z != 0 else "—"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Saturation Intensity")
        st.markdown('<div class="calc-card"><h4>I_sat for a dipole-allowed transition</h4><div class="sub">I_sat = π h c Γ / (3 λ³) for a cycling transition. Above I_sat, the transition is power-broadened.</div>', unsafe_allow_html=True)
        sat_atom = st.selectbox("Common transitions", [
            "¹³³Cs D2 852 nm  (Γ/2π = 5.23 MHz)",
            "⁸⁷Rb D2 780 nm   (Γ/2π = 6.07 MHz)",
            "²³Na D2 589 nm   (Γ/2π = 9.80 MHz)",
            "⁶Li  D2 671 nm   (Γ/2π = 5.87 MHz)",
            "Custom",
        ], key="sat_atom")
        presets = {
            "¹³³Cs D2 852 nm  (Γ/2π = 5.23 MHz)": (852.0, 5.23),
            "⁸⁷Rb D2 780 nm   (Γ/2π = 6.07 MHz)": (780.0, 6.07),
            "²³Na D2 589 nm   (Γ/2π = 9.80 MHz)": (589.0, 9.80),
            "⁶Li  D2 671 nm   (Γ/2π = 5.87 MHz)": (671.0, 5.87),
        }
        if sat_atom != "Custom":
            lam_sat_nm, Gamma_sat_MHz = presets[sat_atom]
        else:
            lam_sat_nm  = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="sat_lam_c")
            Gamma_sat_MHz = st.number_input("Linewidth Γ/2π (MHz)", 0.001, 100.0, 5.23, 0.01, key="sat_G_c")
        lam_sat_m = nm(lam_sat_nm)
        Gamma_sat = MHz(Gamma_sat_MHz) * 2 * math.pi   # rad/s
        I_sat = math.pi * h * c * Gamma_sat / (3 * lam_sat_m**3)
        tau   = 1.0 / (Gamma_sat / (2 * math.pi))   # spontaneous lifetime
        formula("I_sat = π h c Γ / (3 λ³)   (cycling transition)\n"
                "τ = 1 / (Γ/2π)")
        result_box(
            ("Saturation intensity I_sat", f"{I_sat:.2f} W/m²  =  {I_sat*0.1:.4f} mW/cm²"),
            ("Excited-state lifetime τ", f"{tau*1e9:.2f} ns"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Thermal Velocity Distribution")
        st.markdown('<div class="calc-card"><h4>Maxwell–Boltzmann rms and most-probable speed</h4>', unsafe_allow_html=True)
        A_thv = st.number_input("Atomic mass A (amu)", 1.0, 250.0, 133.0, 1.0, key="thv_A")
        T_thv = st.number_input("Temperature T (µK)", 0.01, 1e9, 100.0, 1.0, key="thv_T")
        m_thv = A_thv * amu;  T_thv_K = T_thv * 1e-6
        v_rms  = math.sqrt(3 * kB * T_thv_K / m_thv)
        v_prob = math.sqrt(2 * kB * T_thv_K / m_thv)   # most probable
        v_mean = math.sqrt(8 * kB * T_thv_K / (math.pi * m_thv))
        formula("v_rms  = √(3k_BT/m)\n"
                "v_prob = √(2k_BT/m)   (most probable)\n"
                "v_mean = √(8k_BT/πm)  (mean speed)")
        result_box(
            ("v_rms",  f"{v_rms:.4f} m/s"),
            ("v_prob", f"{v_prob:.4f} m/s"),
            ("v_mean", f"{v_mean:.4f} m/s"),
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAP & CAVITY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Trap & Cavity Calculators")
    col1, col2 = st.columns(2, gap="large")

    # ── 4a. Trap frequencies ──────────────────────────────────────────────────
    with col1:
        st.markdown("### Optical Tweezer Trap Frequencies")
        st.markdown('<div class="calc-card"><h4>Radial and axial trap frequencies from trap depth</h4><div class="sub">Harmonic approximation near the trap minimum. U₀ = trap depth; w₀ = beam waist; z_R = Rayleigh range.</div>', unsafe_allow_html=True)
        U0_tf = st.number_input("Trap depth U₀ (mK)", 0.001, 100.0, 0.5, 0.001, format="%.4f", key="tf_U0")
        w0_tf = st.number_input("Beam waist w₀ (µm)", 0.1, 50.0, 0.95, 0.01, key="tf_w0")
        A_tf  = st.number_input("Atomic mass A (amu)", 1.0, 250.0, 133.0, 1.0, key="tf_A")
        lam_tf = st.number_input("Wavelength λ (nm) — for z_R", 400.0, 2000.0, 1064.0, 1.0, key="tf_lam")
        m_tf  = A_tf * amu
        U0_J  = U0_tf * 1e-3 * kB
        w0_m  = um(w0_tf)
        zR_tf = math.pi * w0_m**2 / nm(lam_tf)
        omega_r = math.sqrt(4 * U0_J / (m_tf * w0_m**2))
        omega_z = math.sqrt(2 * U0_J / (m_tf * zR_tf**2))
        fr = omega_r / (2 * math.pi)
        fz = omega_z / (2 * math.pi)
        formula("ω_r = √(4 U₀ / m w₀²)   (radial)\n"
                "ω_z = √(2 U₀ / m z_R²)  (axial)\n"
                "z_R = π w₀² / λ")
        result_box(
            ("Radial ω_r/2π", f"{fr/1e3:.2f} kHz"),
            ("Axial ω_z/2π",  f"{fz:.1f} Hz  =  {fz/1e3:.4f} kHz"),
            ("Rayleigh range z_R", f"{zR_tf*1e3:.2f} mm"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Lamb–Dicke Parameter")
        st.markdown('<div class="calc-card"><h4>η = k x_zpf — sideband cooling regime criterion</h4><div class="sub">Resolved sideband cooling requires η ≪ 1. x_zpf = √(ħ/2mω) is the zero-point motion amplitude.</div>', unsafe_allow_html=True)
        lam_ld = st.number_input("Probe wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="ld_lam")
        A_ld   = st.number_input("Atomic mass A (amu)", 1.0, 250.0, 133.0, 1.0, key="ld_A")
        f_ld   = st.number_input("Trap frequency f_trap (kHz)", 0.01, 5000.0, 100.0, 1.0, key="ld_f")
        theta_ld = st.number_input("Beam angle to trap axis θ (°)", 0.0, 90.0, 45.0, 1.0, key="ld_th",
                                    help="0° = beam along trap axis (axial LD). 90° = beam perpendicular (radial LD).")
        m_ld   = A_ld * amu
        k_ld   = 2 * math.pi / nm(lam_ld)
        omega_ld = 2 * math.pi * f_ld * 1e3
        x_zpf  = math.sqrt(hbar / (2 * m_ld * omega_ld))
        eta    = k_ld * x_zpf * math.cos(math.radians(theta_ld))
        n_min  = (1.0 / (4 * eta**2)) * ((MHz(0.00523) * 2 * math.pi) / omega_ld) ** 2 if eta > 0 else float("nan")
        formula("x_zpf = √(ħ / 2mω_trap)\n"
                "η = k · cos(θ) · x_zpf\n"
                "⟨n⟩_min ≈ (Γ/2ω)²  (RSB cooling limit)")
        result_box(
            ("x_zpf (zero-point motion)", f"{x_zpf*1e12:.2f} pm"),
            ("Lamb–Dicke parameter η", f"{eta:.4f}"),
            ("Regime", "✅ Lamb–Dicke (η≪1)" if eta < 0.3 else "⚠️ Beyond Lamb–Dicke"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 4b. Optical cavity ────────────────────────────────────────────────────
    with col2:
        st.markdown("### Optical Cavity / Fabry–Pérot")
        st.markdown('<div class="calc-card"><h4>FSR, finesse, linewidth, mode spacing</h4><div class="sub">Used for locking lasers, ULE cavities, transfer cavities, EOM sidebands.</div>', unsafe_allow_html=True)
        L_cav = st.number_input("Cavity length L (mm)", 0.1, 1e6, 100.0, 0.1, key="cav_L")
        R_cav = st.number_input("Mirror reflectivity R (each mirror)", 0.001, 0.9999, 0.9999, 0.0001, format="%.6f",
                                  key="cav_R", help="For high-finesse: R = 1 − T − L_loss where T is transmission.")
        n_cav = st.number_input("Refractive index n", 1.0, 5.0, 1.0, 0.01, key="cav_n",
                                  help="n=1 for air/vacuum cavities.")
        L_m   = L_cav * 1e-3
        FSR   = c / (2 * n_cav * L_m)
        F     = math.pi * R_cav**0.5 / (1 - R_cav)
        lw    = FSR / F
        T_rt  = 2 * n_cav * L_m / c   # round-trip time
        formula("FSR = c / (2nL)\n"
                "F = π√R / (1−R)\n"
                "δν = FSR / F  (cavity linewidth FWHM)")
        result_box(
            ("FSR", f"{FSR/1e9:.4f} GHz  =  {FSR/1e6:.2f} MHz"),
            ("Finesse F", f"{F:.0f}"),
            ("Linewidth δν", f"{lw/1e6:.4f} MHz  =  {lw/1e3:.2f} kHz"),
            ("Round-trip time", f"{T_rt*1e9:.3f} ns"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Gaussian Beam Mode Matching")
        st.markdown('<div class="calc-card"><h4>Coupling efficiency into a fiber or cavity mode</h4><div class="sub">Overlap integral for two Gaussian modes with (possibly) different waists and waist positions.</div>', unsafe_allow_html=True)
        w1 = st.number_input("Mode 1 waist w₁ (µm)", 0.01, 1000.0, 5.0, 0.1, key="mm_w1")
        w2 = st.number_input("Mode 2 waist w₂ (µm)", 0.01, 1000.0, 4.8, 0.1, key="mm_w2")
        dz = st.number_input("Waist separation Δz (µm)", 0.0, 10000.0, 0.0, 1.0, key="mm_dz",
                               help="0 = waists coincide. Non-zero reduces coupling.")
        lam_mm = st.number_input("Wavelength λ (nm)", 400.0, 2000.0, 852.0, 1.0, key="mm_lam")
        w1_m = um(w1);  w2_m = um(w2);  dz_m = um(dz);  lam_mm_m = nm(lam_mm)
        zR1  = math.pi * w1_m**2 / lam_mm_m
        zR2  = math.pi * w2_m**2 / lam_mm_m
        # Overlap integral: η = |∫ψ₁* ψ₂ dA|²
        # For pure waist mismatch (Δz=0):  η = (2w₁w₂/(w₁²+w₂²))²
        # For Δz ≠ 0, include Gouy phase and wavefront curvature mismatch
        # w(z) of beam 2 at z=dz from its waist
        w2_at_z = w2_m * math.sqrt(1 + (dz_m / zR2)**2)
        R2_at_z = dz_m * (1 + (zR2 / dz_m)**2) if dz_m != 0 else float("inf")
        # Simplified overlap (paraxial Gaussian, scalar):
        # η = (2/(w1/w2_at_z + w2_at_z/w1))² × 1/(1 + (w1² - w2_at_z²)²/(something))
        # Use the analytic formula:
        # η = 4 / ((w1/w2_az + w2_az/w1)² + (π w1 w2_az / λ / R2_at_z)²)
        # For Δz=0, R2_at_z → ∞, giving η = (2w1 w2/(w1²+w2²))²
        if dz_m == 0:
            eta_mm = (2 * w1_m * w2_m / (w1_m**2 + w2_m**2))**2
        else:
            A = (w1_m / w2_at_z + w2_at_z / w1_m)**2
            B = (math.pi * w1_m * w2_at_z / (lam_mm_m * R2_at_z))**2 if abs(R2_at_z) < 1e10 else 0
            eta_mm = 4.0 / (A + B)
        formula("η = (2w₁w₂/(w₁²+w₂²))²   [Δz=0]\n"
                "Reduced by wavefront curvature mismatch for Δz≠0")
        result_box(
            ("Mode overlap η", f"{eta_mm*100:.2f} %"),
            ("Expected coupling", "✅ Good (>90%)" if eta_mm > 0.9 else ("⚠️ Moderate" if eta_mm > 0.5 else "❌ Poor (<50%)")),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="color:#64748b; font-size:.82rem; line-height:1.7;">
<strong>References:</strong>
Foot (2005) <em>Atomic Physics</em> · Grimm et al. (2000) Adv. At. Mol. Opt. Phys. 42 ·
Yariv (1989) <em>Quantum Electronics</em> · Saleh & Teich (2019) <em>Fundamentals of Photonics</em> ·
NIST CODATA 2018 constants.
</div>
""", unsafe_allow_html=True)
