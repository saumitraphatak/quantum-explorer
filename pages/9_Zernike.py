"""
Zernike Polynomials & Wavefront Optics
========================================
Visualise Zernike modes, build arbitrary wavefronts from Zernike coefficients,
and explore SLM phase-to-far-field Fourier optics (LG beams, OAM, PSF).
"""

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Zernike Polynomials",
    page_icon="🌊",
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

  .info-box { background:#0a1020; border:1px solid #1e3a6e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#93c5fd; font-size:.85rem; }

  .warn { background:#1c1007; border:1px solid #92400e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#fde68a; font-size:.85rem; }

  .mode-chip {
    display:inline-block; background:#0c1e3a; border:1px solid #1e3a5f;
    border-radius:20px; padding:.2rem .7rem; margin:.2rem .2rem;
    color:#7dd3fc; font-size:.82rem; font-family:monospace; cursor:default;
  }

  hr { border-color:#1e293b; margin:1.5rem 0; }
  .stTabs [data-baseweb="tab-list"]  { background:#0f172a; border-radius:8px; padding:4px; }
  .stTabs [data-baseweb="tab"]       { color:#94a3b8; font-weight:600; }
  .stTabs [aria-selected="true"]     { color:#7dd3fc !important; background:#0c1e3a !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── Aberration name table (n, m) → name ──────────────────────────────────────
NAMES = {
    (0,  0): "Piston",
    (1, -1): "Tilt Y",            (1,  1): "Tilt X",
    (2, -2): "Astigmatism 45°",   (2,  0): "Defocus",        (2,  2): "Astigmatism 0°",
    (3, -3): "Trefoil Y",         (3, -1): "Coma Y",
    (3,  1): "Coma X",            (3,  3): "Trefoil X",
    (4, -4): "Tetrafoil Y",       (4, -2): "Sec. Astig. Y",
    (4,  0): "Primary Spherical", (4,  2): "Sec. Astig. X",   (4,  4): "Tetrafoil X",
    (5, -5): "Pentafoil Y",       (5, -3): "Sec. Trefoil Y",  (5, -1): "Sec. Coma Y",
    (5,  1): "Sec. Coma X",       (5,  3): "Sec. Trefoil X",  (5,  5): "Pentafoil X",
    (6,  0): "Sec. Spherical",
}

# ─── Zernike computation ──────────────────────────────────────────────────────

def valid_m(n):
    return list(range(-n, n + 1, 2))

def zernike_radial(n, m_abs, rho):
    """Radial polynomial R_n^{|m|}(ρ)."""
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        c = ((-1)**s * math.factorial(n - s)
             / (math.factorial(s)
                * math.factorial((n + m_abs) // 2 - s)
                * math.factorial((n - m_abs) // 2 - s)))
        R += c * rho ** (n - 2 * s)
    return R

def zernike(n, m, rho, theta):
    """
    OSA/ANSI-normalised Zernike polynomial Z_n^m.
    Norm: ∫|Z|² ρ dρ dθ / π = 1 over the unit disk.
    """
    m_abs = abs(m)
    norm  = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    R     = zernike_radial(n, m_abs, rho)
    if   m > 0: return norm * R * np.cos(m * theta)
    elif m < 0: return norm * R * np.sin(m_abs * theta)
    else:       return norm * R

def make_polar_grid(N=320):
    x       = np.linspace(-1, 1, N)
    X, Y    = np.meshgrid(x, x)
    rho     = np.sqrt(X**2 + Y**2)
    theta   = np.arctan2(Y, X)
    mask    = rho <= 1.0
    return X, Y, rho, theta, mask

# ─── Plotly helpers ───────────────────────────────────────────────────────────

DARK = dict(
    plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
    font=dict(color="#cbd5e1", family="Inter"),
)
AXIS_OFF = dict(showticklabels=False, showgrid=False, zeroline=False)

# Cyclic colorscale for phase (HSV-like)
PHASE_CS = [
    [0.00, "rgb(255, 0, 0)"],   [0.17, "rgb(255,165, 0)"],
    [0.33, "rgb(255,255, 0)"],  [0.50, "rgb(0, 200, 80)"],
    [0.67, "rgb(0, 120,255)"],  [0.83, "rgb(180,  0,255)"],
    [1.00, "rgb(255, 0, 0)"],
]

def wf_heatmap(Z_masked, title, colorscale="RdBu_r", zmid=0):
    fig = go.Figure(go.Heatmap(
        z=Z_masked, colorscale=colorscale, zmid=zmid,
        showscale=True,
        colorbar=dict(title="λ", len=0.8, thickness=12),
    ))
    fig.update_layout(
        **DARK, title=dict(text=title, font=dict(size=13, color="#94a3b8")),
        height=340, margin=dict(l=5, r=5, t=32, b=5),
        xaxis={**AXIS_OFF, "scaleanchor": "y"}, yaxis=AXIS_OFF,
    )
    return fig

def intensity_heatmap(I, title):
    I_norm = I / (np.nanmax(I) + 1e-30)
    fig = go.Figure(go.Heatmap(
        z=I_norm, colorscale="Inferno", zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title="Norm.", len=0.8, thickness=12),
    ))
    fig.update_layout(
        **DARK, title=dict(text=title, font=dict(size=13, color="#94a3b8")),
        height=340, margin=dict(l=5, r=5, t=32, b=5),
        xaxis={**AXIS_OFF, "scaleanchor": "y"}, yaxis=AXIS_OFF,
    )
    return fig

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

def info(s):
    st.markdown(f'<div class="info-box">ℹ️ {s}</div>', unsafe_allow_html=True)

# ─── Page ─────────────────────────────────────────────────────────────────────

st.title("🌊 Zernike Polynomials & Wavefront Optics")
st.markdown(
    "Zernike polynomials are the standard orthogonal basis for describing optical "
    "wavefront aberrations over a circular pupil — used in adaptive optics, "
    "interferometry, and SLM beam-shaping."
)

tab1, tab2, tab3 = st.tabs([
    "🌀 Single Mode",
    "🔧 Wavefront Builder",
    "📡 SLM & Far Field",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Single mode viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.expander("📖 Definition & properties", expanded=False):
        st.markdown("### Zernike polynomials")
        theory_box(
            "Zernike polynomials Z_n^m(ρ, θ) are defined over the unit disk ρ ≤ 1 "
            "in polar coordinates. They separate into a radial part R_n^|m|(ρ) "
            "and an azimuthal part:"
        )
        formula(
            "Z_n^m(ρ,θ) =  √(2(n+1)) · R_n^|m|(ρ) · cos(mθ)    m > 0\n"
            "            =  √(2(n+1)) · R_n^|m|(ρ) · sin(|m|θ)  m < 0\n"
            "            =  √(n+1)    · R_n^0(ρ)                 m = 0\n\n"
            "Radial polynomial:\n"
            "R_n^m(ρ) = Σ_{s=0}^{(n-m)/2}  (−1)^s (n−s)! ρ^{n−2s}\n"
            "           ────────────────────────────────────────────\n"
            "               s! · ((n+m)/2−s)! · ((n−m)/2−s)!"
        )
        theory_box(
            "OSA/ANSI normalisation: <b>∫|Z_n^m|² ρ dρ dθ / π = 1</b> over the unit disk. "
            "The polynomials form a complete orthonormal set; any wavefront W(ρ,θ) can be expanded as "
            "W = Σ c_n^m Z_n^m, and the RMS wavefront error is simply √(Σ c²)."
        )
        st.markdown("### OSA standard mode names")
        chips = "".join(
            f'<span class="mode-chip">Z_{n}^{{{m:+d}}} {name}</span>'
            for (n, m), name in NAMES.items()
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("### Mode selector")
    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        n = st.slider("Radial order  n", 0, 7, 2)
        m_list   = valid_m(n)
        m_labels = [f"m = {m:+d}  {NAMES.get((n, m), '')}" for m in m_list]
        m_sel    = st.selectbox("Azimuthal frequency  m", m_labels,
                                index=m_list.index(0) if 0 in m_list else 0)
        m = m_list[m_labels.index(m_sel)]

        name = NAMES.get((n, m), f"Z_{n}^{m:+d}")
        st.markdown(f"**{name}**")
        st.markdown(f"n = {n},  m = {m:+d}")

        show_phase = st.checkbox("Show radial cross-section", value=True)

    X, Y, rho, theta, mask = make_polar_grid(320)
    Z = zernike(n, m, rho, theta)
    Z_masked = np.where(mask, Z, np.nan)
    rms = np.sqrt(np.nanmean(Z_masked**2))

    with col_plot:
        st.plotly_chart(
            wf_heatmap(Z_masked, f"Z_{n}^{{{m:+d}}}  —  {name}  (OSA normalised, λ)"),
            use_container_width=True,
        )

    result_box(
        ("RMS over unit disk:", f"{rms:.4f} λ"),
        ("Peak-to-valley:", f"{np.nanmax(Z_masked) - np.nanmin(Z_masked):.4f} λ"),
    )

    if show_phase:
        # Radial cross-section at θ = 0 (or θ = π/4 for m<0)
        angle = 0.0 if m >= 0 else math.pi / 4
        rho_1d = np.linspace(0, 1, 400)
        Z_1d   = zernike(n, m, rho_1d, np.full_like(rho_1d, angle))
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=rho_1d, y=Z_1d, mode="lines",
            line=dict(color="#0ea5e9", width=2),
            name=f"θ = {angle:.2f} rad",
        ))
        fig_r.add_hline(y=0, line_color="#334155", line_dash="dot")
        fig_r.update_layout(
            **DARK, height=200, margin=dict(l=40, r=10, t=20, b=30),
            xaxis=dict(title="ρ", gridcolor="#1e293b"),
            yaxis=dict(title="Z (λ)", gridcolor="#1e293b"),
            showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Wavefront builder
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Wavefront from Zernike expansion")
    theory_box(
        "Any wavefront over a circular aperture can be written as "
        "<b>W(ρ,θ) = Σ c_n^m · Z_n^m(ρ,θ)</b> where coefficients cₙᵐ "
        "are in units of waves (λ). The RMS wavefront error is √(Σ cₙᵐ²) "
        "and the Strehl ratio (Maréchal approximation) is "
        "S ≈ exp(−(2π · σ_W)²)."
    )
    info("Coefficients are in waves λ. PV = peak-to-valley; WFE = wavefront error.")

    n_modes = st.slider("Number of Zernike terms", 1, 8, 3)
    modes   = []

    col_h1, col_h2, col_h3 = st.columns([1.2, 1.8, 0.8])
    col_h1.markdown("**n**"); col_h2.markdown("**m  (aberration)**"); col_h3.markdown("**c (λ)**")

    for i in range(n_modes):
        c1, c2, c3 = st.columns([1.2, 1.8, 0.8])
        with c1:
            ni = st.selectbox("n", list(range(8)), index=min(i+1, 7), key=f"wf_n{i}",
                              label_visibility="collapsed")
        with c2:
            mi_list   = valid_m(ni)
            mi_labels = [f"m={m:+d}  {NAMES.get((ni,m),'')}" for m in mi_list]
            mi_sel    = st.selectbox("m", mi_labels, key=f"wf_m{i}",
                                     label_visibility="collapsed")
            mi = mi_list[mi_labels.index(mi_sel)]
        with c3:
            defaults = [0.5, -0.3, 0.2, 0.1, -0.15, 0.1, 0.05, 0.05]
            ci = st.number_input("c", value=defaults[i], step=0.05, format="%.3f",
                                  key=f"wf_c{i}", label_visibility="collapsed")
        modes.append((ni, mi, ci))

    st.markdown("---")
    X, Y, rho, theta, mask = make_polar_grid(300)
    W = sum(ci * zernike(ni, mi, rho, theta) for ni, mi, ci in modes)
    W_masked = np.where(mask, W, np.nan)

    rms_wfe = math.sqrt(float(np.nanmean(W_masked**2)))
    pv      = float(np.nanmax(W_masked) - np.nanmin(W_masked))
    strehl  = math.exp(-(2 * math.pi * rms_wfe)**2)

    # PSF: pupil function P = A·exp(i·2π·W) then FFT
    pad  = 3
    N    = 300
    E    = mask.astype(complex) * np.exp(1j * 2 * math.pi * W)
    E_ff = np.fft.fftshift(np.fft.fft2(E, s=(N * pad, N * pad)))
    PSF  = np.abs(E_ff)**2
    c0   = N * pad // 2;  hw = N // 2
    PSF_crop = PSF[c0 - hw : c0 + hw, c0 - hw : c0 + hw]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            wf_heatmap(W_masked, "Combined wavefront W(ρ,θ)  (waves λ)"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            intensity_heatmap(PSF_crop, "Point Spread Function  |FT{P(ρ,θ)}|²"),
            use_container_width=True,
        )

    result_box(
        ("RMS WFE:", f"{rms_wfe:.4f} λ"),
        ("Peak-to-valley:", f"{pv:.4f} λ"),
        ("Strehl (Maréchal):", f"{strehl:.4f}"),
    )
    if strehl > 0.8:
        st.markdown('<div class="info-box">✅ Strehl > 0.8 — diffraction-limited (Rayleigh criterion).</div>',
                    unsafe_allow_html=True)
    elif strehl > 0.3:
        st.markdown('<div class="warn">⚠️ Moderate aberration — PSF broadened but recognisable.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn">⚠️ Strong aberration — PSF severely degraded.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — SLM & Far Field
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### SLM Phase → Far-Field Intensity")

    theory_box(
        "A spatial light modulator (SLM) imprints a programmable phase φ(x,y) onto an "
        "input beam. In the far field (focal plane of a lens), the complex field is the "
        "<b>Fourier transform</b> of the SLM-modulated input:"
    )
    formula(
        "E_ff(u,v)  =  FT{ E_in(x,y) · exp[iφ(x,y)] }\n"
        "I_ff(u,v)  =  |E_ff(u,v)|²\n\n"
        "For a helical phase φ = l·θ applied to a Gaussian beam,\n"
        "the far field approximates a Laguerre–Gaussian LG₀ˡ mode:\n"
        "  |LG₀ˡ|²  ∝  (r/w)^{2|l|} exp(−2r²/w²)   [donut for l≠0, Gaussian for l=0]"
    )

    st.markdown("---")
    col_ctrl, col_plots = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Input beam**")
        beam_w = st.slider("Beam width w (×pupil radius)", 0.15, 1.5, 0.65, 0.05,
                           help="1/e² radius of input Gaussian relative to pupil")

        st.markdown("**Phase pattern**")
        preset = st.selectbox("Preset", [
            "Flat (no phase)",
            "Helical — OAM",
            "Helical + defocus",
            "Helical + astigmatism",
            "Blazed grating",
            "Custom Zernike",
        ])

        l_oam = 0
        z_n, z_m, z_a = 2, 0, 0.0
        grating_k = 0.0

        if "Helical" in preset or "OAM" in preset:
            l_oam = st.slider("OAM order  l", -4, 4, 1, key="slm_l")

        if "defocus" in preset:
            z_n, z_m = 2, 0
            z_a = st.slider("Defocus amplitude (waves)", -3.0, 3.0, 0.5, 0.1, key="slm_def")

        if "astigmatism" in preset:
            z_n, z_m = 2, 2
            z_a = st.slider("Astigmatism amplitude (waves)", -3.0, 3.0, 0.5, 0.1, key="slm_astig")

        if "Blazed grating" in preset:
            grating_k = st.slider("Grating spatial freq. (cycles/pupil)", 0.5, 8.0, 3.0, 0.5)

        if "Custom Zernike" in preset:
            l_oam = st.slider("OAM order  l", -4, 4, 0, key="slm_l_cz")
            z_n   = st.selectbox("Zernike n", list(range(6)), index=2, key="slm_zn")
            z_m_opts   = valid_m(z_n)
            z_m_labels = [f"m={m:+d}  {NAMES.get((z_n,m),'')}" for m in z_m_opts]
            z_m_sel    = st.selectbox("Zernike m", z_m_labels, key="slm_zm")
            z_m        = z_m_opts[z_m_labels.index(z_m_sel)]
            z_a        = st.slider("Amplitude (waves)", -3.0, 3.0, 1.0, 0.1, key="slm_za")

    # ── Computation ────────────────────────────────────────────────────────────
    N    = 256
    pad  = 4
    xg   = np.linspace(-1, 1, N)
    Xg, Yg  = np.meshgrid(xg, xg)
    Rg       = np.sqrt(Xg**2 + Yg**2)
    Tg       = np.arctan2(Yg, Xg)
    mask_slm = Rg <= 1.0

    # Input amplitude: Gaussian clipped to pupil
    E_in = np.exp(-Rg**2 / beam_w**2) * mask_slm

    # Phase pattern
    phase = np.zeros((N, N))
    if "Helical" in preset or "OAM" in preset or "Custom" in preset:
        phase += l_oam * Tg
    if "defocus" in preset or "astigmatism" in preset:
        phase += z_a * 2 * math.pi * zernike(z_n, z_m, Rg, Tg) * mask_slm
    if "Blazed grating" in preset:
        phase += grating_k * math.pi * Xg   # linear phase ramp along x
    if "Custom Zernike" in preset:
        phase += z_a * 2 * math.pi * zernike(z_n, z_m, Rg, Tg) * mask_slm

    # SLM output field
    E_slm = E_in * np.exp(1j * phase)

    # Far field via zero-padded FFT
    E_ff   = np.fft.fftshift(np.fft.fft2(E_slm, s=(N * pad, N * pad)))
    I_ff   = np.abs(E_ff)**2
    c0     = N * pad // 2;  hw = N // 2
    I_crop = I_ff[c0 - hw : c0 + hw, c0 - hw : c0 + hw]

    # Phase display: wrap to [0, 2π] then mod for cyclic colormap
    phase_display = np.where(mask_slm, np.mod(phase, 2 * math.pi) / (2 * math.pi), np.nan)
    I_in_display  = E_in**2

    # ── Three-panel plot ────────────────────────────────────────────────────────
    with col_plots:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Input intensity", "Phase pattern  φ (mod 2π)", "Far-field intensity"],
            horizontal_spacing=0.04,
        )
        fig.add_trace(go.Heatmap(z=I_in_display,    colorscale="Inferno",
                                 showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=phase_display,   colorscale=PHASE_CS,
                                 zmin=0, zmax=1, showscale=True,
                                 colorbar=dict(title="× 2π", len=0.7, thickness=10,
                                               x=0.67, tickvals=[0, 0.5, 1],
                                               ticktext=["0", "π", "2π"])), 1, 2)
        fig.add_trace(go.Heatmap(z=I_crop / (np.max(I_crop) + 1e-30),
                                 colorscale="Inferno", showscale=True,
                                 colorbar=dict(title="Norm.", len=0.7, thickness=10,
                                               x=1.01)), 1, 3)

        for col in [1, 2, 3]:
            fig.update_xaxes(**AXIS_OFF, scaleanchor=f"y{'' if col==1 else col}", row=1, col=col)
            fig.update_yaxes(**AXIS_OFF, row=1, col=col)

        fig.update_layout(
            **DARK, height=380, margin=dict(l=5, r=10, t=40, b=5),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Radial profile of far field ─────────────────────────────────────────
        with st.expander("📈 Far-field radial profile"):
            I_norm = I_crop / np.max(I_crop)
            cx, cy = N // 2, N // 2
            r_max  = N // 2
            r_vals = np.arange(r_max)
            I_r    = np.array([
                np.mean(I_norm[cy, cx:cx+1])  # placeholder — do proper azimuthal average
                for _ in r_vals
            ])
            # Proper azimuthal average
            r_2d   = np.sqrt((np.arange(N) - cx)**2 + (np.arange(N)[:, None] - cy)**2)
            I_az   = np.array([
                np.mean(I_norm[np.round(r_2d).astype(int) == r]) if np.any(np.round(r_2d).astype(int) == r) else 0
                for r in r_vals
            ])
            fig_r = go.Figure(go.Scatter(x=r_vals, y=I_az, mode="lines",
                                          line=dict(color="#0ea5e9", width=2)))
            fig_r.update_layout(
                **DARK, height=200, margin=dict(l=40, r=10, t=20, b=30),
                xaxis=dict(title="Pixel radius from centre", gridcolor="#1e293b"),
                yaxis=dict(title="Norm. intensity", gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    st.markdown("### LG mode reference")
    col_a, col_b = st.columns(2)

    with col_a:
        theory_box(
            "<b>Laguerre–Gaussian LG_p^l beams</b> carry orbital angular momentum l·ℏ per photon. "
            "At the beam waist:<br><br>"
            "|LG_p^l(r,θ)| ∝ (r√2/w)^|l| · L_p^|l|(2r²/w²) · exp(−r²/w²)<br><br>"
            "For p=0: intensity ring at r = w√(|l|/2), gets larger with |l|."
        )
        formula(
            "SLM recipe for LG₀ˡ:\n"
            "  1. Input: Gaussian beam (any size)\n"
            "  2. Phase: φ = l·θ  (helical/vortex phase)\n"
            "  3. Far field ≈ LG₀ˡ  (pure donut for l≠0)\n\n"
            "For higher purity: also modulate amplitude\n"
            "  → use (r/w)^|l| · exp(−r²/w²) input profile"
        )

    with col_b:
        st.markdown("""
| l | Name | Far-field intensity |
|---|------|---------------------|
| 0 | Gaussian (LG₀⁰) | Central peak |
| ±1 | First-order vortex | Single ring |
| ±2 | Second-order vortex | Larger ring |
| ±3 | Third-order vortex | Even larger ring |
| 0 (+ defocus) | Defocused Gaussian | Broadened peak |
| 0 (+ astigmatism) | Astigmatic | Elongated spot |

Adding a blazed grating **l·θ + k·x** to the phase shifts the donut off-axis —
useful for separating OAM orders spatially.
""")
