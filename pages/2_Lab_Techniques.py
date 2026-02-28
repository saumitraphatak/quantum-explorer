"""
AMO Lab Techniques
==================
Practical instrumentation guide for ultracold-atom / optical-tweezer experiments.
Based on Chapter 6 of: Phatak (2025), "Cooling Lithium and Cesium Single Atoms
in Optical Tweezers", PhD Thesis, Purdue University.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="AMO Lab Techniques",
    page_icon="⚗️",
    layout="wide",
)

# ── Shared CSS ───────────────────────────────────────────────────────────────
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
.warn-box {
    background: #2a1a00;
    border-left: 4px solid #ffaa00;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ffe0a0;
    font-size: 0.93rem;
}
.tip-box {
    background: #0a2a1a;
    border-left: 4px solid #44cc88;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #a0ffcc;
    font-size: 0.93rem;
}
.vendor-card {
    background: #0f0f28;
    border: 1px solid #2a2a5e;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.91rem;
    color: #ccc;
}
.formula-box {
    background: #0a0a20;
    border: 1px solid #3a3a6e;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 8px 0;
    font-family: monospace;
    color: #aad4ff;
    font-size: 1.02rem;
    text-align: center;
}
h1, h2, h3 { color: #c8b8ff; }
a { color: #7b68ee; }
.tag {
    display: inline-block;
    background: #1e3a5f;
    color: #7ec8e3;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.82rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.title("⚗️ AMO Lab Techniques")
st.markdown("#### Practical instrumentation for optical-tweezer and ultracold-atom experiments")

st.markdown("""
<div class='concept-box'>
This page is a practical guide for anyone entering an AMO lab — covering the <b>how</b>
and <b>why</b> behind the core techniques used in optical-tweezer experiments with
laser-cooled atoms.  The emphasis is on physical reasoning: not just what to do,
but why each component is designed the way it is and what breaks when it goes wrong.
The material is drawn from Chapter 6 of
<a href='#ref-thesis'>Phatak (2025)</a>
(PhD thesis, Purdue University — Hood Lab, Li–Cs optical tweezers).
Use the tabs below to jump to a topic.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔦 Beam Delivery",
    "🌀 Optical Pumping",
    "📡 RF & Electronics",
    "💻 Computational Tools",
    "🔴 Laser Systems",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — BEAM DELIVERY
# ═══════════════════════════════════════════════════════════════
with tab1:

    # ── 1.1 Fiber Coupling ────────────────────────────────────
    st.markdown("## 1 · Optical Fiber Coupling")
    st.markdown("""
<div class='concept-box'>
Almost every laser beam in an AMO experiment is delivered to the optical table via
optical fiber.  This is not merely a convenience: fibers <b>spatially filter</b> the
beam, <b>mechanically decouple</b> the laser table from the experiment table, and
allow laser sources to be swapped without realigning downstream optics.
The practical consequence is that <em>fiber coupling efficiency determines beam quality
at the atom</em>.
<br><br>
Three fiber types appear in a typical lab:
<ul>
  <li><b>Single-mode (SM)</b> — supports only the fundamental HE₁₁ mode; any
  higher-order input content is rejected.  Essential wherever spatial coherence
  matters (cooling beams, probe beams).</li>
  <li><b>Polarisation-maintaining (PM)</b> — SM fiber with stress rods that introduce
  birefringence; preserves the input polarisation when aligned to the principal axis.
  Every beam requiring a defined polarisation at the atom (optical pumping, Λ-GM
  cooling, the 685 nm quadrupole drive) must travel on PM fiber.</li>
  <li><b>Multimode (MM)</b> — large core, easy to couple; used for wavemeter pick-offs,
  diagnostics, and detection paths where polarisation is unimportant.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with st.expander("**Mode-matching: the key formula**"):
        st.markdown("""
A lens of focal length **f** converts an input Gaussian beam of 1/e² radius **w_in**
into a focused waist:

&emsp; **w₀ ≈ λ f / (π w_in)**

The coupling condition is **w₀ = w_f** , where **w_f = MFD/2** is the fiber mode radius.
Solving for f:

&emsp; **f = π w_in w_f / λ**

Trade-off: shorter f → tighter focus, higher peak efficiency, but stricter
transverse/axial alignment tolerances.  Longer f → more robust over temperature drifts.
""")

        st.markdown("#### 🔧 Mode-matching calculator")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lam_nm = st.number_input("Wavelength λ (nm)", 400, 1100, 780, step=1)
        with col_b:
            w_in_mm = st.number_input("Input beam 1/e² radius w_in (mm)", 0.5, 5.0, 1.5, step=0.1)
        with col_c:
            mfd_um = st.number_input("Fiber MFD (μm)  [SM-780HP ≈ 5 μm]", 2.0, 50.0, 5.0, step=0.5)

        lam  = lam_nm * 1e-9
        w_in = w_in_mm * 1e-3
        w_f  = (mfd_um * 1e-6) / 2.0
        f_req = np.pi * w_in * w_f / lam
        w0_at_f = lam * f_req / (np.pi * w_in)

        c1, c2 = st.columns(2)
        c1.metric("Required focal length", f"{f_req*1e3:.1f} mm")
        c2.metric("Focused waist w₀", f"{w0_at_f*1e6:.2f} μm")

        # Show a few standard Thorlabs lenses and their waist
        st.markdown("**Standard Thorlabs aspheric lenses** (at your settings):")
        lenses = [("C220TMD", 11.0), ("C230TMD", 4.51), ("C240TMD", 8.0), ("A220TM", 11.0)]
        rows = []
        for name, f_mm in lenses:
            f_m = f_mm * 1e-3
            w0  = lam * f_m / (np.pi * w_in)
            eff_note = "✅ good match" if abs(w0 - w_f) / w_f < 0.2 else ("⚠️ tight" if w0 < w_f else "⬆️ oversized")
            rows.append(f"| [{name}](https://www.thorlabs.com/search/thorsearch.cfm?search={name}) | {f_mm} mm | {w0*1e6:.2f} μm | {eff_note} |")
        st.markdown("| Lens | f | w₀ | Match |")
        st.markdown("|---|---|---|---|")
        for r in rows:
            st.markdown(r)

        st.markdown("""
<div class='tip-box'>
<b>Target:</b> 50–80% coupling into SM/PM fiber; >80% into MM.  PM fiber ER target: >20 dB (P∥/P⊥ > 100), which means <1% power in the wrong polarisation axis.
</div>
""", unsafe_allow_html=True)

    with st.expander("**PM fiber alignment procedure**"):
        st.markdown("""
1. Use a PBS to prepare clean linear polarisation upstream.
2. Rotate a HWP after the PBS to align the polarisation to the fiber's keyed (slow/fast) axis.
3. Optimise coupling for maximum throughput.
4. Measure extinction ratio (ER) at the output through an analyser PBS.
5. Iterate HWP angle to maximise ER.

**Target ER > 20 dB.** Residual polarisation impurity (leakage into the orthogonal axis) is the
dominant limit on optical-pumping fidelity and on the σ⁺–σ⁻ purity of GM imaging beams.
""")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("""
<div class='vendor-card'>
📦 <b>Thorlabs — SM & PM Fibers</b><br>
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10760'>SM-780HP</a> (780 nm, MFD 5.0 μm) ·
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=740'>PM780-HP</a> (PM, 780 nm) ·
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10760'>P1-630A-FC</a> (630 nm)
</div>
""", unsafe_allow_html=True)
    with col_v2:
        st.markdown("""
<div class='vendor-card'>
📦 <b>Thorlabs — Coupling Lenses</b><br>
<a href='https://www.thorlabs.com/thorproduct.cfm?partnumber=C220TMD-B'>C220TMD</a> (f=11 mm, 350–700 nm) ·
<a href='https://www.thorlabs.com/thorproduct.cfm?partnumber=C230TMD-B'>C230TMD</a> (f=4.51 mm) ·
<a href='https://www.thorlabs.com/thorproduct.cfm?partnumber=C240TMD-B'>C240TMD</a> (f=8 mm)
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 1.2 Double-Pass AOMs ──────────────────────────────────
    st.markdown("## 2 · Double-Pass Acousto-Optic Modulators (AOMs)")
    st.markdown("""
<div class='concept-box'>
An AOM diffracts light off a moving acoustic grating, shifting the optical frequency
by f_RF while deflecting the beam by the Bragg angle.  The problem with a single pass:
the Bragg angle depends on f_RF, so <em>tuning the frequency also steers the beam</em>.
This coupling is unacceptable for precision experiments.<br><br>
The solution is <b>double-passing</b>: retroreflect the first-order beam back through the
same crystal.  The frequency shift doubles to <b>2 f_RF</b> and the angular deviations
cancel exactly — the output beam direction is <em>independent</em> of f_RF.  Every
tunable beam in the experiment uses a double-pass AOM.
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        with st.expander("**Cat's-eye retroreflector geometry**"):
            st.markdown("""
The most compact double-pass geometry:

1. AOM crystal → first-order beam deflected by Bragg angle
2. Lens (f = one focal length from crystal) collimates and focuses
3. QWP + mirror at one focal length from lens — retroreflects beam
4. QWP traversed twice → **polarisation rotated 90°**
5. Second pass through AOM → frequency shifted by another f_RF
6. PBS transmits the 2f_RF output (orthogonal polarisation) and rejects zero-order

The cat's-eye geometry makes retroreflection **insensitive to mirror tilt**, which
is why it is preferred over simpler flat-mirror configurations.

**Typical performance:**
- Single-pass efficiency: ~90%
- Double-pass efficiency: ~80%
- Extinction (AOM off): 10⁻⁴ – 10⁻⁵ (40–50 dB)
""")

        with st.expander("**Longitudinal vs. shear-wave AOMs**"):
            st.markdown("""
| Property | Longitudinal | Shear-wave |
|---|---|---|
| Acoustic mode | Atoms oscillate ∥ propagation | Atoms oscillate ⊥ propagation |
| Sound velocity | ~4–6 km/s | ~1–2 km/s |
| Deflection angle | Larger (AOD applications) | Smaller |
| RF bandwidth | Broader | Narrower |
| Diffraction efficiency | Lower at peak | Higher at peak |
| Acoustic power | More | Less |

**In the lab:** Cooling/imaging beams (frequency shifting + switching) → shear-wave AOM.
Tweezer array generation → longitudinal AOM (AOD), because large deflection range is needed
to position hundreds of tweezers.
""")

    with col_r:
        st.markdown("#### 🔧 Switching-time calculator")
        v_s = st.selectbox("Acoustic material", ["TeO₂ (4.0 mm/μs)", "Fused silica (5.9 mm/μs)", "PbMoO₄ (3.6 mm/μs)"])
        vs_dict = {"TeO₂ (4.0 mm/μs)": 4.0, "Fused silica (5.9 mm/μs)": 5.9, "PbMoO₄ (3.6 mm/μs)": 3.6}
        vs = vs_dict[v_s]
        d_beam = st.slider("Beam 1/e² diameter at crystal (μm)", 20, 500, 100, step=10)
        t_rise = (d_beam * 1e-6) / (vs * 1e3) * 1e9  # ns
        st.metric("Rise time (10–90%)", f"{t_rise:.0f} ns")
        if t_rise < 10:
            st.markdown("<div class='tip-box'>Faster than 10 ns — consider an EOM instead (electro-optic effect is instantaneous).</div>", unsafe_allow_html=True)
        elif t_rise > 200:
            st.markdown("<div class='warn-box'>Rise time > 200 ns — focus beam tighter inside crystal.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='tip-box'>Good switching speed for most AMO sequences.</div>", unsafe_allow_html=True)
        st.caption("Formula: t_rise = d_beam / v_s")

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.markdown("""
<div class='vendor-card'>
📦 <b>AOM vendors</b><br>
<a href='https://www.goochandhousego.com/products/acousto-optic-modulators/'>Gooch & Housego</a> — single/double-pass AOMs ·
<a href='https://www.aaoptoelectronic.com/acousto-optics/aom/'>AA Opto-Electronic</a> — MT series AOMs ·
<a href='https://isomet.com/aom.html'>Isomet</a> — wideband AOMs
</div>
""", unsafe_allow_html=True)
    with col_v4:
        st.markdown("""
<div class='vendor-card'>
📦 <b>EOM vendors (fast switching)</b><br>
<a href='https://www.qubig.com/electro-optic-modulator.html'>Qubig</a> — free-space EOMs ·
<a href='https://www.eospace.com/electro-optic-modulator'>EOSpace</a> — guided-wave EOMs ·
<a href='https://www.jenoptik.com/products/optical-systems/electro-optical-components'>Jenoptik</a>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 1.3 Polarimetry ──────────────────────────────────────
    st.markdown("## 3 · Polarimetry — Stokes Parameters & the Poincaré Sphere")
    st.markdown("""
<div class='concept-box'>
Polarisation purity is one of the most critical and least forgiving parameters in AMO
experiments.  A <b>1% admixture</b> of the wrong circular component in an optical-pumping
beam can reduce state-preparation fidelity from >99% to <95%.  A few percent of linear
polarisation in a σ⁺–σ⁻ gray-molasses beam opens decoherence pathways that significantly
degrade cooling.  Measuring polarisation quantitatively — not just qualitatively — is essential.
</div>
""", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("""
#### Stokes parameters

The complete polarisation state is described by four intensity measurements:

| Parameter | Meaning |
|---|---|
| S₀ = I_total | Total power |
| S₁ = I_H − I_V | Horizontal vs. vertical linear |
| S₂ = I₊₄₅ − I₋₄₅ | Diagonal linear |
| S₃ = I_RCP − I_LCP | Right vs. left circular |

Degree of polarisation: **P = √(S₁² + S₂² + S₃²) / S₀**

All four are *intensities* — no electric-field measurement required.

#### Rotating-QWP method (lab measurement)

Place a QWP on a rotation stage before a PBS analyser.
As QWP angle θ is swept, transmitted intensity follows:

**I(θ) = ½(A + B sin2θ + C cos4θ + D sin4θ)**

Stokes parameters extracted by Fourier fit:
S₀ = A − C,  S₁ = 2C,  S₂ = 2D,  S₃ = B

Use 50–100 evenly spaced angles for reliable results.

#### QWP retardance calibration

**δ = 2 cos⁻¹(√(I_min / I_max))**

Measure between two PBS ports with linear input. Any deviation from 90° is corrected
in the fitting procedure. Calibrate every QWP in optical-pumping and imaging paths.
""")

    with col_right:
        st.markdown("#### 🔧 Poincaré sphere — visualise your polarisation state")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            s1_val = st.slider("S₁ (H−V linear)", -1.0, 1.0, 0.0, 0.05)
        with sc2:
            s2_val = st.slider("S₂ (±45° linear)", -1.0, 1.0, 0.0, 0.05)
        with sc3:
            s3_val = st.slider("S₃ (RCP−LCP)", -1.0, 1.0, 1.0, 0.05)

        # Normalise to unit sphere
        mag = np.sqrt(s1_val**2 + s2_val**2 + s3_val**2)
        if mag > 1.0:
            s1n, s2n, s3n = s1_val/mag, s2_val/mag, s3_val/mag
        else:
            s1n, s2n, s3n = s1_val, s2_val, s3_val

        # Build Poincaré sphere
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(len(u)), np.cos(v))

        fig_ps = go.Figure()
        fig_ps.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.12,
                                    colorscale=[[0,"#1a1a4e"],[1,"#2a2a7e"]],
                                    showscale=False))
        # Axes
        for axis, label, col in [([1.3,0,0],["S₁ (H)",""],"#ff8888"),
                                  ([0,1.3,0],["S₂ (+45°)",""],"#88ff88"),
                                  ([0,0,1.3],["S₃ (RCP)",""],"#8888ff")]:
            fig_ps.add_trace(go.Scatter3d(x=[0,axis[0]], y=[0,axis[1]], z=[0,axis[2]],
                mode="lines+text", text=["", label[0]],
                textfont=dict(color=col, size=11),
                line=dict(color=col, width=3), showlegend=False))
        # State point
        fig_ps.add_trace(go.Scatter3d(
            x=[s1n], y=[s2n], z=[s3n], mode="markers",
            marker=dict(size=10, color="#ffcc00", symbol="circle"),
            name="Polarisation state"))
        # Line from origin
        fig_ps.add_trace(go.Scatter3d(x=[0,s1n], y=[0,s2n], z=[0,s3n],
            mode="lines", line=dict(color="#ffcc00", width=4), showlegend=False))
        # Labels for poles
        for pos, label in [([0,0,1.1], "RCP (σ⁺)"), ([0,0,-1.1], "LCP (σ⁻)"),
                            ([1.1,0,0], "H"), ([-1.1,0,0], "V"),
                            ([0,1.1,0], "+45°"), ([0,-1.1,0], "−45°")]:
            fig_ps.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="text", text=[label],
                textfont=dict(color="#aaa", size=10), showlegend=False))

        dop = np.sqrt(s1_val**2 + s2_val**2 + s3_val**2)
        pol_type = "Pure circular (σ⁺)" if s3_val > 0.95 and dop > 0.9 else \
                   "Pure circular (σ⁻)" if s3_val < -0.95 and dop > 0.9 else \
                   "Linear" if abs(s3_val) < 0.05 and dop > 0.9 else \
                   "Elliptical" if dop > 0.5 else "Partially polarised"

        fig_ps.update_layout(
            scene=dict(
                xaxis=dict(title=dict(text="S₁", font=dict(color="#aaa")), showgrid=False,
                           zeroline=False, tickfont=dict(color="#aaa"), backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title=dict(text="S₂", font=dict(color="#aaa")), showgrid=False,
                           zeroline=False, tickfont=dict(color="#aaa"), backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title=dict(text="S₃", font=dict(color="#aaa")), showgrid=False,
                           zeroline=False, tickfont=dict(color="#aaa"), backgroundcolor="rgba(0,0,0,0)"),
                bgcolor="rgba(5,5,20,0.9)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0), height=340,
            legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ps, use_container_width=True, key="poincare")
        st.markdown(f"**State type:** {pol_type} &nbsp;|&nbsp; **Degree of polarisation:** {dop:.2f}")
        st.caption("RCP (σ⁺) = north pole · LCP (σ⁻) = south pole · equator = linear states")

    st.markdown("---")

    # ── 1.4 Laser Frequency Stabilisation ─────────────────────
    st.markdown("## 4 · Laser Frequency Stabilisation")
    st.markdown("""
<div class='concept-box'>
Free-running diode lasers drift by ~MHz per minute due to temperature and current
fluctuations.  Atomic transitions have natural linewidths of 5–6 MHz (D1, D2 lines) or
even ~3.5 Hz (Cs 5D₅/₂ quadrupole).  Three complementary strategies cover all
stabilisation needs in the lab, forming a hierarchy.
</div>
""", unsafe_allow_html=True)

    lock_tabs = st.tabs(["🔬 SAS Locking", "🎵 Beat-Note Locking", "🏗️ PDH Cavity Locking", "🗺️ Hierarchy"])

    with lock_tabs[0]:
        st.markdown("""
### Saturated-Absorption Spectroscopy (SAS) Locking

Two counter-propagating beams through a **heated vapour cell** create a Doppler-free
Lamb dip: only atoms with near-zero velocity see both beams resonantly, so a narrow
(~linewidth) feature appears on the Doppler-broadened background.

**Error signal generation:** FM dithering (current modulation or EOM sideband) +
phase-sensitive detection (lock-in) converts the Lamb dip into an error signal that
crosses zero at line centre.

**Used for:** Cs 852 nm D2 laser · Li 671 nm D1 laser (direct atomic references).

**Not suitable for:** Cs 685 nm quadrupole line (I_sat ≈ 2.3 W/cm², impractical for SAS)
→ use PDH cavity lock instead.
""")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
<div class='vendor-card'>
📦 <b>Vapour cell suppliers</b><br>
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=6816'>Thorlabs Cs/Rb cells</a> ·
<a href='https://www.alvarezresearch.com/'>Alvarez Research</a>
</div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown("""
<div class='vendor-card'>
📦 <b>Lock electronics</b><br>
<a href='https://vescent.com/products/electronics/d2-125-laser-servo/'>Vescent D2-125</a> laser servo ·
<a href='https://www.toptica.com/products/tunable-diode-lasers/stabilization-accessories/'>Toptica DLC Pro</a>
</div>""", unsafe_allow_html=True)

    with lock_tabs[1]:
        st.markdown("""
### Beat-Note (Offset) Locking

When two lasers need a fixed frequency difference, overlap them on a **fast photodiode**
and compare the heterodyne beat-note to an RF reference.

**Example:** Li D2 laser offset phase-locked to Li D1 laser (Vescent electronics),
inheriting the D1 absolute reference while maintaining ~10–100 Hz relative linewidth
for coherent Raman processes.

**Advantages:**
- Tunable offset (change RF reference frequency)
- Fast relative linewidth (~Hz level)
- No vacuum infrastructure required

**Requirement:** One well-stabilised **master** laser as absolute reference.

**Common topology in a dual-species lab:**
```
Li D1 (SAS locked, absolute) ──── beat-lock ──→ Li D2
Cs D2 (SAS locked, absolute) ──── beat-lock ──→ Cs repumper, other Cs beams
```
""")
        st.markdown("""
<div class='vendor-card'>
📦 <b>Beat-note lock electronics</b><br>
<a href='https://vescent.com/products/electronics/d2-135-offset-phase-lock-servo/'>Vescent D2-135</a> offset PLL ·
<a href='https://www.analog.com/en/products/hmc984.html'>Analog Devices HMC984</a> PFD ·
<a href='https://www.minicircuits.com/WebStore/RF-Mixers.html'>Mini-Circuits</a> RF mixers
</div>""", unsafe_allow_html=True)

    with lock_tabs[2]:
        st.markdown("""
### Pound-Drever-Hall (PDH) Cavity Locking

For wavelengths without accessible atomic references, or when **sub-kHz linewidth**
is required, lock to a high-finesse Fabry-Pérot cavity.

**Principle:** Phase-modulate the input beam (EOM at Ω) → detect reflected light on
fast photodiode → demodulate at Ω → antisymmetric error signal that crosses zero
exactly at cavity resonance.

**Key parameters for 685 nm lock (Phatak 2025):**
- Cavity length L = 77.5 mm → ν_FSR ≈ 1.93 GHz
- Finesse F ≈ 15 000 → cavity linewidth ~100 kHz
- ULE glass cavity with vacuum enclosure + mK-level temperature stabilisation
- Cavity drift ≈ 2.5 kHz per 0.01°C temperature excursion
- Resulting laser linewidth: ~1 kHz

**Optical chain:**
Diode → prism pair (astigmatism) → isolator (30–40 dB) → EOM (27 MHz) →
mode-matching telescope → cavity

**Two-loop feedback:** Fast path to laser current (MHz BW) + slow path to PZT (kHz BW)
""")
        st.markdown("""
<div class='warn-box'>
<b>Critical:</b> fractional length stability for 1 kHz laser linewidth is δL/L ~ 10⁻¹², corresponding to picometer-scale cavity length control.  Thermal isolation at the mK level is non-negotiable.
</div>""", unsafe_allow_html=True)
        col_r3, col_r4 = st.columns(2)
        with col_r3:
            st.markdown("""
<div class='vendor-card'>
📦 <b>ULE cavities</b><br>
<a href='https://www.advanced-thin-films.com/products/optical-cavities/'>Advanced Thin Films</a> ·
<a href='https://www.stable-laser.com/'>Stable Laser Systems</a> ·
<a href='https://www.refoc.us/'>REFoc.us</a>
</div>""", unsafe_allow_html=True)
        with col_r4:
            st.markdown("""
<div class='vendor-card'>
📦 <b>Reference reading</b><br>
<a href='https://arxiv.org/abs/physics/0109098'>Black (2001)</a> — PDH tutorial (Am. J. Phys.) ·
<a href='https://doi.org/10.1103/RevModPhys.87.637'>Ludlow et al. (2015)</a> — Optical atomic clocks
</div>""", unsafe_allow_html=True)

    with lock_tabs[3]:
        st.markdown("""
### Lock hierarchy — how all lasers are referenced

The three methods form a complementary hierarchy:

| Method | Absolute ref? | Linewidth | Infrastructure |
|---|---|---|---|
| SAS | ✅ Yes (atomic line) | ~MHz | Vapour cell + lock-in |
| Beat-note | Via master laser | ~Hz (relative) | Fast PD + RF electronics |
| PDH cavity | Via cavity (ULE) | ~kHz | Vacuum + thermal control |

**Hood Lab hierarchy (typical):**
```
Li D1 (671 nm) ─── SAS locked ────────────── absolute reference
   └── Li D2 (671 nm) ─────── beat lock
Cs D2 (852 nm) ─── SAS locked ────────────── absolute reference
   └── Cs repumper ─────────── beat lock
Cs 685 nm ──────── PDH (ULE cavity) ──────── <1 kHz linewidth
1064 nm tweezer ─── free-running (stable laser)
```

In general: the D1 Li laser and the 852 nm Cs laser carry the absolute references;
all other lasers are derived through beat locks or share the ULE cavity reference.
""")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — OPTICAL PUMPING
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Optical Pumping — Internal-State Preparation")
    st.markdown("""
<div class='concept-box'>
Optical pumping drives an atom into a specific Zeeman sublevel via repeated photon absorption
and spontaneous emission cycles.  In a tweezer experiment, its purposes are twofold:
<ol>
  <li><b>State preparation</b> — put the atom in a single, well-defined |F, m_F⟩ state before
  any coherent manipulation (Raman transitions, quadrupole excitation, Rydberg gates).</li>
  <li><b>Detection preparation</b> — define the initial condition for state-selective
  fluorescence imaging.</li>
</ol>
Imperfect optical pumping is a direct source of systematic error in lifetime measurements,
qubit state detection, and gate fidelity.
</div>
""", unsafe_allow_html=True)

    op_tab1, op_tab2 = st.tabs(["🔵 Cesium", "🔴 Lithium"])

    with op_tab1:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("""
### Cesium: pumping to |F=4, m_F=+4⟩

**Scheme:** σ⁺-polarised light driving **F=4 → F′=4** has no allowed absorption for an atom
already in m_F=+4 (which would require Δm_F=+1, but no m′_F=+5 exists in the excited state).
That state is *dark*; all other m_F sublevels are continuously depopulated until the atom
accumulates in m_F=+4.

**Repumper:** A beam resonant with F=3→F′=4 returns any population that decayed to F=3.

**Why the magnetic-field direction matters:**
The quantisation axis is defined by the **local B-field**, not the laser k-vector.
If the bias field is misaligned with the beam, the σ⁺ in the lab frame decomposes into
σ⁺, π, and σ⁻ in the atom's frame — pumping into a single m_F is compromised.
Apply a bias field of ~6 G **along the optical axis** of the pumping beam.

**Diagnostic: the depumping-ratio test**
1. Drive F=4→F′=4 *without* the repumper — atoms in F=4 scatter photons and heat out of the trap.
2. Under aligned B-field: atoms in the dark state |4,+4⟩ survive for ~1 ms (off-resonant scattering only).
3. Under deliberately misaligned B-field (~45°): σ⁺ acquires σ⁻ component → atoms depumped in ~10 μs.
4. **Target depumping ratio > 100** (aligned/misaligned survival times).

**Adjustment knobs:** laser frequency (exact line centre), QWP orientation (polarisation purity),
bias field direction.  Together these achieve >99% pumping fidelity.
""")
        with col_r:
            st.markdown("""
<div class='tip-box'>
<b>Quick diagnosis checklist:</b><br>
✅ ER of pumping beam fiber > 20 dB<br>
✅ Bias field coil along pump beam axis<br>
✅ Pump laser on F=4→F′=4 (not F=3→F′=4)<br>
✅ Repumper on F=3→F′=4<br>
✅ Depumping ratio > 100<br>
✅ Pump pulse duration > 5 × (1/Γ_scatter)
</div>""", unsafe_allow_html=True)
            st.markdown("""
<div class='vendor-card'>
📦 <b>Bias coil drivers</b><br>
<a href='https://www.teamwavelength.com/products/temperature-controllers-laser-diode-drivers/laser-diode-drivers/'>Wavelength Electronics</a> low-noise current sources ·
<a href='https://www.itech.de/en/products/power-supplies/'>itech</a> programmable supplies
</div>""", unsafe_allow_html=True)
            st.markdown("""
<div class='vendor-card'>
📦 <b>Key optics</b><br>
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=753'>Thorlabs PBS cubes</a> ·
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5128'>Thorlabs QWPs</a> ·
<a href='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=756'>Thorlabs HWPs</a>
</div>""", unsafe_allow_html=True)

    with op_tab2:
        st.markdown("""
### Lithium: D2-line optical pumping

**Challenge:** The ⁶Li D2 excited-state hyperfine splitting is only ~5 MHz — comparable to Γ.
Different F′ components overlap, making it impossible to address a single excited hyperfine
level cleanly.  In practice, a combination of F=3/2→F′=3/2 and F=1/2→F′=3/2 light is used.

**Procedure:**
1. Coarse beam alignment to MOT on the diagonal camera.
2. Fine alignment on a single trapped atom.
3. Verify resonance by scanning laser frequency over the atom-loss signal.
4. Characterise fidelity with the depumping-ratio test (same as Cs).

**Advantage:** Ground-state hyperfine splitting of 228 MHz ≫ D2 linewidth → state-selective
probing is clean and free of cross-talk.

**Practical note:** In a dual-species (Li+Cs) setup, the Li pumping beams co-propagate with
the Cs beams — beam geometry and PM-fiber infrastructure are shared, so only minor frequency
and alignment adjustments are needed once Cs pumping is optimised.
""")
        st.markdown("""
<div class='vendor-card'>
📄 <b>Reference</b> ·
<a href='https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf'>Gehm (2003) — Properties of ⁶Li (NCSU)</a> ·
<a href='https://steck.us/alkalidata/cesiumnumbers.pdf'>Steck — Cs D line data</a>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — RF & ELECTRONICS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## RF & Electronics Infrastructure")

    st.markdown("""
<div class='concept-box'>
Coherent control of atomic hyperfine states requires delivering oscillating magnetic fields
at precise frequencies — 228 MHz for ⁶Li ground-state splitting, ~76 MHz in the
Paschen-Back (high-field) regime, 9.2 GHz for Cs.  The engineering challenge is
delivering enough B-field amplitude at the atom while fitting inside an existing vacuum
apparatus.  Safety-critical high-current electronics (Feshbach coils) demand dedicated
hardware interlocks that are independent of the computer control system.
</div>
""", unsafe_allow_html=True)

    rf_tab1, rf_tab2 = st.tabs(["📡 Hyperfine Antenna Design", "⚡ Safety & High-Current"])

    with rf_tab1:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("""
### Loop antenna for Li hyperfine transitions

A **single-turn loop** radiates primarily through its magnetic dipole field.  The radiation
resistance of a small loop of area A at frequency f is:

**R_rad ≈ 20π² (2πA/λ²)² Ω**

For any loop that fits near a vacuum cell, R_rad ≪ 50 Ω.  Efficient power delivery
from a 50 Ω source therefore requires an **impedance-matching network**.

**Three approaches tested (with VNA):**
1. **Capacitive loading** — series or parallel capacitor shifts resonance.
   *Result: parallel ~47 pF proved most effective for 76 MHz, compact geometry.*
2. **Transmission-line stub matching** — moves impedance on Smith chart.
   *Works in principle, but spurious resonances of the stub can complicate things.*
3. **Discrete LC networks** — more design freedom but more components.

**Achieved:** reflection minimum ~10 dB, sufficient B-field at the atom with ~100 W amplifier.

**Lessons learned:**
- Lead length (connector to loop) contributes parasitic inductance at 100 MHz — non-negligible.
- Simulate with **SimSmith** before building every iteration.
- The resonant frequency scales inversely with circumference at fixed inductance.
""")
        with col_r:
            st.markdown("#### 🔧 Loop antenna quick estimate")
            freq_mhz = st.number_input("Target frequency (MHz)", 10.0, 500.0, 76.0, step=1.0)
            loop_diam_cm = st.number_input("Loop diameter (cm)", 0.5, 20.0, 3.0, step=0.5)
            A = np.pi * (loop_diam_cm * 1e-2 / 2)**2
            lam = 3e8 / (freq_mhz * 1e6)
            R_rad = 20 * np.pi**2 * (2 * np.pi * A / lam**2)**2
            L_nH = 12.566 * loop_diam_cm / 2  # approx inductance single-turn loop (nH)
            C_match = 1e12 / ((2*np.pi*freq_mhz*1e6)**2 * L_nH*1e-9)
            st.metric("Radiation resistance R_rad", f"{R_rad*1000:.2f} mΩ")
            st.metric("Loop inductance (approx)", f"{L_nH:.0f} nH")
            st.metric("Resonating capacitance", f"{C_match:.0f} pF")
            st.caption("Approximate formulas — always verify with VNA measurement.")

            st.markdown("""
<div class='vendor-card'>
📦 <b>Tools & instruments</b><br>
<a href='https://www.siglentna.com/spectrum-vna/ssa3000x-plus-series-spectrum-analyzer-and-svna/'>Siglent SVA1015X</a> VNA ·
<a href='https://www.ae6ty.com/smith_charts.html'>SimSmith</a> (free Smith chart simulator) ·
<a href='https://www.minicircuits.com/WebStore/RF-Amplifiers.html'>Mini-Circuits RF amplifiers</a>
</div>""", unsafe_allow_html=True)

    with rf_tab2:
        st.markdown("""
### Feshbach coil safety interlocks

Feshbach coils produce fields of order **1000 G** by carrying large DC currents.
Sudden current interruption generates inductive voltage spikes that can damage
power supplies and coils.

**Hardware interlock logic (essential rules):**
- Monitor: coil temperature, current level, supply voltage
- On threshold exceeded: **ramp down smoothly**, do not switch off abruptly
- All interlock logic implemented in **relay hardware**, independent of computer control
- Interlock circuit must be untriggerable by software bugs

**Why hardware, not software?**  A software interlock can fail if the control computer
crashes, loses communication, or executes a bug.  A relay that physically disconnects
the gate signal of the MOSFET driver cannot be bypassed by software.
""")
        st.markdown("""
<div class='warn-box'>
<b>⚠️ Never bypass a hardware interlock for convenience.</b>
Inductive spikes from sudden coil switching have destroyed power supplies and
cracked coil epoxy in labs worldwide.  The hardware interlock is not optional.
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class='vendor-card'>
📦 <b>High-current supplies</b><br>
<a href='https://www.kepco.com/products/dc-power-supplies/'>Kepco BOP bipolar</a> ·
<a href='https://www.iseg-hv.com/'>iSeg</a> precision current sources ·
<a href='https://www.ametek-programmablepower.com/'>AMETEK Programmable Power</a>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — COMPUTATIONAL TOOLS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Computational Tools for AMO Experiments")

    comp_tab1, comp_tab2 = st.tabs(["🧠 Bayesian Optimisation", "🔬 QuTiP Simulation"])

    with comp_tab1:
        st.markdown("""
<div class='concept-box'>
Optimising an ultracold-atom experiment means navigating a high-dimensional continuous
parameter space where each measurement takes seconds to minutes.  A MOT has ≥5 knobs
(gradient, powers, detunings); a tweezer experiment adds trap depth, cooling polarisation,
B-field direction.  Naïve grid search over 5 variables × 10 points = 10⁵ trials —
months of run time.  <b>Bayesian optimisation (BO)</b> converges in 30–100 evaluations.
</div>
""", unsafe_allow_html=True)

        col_l, col_r = st.columns([2, 3])
        with col_l:
            st.markdown("""
### How it works

**Gaussian-Process Regression (GPR):** a non-parametric Bayesian model that
maintains a probabilistic map of the response surface (e.g. atom survival vs.
laser detuning + power).  At each step it returns a predicted value **and an
uncertainty** — unexplored regions have high uncertainty.

**Acquisition function:** selects the next measurement point by trading off:
- *Exploitation* — sample near current optimum
- *Exploration* — reduce uncertainty in poorly sampled regions

**Common acquisition functions:**
| Name | When to use |
|---|---|
| Expected Improvement (EI) | Default; works well near optimum |
| Upper Confidence Bound (UCB) | When signal is absent, need broad search |
| Probability of Improvement (PI) | Conservative; avoids risk |

**Workflow:**
1. Collect 5–20 initial points from Latin hypercube or random design
2. Train GPR model; inspect posterior mean + uncertainty
3. Select next point by maximising EI/UCB
4. Run experiment, append data, retrain, repeat

**Convergence:** 30–100 evaluations for 4–8 dimensional spaces
(vs. thousands for grid search).
""")
        with col_r:
            st.markdown("#### 🔧 1D Bayesian optimisation demo")
            st.markdown("*Example: optimising atom survival vs. cooling detuning*")

            np.random.seed(42)
            x_true = np.linspace(-5, 5, 300)
            y_true = np.exp(-0.3*(x_true-1.2)**2) * 0.85 + 0.1*np.exp(-0.8*(x_true+2)**2)

            n_obs = st.slider("Number of observations so far", 3, 20, 6)
            obs_x = np.array([-4.2, -1.8, 0.5, 1.1, 2.3, 3.8, -3.1, 0.0, 1.8, 2.8,
                              -2.5, 1.4, -0.5, 3.2, -1.0, 4.1, 0.9, -3.8, 1.6, 2.1])[:n_obs]
            obs_y = np.interp(obs_x, x_true, y_true) + np.random.randn(n_obs)*0.03

            # Simple RBF kernel GP
            l, sigma_n = 1.5, 0.02
            def rbf(x1, x2): return np.exp(-0.5*((x1[:,None]-x2[None,:])/l)**2)
            K_xx = rbf(obs_x, obs_x) + sigma_n**2 * np.eye(n_obs)
            K_xs = rbf(obs_x, x_true)
            K_ss = rbf(x_true, x_true)
            try:
                K_inv = np.linalg.inv(K_xx)
                mu = K_xs.T @ K_inv @ obs_y
                cov_diag = np.diag(K_ss - K_xs.T @ K_inv @ K_xs)
                std = np.sqrt(np.maximum(cov_diag, 0))
            except Exception:
                mu = np.zeros_like(x_true)
                std = np.ones_like(x_true)

            fig_bo = go.Figure()
            fig_bo.add_trace(go.Scatter(x=x_true, y=y_true, mode="lines",
                line=dict(color="#555", dash="dot", width=1.5), name="True function (hidden)"))
            fig_bo.add_trace(go.Scatter(x=np.concatenate([x_true, x_true[::-1]]),
                y=np.concatenate([mu+2*std, (mu-2*std)[::-1]]),
                fill="toself", fillcolor="rgba(123,104,238,0.18)",
                line=dict(color="rgba(0,0,0,0)"), name="GP ±2σ", showlegend=True))
            fig_bo.add_trace(go.Scatter(x=x_true, y=mu, mode="lines",
                line=dict(color="#7b68ee", width=2), name="GP mean"))
            fig_bo.add_trace(go.Scatter(x=obs_x, y=obs_y, mode="markers",
                marker=dict(size=8, color="#ffcc44"), name="Observations"))
            best_x = x_true[np.argmax(mu)]
            fig_bo.add_vline(x=float(best_x), line_dash="dash", line_color="#44ff88",
                             annotation_text=f"Next query: {best_x:.1f}", annotation_font_color="#44ff88")
            fig_bo.update_layout(
                xaxis_title="Cooling detuning (Γ units)",
                yaxis_title="Atom survival",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,30,0.9)",
                font=dict(color="#ccc"), height=280,
                margin=dict(l=20, r=20, t=20, b=30),
                legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="rgba(100,100,150,0.2)", color="#aaa"),
                yaxis=dict(gridcolor="rgba(100,100,150,0.2)", color="#aaa"),
            )
            st.plotly_chart(fig_bo, use_container_width=True, key="bayes_opt")
            st.caption("Grey dotted line = true optimum (unknown to the algorithm). The GP explores and exploits to find it.")

        st.markdown("""
<div class='vendor-card'>
📦 <b>Software</b> ·
<a href='https://scikit-learn.org/stable/modules/gaussian_process.html'>scikit-learn GPR</a> ·
<a href='https://github.com/fmfn/BayesianOptimization'>BayesianOptimization (Python)</a> ·
<a href='https://ax.dev/'>Meta Ax platform</a> ·
<a href='https://arxiv.org/abs/1807.02811'>Shahriari et al. (2016) — BO tutorial arXiv</a>
</div>""", unsafe_allow_html=True)

    with comp_tab2:
        st.markdown("""
<div class='concept-box'>
The master-equation simulations in the Li and Cs chapters are implemented in
<b>QAtomTweezer</b> — a purpose-built Python library built on top of QuTiP.
The key idea is to treat a single atom with <em>full hyperfine structure</em> coupled
to a <b>1D harmonic oscillator</b> representing one motional axis of the tweezer.
</div>
""", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
### Physical model

**Hilbert space:** H = H_internal ⊗ H_HO

- H_internal: all hyperfine and Zeeman sublevels (typically 12–24 states for D1 of an alkali)
- H_HO: harmonic oscillator, truncated at N_HO = 10–20 Fock states
- Total dimension: d_int × N_HO (e.g. 12 × 12 = 144 for Li D1)

**Bare Hamiltonian** includes hyperfine offsets for ground/excited manifolds + ℏω â†â.

**Laser interaction:** each laser contributes a term proportional to the
polarisation-resolved lowering operator ⊗ recoil operator R̂ = e^(iη(â+â†)),
where **η = k x_ho** is the Lamb-Dicke parameter.

**Spontaneous emission:** Lindblad collapse operators that simultaneously change
internal state and kick the motional state.

**Key feature:** the full matrix exponential is used for R̂ (not the Lamb-Dicke
expansion), so the code is valid beyond the strict Lamb-Dicke regime.
""")
        with col_r:
            st.markdown("""
### Code structure

```python
# Entry points:
QAtomTweezer.py          # Physics library
QAtomTweezer_SingleLevel.py  # Driver/scanner

# Main callable:
SteadyStateTweezer(
    x,      # [δ1, δ2, Ω1, Ω2, φ1, φ2]
    wh,     # trap freq in units of Γ
    Nh,     # HO truncation
    atom,   # AtomSettings object
    eta,    # Lamb-Dicke parameter
    pol,    # polarisation config
)
# Returns: ⟨n⟩, P(n), p_e
```

### Validation (3 cross-checks)

1. **Fock distribution** P(n) — fit to Boltzmann form to extract T_eff;
   non-thermal features flag dark-state structure.
2. **Excited-state fraction** p_e — compare to measured photon rate
   (after collection efficiency correction).
3. **Temperature minimum** — verify location in 2D parameter scans.

### Performance

- 144×144 Hilbert space: ~1–5 s per steady-state evaluation
- 2D scan (50×50): ~10–30 min on modern laptop with joblib parallelism
- Auto-warning if P(N_HO−1) > 10⁻³ (truncation error)
""")

        st.markdown("""
<div class='vendor-card'>
📦 <b>Software stack</b> ·
<a href='https://qutip.org/docs/latest/'>QuTiP documentation</a> ·
<a href='https://joblib.readthedocs.io/'>joblib (parallelism)</a> ·
<a href='https://scikit-learn.org/stable/modules/gaussian_process.html'>scikit-learn (GP optimiser)</a> ·
<a href='https://arxiv.org/abs/1211.6518'>Johansson et al. (2013) — QuTiP 2 paper</a>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — LASER SYSTEMS
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## The Laser System")
    st.markdown("""
<div class='concept-box'>
AMO experiments for Li–Cs tweezer work require four distinct laser wavelengths, each with
its own technical demands.  All are produced by <b>external-cavity diode lasers (ECDLs)</b>
in Littrow configuration, except the 1064 nm tweezer which uses a commercial Nd:YAG/fiber laser.
This section explains the ECDL operating principles and the specific choices made for each wavelength.
</div>
""", unsafe_allow_html=True)

    with st.expander("**External-Cavity Diode Lasers (ECDLs) — general principles**"):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
A bare semiconductor diode lases on multiple longitudinal modes, with center frequency
set by chip geometry and refractive index.

**Two tuning mechanisms in a bare diode:**
- Injection current: ∂ν/∂I ~ 1–3 GHz/A (fast but noisy)
- Temperature: ∂ν/∂T ~ −20 to −40 GHz/K (slow, hysteretic)

Neither alone provides the narrow linewidth (<100 kHz) or mode-hop-free tuning range
(>1 GHz) required for AMO experiments.

**The Littrow ECDL solution:**
- Holographic grating at Littrow angle feeds first-order diffraction back into the diode
- Selects one longitudinal mode of the extended cavity
- Grating angle (PZT-tuned) + injection current → mode-hop-free tuning over 1–2 GHz
- AR coating on front facet suppresses internal Fabry-Pérot resonances

**Typical performance:**
- Free-running linewidth: ~100 kHz (cavity-limited)
- Mode-hop-free scan: 1–2 GHz
- Tuning: PZT (fine, ~30 MHz/V) + temperature (coarse, ~GHz/°C)
""")
        with col_r:
            st.markdown("""
**Common failure modes:**
- Mode hops: usually from temperature drift of the diode chip; cure with better temperature control
- Reduced output power: check AR coating; diodes degrade with age and excessive current
- Multiple modes: grating feedback misaligned; realign with DC current scan while monitoring mode on wavemeter
- Linewidth broadening: current noise from noisy driver; use low-noise current source

**ECDL alignment tips:**
1. Set temperature for approximate target wavelength (check Thorlabs/Toptica diode specs)
2. Coarsely align grating to first-order feedback with IR card
3. Monitor wavelength on wavemeter; find single-mode region
4. Maximise mode-hop-free range by co-scanning PZT + current (feed-forward)
""")
            st.markdown("""
<div class='vendor-card'>
📦 <b>ECDL vendors</b><br>
<a href='https://www.toptica.com/products/tunable-diode-lasers/ecdl-dfb-lasers/dl-pro/'>Toptica DL Pro</a> ·
<a href='https://www.sacher-laser.com/home/home/product_detail/lion/model/lion.html'>Sacher LION</a> ·
<a href='https://www.moglabs.com/products/laser-systems/external-cavity-diode-laser'>MOGLabs CEL</a> ·
<a href='https://www.riotphotonics.com/products/laser-systems/'>RIO photonics</a> (DFB)
</div>""", unsafe_allow_html=True)

    laser_data = {
        "671 nm — Lithium D-lines": {
            "species": "⁶Li / ⁷Li",
            "transitions": "D1 (670.992 nm) and D2 (670.977 nm)",
            "role": "MOT (D2), Zeeman slower (D2), Λ-GM cooling and tweezer imaging (D1)",
            "linewidth": "Γ/2π = 5.87 MHz (D1 and D2 nearly identical)",
            "challenge": "D1 and D2 lines separated by only ~10 GHz — both needed; D2 laser offset-locked to D1. Lightest alkali: recoil temperature 3.54 μK, largest recoil in the alkalis.",
            "lock": "D1: SAS locked to vapour cell. D2: beat-note locked to D1 (Vescent D2-135).",
            "diode": "Toptica or home-built ECDL at 671 nm; TA amplifier often needed for MOT power",
            "thorlabs": "https://www.thorlabs.com/thorproduct.cfm?partnumber=L671P200",
            "color": "#ff8844",
        },
        "852 nm — Cesium D2": {
            "species": "¹³³Cs",
            "transitions": "D2 (852.347 nm); D1 at 894.6 nm also used in some labs",
            "role": "MOT, fluorescence imaging, repumper",
            "linewidth": "Γ/2π = 5.23 MHz",
            "challenge": "Large hyperfine splitting (9.193 GHz) means cooler and repumper must be offset by ~9.2 GHz — use AOM chain or separate laser with beat lock.",
            "lock": "SAS locked directly to Cs D2 line in vapour cell.",
            "diode": "Readily available 852 nm diodes from Eagleyard, Laser Components, Thorlabs",
            "thorlabs": "https://www.thorlabs.com/thorproduct.cfm?partnumber=L852P150",
            "color": "#44cc88",
        },
        "685 nm — Cesium quadrupole (5D₅/₂)": {
            "species": "¹³³Cs",
            "transitions": "6S₁/₂ → 5D₅/₂ (electric quadrupole, Γ/2π ≈ 117.6 kHz = 3.5 Hz natural linewidth)",
            "role": "Narrow-line sideband cooling of Cs in tweezer; excited-state lifetime measurement",
            "linewidth": "Γ/2π = 117.6 kHz  (3.5 Hz natural; resolved sidebands at typical trap freq)",
            "challenge": "Forbidden E2 transition → I_sat ≈ 2.3 W/cm² (much higher than D-lines). No SAS possible. Requires PDH lock to ULE cavity for <1 kHz linewidth. Astigmatism correction needed (prism pair before cavity).",
            "lock": "PDH locked to ULE cavity (L=77.5 mm, F≈15000, linewidth ~100 kHz). Laser linewidth ~1 kHz.",
            "diode": "Commercial 685 nm diode; AR-coated front facet strongly preferred for stable ECDL operation",
            "thorlabs": "https://www.thorlabs.com/thorproduct.cfm?partnumber=L685P010",
            "color": "#7b68ee",
        },
        "1064 nm — Optical tweezer trap": {
            "species": "Li, Cs (both)",
            "transitions": "Far-detuned (no resonant absorption); acts as conservative dipole trap",
            "role": "Creates the optical tweezer potential; all atoms trapped in the 1064 nm focus",
            "linewidth": "N/A — intensity noise matters, not frequency noise",
            "challenge": "Intensity noise at trap frequencies (kHz) causes parametric heating. RIN (relative intensity noise) target: <−130 dBc/Hz at trap sidebands. Needs intensity stabilisation (AOM servo on a pick-off PD).",
            "lock": "Free-running (stable Nd:YAG or fiber laser); intensity servo via AOM.",
            "diode": "Coherent Mephisto, NKT Photonics Koheras, or Azurlight fiber amplifier",
            "thorlabs": "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=15",
            "color": "#ffcc44",
        },
    }

    for laser_name, info in laser_data.items():
        with st.expander(f"**{laser_name}**"):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"**Species:** {info['species']}  &nbsp;|&nbsp;  **Transitions:** {info['transitions']}")
                st.markdown(f"**Role in experiment:** {info['role']}")
                st.markdown(f"**Linewidth:** {info['linewidth']}")
                st.markdown(f"**Key challenge:** {info['challenge']}")
                st.markdown(f"**Locking strategy:** {info['lock']}")
            with c2:
                st.markdown(f"**Laser source:** {info['diode']}")
                st.markdown(f"📦 [Thorlabs diode example]({info['thorlabs']})")
                st.markdown("""
<div class='vendor-card'>
📦 <b>Other diode sources</b><br>
<a href='https://www.eagleyard.com/products/'>Eagleyard Photonics</a> ·
<a href='https://www.lasercomponents.com/us/laser-diodes/'>Laser Components</a> ·
<a href='https://www.coherent.com/'>Coherent</a>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# REFERENCES
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📚 References & Further Reading")

col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.markdown("""
### Primary source
- <a id='ref-thesis'></a>[Phatak (2025) — PhD Thesis, Purdue (Chapter 6)](https://hoodlab.physics.purdue.edu)
  *"Cooling Lithium and Cesium Single Atoms in Optical Tweezers"*

### Textbooks
- Metcalf & van der Straten — *Laser Cooling and Trapping* (1999)
- Foot — *Atomic Physics* (OUP, 2005)
- Saleh & Teich — *Fundamentals of Photonics* (Wiley)
- Pozar — *Microwave Engineering* (Wiley, 4th ed.)
""", unsafe_allow_html=True)

with col_r2:
    st.markdown("""
### Key papers & notes
- [Black (2001) — PDH locking tutorial](https://arxiv.org/abs/physics/0109098)
- [Johansson et al. (2013) — QuTiP 2](https://arxiv.org/abs/1211.6518)
- [Kaufman & Ni (2021) — Tweezer arrays review](https://arxiv.org/abs/2009.07073)
- [Shahriari et al. (2016) — Bayesian optimisation](https://arxiv.org/abs/1807.02811)
- [Torrontegui et al. (2023) — ML for AMO (review)](https://arxiv.org/abs/2104.05648)
""")

with col_r3:
    st.markdown("""
### Vendor application notes
- [Thorlabs fiber coupling guide](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10760)
- [AA Opto AOM selection guide](https://www.aaoptoelectronic.com/acousto-optics/aom/)
- [Vescent laser locking app note](https://vescent.com/resources/)
- [SimSmith Smith chart tool](https://www.ae6ty.com/smith_charts.html)
- [QuTiP documentation](https://qutip.org/docs/latest/)
- [scikit-learn GP regression](https://scikit-learn.org/stable/modules/gaussian_process.html)
""")

st.markdown("---")
st.caption("Built for curious96.com · Based on Phatak (2025) PhD Thesis Chapter 6, Purdue University Hood Lab · All vendor links for reference only")
