"""
Quantum Computing Landscape — curious96.com
Based on Chapter 8 of Phatak (2025) PhD Thesis, Purdue University Hood Lab
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Computing Landscape",
    page_icon="⚛️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0a0a0f; color: #e2e8f0; }
  .block-container { padding: 2rem 3rem; max-width: 1400px; }

  h1 { font-size: 2.4rem; font-weight: 700; color: #f0f4ff;
       border-bottom: 2px solid #7c3aed; padding-bottom: .5rem; }
  h2 { color: #c4b5fd; font-size: 1.5rem; }
  h3 { color: #a78bfa; font-size: 1.15rem; }

  /* concept cards */
  .concept-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: .8rem 0;
  }
  .concept-box h4 { color: #c7d2fe; margin: 0 0 .4rem 0; font-size: 1rem; }
  .concept-box p  { color: #a5b4fc; margin: 0; font-size: .9rem; line-height: 1.55; }

  /* platform cards */
  .platform-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: .6rem 0;
    transition: border-color .2s;
  }
  .platform-card:hover { border-color: #7c3aed; }
  .platform-card h4 { color: #f1f5f9; margin: 0 0 .5rem 0; font-size: 1.05rem; }
  .platform-card p  { color: #94a3b8; margin: 0; font-size: .88rem; line-height: 1.55; }

  /* company pills */
  .company-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #3b82f6;
    border-radius: 20px;
    padding: .2rem .75rem;
    font-size: .8rem;
    margin: .2rem .2rem;
    font-weight: 600;
  }

  /* metric badges */
  .metric-green {
    display: inline-block; background:#064e3b; color:#6ee7b7;
    border:1px solid #059669; border-radius:6px;
    padding:.15rem .6rem; font-size:.78rem; margin:.1rem;
  }
  .metric-yellow {
    display: inline-block; background:#451a03; color:#fcd34d;
    border:1px solid #d97706; border-radius:6px;
    padding:.15rem .6rem; font-size:.78rem; margin:.1rem;
  }
  .metric-red {
    display: inline-block; background:#450a0a; color:#fca5a5;
    border:1px solid #dc2626; border-radius:6px;
    padding:.15rem .6rem; font-size:.78rem; margin:.1rem;
  }

  /* DiVincenzo criteria */
  .divincenzo-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: .4rem 0;
  }
  .divincenzo-box h5 { color: #e2e8f0; margin: 0 0 .3rem 0; font-size:.95rem; }
  .divincenzo-box p  { color: #94a3b8; margin: 0; font-size:.85rem; }

  /* formula box */
  .formula-box {
    background: #0c1a0c;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin: .7rem 0;
    font-family: 'Courier New', monospace;
    color: #86efac;
    font-size: .88rem;
  }

  /* info alert */
  .info-box {
    background: #0c1e3a;
    border: 1px solid #1e40af;
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin: .7rem 0;
    color: #bfdbfe;
    font-size: .9rem;
  }

  /* warning */
  .warn-box {
    background: #1c1007;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin: .7rem 0;
    color: #fde68a;
    font-size: .9rem;
  }

  /* section separator */
  hr { border-color: #1e293b; margin: 2rem 0; }

  /* thesis quote */
  .thesis-quote {
    background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%);
    border-left: 4px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    color: #c4b5fd;
    font-style: italic;
    font-size: .92rem;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 8px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    color: #c4b5fd !important;
    background: #1e1b4b !important;
    border-radius: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.title("⚛️ Quantum Computing Landscape")
st.markdown("""
<div style='color:#94a3b8; font-size:1rem; margin-bottom:1.5rem;'>
A researcher's guide to the hardware race — platforms, companies, metrics, and where neutral atoms fit in.<br>
<span style='color:#6366f1; font-size:.85rem;'>
Based on Chapter 8 · Phatak (2025), Purdue University Hood Lab
</span>
</div>
""", unsafe_allow_html=True)

# ── NISQ intro ────────────────────────────────────────────────────────────────
st.markdown("## The NISQ Era: Where We Are")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class='concept-box'>
    <h4>🎯 The Goal</h4>
    <p>Fault-tolerant quantum computers that run Shor's algorithm on RSA-2048 (~10⁶ physical qubits,
    error rate &lt;10⁻³) and simulate industrial molecules (FeMoco, nitrogenase) with ~100 logical qubits
    at error rates below 10⁻⁴.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='concept-box'>
    <h4>📍 Where We Are (NISQ)</h4>
    <p>Noisy Intermediate-Scale Quantum devices — tens to hundreds of physical qubits, no error
    correction, circuit depths limited by decoherence. Google's 53-qubit supremacy (2019) and photonic
    Gaussian boson sampling confirmed quantum advantage on specific sampling tasks, but no practically
    useful algorithms yet. <em>— Preskill (2018)</em></p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='concept-box'>
    <h4>🚧 The Gap</h4>
    <p>Current best two-qubit error rates are 10⁻³ to 10⁻². Surface codes need &lt;10⁻³ per gate.
    At ~441 physical qubits per logical qubit (distance-21 surface code), RSA-2048 breaking requires
    ~2×10⁶ physical qubits — 3–4 orders of magnitude beyond today.</p>
    </div>
    """, unsafe_allow_html=True)

# ── DiVincenzo criteria ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## The DiVincenzo Criteria")
st.markdown("""
<div class='info-box'>
<strong>DiVincenzo (2000)</strong> — A physical system is a viable qubit platform only if it satisfies
all five criteria <em>simultaneously, at scale, and with fault-tolerance margins</em>.
No platform has fully satisfied all five yet — this is what makes quantum computing hard.
</div>
""", unsafe_allow_html=True)

dv_cols = st.columns(5)
criteria = [
    ("1️⃣ Scalable Register",
     "Well-defined |0⟩,|1⟩ two-level subspace, replicable in large numbers without performance degradation."),
    ("2️⃣ Initialisation",
     "Reliable preparation of the fiducial |000…0⟩ state before each computation. Requires optical pumping + ground-state cooling."),
    ("3️⃣ Long Coherence",
     "T₂ ≫ t_gate. Decoherence must be slow vs. gate time. For neutral atoms: T₂ > 1 s is achievable; tweezer photon scattering erodes it."),
    ("4️⃣ Universal Gates",
     "High-fidelity single-qubit rotations + at least one entangling two-qubit gate (e.g. Rydberg blockade for neutral atoms)."),
    ("5️⃣ Qubit Readout",
     "Site-specific measurement in the computational basis with near-unity fidelity, without disturbing neighbours."),
]

for col, (title, desc) in zip(dv_cols, criteria):
    with col:
        st.markdown(f"""
        <div class='divincenzo-box'>
        <h5>{title}</h5>
        <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Analog vs Digital ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Analog vs. Digital Quantum Computing")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class='platform-card'>
    <h4>🔢 Digital (Gate-Based)</h4>
    <p>Programs decomposed into discrete unitary gates from a universal gate set, applied to qubits,
    then measured. Mathematically clean and compatible with quantum error correction. <br><br>
    <strong>Cost:</strong> Error correction needs 100s–1000s of physical qubits per logical qubit.
    Every gate must exceed the fault-tolerance threshold (~10⁻³ for surface codes).<br><br>
    <strong>Who:</strong> IBM, Google, IonQ, Quantinuum, QuEra (Rydberg gates)</p>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class='platform-card'>
    <h4>🌊 Analog (Simulation / Annealing)</h4>
    <p>Engineer a Hamiltonian whose ground state encodes the answer, then adiabatically evolve
    from an easy initial state. Special-purpose, not universally programmable, but far larger
    system sizes are accessible in the NISQ era.<br><br>
    <strong>Examples:</strong> D-Wave's 5000+ superconducting flux qubit annealers; neutral-atom
    arrays studying quantum magnetism on 2D lattices of hundreds of atoms.<br><br>
    <strong>Neutral atoms are unique:</strong> the same array can run Rydberg gates (digital) OR evolve
    a programmable spin Hamiltonian (analog) — just by changing the control protocol.
    <em>Pasqal & QuEra exploit this dual capability as a near-term strategy.</em></p>
    </div>
    """, unsafe_allow_html=True)

# ── Platform Comparison Radar ────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Platform Comparison")
st.markdown("*Scores are normalised (0–10) across six dimensions. Hover for details.*")

categories = [
    "Gate Fidelity", "Coherence Time", "Qubit Count",
    "Gate Speed", "CMOS Integration", "Reconfigurability"
]

platforms_data = {
    "Superconducting": [9.5, 4, 8.5, 9.5, 6, 3],
    "Trapped Ions":    [9.9, 9, 4,   3,   2, 5],
    "Neutral Atoms":   [8.5, 9, 9,   7,   3, 9.5],
    "Silicon Spins":   [9.8, 7, 2,   9,   9.5, 2],
    "Photonic":        [6,   9.5, 7, 9.5, 8, 6],
    "NV / Topological":[6,   9.5, 1, 3,   4, 2],
}
colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899", "#14b8a6"]

def hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

fig_radar = go.Figure()
for (name, scores), color in zip(platforms_data.items(), colors):
    fig_radar.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=name,
        line_color=color,
        fillcolor=hex_to_rgba(color, 0.15),
        opacity=0.85,
    ))

fig_radar.update_layout(
    polar=dict(
        bgcolor="#0f172a",
        radialaxis=dict(visible=True, range=[0, 10], color="#475569",
                        gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10)),
        angularaxis=dict(color="#94a3b8", gridcolor="#1e293b",
                         tickfont=dict(color="#c4b5fd", size=11)),
    ),
    paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
    font=dict(color="#e2e8f0"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155", borderwidth=1,
                font=dict(color="#e2e8f0", size=11)),
    margin=dict(l=60, r=60, t=40, b=40),
    height=480,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Qubit count bar chart ─────────────────────────────────────────────────────
st.markdown("### Current Qubit Counts & Gate Fidelities (≈ 2024–25)")

tab_bar1, tab_bar2 = st.tabs(["Qubit Count", "2-Qubit Gate Fidelity (%)"])

with tab_bar1:
    systems = [
        ("IBM Condor", "Superconducting", 1121),
        ("IBM Heron r2", "Superconducting", 156),
        ("Google Willow", "Superconducting", 105),
        ("QuEra Aquila", "Neutral Atoms", 256),
        ("Atom Computing Phoenix", "Neutral Atoms", 1180),
        ("Pasqal Fresnel", "Neutral Atoms", 100),
        ("IonQ Forte", "Trapped Ions", 36),
        ("Quantinuum H2", "Trapped Ions", 56),
        ("Intel Tunnel Falls", "Silicon Spins", 12),
        ("PsiQuantum", "Photonic", 0),  # not disclosed / fusion-based
    ]
    sys_names = [s[0] for s in systems]
    sys_counts = [s[2] for s in systems]
    sys_types  = [s[1] for s in systems]
    color_map = {
        "Superconducting": "#3b82f6",
        "Neutral Atoms": "#8b5cf6",
        "Trapped Ions": "#10b981",
        "Silicon Spins": "#f59e0b",
        "Photonic": "#ec4899",
    }
    bar_colors = [color_map[t] for t in sys_types]

    fig_bar = go.Figure(go.Bar(
        x=sys_names, y=sys_counts,
        marker_color=bar_colors,
        text=sys_counts, textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(color="#94a3b8", gridcolor="#1e293b", tickangle=-30),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b", title="Physical Qubits"),
        margin=dict(t=30, b=80),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Note: Atom Computing Phoenix (1,180 atoms) demonstrated array loading; not all atoms used as qubits simultaneously. PsiQuantum uses photonic fusion gates — qubit count metric is not directly comparable.")

with tab_bar2:
    platforms_fid = [
        ("Superconducting\n(IBM/Google)", 99.5),
        ("Trapped Ions\n(Quantinuum)", 99.9),
        ("Neutral Atoms\n(best demo)", 99.5),
        ("Silicon Spins\n(Si/SiGe)", 99.8),
        ("Photonic\n(linear optical)", 93),
    ]
    pf_names = [p[0] for p in platforms_fid]
    pf_vals  = [p[1] for p in platforms_fid]
    pf_colors= ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"]

    fig_fid = go.Figure(go.Bar(
        x=pf_names, y=pf_vals,
        marker_color=pf_colors,
        text=[f"{v}%" for v in pf_vals], textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
    ))
    fig_fid.add_hline(y=99.9, line_dash="dot", line_color="#ef4444",
                      annotation_text="Surface code threshold ~99.9%",
                      annotation_font_color="#fca5a5")
    fig_fid.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b",
                   range=[88, 100.5], title="2-Qubit Gate Fidelity (%)"),
        margin=dict(t=30, b=40),
        height=380,
    )
    st.plotly_chart(fig_fid, use_container_width=True)
    st.caption("Best published values. Superconducting: IBM Heron/Google Willow. Trapped ions: Quantinuum H-series. Neutral atoms: individual atom pair demos (Evered et al. 2023, Harvard). Silicon spins: isotopically-purified ²⁸Si.")

# ── Platform deep dives ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Platform Deep Dives")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔵 Superconducting",
    "🟢 Trapped Ions",
    "🟣 Neutral Atoms",
    "🟡 Silicon Spins",
    "🩷 Photonic",
    "🩵 Defects & Topological",
])

# ─ Superconducting ─
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Superconducting Qubits")
        st.markdown("""
        <div class='platform-card'>
        <h4>How it works</h4>
        <p>Transmon qubits — Josephson junction circuits cooled to ~10 mK in dilution refrigerators.
        Anharmonic oscillators whose two lowest energy levels form the qubit. Microwave pulses drive
        single- and two-qubit gates. The Josephson junction provides the nonlinearity that makes the
        energy ladder anharmonic, isolating the |0⟩↔|1⟩ transition.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Key Metrics (≈ 2024)</h4>
        <p>
        <span class='metric-green'>Gate time: 10–200 ns</span>
        <span class='metric-green'>1Q fidelity: &gt;99.9%</span>
        <span class='metric-yellow'>2Q fidelity: ~99.5%</span>
        <span class='metric-yellow'>T₁, T₂: 50–500 µs</span>
        <span class='metric-yellow'>Ops/T₂: ~10³</span>
        <span class='metric-green'>Qubit count: 100–1000+</span>
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Principal Bottleneck</h4>
        <p>Classical wiring infrastructure inside the dilution refrigerator. Each qubit needs multiple
        coaxial lines for control and readout. Heat load + cable density become prohibitive beyond
        ~few thousand qubits. Solutions: cryo-CMOS multiplexing inside the fridge,
        microwave-to-optical transduction for remote coupling.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
# Josephson energy sets qubit frequency
E_J = (Φ₀/2π)² / (2L_J)   [Josephson inductance]
ω_q ≈ √(8 E_J E_C) / ℏ    [transmon frequency]
where E_C = e²/2C_Σ        [charging energy]
Anharmonicity: α = ω_12 - ω_01 ≈ -E_C/ℏ ≈ -200 to -350 MHz
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Companies")
        companies_sc = [
            ("IBM Quantum", "Eagle (127Q), Osprey (433Q), Condor (1121Q), Heron r2 (156Q, 2024). Quantum volume roadmap. Cloud access via IBM Quantum Platform.", "https://quantum.ibm.com"),
            ("Google Quantum AI", "Sycamore (53Q, 2019 supremacy), Willow (105Q, 2024 — below threshold error correction demonstration).", "https://quantumai.google"),
            ("Rigetti Computing", "Aspen/Ankaa series. Publicly traded, cloud hybrid classical–quantum.", "https://rigetti.com"),
            ("IQM Quantum Computers", "Finnish startup. Resonance (20Q), modular architecture. Focus on HPC integration.", "https://meetiqm.com"),
            ("Alice & Bob", "Cat qubit approach — biased noise qubits to reduce overhead for error correction.", "https://alice-bob.com"),
            ("D-Wave", "Quantum annealing (5000+ flux qubits). Advantage2 processor. Not gate-based.", "https://dwavesys.com"),
        ]
        for name, desc, url in companies_sc:
            st.markdown(f"""
            <div class='platform-card'>
            <h4><a href='{url}' target='_blank' style='color:#93c5fd;text-decoration:none;'>{name} ↗</a></h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Key Papers")
        st.markdown("""
        - [Google supremacy (Arute et al., *Nature* 2019)](https://www.nature.com/articles/s41586-019-1666-5)
        - [Surface code logical qubit (Google, *Nature* 2023)](https://www.nature.com/articles/s41586-022-05434-1)
        - [Willow chip — below-threshold error correction (Google, *Nature* 2024)](https://www.nature.com/articles/s41586-024-08449-y)
        - [IBM Heron r2 — 156Q heavy-hex (IBM, 2024)](https://research.ibm.com/blog/ibm-quantum-heron)
        """)

# ─ Trapped Ions ─
with tab2:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Trapped Ion Qubits")
        st.markdown("""
        <div class='platform-card'>
        <h4>How it works</h4>
        <p>Hyperfine "clock" states of ¹⁷¹Yb⁺ or ⁴³Ca⁺ ions confined in radiofrequency Paul traps.
        Ions are laser-cooled to the motional ground state; shared vibrational modes of the ion chain
        act as a quantum bus. The Mølmer–Sørensen gate drives an entangling operation via collective
        phonon modes.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Key Metrics (≈ 2024)</h4>
        <p>
        <span class='metric-green'>Coherence T₂: 1 s – minutes</span>
        <span class='metric-green'>2Q fidelity: ~99.9%</span>
        <span class='metric-yellow'>Gate time: 20–200 µs</span>
        <span class='metric-green'>1Q fidelity: &lt;10⁻⁴ error</span>
        <span class='metric-yellow'>Qubit count: 30–56</span>
        <span class='metric-green'>Ops/T₂: &gt;10⁴</span>
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Principal Bottleneck</h4>
        <p>Motional-mode crowding: adding ions to a linear chain increases shared vibrational modes,
        making gates slower and harder to address selectively. Scaling path: the quantum charge-coupled
        device (QCCD) architecture — ions shuttled between storage and gate zones in segmented traps.
        Photonic interconnects link separate traps into a modular network.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
# Mølmer-Sørensen gate (bichromatic light)
H_MS = ℏΩ(σ⁺ₐσ⁺ᵦ + σ⁻ₐσ⁻ᵦ) · cos(δt)
where δ = ω_laser - ω_qubit - ω_phonon (sideband detuning)
Gate time: τ_g ~ 2π/η²Ω  [η = Lamb-Dicke parameter]
Fidelity limited by: Rabi freq. stability, heating rate, mode crowding
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Companies")
        companies_ti = [
            ("IonQ", "Forte (36 qubits). NASDAQ-listed (IONQ). #AQ metric for algorithmic performance. Partners with AWS, Azure.", "https://ionq.com"),
            ("Quantinuum", "H2 (56 trapped Yb⁺ qubits). Joint venture Honeywell + Cambridge Quantum. Highest algorithmic fidelity demonstrated. H-series roadmap.", "https://quantinuum.com"),
            ("Oxford Ionics", "Uses electronic microwave signals (no lasers). Built on existing semiconductor fab lines. Quieter than laser-driven gates.", "https://oxfordionics.com"),
            ("AQT (Alpine Quantum Technologies)", "Ca⁺ ion trap systems. European focus. Partners with CERN for HPC integration.", "https://aqt.eu"),
        ]
        for name, desc, url in companies_ti:
            st.markdown(f"""
            <div class='platform-card'>
            <h4><a href='{url}' target='_blank' style='color:#6ee7b7;text-decoration:none;'>{name} ↗</a></h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Key Papers")
        st.markdown("""
        - [Mølmer–Sørensen gate (Sørensen & Mølmer, *PRL* 1999)](https://link.aps.org/doi/10.1103/PhysRevLett.82.1971)
        - [Quantinuum H2 — 99.9% 2Q fidelity (2023)](https://arxiv.org/abs/2305.03828)
        - [QCCD architecture (Kielpinski et al., *Nature* 2002)](https://www.nature.com/articles/nature00784)
        - [56-qubit Quantinuum H2 demonstration (2024)](https://arxiv.org/abs/2404.14989)
        """)

# ─ Neutral Atoms ─
with tab3:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Neutral Atom Qubits (Optical Tweezers)")
        st.markdown("""
        <div class='platform-card'>
        <h4>How it works</h4>
        <p>Individual neutral atoms (Rb, Cs, Yb, Sr) trapped in tightly focused laser beams (optical
        tweezers) — intensity gradient provides a restoring force. Hyperfine ground states form the
        qubit. Two-qubit gates use the <strong>Rydberg blockade</strong>: transient excitation to
        highly-excited Rydberg states (n ~ 60–100) whose strong dipole-dipole interaction (V_dd ~ n⁷)
        prevents simultaneous excitation of two nearby atoms.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Key Metrics (≈ 2024)</h4>
        <p>
        <span class='metric-green'>Qubit count: 50–1000+</span>
        <span class='metric-green'>Coherence T₂: 1–10 s</span>
        <span class='metric-yellow'>2Q fidelity: 97–99.5%</span>
        <span class='metric-green'>Gate time: 0.2–5 µs</span>
        <span class='metric-green'>Identical qubits: atoms are perfect copies</span>
        <span class='metric-green'>Reconfigurable geometry: real-time rearrangement</span>
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Three Bottlenecks (from this thesis)</h4>
        <p>
        <strong>① Imaging fidelity:</strong> Must scatter photons for readout while atom survives.
        Prior work on ⁶Li: 90–97% per-image survival. Chapter 3 achieves 99.950(2)% survival over
        2000 consecutive images via Λ-enhanced gray molasses.<br><br>
        <strong>② Qubit temperature:</strong> Rydberg gate fidelity ∝ 1 − α⟨n⟩. At ω_trap ~
        2π×100 kHz, going from 20 µK → 5 µK cuts motional error by 4×. Chapter 4 demonstrates
        ⟨n⟩ ≈ 0.01 for Cs via narrow-line sideband cooling on the 685 nm quadrupole transition.<br><br>
        <strong>③ Precision benchmarking:</strong> τ(5D₅/₂) for Cs measured to 1% accuracy
        (Chapter 5), pinning the saturation intensity, scattering rate, and magic-trap condition
        to the level needed for systematic gate optimisation.
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
# Rydberg blockade condition
V_dd(r) = C₆/r⁶   [van der Waals interaction, C₆ ∝ n¹¹]
Blockade radius: r_b = (C₆/Ω_Ryd)^(1/6)
Typical r_b ~ 5–15 µm for n = 60–100

# Gate fidelity vs temperature (Phatak 2025, §8.5.2)
F ≈ 1 - α·⟨n⟩   [motional error per Rydberg π-pulse]
⟨n⟩ = k_B T / (ℏ ω_trap)

# Thesis result (Cs, Chapter 4):
T ≈ 5 µK,  ⟨n⟩ ≈ 0.01,  ~99% ground-state population
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='thesis-quote'>
        "Neutral-atom optical tweezer arrays have evolved from a laboratory curiosity to a commercial
        quantum computing platform in less than a decade. The platform's appeal rests on several features
        that are structurally difficult to replicate with other technologies: the atoms are perfectly
        identical, ground-state hyperfine coherence times exceed 1 second, and the array geometry is
        reconfigurable in real time." — Phatak (2025), §8.5
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Companies")
        companies_na = [
            ("QuEra Computing", "Aquila (256 atoms). Spin-out from Harvard/MIT (Lukin/Greiner groups). Analog + digital modes. AWS partnership. Largest publicly accessible neutral-atom QPU.", "https://quera.com"),
            ("Pasqal", "Fresnel (100 Rb atoms). French startup (Browaeys/Lahaye group). Hybrid analog-digital. HPC integration via EDF, BASF partnerships.", "https://pasqal.com"),
            ("Atom Computing", "Phoenix (1180 Sr atoms demonstrated). Focus on ⁸⁷Sr nuclear spin qubits (longer T₂). Modular architecture.", "https://atom-computing.com"),
            ("Infleqtion", "Formerly ColdQuanta. Rb & Cs platforms. Acquired SuperTech. Cloud access & quantum networking.", "https://infleqtion.com"),
        ]
        for name, desc, url in companies_na:
            st.markdown(f"""
            <div class='platform-card'>
            <h4><a href='{url}' target='_blank' style='color:#c4b5fd;text-decoration:none;'>{name} ↗</a></h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Key Papers")
        st.markdown("""
        - [Tweezer arrays review (Kaufman & Ni, *Annu. Rev. Chem.* 2021)](https://arxiv.org/abs/2109.10087)
        - [Rydberg blockade gate (Evered et al., Harvard *Nature* 2023 — 99.5% fidelity)](https://www.nature.com/articles/s41586-023-06481-y)
        - [QuEra 256-qubit analog (Ebadi et al., *Nature* 2021)](https://www.nature.com/articles/s41586-021-03582-4)
        - [⁶Li imaging 2000× (Blodgett, Phatak et al., *PRL* 2023)](https://link.aps.org/doi/10.1103/PhysRevLett.131.083001)
        - [Cs quadrupole cooling (Blodgett, Phatak et al., *PRA* 2025)](https://arxiv.org/abs/2505.10540)
        - [Generalized cooling theory (Phatak et al., *PRA* 2024)](https://link.aps.org/doi/10.1103/PhysRevA.110.043116)
        """)

# ─ Silicon Spins ─
with tab4:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Silicon Spin Qubits")
        st.markdown("""
        <div class='platform-card'>
        <h4>How it works</h4>
        <p>Electron or hole spins confined in silicon/silicon-germanium quantum dots. An
        electrostatic gate defines a potential well holding a single electron; its spin-up/down states
        form the qubit. Single-qubit gates: microwave-driven electron spin resonance. Two-qubit gates:
        exchange interaction tuned by gate voltage. In isotopically purified ²⁸Si (nuclear-spin-free),
        the dominant dephasing source (background ²⁹Si nuclear spins) is eliminated.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Key Metrics (≈ 2024)</h4>
        <p>
        <span class='metric-green'>Gate time: 20–50 ns</span>
        <span class='metric-green'>2Q fidelity: 99.8% (²⁸Si)</span>
        <span class='metric-green'>T₂: &gt;1 ms in ²⁸Si</span>
        <span class='metric-green'>Ops/T₂: ~10⁴–10⁵</span>
        <span class='metric-red'>Qubit count: 6–12 (2024)</span>
        <span class='metric-red'>Variability: no two dots are identical</span>
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Why it's compelling long-term</h4>
        <p>Integration density: millions of quantum dots can in principle be fabricated using existing
        CMOS processes (same fabs that make Intel chips). This is the most credible path to
        millions of physical qubits if variability can be solved. Automated tuning protocols —
        machine learning for quantum dot tuning — are the active research frontier.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Companies")
        companies_si = [
            ("Intel", "Tunnel Falls (12Q, 2023). Fab-compatible process. Cryo-CMOS Horse Ridge controller chip. Integration with Intel's existing 300mm fabs.", "https://intel.com/quantumcomputing"),
            ("Quantum Motion", "UK startup. CMOS-compatible Si/SiO₂ platform. Raises Series B 2023. Focus on scalable qubit arrays.", "https://quantummotion.tech"),
            ("Silicon Quantum Computing", "Australian national initiative (UNSW). Precision atom placement in Si:P. Sub-nm gate control.", "https://sqc.com.au"),
            ("Equal1", "Irish startup. Full stack on a single CMOS chip — qubits + classical control at 4K.", "https://equal1.com"),
        ]
        for name, desc, url in companies_si:
            st.markdown(f"""
            <div class='platform-card'>
            <h4><a href='{url}' target='_blank' style='color:#fcd34d;text-decoration:none;'>{name} ↗</a></h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Key Papers")
        st.markdown("""
        - [Six-qubit Si/SiGe processor (Philips et al., *Nature* 2022)](https://www.nature.com/articles/s41586-022-04903-3)
        - [99.8% 2Q fidelity in ²⁸Si (Noiri et al., *Nature* 2022)](https://www.nature.com/articles/s41586-022-04566-0)
        - [Intel Tunnel Falls 12-qubit (2023)](https://arxiv.org/abs/2309.14781)
        - [ML-assisted dot tuning (Zwolak et al., *PRApplied* 2021)](https://link.aps.org/doi/10.1103/PhysRevApplied.15.034011)
        """)

# ─ Photonic ─
with tab5:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Photonic Qubits")
        st.markdown("""
        <div class='platform-card'>
        <h4>How it works</h4>
        <p>Qubits encoded in polarisation, path, or time-bin modes of single photons propagating
        through integrated waveguide chips. Decoherence is negligible in transmission; photonic
        chips can integrate millions of modes on a wafer. Single-qubit gates: beam splitters and
        phase shifters. The fundamental difficulty: photons don't interact — entangling gates must
        be probabilistic (linear optical quantum computing, KLM protocol) or mediated through
        engineered nonlinearities.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Fusion-Based QC (PsiQuantum approach)</h4>
        <p>Rather than deterministic gates, resource states (small entangled photon clusters) are
        fused by Bell measurements. Failures are heralded and corrected by the architecture itself.
        Key advantage: photons are naturally flying qubits — perfect for quantum networking and
        modular QC with optical interconnects. Error rates per fusion: ~1–10%.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Near-term Value</h4>
        <p>Quantum communication / QKD; optical interconnects between otherwise isolated qubit
        modules; Gaussian boson sampling demonstrations (Xanadu Borealis, ~216 squeezed modes,
        2022 — beyond-classical throughput on specific sampling problems).</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Companies")
        companies_ph = [
            ("PsiQuantum", "Largest-funded photonic QC startup (~$700M). Fusion-based QC on GlobalFoundries silicon photonics. Target: 1M+ photonic qubits.", "https://psiquantum.com"),
            ("Xanadu", "Borealis (216 squeezed modes). PennyLane open-source framework. Gaussian boson sampling. Amazon Braket partner.", "https://xanadu.ai"),
            ("QuiX Quantum", "Boson sampling processors. Si₃N₄ waveguide chips. European quantum flagship.", "https://quixquantum.com"),
            ("Quandela", "Semiconductor quantum dot single-photon sources (InGaAs). Muse cloud platform.", "https://quandela.com"),
        ]
        for name, desc, url in companies_ph:
            st.markdown(f"""
            <div class='platform-card'>
            <h4><a href='{url}' target='_blank' style='color:#f9a8d4;text-decoration:none;'>{name} ↗</a></h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Key Papers")
        st.markdown("""
        - [KLM linear optical QC (Knill et al., *Nature* 2001)](https://www.nature.com/articles/35106030)
        - [Photonic quantum advantage: Gaussian boson sampling (Zhong et al., *Science* 2020)](https://www.science.org/doi/10.1126/science.abe8770)
        - [Borealis — programmable photonic advantage (Madsen et al., *Nature* 2022)](https://www.nature.com/articles/s41586-022-04725-x)
        - [Fusion-based QC (Bartolucci et al., *Nature Commun.* 2023)](https://www.nature.com/articles/s41467-023-36493-1)
        """)

# ─ Defects & Topological ─
with tab6:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Solid-State Defects & Topological Qubits")
        st.markdown("""
        <div class='platform-card'>
        <h4>NV Centres & Diamond Defects</h4>
        <p>Nitrogen-vacancy centres in diamond — a nitrogen atom adjacent to a lattice vacancy — have
        an electron spin with millisecond-to-hour coherence at <em>room temperature</em>. Unique
        among qubit platforms. Optically addressable; near-room-temperature operation. Principal
        near-term role: quantum repeater nodes in distributed networks and precision sensing
        (magnetometry at nanoscale), rather than dense on-chip processors. Related: SiV centres
        in diamond (Lukin group, Harvard), defects in SiC.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='platform-card'>
        <h4>Topological Qubits (Majorana)</h4>
        <p>Theoretically compelling — Majorana zero modes at the ends of 1D topological superconductors
        encode quantum information non-locally, giving intrinsic protection from local perturbations.
        Error rates theoretically far below 10⁻⁶. Huge overhead reduction for fault-tolerance.<br><br>
        <strong>Status:</strong> Unambiguous Majorana detection remains controversial; no braiding
        operation has been demonstrated. Physics frontier, not yet an engineering option.
        Microsoft announced topological qubit progress (2023–2025) but peer-reviewed braiding
        demonstrations are awaited.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='warn-box'>
        ⚠️ <strong>Topological qubits remain undemonstrated at the computational level.</strong>
        Microsoft's announcements have faced scrutiny; independent confirmation of Majorana modes
        is ongoing (2025). Treat timelines with appropriate skepticism.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Companies & Groups")
        companies_def = [
            ("Microsoft Azure Quantum", "Topological qubit programme (Station Q). InAs/Al heterostructures. Announced topological qubit chip (2025). Also partners with IonQ, Quantinuum on Azure.", "https://azure.microsoft.com/en-us/solutions/quantum-computing"),
            ("Quantum Brilliance", "Room-temperature NV-centre quantum accelerators. Diamond-based QPU. Partnered with Oak Ridge National Lab.", "https://quantumbrilliance.com"),
            ("Q-Next / Argonne", "US DOE quantum center. NV centres for quantum repeaters. Quantum network testbed Chicago–Argonne.", "https://q-next.org"),
        ]
        for name, desc, url in companies_def:
            st.markdown(f"""
            <div class='platform-card'>
            <h4><a href='{url}' target='_blank' style='color:#67e8f9;text-decoration:none;'>{name} ↗</a></h4>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Key Papers")
        st.markdown("""
        - [NV centre qubit review (Doherty et al., *Phys. Rep.* 2013)](https://www.sciencedirect.com/science/article/pii/S0370157313000562)
        - [Diamond NV quantum network (Hensen et al., *Nature* 2015)](https://www.nature.com/articles/nature15759)
        - [Topological superconductor theory (Kitaev 2001)](https://arxiv.org/abs/cond-mat/0010440)
        - [Microsoft topological qubit chip (2025)](https://arxiv.org/abs/2503.12097)
        """)

# ── Fault-tolerance roadmap ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Road to Fault-Tolerant Quantum Computing")

st.markdown("""
<div class='info-box'>
<strong>Surface code</strong> — The leading error correction scheme. Encodes 1 logical qubit in d²
physical qubits (d = code distance). Can correct any error affecting fewer than d/2 qubits per
syndrome round. At physical error rate ~10⁻³, a distance-21 surface code gives one logical qubit
at 10⁻¹⁰ logical error rate — at the cost of 441 physical qubits and continuous syndrome measurement.
Running Shor's algorithm on RSA-2048 requires ~4000 logical qubits → <strong>~2×10⁶ physical qubits</strong>.
</div>
""", unsafe_allow_html=True)

# Timeline figure
milestones = {
    "Superconducting": [
        (2019, "Google 53Q supremacy"),
        (2023, "Surface code logical qubit (Google)"),
        (2024, "Willow: below-threshold 1 logical qubit"),
        (2027, "~10 logical qubits (projected)"),
        (2030, "1000 logical qubits (projected)"),
    ],
    "Trapped Ions": [
        (2021, "IonQ 32-qubit system"),
        (2023, "Quantinuum 99.9% 2Q gate"),
        (2024, "56-qubit H2"),
        (2027, "Modular QCCD network (projected)"),
        (2032, "Fault-tolerant logical qubits (projected)"),
    ],
    "Neutral Atoms": [
        (2021, "QuEra 256-atom analog processor"),
        (2023, "Harvard 99.5% Rydberg 2Q gate"),
        (2023, "Li 2000× imaging (Phatak et al.)"),
        (2025, "Cs ground-state cooling ⟨n⟩≈0.01 (Phatak et al.)"),
        (2026, "Mid-circuit measurement arrays (projected)"),
        (2029, "Surface code on neutral atoms (projected)"),
    ],
}

timeline_colors = {"Superconducting": "#3b82f6", "Trapped Ions": "#10b981", "Neutral Atoms": "#8b5cf6"}

fig_tl = go.Figure()
y_map = {"Superconducting": 2, "Trapped Ions": 1, "Neutral Atoms": 0}
y_labels = {2: "Superconducting", 1: "Trapped Ions", 0: "Neutral Atoms"}

for platform, events in milestones.items():
    for year, label in events:
        is_projected = "(projected)" in label
        fig_tl.add_trace(go.Scatter(
            x=[year], y=[y_map[platform]],
            mode="markers+text",
            marker=dict(
                size=14 if not is_projected else 10,
                color=timeline_colors[platform],
                symbol="circle" if not is_projected else "circle-open",
                line=dict(width=2, color=timeline_colors[platform]),
            ),
            text=[label.replace(" (projected)", " ⋯")],
            textposition="top center",
            textfont=dict(color="#e2e8f0" if not is_projected else "#64748b", size=9),
            name=platform,
            showlegend=(label == list(milestones[platform])[-1][1]),
            hovertemplate=f"<b>{platform}</b><br>{year}: {label}<extra></extra>",
        ))

for platform, y in y_map.items():
    years = sorted([e[0] for e in milestones[platform]])
    fig_tl.add_trace(go.Scatter(
        x=years, y=[y]*len(years),
        mode="lines",
        line=dict(color=timeline_colors[platform], width=2, dash="solid"),
        showlegend=False,
        hoverinfo="skip",
    ))

fig_tl.update_layout(
    paper_bgcolor="#0a0a0f", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0"),
    xaxis=dict(title="Year", color="#94a3b8", gridcolor="#1e293b", range=[2018, 2033],
               dtick=2),
    yaxis=dict(tickvals=[0,1,2],
               ticktext=["Neutral Atoms", "Trapped Ions", "Superconducting"],
               color="#94a3b8", gridcolor="#1e293b"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155", borderwidth=1,
                font=dict(color="#e2e8f0")),
    margin=dict(t=20, b=40),
    height=360,
)
st.plotly_chart(fig_tl, use_container_width=True)
st.caption("Open markers = projected milestones. Projected dates are speculative estimates based on current roadmaps (2025).")

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Platform Snapshot Table (≈ 2024)")
st.markdown("""
*Reproduced and updated from Table 8.1 in Phatak (2025). Gate times and fidelities are for two-qubit
operations unless noted. T₂ is the dephasing time; ops/T₂ is two-qubit gates per coherence time.*
""")

table_html = """
<style>
  .qt { width:100%; border-collapse:collapse; font-size:.88rem; }
  .qt th { background:#1e1b4b; color:#c4b5fd; padding:.6rem .8rem; text-align:left; border-bottom:2px solid #312e81; }
  .qt td { color:#e2e8f0; padding:.5rem .8rem; border-bottom:1px solid #1e293b; }
  .qt tr:hover td { background:#0f172a; }
  .qt tr:nth-child(3) td { color:#c4b5fd; font-weight:600; }  /* neutral atoms = thesis platform */
</style>
<table class='qt'>
<tr>
  <th>Platform</th><th>Qubits (2024)</th><th>Gate time (2Q)</th>
  <th>2Q Fidelity</th><th>T₂</th><th>Ops / T₂</th><th>Key Bottleneck</th>
</tr>
<tr>
  <td>Superconducting</td><td>100–1000+</td><td>50–200 ns</td>
  <td>99.5%</td><td>50 µs – 0.5 ms</td><td>~10³</td><td>Classical wiring at mK</td>
</tr>
<tr>
  <td>Trapped Ions</td><td>30–56</td><td>20–200 µs</td>
  <td>99.9%</td><td>1 s – minutes</td><td>&gt;10⁴</td><td>Mode crowding & gate speed</td>
</tr>
<tr>
  <td>⭐ Neutral Atoms</td><td>50–1180</td><td>0.2–5 µs</td>
  <td>97–99.5%</td><td>1–10 s</td><td>10³–10⁴</td><td>Gate fidelity, mid-circuit readout</td>
</tr>
<tr>
  <td>Silicon Spins</td><td>6–12</td><td>20–50 ns</td>
  <td>99.8%</td><td>&gt;1 ms (²⁸Si)</td><td>~10⁴–10⁵</td><td>Variability & tuning</td>
</tr>
<tr>
  <td>Photonic</td><td>&gt;100 modes</td><td>probabilistic</td>
  <td>90–97%</td><td>N/A</td><td>limited</td><td>No photon-photon interaction</td>
</tr>
<tr>
  <td>NV / Topological</td><td>1–few</td><td>µs–ms</td>
  <td>~99% (NV)</td><td>ms – hours</td><td>varies</td><td>Scale / Majorana undemonstrated</td>
</tr>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)
st.markdown("⭐ = Neutral atoms: the platform studied in this thesis.")

# ── Neutral atom roadmap ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Neutral Atom Roadmap: What Needs to Happen")

roadmap_items = [
    ("① Gate Fidelity → 99.9%", "in_progress",
     "Optimal-control pulse shaping + ground-state-cooled atoms (⟨n⟩ < 0.1). Chapters 3 & 4 address this directly for Li and Cs."),
    ("② Scale to ~10⁴ atoms", "near_term",
     "Multiplexed AOD arrays + parallel imaging. Atom Computing Phoenix demonstrated 1180 atoms loaded (2023)."),
    ("③ Mid-circuit measurement", "near_term",
     "Feed-forward operations demonstrated at few-qubit level. Essential for active error correction."),
    ("④ Photonic interconnects", "future",
     "Modular scaling beyond a single array. Photonic links between tweezer modules. Research stage."),
    ("⑤ Surface code demo", "future",
     "First demonstrations expected following similar achievements on superconducting (Google 2023) and trapped-ion systems."),
]

status_colors = {
    "in_progress": ("#064e3b", "#6ee7b7", "#059669", "Active"),
    "near_term": ("#1c1007", "#fcd34d", "#d97706", "Near-term"),
    "future": ("#1e1b4b", "#a78bfa", "#7c3aed", "Future"),
}

for title, status, desc in roadmap_items:
    bg, txt, border, label = status_colors[status]
    st.markdown(f"""
    <div style='background:{bg}; border-left:4px solid {border}; border-radius:8px;
                padding:.9rem 1.2rem; margin:.5rem 0;'>
    <span style='color:{txt}; font-weight:700; font-size:.95rem;'>{title}</span>
    <span style='float:right; background:{border}22; color:{txt}; border:1px solid {border};
                 border-radius:12px; padding:.1rem .6rem; font-size:.75rem;'>{label}</span>
    <p style='color:#94a3b8; margin:.4rem 0 0 0; font-size:.88rem;'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

# ── More page suggestions ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Other Pages on This Site")
st.markdown("""
<div style='display:flex; gap:1rem; flex-wrap:wrap;'>
<a href='/Atom_Library' style='text-decoration:none;'>
  <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:.8rem 1.2rem;
              color:#e2e8f0;min-width:180px;'>
  <div style='color:#60a5fa;font-weight:700;'>📚 Atom Library</div>
  <div style='color:#64748b;font-size:.82rem;'>Laser-coolable atoms & datasheets</div>
  </div>
</a>
<a href='/Lab_Techniques' style='text-decoration:none;'>
  <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:.8rem 1.2rem;
              color:#e2e8f0;min-width:180px;'>
  <div style='color:#34d399;font-weight:700;'>🔬 Lab Techniques</div>
  <div style='color:#64748b;font-size:.82rem;'>AMO instrumentation guide</div>
  </div>
</a>
</div>
""", unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='color:#475569; font-size:.82rem; text-align:center; padding:.5rem 0;'>
Built for <a href='https://curious96.com' style='color:#818cf8;'>curious96.com</a> ·
Based on Phatak (2025) PhD Thesis Chapter 8, Purdue University Hood Lab ·
Platform data approximate 2024–25 values · Open markers in timeline = projected
</div>
""", unsafe_allow_html=True)
