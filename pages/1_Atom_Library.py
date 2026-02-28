"""
Atom Library
============
Resource page: laser-coolable atoms, their spectroscopic properties,
canonical data sheets, and US research groups.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Atom Library",
    page_icon="🧪",
    layout="wide",
)

# ── Shared CSS ──────────────────────────────────────────────────
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
.atom-card {
    background: #12122b;
    border: 1px solid #2a2a5e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.group-card {
    background: #0f1028;
    border-left: 3px solid #44aaff;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.92rem;
    color: #ccc;
}
.tag {
    display: inline-block;
    background: #1e3a5f;
    color: #7ec8e3;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.82rem;
    margin: 2px;
}
.tag-red   { background: #3f1010; color: #ff8888; }
.tag-green { background: #0f3020; color: #66cc88; }
.tag-gold  { background: #3f2f00; color: #ffcc44; }
h1, h2, h3 { color: #c8b8ff; }
a { color: #7b68ee; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.title("🧪 Atom Library")
st.markdown("#### Laser-coolable atoms · spectroscopic data · data sheets · US research groups")

st.markdown("""
<div class='concept-box'>
This page is a starting-point resource for anyone entering the field of ultracold atoms
and neutral-atom quantum science.  Choose an atom family below to explore key spectroscopic
properties, download canonical data sheets (Steck and others), and find the leading
US research groups working with each species.  A quick comparison table lets you see
all atoms side-by-side.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MASTER DATA TABLE
# ═══════════════════════════════════════════════════════════════

ATOMS = [
    # name, Z, A, stats, family, cool_λ_nm, Γ_MHz, T_D_uK, T_r_uK, I_nuc, HF_GHz, notes
    ("⁶Li",   3,  6,  "Fermion", "Alkali",         671.0, 5.87,  141,   3.54, "1",   0.228,  "Strong Feshbach res.; Li-Cs molecules"),
    ("⁷Li",   3,  7,  "Boson",   "Alkali",         671.0, 5.87,  141,   3.54, "3/2", 0.804,  "Early BEC; light mass → large recoil"),
    ("²³Na",  11, 23, "Boson",   "Alkali",         589.0, 9.80,  235,   2.40, "3/2", 1.772,  "First BEC (Ketterle 1995, Nobel 2001)"),
    ("³⁹K",   19, 39, "Boson",   "Alkali",         767.0, 6.04,  145,   0.42, "3/2", 0.462,  "Feshbach res.; Fermi-Hubbard model"),
    ("⁴⁰K",   19, 40, "Fermion", "Alkali",         767.0, 6.04,  145,   0.42, "4",   None,   "Fermionic isotope; degenerate Fermi gas"),
    ("⁴¹K",   19, 41, "Boson",   "Alkali",         767.0, 6.04,  145,   0.42, "3/2", 0.254,  "Less common; BEC demonstrated"),
    ("⁸⁵Rb",  37, 85, "Boson",   "Alkali",         780.2, 6.07,  146,   0.36, "5/2", 3.036,  "Feshbach res. for tuneable interactions"),
    ("⁸⁷Rb",  37, 87, "Boson",   "Alkali",         780.2, 6.07,  146,   0.36, "3/2", 6.835,  "Most popular qubit (6.8 GHz clock)"),
    ("¹³³Cs", 55, 133,"Boson",   "Alkali",         852.3, 5.23,  125,   0.20, "7/2", 9.193,  "Defines the SI second; Li-Cs molecules"),
    ("⁸⁸Sr",  38, 88, "Boson",   "Alkaline Earth", 461.0, 32.0,  770,   0.46, "0",   None,   "Narrow 689 nm line → T_D=0.18 μK; optical clock"),
    ("⁸⁷Sr",  38, 87, "Fermion", "Alkaline Earth", 461.0, 32.0,  770,   0.46, "9/2", None,   "10 nuclear spin states; SU(N) physics"),
    ("¹⁷⁴Yb", 70, 174,"Boson",   "Alkaline Earth", 399.0, 28.0,  4.4,   0.20, "0",   None,   "556 nm narrow line; mHz clock transition"),
    ("¹⁷¹Yb", 70, 171,"Fermion", "Alkaline Earth", 399.0, 28.0,  4.4,   0.20, "1/2", None,   "Effective spin-1/2 qubit; clock QC"),
    ("¹⁶⁴Dy", 66, 164,"Boson",   "Magnetic",       421.0, None,  None,  None, "0",   None,   "Largest magnetic moment (10 μB); dipolar physics"),
    ("¹⁶⁸Er", 68, 168,"Boson",   "Magnetic",       401.0, None,  None,  None, "0",   None,   "Large magnetic moment (7 μB); anisotropic interactions"),
]

# Render — is a placeholder for missing data
def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)

rows = []
for (name, Z, A, stats, family, cool_λ, Γ, T_D, T_r, I_nuc, HF, notes) in ATOMS:
    rows.append({
        "Atom": name,
        "Z": Z,
        "A": A,
        "Statistics": stats,
        "Family": family,
        "Cool. λ (nm)": fmt(cool_λ),
        "Γ/2π (MHz)": fmt(Γ),
        "T_D (μK)": fmt(T_D),
        "T_r (μK)": fmt(T_r),
        "Nuclear spin I": I_nuc,
        "HF split (GHz)": fmt(HF),
        "Key notes": notes,
    })

df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
# QUICK COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

st.markdown("## 📊 Quick Comparison")

family_filter = st.multiselect(
    "Filter by atom family",
    ["Alkali", "Alkaline Earth", "Magnetic"],
    default=["Alkali", "Alkaline Earth", "Magnetic"],
)
df_show = df[df["Family"].isin(family_filter)]

st.dataframe(
    df_show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Atom":           st.column_config.TextColumn("Atom", width="small"),
        "Statistics":     st.column_config.TextColumn("Statistics", width="small"),
        "Family":         st.column_config.TextColumn("Family", width="small"),
        "Key notes":      st.column_config.TextColumn("Key notes", width="large"),
    },
)

st.caption("T_D = Doppler temperature limit  |  T_r = recoil temperature  |  HF = ground-state hyperfine splitting  |  — = not applicable or varies by isotope")


# ═══════════════════════════════════════════════════════════════
# VISUAL: Doppler temperature bar chart
# ═══════════════════════════════════════════════════════════════

with st.expander("📈 Visual comparison — Doppler temperature & linewidth"):
    plot_atoms = [r for r in ATOMS if isinstance(r[6], float) and isinstance(r[7], (int, float))]
    names  = [r[0] for r in plot_atoms]
    T_Ds   = [r[7] for r in plot_atoms]
    gammas = [r[6] for r in plot_atoms]
    colors_bar = ["#7b68ee" if r[4] == "Alkali" else "#44aaff" if r[4] == "Alkaline Earth" else "#ff6b6b"
                  for r in plot_atoms]

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_td = go.Figure(go.Bar(
            x=names, y=T_Ds, marker_color=colors_bar,
            text=[f"{v:.0f}" for v in T_Ds], textposition="outside",
            textfont=dict(color="#fff"),
        ))
        fig_td.update_layout(
            title=dict(text="Doppler temperature T_D (μK)", font=dict(color="#c8b8ff")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.9)",
            yaxis=dict(title="T_D (μK)", tickfont=dict(color="#aaa"),
                       gridcolor="rgba(100,100,150,0.2)"),
            xaxis=dict(tickfont=dict(color="#fff")),
            margin=dict(l=20, r=20, t=40, b=20), height=300,
        )
        st.plotly_chart(fig_td, use_container_width=True, key="td_bar")

    with col_v2:
        fig_gm = go.Figure(go.Bar(
            x=names, y=gammas, marker_color=colors_bar,
            text=[f"{v:.1f}" for v in gammas], textposition="outside",
            textfont=dict(color="#fff"),
        ))
        fig_gm.update_layout(
            title=dict(text="Natural linewidth Γ/2π (MHz)", font=dict(color="#c8b8ff")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.9)",
            yaxis=dict(title="Γ/2π (MHz)", tickfont=dict(color="#aaa"),
                       gridcolor="rgba(100,100,150,0.2)"),
            xaxis=dict(tickfont=dict(color="#fff")),
            margin=dict(l=20, r=20, t=40, b=20), height=300,
        )
        st.plotly_chart(fig_gm, use_container_width=True, key="gm_bar")

    st.markdown("""
**Purple = Alkali &nbsp;|&nbsp; Blue = Alkaline Earth &nbsp;|&nbsp; Red = Magnetic**

Alkaline-earth atoms (Sr, Yb) have *narrow intercombination lines* not shown here —
the 689 nm Sr line has Γ/2π = 7.6 kHz (T_D = 0.18 μK!) and the Yb 556 nm line
has Γ/2π = 182 kHz, enabling sub-Doppler temperatures without extra sub-Doppler stages.
""")


# ═══════════════════════════════════════════════════════════════
# ATOM FAMILY TABS
# ═══════════════════════════════════════════════════════════════

st.markdown("## 🔬 Atom Details")
tab_alk, tab_ae, tab_mag = st.tabs(["⚗️ Alkali Atoms", "🌟 Alkaline-Earth & Yb", "🧲 Magnetic Atoms"])


# ── ALKALI ──────────────────────────────────────────────────────
with tab_alk:
    st.markdown("""
<div class='concept-box'>
Alkali atoms (Li, Na, K, Rb, Cs) have a single valence electron, giving a simple
hydrogen-like level structure.  Their D1 and D2 lines in the visible/near-IR are
accessible with diode lasers.  Hyperfine ground states form natural two-level
qubits.  <b>Rb-87</b> (6.8 GHz clock) and <b>Cs-133</b> (9.2 GHz, defines the SI
second) are the two most widely used qubit atoms.
</div>
""", unsafe_allow_html=True)

    alkali_detail = {
        "⁶Li / ⁷Li  — Lithium": {
            "why": "Lightest alkali. Large recoil energy enables efficient sub-Doppler cooling. Strong Feshbach resonances allow tunable interactions. Li-6 is the primary fermionic atom for strongly-correlated physics.",
            "lines": "D1: 670.992 nm  |  D2: 670.977 nm  |  Γ/2π = 5.87 MHz",
            "qubit": "Li-7 |F=1⟩↔|F=2⟩ hyperfine qubit (803 MHz); Li-6 used for fermionic qubits",
            "steck_6": "https://steck.us/alkalidata/lithium6numbers.pdf",
            "steck_7": "https://steck.us/alkalidata/lithiumdata.pdf",
            "arxiv": "arXiv:1007.3348 — Gehm, Properties of ⁶Li",
            "color": "#ff8844",
        },
        "²³Na  — Sodium": {
            "why": "First atom used to achieve BEC (Ketterle group, MIT, 1995, Nobel Prize 2001). Yellow D-line at 589 nm. Larger scattering length suitable for BEC studies. Being revisited for molecule formation (NaLi, NaK, NaRb).",
            "lines": "D1: 589.756 nm  |  D2: 588.995 nm  |  Γ/2π = 9.80 MHz",
            "qubit": "|F=1⟩↔|F=2⟩ hyperfine qubit (1772 MHz)",
            "steck_6": "https://steck.us/alkalidata/sodiumnumbers.pdf",
            "steck_7": None,
            "arxiv": "Steck data sheet (above) is the canonical reference",
            "color": "#ffcc44",
        },
        "³⁹K / ⁴⁰K / ⁴¹K  — Potassium": {
            "why": "K-40 is the only naturally abundant fermionic alkali; key for Fermi-Hubbard model simulations. K-39 has accessible Feshbach resonances. All isotopes share the same 767/770 nm D-lines (diode laser accessible).",
            "lines": "D1: 770.108 nm  |  D2: 766.701 nm  |  Γ/2π = 6.04 MHz",
            "qubit": "K-39: |1,−1⟩↔|2,2⟩ clock-like transition",
            "steck_6": "https://steck.us/alkalidata/potassiumnumbers.pdf",
            "steck_7": None,
            "arxiv": "arXiv:0906.2888 — Falke et al., K spectroscopy",
            "color": "#aa44ff",
        },
        "⁸⁵Rb / ⁸⁷Rb  — Rubidium": {
            "why": "Rb-87 is the most widely used quantum computing atom — large 6.8 GHz hyperfine splitting, convenient 780 nm lasers, and well-understood collisional properties. The backbone of most Rydberg tweezer quantum computers today.",
            "lines": "D1: 794.979 nm  |  D2: 780.241 nm  |  Γ/2π = 6.07 MHz",
            "qubit": "Rb-87 |0,0⟩↔|1,1⟩ or |1,−1⟩↔|2,1⟩ 'clock' qubit (6835 MHz)",
            "steck_6": "https://steck.us/alkalidata/rubidium87numbers.pdf",
            "steck_7": "https://steck.us/alkalidata/rubidium85numbers.pdf",
            "arxiv": "arXiv:1312.3632 — Steck Rb-87 data sheet (v2.2)",
            "color": "#44aaff",
        },
        "¹³³Cs  — Caesium": {
            "why": "Largest hyperfine splitting of the alkalis (9.19 GHz — defines the SI second). Excellent for optical tweezer work (heavy mass → low recoil, tight confinement). Used in Li-Cs molecular assembly experiments.",
            "lines": "D1: 894.593 nm  |  D2: 852.347 nm  |  Γ/2π = 5.23 MHz",
            "qubit": "|3,0⟩↔|4,0⟩ 'clock' transition (9193 MHz, field-insensitive)",
            "steck_6": "https://steck.us/alkalidata/cesiumnumbers.pdf",
            "steck_7": None,
            "arxiv": "arXiv:1601.06691 — Steck Cs data sheet",
            "color": "#44ff88",
        },
    }

    for atom_name, info in alkali_detail.items():
        with st.expander(f"**{atom_name}**"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Why it's used:** {info['why']}")
                st.markdown(f"**Key lines:** `{info['lines']}`")
                st.markdown(f"**Qubit transition:** {info['qubit']}")
                st.markdown(f"**Further reading:** {info['arxiv']}")
            with c2:
                st.markdown("**Data sheets:**")
                st.markdown(f"📄 [Primary Steck data sheet]({info['steck_6']})")
                if info["steck_7"]:
                    st.markdown(f"📄 [Second isotope data sheet]({info['steck_7']})")
                st.markdown("📚 [All Steck alkali data](https://steck.us/alkalidata/)")
                st.markdown("🔗 [NIST Atomic Spectra DB](https://physics.nist.gov/asd)")


# ── ALKALINE EARTH ───────────────────────────────────────────────
with tab_ae:
    st.markdown("""
<div class='concept-box'>
Alkaline-earth and alkaline-earth-like atoms (Ca, Sr, Yb) have two valence electrons,
giving rich level structures including ultra-narrow <em>intercombination</em> and
<em>clock</em> transitions.  These enable <b>sub-recoil cooling</b> on the narrow line,
exceptional coherence times, and optical-lattice clocks accurate to 1 part in 10¹⁸.
Fermionic isotopes (Sr-87, Yb-171) offer nuclear spin qubits decoupled from the
electronic state — ideal for optical-clock quantum computing.
</div>
""", unsafe_allow_html=True)

    ae_detail = {
        "⁸⁸Sr / ⁸⁷Sr  — Strontium": {
            "why": "Sr has two laser-cooling stages: the broad 461 nm blue line (Doppler limit 770 μK) and the 689 nm red intercombination line (Γ/2π = 7.6 kHz, T_D = 0.18 μK). The 698 nm clock transition has a linewidth of ~1 mHz. Sr-87 (I=9/2) gives 10 nuclear spin states for SU(N) physics and quantum simulation.",
            "lines": "Broad: 461 nm (Γ/2π=32 MHz)  |  Narrow: 689 nm (Γ/2π=7.6 kHz)  |  Clock: 698 nm (~1 mHz)",
            "qubit": "Sr-87 nuclear spin qubit |mI=−9/2⟩↔|mI=−7/2⟩ via the 698 nm clock transition",
            "resources": [
                ("📄 Boyd PhD thesis (Ye Lab) — Sr spectroscopy", "https://jila.colorado.edu/sites/default/files/assets/files/thesis_dissertation/Boyd_Thesis.pdf"),
                ("📄 Stellmer review — Sr BEC and degenerate gases", "https://arxiv.org/abs/1307.0506"),
                ("🔗 NIST Atomic Spectra — Sr", "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Sr"),
                ("📦 ARC library (Python)", "https://arc-alkali-rydberg-calculator.readthedocs.io/"),
            ],
        },
        "¹⁷⁴Yb / ¹⁷¹Yb  — Ytterbium": {
            "why": "Yb combines broad (399 nm, Γ/2π=28 MHz) and narrow (556 nm, Γ/2π=182 kHz) cooling lines with a mHz-linewidth clock transition at 578 nm. Yb-171 (I=1/2) is effectively a perfect two-level nuclear-spin qubit. Magic wavelengths at 759 nm make optical lattice clocks insensitive to light shifts.",
            "lines": "Broad: 399 nm (Γ/2π=28 MHz)  |  Narrow: 556 nm (Γ/2π=182 kHz)  |  Clock: 578 nm (mHz)",
            "qubit": "Yb-171 |mI=+1/2⟩↔|mI=−1/2⟩ nuclear spin qubit (zero magnetic field insensitive)",
            "resources": [
                ("📄 Ludlow et al. — Optical atomic clocks (RMP 2015)", "https://arxiv.org/abs/1407.3493"),
                ("📄 Taichenachev et al. — Yb spectroscopy", "https://arxiv.org/abs/0611719"),
                ("🔗 NIST Atomic Spectra — Yb", "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Yb"),
                ("📦 ARC library (Python)", "https://arc-alkali-rydberg-calculator.readthedocs.io/"),
            ],
        },
    }

    for atom_name, info in ae_detail.items():
        with st.expander(f"**{atom_name}**"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Why it's used:** {info['why']}")
                st.markdown(f"**Key lines:** `{info['lines']}`")
                st.markdown(f"**Qubit transition:** {info['qubit']}")
            with c2:
                st.markdown("**Resources:**")
                for label, url in info["resources"]:
                    st.markdown(f"[{label}]({url})")


# ── MAGNETIC ─────────────────────────────────────────────────────
with tab_mag:
    st.markdown("""
<div class='concept-box'>
Highly magnetic atoms (Cr, Dy, Er) have large magnetic moments (6–10 μB), making
their inter-particle interactions strongly anisotropic and long-range.  This opens
the door to <b>dipolar quantum simulation</b> — phenomena impossible with contact-
interaction BECs.  Dy has the largest magnetic moment of any element (10 μB).
</div>
""", unsafe_allow_html=True)

    mag_detail = {
        "¹⁶⁴Dy  — Dysprosium": {
            "why": "Dy-164 has the largest magnetic moment of any element (10 μB), enabling strong dipolar interactions and anisotropic collisional physics. Used for dipolar BEC, quantum droplets, and supersolid phases. Cooled on a broad 421 nm line.",
            "lines": "Main: 421 nm (broad)  |  421 nm, 598 nm, 626 nm intercombination lines",
            "resources": [
                ("📄 Lu et al. (Lev) — Strongly dipolar Bose-Einstein condensate", "https://arxiv.org/abs/1101.4626"),
                ("📄 Chomaz et al. — Dipolar physics review (RMP 2023)", "https://arxiv.org/abs/2201.02672"),
            ],
        },
        "¹⁶⁸Er  — Erbium": {
            "why": "Er-168 has a magnetic moment of 7 μB and a rich level structure. Dipolar BEC demonstrated by Ferrier-Barbut/Pfau group (Stuttgart) and Grimm group (Innsbruck). Anisotropic scattering leads to distinctive many-body phases.",
            "lines": "Main: 401 nm  |  583 nm intercombination line",
            "resources": [
                ("📄 Aikawa et al. — Bose-Einstein condensation of Er", "https://arxiv.org/abs/1205.6813"),
                ("📄 Chomaz et al. — Dipolar physics review (RMP 2023)", "https://arxiv.org/abs/2201.02672"),
            ],
        },
    }

    for atom_name, info in mag_detail.items():
        with st.expander(f"**{atom_name}**"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Why it's used:** {info['why']}")
                st.markdown(f"**Key lines:** `{info['lines']}`")
            with c2:
                st.markdown("**Resources:**")
                for label, url in info["resources"]:
                    st.markdown(f"[{label}]({url})")


# ═══════════════════════════════════════════════════════════════
# US RESEARCH GROUPS
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🏛️ US Research Groups in Neutral-Atom Quantum Science")
st.markdown("""
<div class='concept-box'>
The following groups work on neutral-atom experiments — quantum computing, quantum
simulation, precision measurement, and ultracold chemistry — organised by their
primary atom(s).  Click a group name to visit their lab website.
</div>
""", unsafe_allow_html=True)

groups = {
    "⁸⁷Rb — Quantum Computing & Simulation": [
        ("Lukin Lab", "Harvard", "Rydberg tweezer quantum processor; many-body physics", "https://lukin.physics.harvard.edu"),
        ("Greiner Lab", "Harvard", "Quantum simulation, Hubbard model, quantum gas microscope", "https://greiner.physics.harvard.edu"),
        ("Saffman Lab", "Wisconsin", "Rydberg two-qubit gates, trapped-atom qubits", "https://saffman.physics.wisc.edu"),
        ("Kaufman Lab", "Colorado/JILA", "Tweezer arrays, quantum optics with atoms", "https://www.colorado.edu/lab/kaufman"),
        ("Bernien Lab", "UChicago", "Programmable quantum matter, Rydberg arrays", "https://bernien.uchicago.edu"),
        ("Weiss Lab", "Penn State", "Neutral atom qubits, quantum computing", "https://www.phys.psu.edu/people/daw11"),
    ],
    "¹³³Cs — Quantum Computing & Molecules": [
        ("Chin Lab", "UChicago", "Strongly correlated gases, Efimov physics, BEC", "https://ultracold.uchicago.edu"),
        ("Hood Lab", "Purdue", "Li-Cs molecule assembly, optical tweezers, single-atom control", "https://hoodlab.physics.purdue.edu"),
    ],
    "⁶Li / ⁷Li — Fermi Gases & Molecules": [
        ("Hulet Lab", "Rice", "Li-6 Fermi gases, BEC-BCS crossover, solitons", "https://hulet.rice.edu"),
        ("Zwierlein Lab", "MIT", "Degenerate Fermi gases, fermionic superfluidity", "https://www.mit.edu/~zwierlein"),
        ("DeMarco Lab", "UIUC", "Fermi-Hubbard model, disordered lattices", "https://deMarco.physics.illinois.edu"),
        ("Hood Lab", "Purdue", "Li-Cs molecules in tweezers", "https://hoodlab.physics.purdue.edu"),
    ],
    "²³Na & ¹⁹K — Molecules & BEC": [
        ("Ketterle Lab", "MIT", "First BEC (Nobel 2001); spinor BECs; NaLi molecules", "https://ketterle.mit.edu"),
        ("Park Lab", "MIT", "NaLi and NaK ultracold molecules", "https://www.rle.mit.edu/park-lab"),
        ("Ni Lab", "Harvard", "Ultracold polar molecules (NaK, KRb)", "https://ni.chem.harvard.edu"),
    ],
    "⁸⁸Sr / ⁸⁷Sr — Optical Clocks & Simulation": [
        ("Ye Lab", "JILA/Colorado", "World-leading optical lattice clock; Sr tweezer arrays; quantum simulation", "https://jila.colorado.edu/yelab"),
        ("Killian Lab", "Rice", "Sr BEC and tweezer arrays, Rydberg excitation", "https://ultracold.rice.edu"),
        ("Thompson Lab", "Princeton", "Sr cavity QED, quantum networking", "https://thompsonlab.physics.princeton.edu"),
        ("Covey Lab", "UIUC", "Tweezer arrays with alkaline-earth atoms", "https://covey.physics.illinois.edu"),
        ("Rey Lab (theory)", "JILA/Colorado", "AMO theory for Sr, Yb quantum simulation", "https://jila.colorado.edu/reygroup"),
    ],
    "¹⁷⁴Yb / ¹⁷¹Yb — Clocks & Quantum Computing": [
        ("Ye Lab", "JILA/Colorado", "Yb optical lattice clock; quantum simulation with Yb", "https://jila.colorado.edu/yelab"),
        ("Kaufman Lab", "Colorado", "Yb tweezer arrays; quantum optics", "https://www.colorado.edu/lab/kaufman"),
        ("Porto/Spielman Lab", "NIST/Maryland", "Optical lattices, synthetic gauge fields", "https://www.nist.gov/pml/quantum-measurement/laser-cooling-and-trapping-group"),
        ("Thompson Lab", "Princeton", "Yb cavity QED", "https://thompsonlab.physics.princeton.edu"),
    ],
    "¹⁶⁴Dy / ¹⁶⁸Er — Dipolar Physics": [
        ("Lev Lab", "Stanford", "Dy and Er dipolar gases, quantum magnetism", "https://levlab.stanford.edu"),
    ],
}

for section, group_list in groups.items():
    with st.expander(f"**{section}**  ({len(group_list)} groups)"):
        cols = st.columns(2)
        for i, (name, inst, focus, url) in enumerate(group_list):
            with cols[i % 2]:
                st.markdown(f"""
<div class='group-card'>
<b><a href="{url}" target="_blank">🔗 {name}</a></b> — <em>{inst}</em><br>
<span style='color:#aaa;font-size:0.88rem'>{focus}</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# RESOURCES & DATA TOOLS
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📚 Essential Resources & Tools")

col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    st.markdown("### Data Sheets")
    st.markdown("""
**Steck Alkali Data Sheets** *(the gold standard)*
- [Li-6](https://steck.us/alkalidata/lithium6numbers.pdf)
- [Li-7](https://steck.us/alkalidata/lithiumdata.pdf)
- [Na-23](https://steck.us/alkalidata/sodiumnumbers.pdf)
- [K-39/40/41](https://steck.us/alkalidata/potassiumnumbers.pdf)
- [Rb-85](https://steck.us/alkalidata/rubidium85numbers.pdf)
- [Rb-87](https://steck.us/alkalidata/rubidium87numbers.pdf)
- [Cs-133](https://steck.us/alkalidata/cesiumnumbers.pdf)
- [All Steck sheets →](https://steck.us/alkalidata/)
""")

with col_r2:
    st.markdown("### Databases & Tools")
    st.markdown("""
**Spectroscopy Databases**
- [NIST Atomic Spectra Database](https://physics.nist.gov/asd)
- [NIST Physical Reference Data](https://physics.nist.gov/PhysRefData/)

**Python Libraries**
- [ARC — Alkali Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/)
- [QuTiP — Open quantum systems](https://qutip.org)
- [pylcp — Laser cooling physics](https://python-laser-cooling-physics.readthedocs.io/)

**arXiv searches**
- [quant-ph neutral atom](https://arxiv.org/search/?query=neutral+atom+qubit&searchtype=all)
- [cond-mat.quant-gas](https://arxiv.org/list/cond-mat.quant-gas/recent)
""")

with col_r3:
    st.markdown("### Textbooks & Reviews")
    st.markdown("""
**Textbooks**
- Metcalf & van der Straten — *Laser Cooling and Trapping* (1999)
- Foot — *Atomic Physics* (2005)
- Pethick & Smith — *BEC in Dilute Gases* (2008)

**Review Articles**
- [Saffman — Quantum computing with Rydberg atoms (RMP 2010)](https://arxiv.org/abs/0909.4777)
- [Kaufman & Ni — Tweezer arrays review (Nat. Phys. 2021)](https://arxiv.org/abs/2009.07073)
- [Ludlow et al. — Optical atomic clocks (RMP 2015)](https://arxiv.org/abs/1407.3493)
- [Chomaz et al. — Dipolar physics (RMP 2023)](https://arxiv.org/abs/2201.02672)
- [Browaeys & Lahaye — Many-body Rydberg physics (Nat. Phys. 2020)](https://arxiv.org/abs/2002.07413)
""")

st.markdown("---")
st.caption("Built for curious96.com · Data compiled from Steck data sheets, NIST ASD, and primary literature · Links open in a new tab")
