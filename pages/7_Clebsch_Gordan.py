"""
Clebsch–Gordan Coefficients
============================
Calculator and reference tables for ⟨j₁m₁; j₂m₂ | JM⟩.
Also covers the Wigner–Eckart theorem.
"""

import math
import streamlit as st
import pandas as pd
from fractions import Fraction

st.set_page_config(
    page_title="Clebsch–Gordan Coefficients",
    page_icon="⚛️",
    layout="wide",
)

# ─── CSS (matching dark theme) ────────────────────────────────────────────────
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
  .result-row {
    background:#052e16; border:1px solid #15803d;
    border-radius:8px; padding:.7rem 1.1rem; margin-top:.6rem;
    display:flex; flex-wrap:wrap; gap:.6rem 2rem; align-items:baseline;
  }
  .result-val { color:#4ade80; font-size:1.35rem; font-weight:700; font-family:monospace; }
  .result-lbl { color:#86efac; font-size:.85rem; }
  .result-zero { color:#94a3b8; font-size:1.25rem; font-weight:700; font-family:monospace; }

  .formula { background:#0a1f0a; border:1px solid #166534;
    border-radius:6px; padding:.55rem 1rem; margin:.6rem 0;
    font-family:'Courier New',monospace; color:#86efac; font-size:.83rem; line-height:1.8; }

  .theory-box { background:#0a0a1f; border:1px solid #1e3a5f;
    border-radius:8px; padding:1rem 1.4rem; margin:.8rem 0;
    color:#cbd5e1; font-size:.9rem; line-height:1.7; }

  .warn { background:#1c1007; border:1px solid #92400e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#fde68a; font-size:.85rem; }

  .info-box { background:#0a1020; border:1px solid #1e3a6e;
    border-radius:6px; padding:.6rem 1rem; margin:.5rem 0;
    color:#93c5fd; font-size:.85rem; }

  hr { border-color:#1e293b; margin:1.5rem 0; }

  .stTabs [data-baseweb="tab-list"]  { background:#0f172a; border-radius:8px; padding:4px; }
  .stTabs [data-baseweb="tab"]       { color:#94a3b8; font-weight:600; }
  .stTabs [aria-selected="true"]     { color:#7dd3fc !important; background:#0c1e3a !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── Pure-Python CG via Racah formula (no external deps) ─────────────────────

def _cg_float(j1, m1, j2, m2, J, M):
    """CG coefficient as float via Racah formula (Condon–Shortley convention)."""
    def F(x):
        n = int(round(x))
        return float(math.factorial(n)) if n >= 0 else None

    fs = [F(j1+j2-J), F(j1-j2+J), F(-j1+j2+J), F(j1+j2+J+1)]
    if None in fs or fs[3] == 0: return 0.0
    delta = fs[0] * fs[1] * fs[2] / fs[3]

    pf = [F(j1+m1), F(j1-m1), F(j2+m2), F(j2-m2), F(J+M), F(J-M)]
    if None in pf: return 0.0
    pref_sq = (2*J + 1) * delta
    for f in pf: pref_sq *= f

    nu_min = max(0, int(round(j2 - J - m1)), int(round(j1 - J + m2)))
    nu_max = int(round(min(j1 + j2 - J, j1 - m1, j2 + m2)))

    total = 0.0
    for nu in range(nu_min, nu_max + 1):
        terms = [F(nu), F(j1+j2-J-nu), F(j1-m1-nu),
                 F(j2+m2-nu), F(J-j2+m1+nu), F(J-j1-m2+nu)]
        if None in terms: continue
        denom = 1.0
        for t in terms: denom *= t
        total += (-1)**nu / denom

    return 0.0 if abs(total) < 1e-15 else math.sqrt(abs(pref_sq)) * total

def _exact_str(value):
    """Express float CG value as ±√(p/q) or integer string."""
    if abs(value) < 1e-10: return "0"
    sign = "−" if value < 0 else ""
    v_sq = value**2
    frac = Fraction(v_sq).limit_denominator(10000)
    p, q = frac.numerator, frac.denominator
    if abs(math.sqrt(p / q) - abs(value)) > 1e-7:
        return f"{sign}{abs(value):.6f}"
    if p == q: return f"{sign}1"
    g = math.gcd(p, q); p //= g; q //= g
    sqp = int(round(math.sqrt(p))); sqq = int(round(math.sqrt(q)))
    if sqp*sqp == p and sqq*sqq == q:   # both perfect squares → integer ratio
        g2 = math.gcd(sqp, sqq)
        return f"{sign}{sqp//g2}/{sqq//g2}"
    if sqp*sqp == p:                     # numerator is perfect square
        return (f"{sign}1/√{q}" if sqp == 1 else f"{sign}{sqp}/√{q}")
    if sqq*sqq == q:                     # denominator is perfect square
        return (f"{sign}√{p}" if sqq == 1 else f"{sign}√{p}/{sqq}")
    return f"{sign}√({p}/{q})"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fmt_j(x):
    """Format half-integer: 0.5→'1/2', 1.0→'1', -0.5→'-1/2'."""
    n = int(round(2 * x))
    sign = "-" if n < 0 else ""
    n = abs(n)
    return sign + (str(n // 2) if n % 2 == 0 else f"{n}/2")

def half_range(j):
    """List of m-values from -j to +j (step 1)."""
    m, out = -j, []
    while m <= j + 1e-9:
        out.append(round(m * 2) / 2)
        m += 1.0
    return out

def allowed_J(j1, j2):
    """List of allowed total J values for j1⊗j2."""
    j = abs(j1 - j2)
    out = []
    while j <= j1 + j2 + 1e-9:
        out.append(round(j * 2) / 2)
        j += 1.0
    return out

def cg(j1, m1, j2, m2, J, M):
    """Return (exact_string, float_value) for ⟨j₁m₁;j₂m₂|JM⟩."""
    if abs(round(2 * (m1 + m2)) - round(2 * M)) != 0:
        return "0", 0.0
    if J < abs(j1 - j2) - 1e-9 or J > j1 + j2 + 1e-9:
        return "0", 0.0
    if abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(M) > J + 1e-9:
        return "0", 0.0
    fval = _cg_float(j1, m1, j2, m2, J, M)
    return _exact_str(fval), fval

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

def warn(s):
    st.markdown(f'<div class="warn">⚠️ {s}</div>', unsafe_allow_html=True)

# ─── Page header ──────────────────────────────────────────────────────────────

st.title("⚛️ Clebsch–Gordan Coefficients")
st.markdown(
    "Coupling coefficients **⟨j₁m₁; j₂m₂ | JM⟩** for addition of angular momenta. "
    "Exact results via the Racah formula — no external dependencies required."
)

# ─── Background expander ──────────────────────────────────────────────────────

with st.expander("📖 Background & definitions", expanded=False):
    st.markdown("### What are Clebsch–Gordan coefficients?")
    theory_box(
        "When two angular momenta <b>j₁</b> and <b>j₂</b> are combined, the coupled eigenstates "
        "|J,M⟩ of the total angular momentum <b>J = j₁ + j₂</b> are linear combinations of "
        "the uncoupled product basis |j₁,m₁⟩⊗|j₂,m₂⟩. The expansion coefficients are the "
        "Clebsch–Gordan (CG) coefficients:"
    )
    formula("|J, M⟩ = Σ_{m₁,m₂}  ⟨j₁m₁; j₂m₂ | J M⟩  |j₁,m₁⟩|j₂,m₂⟩")

    st.markdown("### Selection rules — coefficient is zero unless:")
    st.markdown("""
- **M = m₁ + m₂** (z-component conservation)
- **|j₁ − j₂| ≤ J ≤ j₁ + j₂** (triangle rule)
- **|mᵢ| ≤ jᵢ** for i = 1, 2
""")

    st.markdown("### Allowed J values and dimension check")
    formula(
        "J  ∈  { |j₁−j₂|,  |j₁−j₂|+1,  …,  j₁+j₂ }\n"
        "Σ_J (2J+1)  =  (2j₁+1)(2j₂+1)     [dimension check]\n"
        "Example:  ½ ⊗ ½  →  J=0 (dim 1) ⊕ J=1 (dim 3)  →  1+3 = 4 = 2·2  ✓"
    )

    st.markdown("### Orthogonality and completeness")
    formula(
        "Σ_{m₁m₂}  ⟨j₁m₁;j₂m₂|JM⟩ ⟨j₁m₁;j₂m₂|J'M'⟩  =  δ_{JJ'} δ_{MM'}\n"
        "Σ_{JM}    ⟨j₁m₁;j₂m₂|JM⟩ ⟨j₁m₁';j₂m₂'|JM⟩  =  δ_{m₁m₁'} δ_{m₂m₂'}"
    )

    st.markdown("### Symmetry relations")
    formula(
        "⟨j₁m₁;j₂m₂|JM⟩ = (−1)^{j₁+j₂−J} ⟨j₂m₂;j₁m₁|JM⟩          [exchange]\n"
        "⟨j₁m₁;j₂m₂|JM⟩ = (−1)^{j₁+j₂−J} ⟨j₁−m₁;j₂−m₂|J−M⟩       [time reversal]"
    )

# ─── Tabs ──────���──────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Coefficient",
    "📊 CG Table",
    "🔮 Wigner–Eckart Theorem",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Single coefficient calculator
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Compute ⟨j₁m₁; j₂m₂ | JM⟩")
    info("M is fixed automatically as m₁ + m₂ (the only non-zero case).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**First angular momentum**")
        j1 = st.number_input("j₁", min_value=0.0, max_value=10.0,
                             value=1.0, step=0.5, key="s_j1")
        m1_opts = [fmt_j(m) for m in half_range(j1)]
        m1_sel  = st.selectbox("m₁", m1_opts,
                               index=len(m1_opts) // 2, key="s_m1")
        m1 = half_range(j1)[m1_opts.index(m1_sel)]

    with col2:
        st.markdown("**Second angular momentum**")
        j2 = st.number_input("j₂", min_value=0.0, max_value=10.0,
                             value=0.5, step=0.5, key="s_j2")
        m2_opts = [fmt_j(m) for m in half_range(j2)]
        m2_sel  = st.selectbox("m₂", m2_opts, index=0, key="s_m2")
        m2 = half_range(j2)[m2_opts.index(m2_sel)]

    with col3:
        st.markdown("**Total angular momentum J**")
        J_opts = [fmt_j(Jv) for Jv in allowed_J(j1, j2)]
        J_sel  = st.selectbox("J", J_opts, index=0, key="s_J")
        J  = allowed_J(j1, j2)[J_opts.index(J_sel)]
        M  = round((m1 + m2) * 2) / 2
        st.markdown(f"**M = m₁ + m₂ = {fmt_j(M)}**")

    exact, fval = cg(j1, m1, j2, m2, J, M)
    bra = f"⟨{fmt_j(j1)},{fmt_j(m1)}; {fmt_j(j2)},{fmt_j(m2)} | {fmt_j(J)},{fmt_j(M)}⟩"

    st.markdown("---")
    if exact == "0":
        st.markdown(
            f'<div class="result-row">{bra} &nbsp;= '
            f'<span class="result-zero">0</span> &nbsp; (selection rule)</div>',
            unsafe_allow_html=True,
        )
    else:
        result_box(
            (bra + " =", ""),
            ("Exact:", exact),
            ("≈", f"{fval:.8f}"),
        )

    # Symmetry companions
    if exact != "0":
        with st.expander("Symmetry-related coefficients"):
            # Exchange symmetry
            ex_exact, ex_fval = cg(j2, m2, j1, m1, J, M)
            phase_ex = int(round((j1 + j2 - J) % 2))
            sign_ex  = "+" if phase_ex == 0 else "−"
            formula(
                f"Exchange:      ⟨{fmt_j(j2)},{fmt_j(m2)}; {fmt_j(j1)},{fmt_j(m1)} | {fmt_j(J)},{fmt_j(M)}⟩"
                f"  =  (−1)^{{{fmt_j(j1+j2-J)}}} × {exact}  =  {ex_exact}"
            )
            # Time-reversal
            tr_exact, tr_fval = cg(j1, -m1, j2, -m2, J, -M)
            formula(
                f"Time-reversal: ⟨{fmt_j(j1)},{fmt_j(-m1)}; {fmt_j(j2)},{fmt_j(-m2)} | {fmt_j(J)},{fmt_j(-M)}⟩"
                f"  =  (−1)^{{{fmt_j(j1+j2-J)}}} × {exact}  =  {tr_exact}"
            )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Full CG table
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### CG table for fixed j₁, j₂, J")
    st.markdown(
        "Rows = m₁, columns = m₂. Cell value: **⟨j₁m₁; j₂m₂ | J, m₁+m₂⟩**."
    )
    info("Keep j ≤ 4 for fast computation. Zeros appear when |m₁+m₂| > J.")

    col1, col2, col3 = st.columns(3)
    with col1:
        t_j1 = st.number_input("j₁", min_value=0.0, max_value=6.0,
                               value=1.0, step=0.5, key="t_j1")
    with col2:
        t_j2 = st.number_input("j₂", min_value=0.0, max_value=6.0,
                               value=0.5, step=0.5, key="t_j2")
    with col3:
        t_J_opts = [fmt_j(Jv) for Jv in allowed_J(t_j1, t_j2)]
        t_J_sel  = st.selectbox("J", t_J_opts, index=0, key="t_J")
        t_J = allowed_J(t_j1, t_j2)[t_J_opts.index(t_J_sel)]

    m1_vals = half_range(t_j1)
    m2_vals = half_range(t_j2)

    # Build DataFrame of exact strings
    rows = {}
    for m1v in m1_vals:
        row = {}
        for m2v in m2_vals:
            Mv = round((m1v + m2v) * 2) / 2
            exact_v, _ = cg(t_j1, m1v, t_j2, m2v, t_J, Mv)
            row[fmt_j(m2v)] = exact_v
        rows[fmt_j(m1v)] = row

    df = pd.DataFrame(rows).T
    df.index.name = "m₁ \\ m₂"

    # Style: colour positive / negative / zero differently
    def colour_cg(val):
        if val == "0":
            return "color: #334155"
        try:
            f = float(val.replace("sqrt", "1").replace("(", "").replace(")", ""))
        except Exception:
            f = 1.0
        # Re-evaluate via sympy sign
        if val.startswith("-"):
            return "color: #f87171; font-weight:600"
        return "color: #4ade80; font-weight:600"

    styled = df.style.applymap(colour_cg)
    st.dataframe(styled, use_container_width=True)

    # Coupling decomposition
    st.markdown("---")
    st.markdown("### Angular momentum decomposition")
    J_list = allowed_J(t_j1, t_j2)
    total_dim = int((2 * t_j1 + 1) * (2 * t_j2 + 1))
    decomp = " ⊕ ".join([f"J={fmt_j(Jv)}" for Jv in J_list])
    dim_check = " + ".join([str(int(2 * Jv + 1)) for Jv in J_list])
    formula(
        f"j₁={fmt_j(t_j1)}  ⊗  j₂={fmt_j(t_j2)}   →   {decomp}\n"
        f"Dimension:  {total_dim}  =  {dim_check}  ✓"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Wigner–Eckart Theorem
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Wigner–Eckart Theorem")

    theory_box(
        "The Wigner–Eckart theorem factorises any matrix element of a "
        "<b>rank-k spherical tensor operator</b> T<sup>(k)</sup><sub>q</sub> "
        "into a <em>geometric</em> part (a CG coefficient, carrying all "
        "m-dependence) and a <em>reduced matrix element</em> "
        "⟨α′j′ ‖ T<sup>(k)</sup> ‖ αj⟩ that is independent of m, m′, q:"
    )
    formula(
        "⟨α′, j′, m′ | T(k)_q | α, j, m⟩\n"
        "    =  ⟨j, m; k, q | j′, m′⟩  ×  ⟨α′j′ ‖ T(k) ‖ αj⟩ / √(2j′+1)\n\n"
        "Here α, α′ label all other (non-angular) quantum numbers.\n"
        "The reduced matrix element encodes the physics; the CG coefficient is purely geometric."
    )

    st.markdown("### Why it matters")
    theory_box(
        "Once the reduced matrix element ⟨α′j′ ‖ T<sup>(k)</sup> ‖ αj⟩ is known "
        "(from one measurement or calculation), <b>all</b> (2j+1)(2j′+1) "
        "individual matrix elements ���j′m′|T<sup>(k)</sup><sub>q</sub>|jm⟩ are "
        "determined. This is enormously powerful for selection rules, "
        "hyperfine matrix elements, and optical transition strengths."
    )

    st.markdown("### Selection rules (from the CG coefficient factor)")
    st.markdown("""
- **m′ = m + q** (z-projection conservation)
- **|j − k| ≤ j′ ≤ j + k** (triangle rule — now involving the operator rank k)
- Matrix element is zero unless the above are satisfied, regardless of the physics
""")

    st.markdown("---")
    st.markdown("### Common cases in AMO physics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Electric dipole (E1), k = 1**")
        formula(
            "Δj = 0, ±1    (no 0→0)\n"
            "Δm = 0  (π polarisation)\n"
            "Δm = ±1 (σ± polarisation)"
        )
        st.markdown("**Magnetic dipole (M1), k = 1**")
        formula(
            "Same Δj, Δm rules as E1\n"
            "But Δl = 0  (no parity change)"
        )

    with col2:
        st.markdown("**Electric quadrupole (E2), k = 2**")
        formula(
            "Δj = 0, ±1, ±2  (no 0→0, ½→½)\n"
            "Δm = 0, ±1, ±2"
        )
        st.markdown("**Hyperfine coupling**")
        theory_box(
            "All ⟨F,m_F | T | F′,m_F′⟩ matrix elements reduce to a single "
            "reduced matrix element ⟨F ‖ T ‖ F′⟩, making hyperfine "
            "calculations tractable."
        )

    st.markdown("---")
    st.markdown("### Relation to Wigner 3j symbols")
    formula(
        "⟨j₁m₁; j₂m₂ | J M⟩  =  (−1)^{j₁−j₂+M} √(2J+1)  ×  ⎛ j₁   j₂   J  ⎞\n"
        "                                                          ⎝ m₁   m₂  −M  ⎠\n\n"
        "3j symbols are more symmetric: invariant under even column permutations,\n"
        "acquire phase (−1)^{j₁+j₂+J} under odd permutations or flipping all m-signs."
    )

    theory_box(
        "Wigner 3j symbols are often preferred in theoretical work because their "
        "symmetries make angular momentum algebra cleaner. They are available in "
        "sympy as <code>sympy.physics.wigner.wigner_3j(j1, j2, j3, m1, m2, m3)</code>."
    )
