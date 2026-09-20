"""
BurnRate AI — Tier 5 application layer.

Run from the project root:
    pip install streamlit plotly
    streamlit run app.py

Reads the artefacts the notebooks produce. Every panel degrades gracefully if a
file is missing, and says which notebook produces it.
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
try:
    import tier5
    HAVE_TIER5 = True
except Exception:
    HAVE_TIER5 = False

PROCESSED = "data/processed"

st.set_page_config(page_title="BurnRate AI", page_icon="◔", layout="wide")

INK, SURFACE, EDGE = "#141A2E", "#1E2740", "#2C3755"
TEXT, DIM = "#E7EAF4", "#8E99B8"
BRASS, CORAL = "#C9A227", "#E2674F"

# One style block, no blank lines inside it. A blank line closes the raw-HTML
# block in Streamlit's markdown parser, and every rule after it renders as
# visible text instead of applying.
CSS = (
    "<link href='https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700"
    "&family=Inter:wght@400;500;600&display=swap' rel='stylesheet'>"
    "<style>"
    f".stApp {{background:{INK};}}"
    f"html,body,[class*='css'] {{font-family:'Inter',system-ui,sans-serif;color:{TEXT};}}"
    "#MainMenu,footer,header {visibility:hidden;}"
    ".block-container {padding-top:2rem;padding-bottom:3rem;max-width:1380px;}"
    f"section[data-testid='stSidebar'] {{background:{SURFACE};border-right:1px solid {EDGE};}}"
    f"section[data-testid='stSidebar'] * {{color:{TEXT};}}"
    f"div[data-testid='stVerticalBlockBorderWrapper'] {{background:{SURFACE};"
    f"border:1px solid {EDGE} !important;border-radius:6px;padding:1.1rem 1.25rem;"
    "box-shadow:0 1px 0 rgba(255,255,255,.03) inset;}}"
    f".mast {{display:flex;align-items:baseline;gap:1rem;border-bottom:1px solid {EDGE};"
    "padding-bottom:.9rem;margin-bottom:1.5rem;}"
    f".mark {{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;"
    f"letter-spacing:-.02em;color:{TEXT};}}"
    f".tag {{color:{DIM};font-size:.9rem;margin-left:.95rem;}}"
    f".lbl {{font-size:.82rem;font-weight:500;color:{DIM};margin-bottom:.5rem;}}"
    ".fig {font-family:'Space Grotesk',sans-serif;font-size:2.45rem;font-weight:700;"
    "line-height:1.05;letter-spacing:-.025em;}"
    f".unit {{font-size:.98rem;font-weight:400;color:{DIM};padding-left:.4rem;}}"
    f".sub {{color:{DIM};font-size:.82rem;margin-top:.4rem;line-height:1.5;}}"
    f".rw {{display:flex;justify-content:space-between;align-items:center;padding:.6rem 0;"
    f"border-bottom:1px solid {EDGE};font-size:.92rem;}}"
    ".rw:last-child {border-bottom:none;}"
    f".gain {{font-family:'Space Grotesk',sans-serif;font-weight:700;color:{BRASS};}}"
    f".note {{border-left:2px solid {EDGE};padding:.5rem 0 .5rem .8rem;color:{DIM};"
    "font-size:.81rem;line-height:1.55;margin-top:1rem;}"
    f".note b {{color:{TEXT};font-weight:600;}}"
    f"div[data-baseweb='select'] > div {{background:{INK} !important;"
    f"border-color:{EDGE} !important;color:{TEXT} !important;}}"
    "</style>"
)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load(name):
    p = os.path.join(PROCESSED, name)
    return pd.read_csv(p) if os.path.exists(p) else None


survival = load("survival_scored.csv")
outcomes = load("outcomes.csv")          # carries name + sector for the picker
rhi = load("rhi_scores.csv")
ranks = load("intervention_rankings.csv")
traj = load("synthetic_trajectories.csv")
distress = load("distress_scores.csv")

if survival is None:
    st.error("No `data/processed/survival_scored.csv`. Run notebook 03, then return here.")
    st.stop()

st.markdown(
    "<div class='mast'><span class='mark'>BurnRate AI</span>"
    "<span class='tag'>Cash-runway estimation from sparse funding records</span></div>",
    unsafe_allow_html=True)

pool = survival.copy()
if ranks is not None:
    keep = pool[pool["id"].isin(ranks["object_id"])]
    if not keep.empty:
        pool = keep

# Crunchbase ids ("c:42384") are opaque, so label the picker with the real company
# name and sector and keep the id only as the lookup key.
if outcomes is not None and "name" in outcomes.columns:
    meta = outcomes[["id", "name", "category_code"]].drop_duplicates("id")
    pool = pool.merge(meta, on="id", how="left")
    pool["label"] = pool.apply(
        lambda r: f"{r['name']}"
                  + (f"  ·  {r['category_code']}" if pd.notna(r.get("category_code")) else "")
        if pd.notna(r.get("name")) else r["id"], axis=1)
else:
    pool["label"] = pool["id"]

# Most ventures in the population are genuinely low-risk, so an alphabetical list
# surfaces flat, uninformative curves almost every time. Order by 12-month risk and
# put the figure in the label, so the interesting cases are reachable.
RISK_COL = ("p_exhaust_12m_cond" if "p_exhaust_12m_cond" in pool.columns
            else "p_exhaust_12m" if "p_exhaust_12m" in pool.columns else None)

with st.sidebar:
    st.markdown("<div class='lbl'>Venture</div>", unsafe_allow_html=True)
    if RISK_COL:
        order = st.radio("order", ["Highest risk first", "A \u2192 Z"],
                         label_visibility="collapsed", horizontal=False)
        if order == "Highest risk first":
            pool = pool.sort_values(RISK_COL, ascending=False)
            pool["label"] = pool.apply(
                lambda r: f"{r['label']}  \u2014  {r[RISK_COL]:.0%}", axis=1)
        else:
            pool = pool.sort_values("label")
    pool = pool.head(600)
    label_to_id = dict(zip(pool["label"], pool["id"]))
    picked = st.selectbox("v", list(label_to_id), label_visibility="collapsed")
    choice = label_to_id[picked]
    st.markdown(
        f"<div class='sub'>Record {choice}"
        + (f" \u00b7 {pool.loc[pool['id'] == choice, RISK_COL].iloc[0]:.1%} chance of "
           "exhausting cash within 12 months" if RISK_COL else "")
        + "</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note'>Scores are peer-derived from funding history and sector. "
        "They express relative risk against comparable ventures, not a verdict about "
        "this one.</div>", unsafe_allow_html=True)

row = survival[survival["id"] == choice].iloc[0]

# Prefer CONDITIONAL exhaustion probabilities when src/conditional_survival.py has
# been run. The unconditional columns measure risk from founding, so for a venture
# already years old they sit near zero for everyone and the curve is flat and
# uninformative. The conditional form asks the question a founder actually has:
# given we have survived this long, what happens over the next 3/6/12 months?
COND = all(f"p_exhaust_{d}m_cond" in survival.columns for d in (3, 6, 12))
if COND:
    p3 = float(row["p_exhaust_3m_cond"])
    p6 = float(row["p_exhaust_6m_cond"])
    p12 = float(row["p_exhaust_12m_cond"])
else:
    p3 = np.nan
    p6 = float(row.get("p_exhaust_6m", np.nan))
    p12 = float(row.get("p_exhaust_12m", np.nan))

state = None
if HAVE_TIER5 and traj is not None:
    g = traj[traj["object_id"] == choice].sort_values("month_idx")
    if len(g) >= 3:
        state = tier5.state_from_trajectory(choice, g["synthesized_spend"].values,
                                            cash_multiple=8.0)

runway = tier5.runway_months(state) if state is not None else np.nan
rhi_val = np.nan
if rhi is not None and "id" in rhi.columns:
    hit = rhi[rhi["id"] == choice]
    if len(hit):
        rhi_val = float(hit["RHI"].iloc[0])

urgent = (not np.isnan(rhi_val) and rhi_val < 50) or (not np.isnan(p6) and p6 > 0.4)
accent = CORAL if urgent else BRASS
rgb = f"{int(accent[1:3], 16)},{int(accent[3:5], 16)},{int(accent[5:7], 16)}"

left, right = st.columns([2.05, 1], gap="medium")

with left:
    with st.container(border=True):
        heading = ("Probability of still operating, from today"
                   if COND else "Probability of still operating, from founding")
        st.markdown(f"<div class='lbl'>{heading}</div>", unsafe_allow_html=True)
        months = np.arange(0, 25)
        if COND and not np.isnan(p12):
            # interpolate the survival curve through the three measured points
            # rather than assuming a single exponential rate from p12 alone
            known_t = np.array([0.0, 3.0, 6.0, 12.0])
            known_s = np.array([1.0, 1 - p3, 1 - p6, 1 - p12])
            cum_h = -np.log(np.clip(known_s, 1e-6, 1.0))
            slope = (cum_h[-1] - cum_h[-2]) / (known_t[-1] - known_t[-2])
            h = np.interp(months, known_t, cum_h)
            tail = months > known_t[-1]
            h[tail] = cum_h[-1] + slope * (months[tail] - known_t[-1])
            surv = np.exp(-h)
        elif not np.isnan(p12):
            lam = -np.log(max(1 - p12, 1e-3)) / 12
            surv = np.exp(-lam * months)
        else:
            surv = np.exp(-0.05 * months)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=surv, mode="lines", line=dict(color=accent, width=3),
            fill="tozeroy", fillcolor=f"rgba({rgb},0.10)",
            hovertemplate="month %{x}<br>%{y:.0%} operating<extra></extra>"))
        for m in (3, 6, 12):
            fig.add_trace(go.Scatter(
                x=[m], y=[surv[m]], mode="markers+text", marker=dict(color=TEXT, size=7),
                text=[f"  {1 - surv[m]:.0%} by {m}mo"], textposition="middle right",
                textfont=dict(color=DIM, size=11, family="Space Grotesk"),
                hoverinfo="skip", showlegend=False))
        fig.update_layout(
            height=272, margin=dict(l=0, r=0, t=4, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="months from today" if COND else "months from founding",
                       gridcolor=EDGE, zeroline=False,
                       color=DIM, tickfont=dict(size=11)),
            yaxis=dict(tickformat=".0%", gridcolor=EDGE, zeroline=False, color=DIM,
                       tickfont=dict(size=11), range=[0, 1.04]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    rtxt = f"{runway:.1f}" if not np.isnan(runway) else "—"
    htxt = f"{rhi_val:.0f}" if not np.isnan(rhi_val) else "—"
    with st.container(border=True):
        st.markdown(
            "<div class='lbl'>Runway remaining</div>"
            f"<div class='fig' style='color:{accent}'>{rtxt}"
            "<span class='unit'>months</span></div>"
            "<div class='sub'>Projected from this venture's own burn trend to a zero "
            "cash balance. Cash on hand is a placeholder until founder intake exists.</div>",
            unsafe_allow_html=True)
    st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "<div class='lbl'>Runway Health Index</div>"
            f"<div class='fig' style='color:{accent}'>{htxt}<span class='unit'>of 100</span></div>"
            "<div class='sub'>Fused score. Lower means more urgent.</div>",
            unsafe_allow_html=True)

if not COND:
    st.markdown(
        "<div class='note'>This curve is measured <b>from founding</b>, so it sits near "
        "100% for every established venture and carries little information. Run "
        "<b>python src/conditional_survival.py</b> once to add conditional "
        "probabilities, and this panel will show risk over the next 12 months "
        "instead.</div>", unsafe_allow_html=True)

st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([1.15, 1], gap="medium")

with c1:
    with st.container(border=True):
        st.markdown("<div class='lbl'>What buys the most time</div>", unsafe_allow_html=True)
        if state is not None:
            rows = "".join(
                f"<div class='rw'><span>{r['intervention']}</span>"
                f"<span class='gain'>+{r['gain_weeks']:.1f} weeks</span></div>"
                for r in tier5.rank_interventions(state, pct=0.20))
            st.markdown(rows, unsafe_allow_html=True)
            st.markdown(
                "<div class='note'>Each figure is measured by re-running the projection "
                "with that change applied, not by a linear estimate. <b>Cost-line shares "
                "are an assumption</b>, not measured expenses.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sub'>No trajectory for this venture. Run notebook 06, "
                        "then 13.</div>", unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        st.markdown("<div class='lbl'>Test a change</div>", unsafe_allow_html=True)
        if state is not None:
            cut = st.slider("Reduce monthly burn by", 0, 50, 0, 5, format="%d%%")
            raise_m = st.slider("Raise capital", 0.0, 10.0, 0.0, 0.5, format="$%.1fM")
            out = tier5.simulate_scenario(state, burn_multiplier=1 - cut / 100,
                                          new_raise=raise_m * 1e6)
            delta = out["delta_months"]
            col = BRASS if delta > 0 else DIM
            st.markdown(
                "<div style='display:flex;align-items:baseline;gap:.85rem;margin-top:.4rem'>"
                f"<span class='fig' style='font-size:2rem;color:{col}'>"
                f"{out['scenario_months']:.1f}<span class='unit'>months</span></span>"
                f"<span style=\"color:{col};font-family:'Space Grotesk';font-weight:500\">"
                f"{delta:+.1f} vs today</span></div>"
                "<div class='sub'>Runs the identical projection path, so this is directly "
                "comparable with the reading above.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sub'>Needs a trajectory for this venture.</div>",
                        unsafe_allow_html=True)

st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)

d = "—"
if distress is not None:
    hit = distress[distress["object_id"] == choice]
    if len(hit):
        d = f"{float(hit['distress_score'].iloc[0]):.2f}"

with st.container(border=True):
    st.markdown(
        "<div class='lbl'>What this score is built from</div>"
        "<div class='rw'><span>Consumed by the hazard model</span>"
        f"<span style='color:{TEXT}'>funding history · rounds · sector</span></div>"
        "<div class='rw'><span>Consumed by the projector only</span>"
        f"<span style='color:{DIM}'>cash · burn · revenue · headcount</span></div>"
        "<div class='rw'><span>Distress-language score</span>"
        f"<span style='color:{TEXT}'>{d}</span></div>"
        "<div class='rw'><span>Fitted on</span>"
        f"<span style='color:{DIM}'>Crunchbase 2013 · 98,280 ventures · concordance 0.769"
        "</span></div>", unsafe_allow_html=True)

st.markdown(
    "<div class='note'>Founder-supplied figures reach the cash-flow projection, not the "
    "hazard estimate — the survival model is fitted on funding history and sector alone. "
    "Any venture can request removal of its record; the model is refitted exactly and the "
    "removal verified.</div>", unsafe_allow_html=True)