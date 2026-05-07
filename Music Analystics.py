"""
🎵 Music Streaming Analytics Dashboard
DataDNA Challenge — 2021-2024
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🎵 Streaming Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
    .metric-card {
        background: linear-gradient(135deg, #1c2333, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 6px 0;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #e6edf3; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #e6edf3;
        border-left: 3px solid #58a6ff;
        padding-left: 12px; margin: 24px 0 16px;
    }
    .insight-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #3fb950;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 14px;
    }
    .warning-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #d29922;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 14px;
    }
    .danger-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #f85149;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 14px;
    }
    h1 { color: #e6edf3 !important; }
    h2, h3 { color: #c9d1d9 !important; }
    .stSelectbox label, .stRadio label { color: #8b949e !important; }
    div[data-testid="metric-container"] { background: #161b22; border-radius: 10px; padding: 14px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
import os, pathlib

def _find_data_dir():
    """
    Looks for the CSV data folder in several common locations:
      1. A 'data/' subfolder next to this script
      2. The long DataDNA folder name (unzipped zip) next to this script
      3. The current working directory variants
    Put your CSVs in any of these and it will be found automatically.
    """
    script_dir = pathlib.Path(__file__).parent.resolve()
    candidates = [
        # short 'data/' folder right next to the script  ← easiest option
        script_dir / "data",
        # unzipped folder from the challenge zip
        script_dir / "DataDNA Dataset Challenge - Streaming Analytics User Engagement Subscription Churn Content Performance" / "data",
        # same but from cwd
        pathlib.Path.cwd() / "data",
        pathlib.Path.cwd() / "DataDNA Dataset Challenge - Streaming Analytics User Engagement Subscription Churn Content Performance" / "data",
    ]
    for p in candidates:
        if (p / "fact_listening_session.csv").exists():
            return str(p) + "/"
    # nothing found — show a clear error in the app
    return None

DATA = _find_data_dir()

@st.cache_data
def load_data():
    if DATA is None:
        st.error(
            "❌ **Data folder not found.**\n\n"
            "Place the CSV files in a folder called **`data/`** next to this script, e.g.:\n\n"
            "```\n"
            "Visualization project/\n"
            "├── Music Analystics.py\n"
            "└── data/\n"
            "    ├── fact_listening_session.csv\n"
            "    ├── fact_subscription_event.csv\n"
            "    ├── dim_user.csv\n"
            "    └── ... (all other CSVs)\n"
            "```\n\n"
            "Extract the zip file and copy the `data/` folder here."
        )
        st.stop()
    sessions = pd.read_csv(DATA + "fact_listening_session.csv")
    subs     = pd.read_csv(DATA + "fact_subscription_event.csv")
    users    = pd.read_csv(DATA + "dim_user.csv")
    tracks   = pd.read_csv(DATA + "dim_track.csv")
    artists  = pd.read_csv(DATA + "dim_artist.csv")
    genres   = pd.read_csv(DATA + "dim_genre.csv")
    countries= pd.read_csv(DATA + "dim_country.csv")
    devices  = pd.read_csv(DATA + "dim_device.csv")
    plans    = pd.read_csv(DATA + "dim_subscription_plan.csv")

    sessions["listen_start_ts"] = pd.to_datetime(sessions["listen_start_ts"])
    sessions["year_month"]  = sessions["listen_start_ts"].dt.to_period("M").astype(str)
    sessions["year"]        = sessions["listen_start_ts"].dt.year
    sessions["month"]       = sessions["listen_start_ts"].dt.month
    sessions["month_name"]  = sessions["listen_start_ts"].dt.strftime("%b")

    subs["event_ts"]   = pd.to_datetime(subs["event_ts"])
    subs["year_month"] = subs["event_ts"].dt.to_period("M").astype(str)
    subs["year"]       = subs["event_ts"].dt.year

    users["age_group"] = pd.cut(
        users["age"],
        bins=[0,25,35,45,55,100],
        labels=["<25","25-34","35-44","45-54","55+"]
    )

    # Enriched sessions
    sess_e = sessions.merge(genres[["genre_id","genre_name"]], on="genre_id", how="left")
    sess_e = sess_e.merge(countries[["country_code","country_name"]], on="country_code", how="left")
    sess_e = sess_e.merge(devices[["device_id","device_type"]], on="device_id", how="left")
    sess_e = sess_e.merge(users[["user_id","age_group","is_fraud_cluster"]], on="user_id", how="left")

    return sessions, subs, users, tracks, artists, genres, countries, devices, plans, sess_e

sessions, subs, users, tracks, artists, genres, countries, devices, plans, sess_e = load_data()

fraud_ids = set(users[users["is_fraud_cluster"]==True]["user_id"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Streaming Analytics")
    st.markdown("**DataDNA Challenge · 2021–2024**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Executive Summary",
         "📈 Revenue & MRR",
         "🎧 Listening Behaviour",
         "👥 User Segments",
         "🌍 Geographic Analysis",
         "🎵 Content & Catalogue",
         "⚠️ Fraud & Risk",
         "🔮 Churn Prediction"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"📁 {len(sessions):,} sessions · {len(users):,} users · {len(subs):,} events")
    st.caption("2021-01-01 → 2024-12-31")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("# 📊 Executive Summary")
    st.markdown("*Music Streaming Platform · 4-Year Operational Analysis · 10 Markets*")
    st.divider()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev      = sessions["estimated_revenue_usd"].sum()
    net_mrr        = subs["mrr_change_usd"].sum()
    churn_loss     = subs[subs["event_type"]=="churn"]["mrr_change_usd"].sum()
    upgrade_gain   = subs[subs["event_type"]=="upgrade"]["mrr_change_usd"].sum()
    skip_rate      = sessions["skipped"].mean()
    churn_users    = subs[subs["event_type"]=="churn"]["user_id"].nunique()
    churn_rate     = churn_users / len(users)
    arpu_premium   = sessions[sessions["subscription_tier"]=="Premium"]["estimated_revenue_usd"].sum() / users[users["subscription_tier"]=="Premium"].shape[0]

    with col1:
        st.metric("Total Session Revenue", f"${total_rev:,.0f}", "4-year cumulative")
    with col2:
        st.metric("Net MRR Growth", f"${net_mrr:,.0f}", f"↑ ${upgrade_gain:,.0f} upgrades")
    with col3:
        st.metric("Churned MRR Lost", f"${abs(churn_loss):,.0f}", f"{churn_rate:.0%} users churned", delta_color="inverse")
    with col4:
        st.metric("Overall Skip Rate", f"{skip_rate:.1%}", "Free: 20% | Paid: ~10%")
    with col5:
        st.metric("Premium ARPU", f"${arpu_premium:.2f}", "vs Free $0.83")

    st.divider()

    # Key Findings
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown('<div class="section-header">🔑 Key Findings</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>🚀 Platform growing 4× YoY in sessions</b><br>
        From 6,168 sessions in 2021 to 118,764 in 2024 (+1,824% total). December 2024 alone
        generated 14,795 sessions — the single highest month on record, with a strong seasonal
        pattern showing summer (Jun–Aug) and December as peak periods.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <b>⚠️ 47% of users have churned at least once</b><br>
        453 of 961 users experienced at least one churn event. Free-tier users account for
        66% of churn events (421 of 635). Pre-churn sessions drop 14% in the final 30 days,
        and skip rates rise from 14.3% to 16.6% — a detectable leading indicator.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="danger-box">
        <b>🚨 Fraud cluster represents 70% of sessions but only 50% of users</b><br>
        482 flagged users generate 157,208 sessions (326/user vs 140/user for clean accounts).
        Average listen time is 46% shorter (73s vs 137s), and they account for 67.7% of
        session revenue — suggesting stream-farming behavior inflating royalty calculations.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>🎵 Pop and Rock drive 46% of all revenue</b><br>
        Pop ($209) and Rock ($183) are the top revenue generators. Country-genre over-indexing
        reveals strong geographic affinities: Reggae in South Africa (2.3×), Hip-Hop in
        the US (2.0×), and Latin in Brazil (2.0×) — key signals for geo-targeted content investment.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>💡 869 Free users converted to paid — but conversion is under-leveraged</b><br>
        Heavy free users (top 25% by volume) show 117% higher upgrade rates than average.
        France and South Africa have the highest Free-tier session share (33–34%), representing
        the richest conversion opportunity by market.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">🎯 Top 3 Recommendations</div>', unsafe_allow_html=True)

        st.markdown("**1. Deploy churn early-warning system**")
        st.info("Target users whose skip rate rises >5pp week-on-week OR whose weekly sessions drop >30%. Trigger personalised retention offers 3–4 weeks before predicted churn. Expected impact: recover ~$608/quarter in MRR.")

        st.markdown("**2. Geo-targeted conversion campaigns for France & South Africa**")
        st.info("FR and ZA have the highest free-tier share (34–33%) and both over-index in high-engagement genres (Jazz/FR, Reggae/ZA). A 10% conversion lift in these markets alone would add ~85 paid subscribers.")

        st.markdown("**3. Investigate and remediate fraud cluster**")
        st.warning("482 flagged users represent 70% of sessions with anomalously short listening durations (73s avg). Implement minimum-listen-time royalty gates (≥30s already in model) and review cluster characteristics to prevent revenue manipulation worth ~$576 in misattributed session revenue.")

        st.divider()
        # YoY summary chart
        yearly = sessions.groupby("year").agg(
            sessions=("session_id","count"),
            revenue=("estimated_revenue_usd","sum")
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yearly["year"], y=yearly["sessions"], name="Sessions",
                             marker_color="#58a6ff", opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["revenue"], name="Revenue $",
                                 mode="lines+markers", line=dict(color="#3fb950", width=2),
                                 marker=dict(size=8)), secondary_y=True)
        fig.update_layout(
            title="Year-over-Year Growth", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REVENUE & MRR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Revenue & MRR":
    st.markdown("# 📈 Revenue & MRR Analysis")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    upgrade_mrr = subs[subs["event_type"]=="upgrade"]["mrr_change_usd"].sum()
    churn_mrr   = subs[subs["event_type"]=="churn"]["mrr_change_usd"].sum()
    down_mrr    = subs[subs["event_type"]=="downgrade"]["mrr_change_usd"].sum()
    net_mrr     = subs["mrr_change_usd"].sum()

    with col1: st.metric("Net MRR Expansion", f"${net_mrr:,.0f}", "cumulative 4yr")
    with col2: st.metric("Upgrade Revenue", f"+${upgrade_mrr:,.0f}", "from 1,059 upgrades")
    with col3: st.metric("Churn Revenue Lost", f"${abs(churn_mrr):,.0f}", "635 churn events", delta_color="inverse")
    with col4: st.metric("Downgrade Revenue Lost", f"${abs(down_mrr):,.0f}", "336 downgrades", delta_color="inverse")

    st.divider()

    # Monthly MRR waterfall
    st.markdown('<div class="section-header">Monthly MRR by Event Type</div>', unsafe_allow_html=True)

    mrr_monthly = subs.groupby(["year_month","event_type"])["mrr_change_usd"].sum().unstack(fill_value=0).reset_index()
    mrr_monthly = mrr_monthly.sort_values("year_month").tail(24)

    event_cols  = [c for c in ["upgrade","retention","signup","downgrade","churn"] if c in mrr_monthly.columns]
    colors_map  = {"upgrade":"#3fb950","retention":"#58a6ff","signup":"#a371f7",
                   "downgrade":"#d29922","churn":"#f85149"}

    fig = go.Figure()
    for ec in event_cols:
        if ec in mrr_monthly.columns:
            fig.add_trace(go.Bar(
                x=mrr_monthly["year_month"], y=mrr_monthly[ec],
                name=ec.capitalize(), marker_color=colors_map.get(ec,"#8b949e")
            ))
    fig.update_layout(
        barmode="relative", height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickangle=-45, gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="MRR Change ($)"),
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Revenue by Tier</div>', unsafe_allow_html=True)
        tier_rev = sessions.groupby("subscription_tier")["estimated_revenue_usd"].sum().reset_index()
        tier_users = users.groupby("subscription_tier").size().reset_index(name="users")
        tier_merged = tier_rev.merge(tier_users, on="subscription_tier")
        tier_merged["arpu"] = (tier_merged["estimated_revenue_usd"] / tier_merged["users"]).round(2)
        tier_merged.columns = ["Tier","Session Revenue","Users","ARPU"]

        fig2 = px.pie(tier_merged, values="Session Revenue", names="Tier",
                      color_discrete_sequence=["#58a6ff","#3fb950","#a371f7"],
                      hole=0.55)
        fig2.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(tier_merged.set_index("Tier").style.format({"Session Revenue":"${:.2f}","ARPU":"${:.2f}"}),
                     use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Revenue by Country</div>', unsafe_allow_html=True)
        crev = sessions.merge(countries[["country_code","country_name"]], on="country_code")\
                       .groupby("country_name")["estimated_revenue_usd"].sum()\
                       .sort_values(ascending=True).reset_index()
        fig3 = px.bar(crev, x="estimated_revenue_usd", y="country_name", orientation="h",
                      color="estimated_revenue_usd", color_continuous_scale="Blues",
                      labels={"estimated_revenue_usd":"Revenue ($)","country_name":""})
        fig3.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <b>France leads revenue</b> ($117) despite having the highest Free-tier share (33%).
        Converting even 15% of FR free users to Premium would add ~$17 in ARPU.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LISTENING BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎧 Listening Behaviour":
    st.markdown("# 🎧 Listening Behaviour Analysis")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Sessions", f"{len(sessions):,}")
    with col2: st.metric("Avg Listen Duration", f"{sessions['listen_seconds'].mean():.0f}s")
    with col3: st.metric("Skip Rate", f"{sessions['skipped'].mean():.1%}")
    with col4: st.metric("New Artist Discovery", f"{sessions['new_artist_discovered'].mean():.1%}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Monthly Session Volume (2021–2024)</div>', unsafe_allow_html=True)
        monthly = sessions.groupby("year_month").agg(
            sessions=("session_id","count"),
            revenue=("estimated_revenue_usd","sum")
        ).reset_index().sort_values("year_month")

        fig = px.area(monthly, x="year_month", y="sessions",
                      color_discrete_sequence=["#58a6ff"], labels={"year_month":"Month","sessions":"Sessions"})
        fig.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), xaxis=dict(tickangle=-45, gridcolor="#21262d", showticklabels=False),
            yaxis=dict(gridcolor="#21262d"), margin=dict(l=0,r=0,t=10,b=0)
        )
        fig.update_traces(fill="tozeroy", line_color="#58a6ff", fillcolor="rgba(88,166,255,0.15)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Seasonality — Avg Sessions by Month</div>', unsafe_allow_html=True)
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        seasonal = sessions.copy()
        seasonal["month_abbr"] = pd.Categorical(
            sessions["listen_start_ts"].dt.strftime("%b"), categories=month_order, ordered=True
        )
        seasonal_agg = seasonal.groupby("month_abbr", observed=True).size().reset_index(name="sessions")
        seasonal_agg["pct_vs_avg"] = (seasonal_agg["sessions"] / seasonal_agg["sessions"].mean() - 1) * 100

        colors_bar = ["#3fb950" if v >= 0 else "#f85149" for v in seasonal_agg["pct_vs_avg"]]
        fig2 = go.Figure(go.Bar(
            x=seasonal_agg["month_abbr"], y=seasonal_agg["pct_vs_avg"],
            marker_color=colors_bar,
            text=seasonal_agg["pct_vs_avg"].apply(lambda x: f"{x:+.0f}%"),
            textposition="outside"
        ))
        fig2.add_hline(y=0, line_color="#8b949e", line_dash="dash")
        fig2.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), yaxis=dict(gridcolor="#21262d", title="% vs Average"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=10,b=0), showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Skip Rate & Listen Duration by Tier</div>', unsafe_allow_html=True)
        tier_engage = sessions.groupby("subscription_tier").agg(
            skip_rate=("skipped","mean"),
            avg_listen=("listen_seconds","mean"),
            sessions=("session_id","count")
        ).reset_index()

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(
            x=tier_engage["subscription_tier"], y=tier_engage["skip_rate"]*100,
            name="Skip Rate %", marker_color=["#f85149","#3fb950","#58a6ff"]
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=tier_engage["subscription_tier"], y=tier_engage["avg_listen"],
            name="Avg Listen (s)", mode="lines+markers",
            line=dict(color="#d29922", width=2), marker=dict(size=10)
        ), secondary_y=True)
        fig3.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d", title="Skip Rate %"),
            yaxis2=dict(title="Avg Listen (s)"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""<div class="warning-box">
        Free tier skips at <b>20%</b> — 2× the Family tier (9.2%). This friction
        directly correlates with lower session duration (78s vs 99s) and is a key
        driver of pre-churn disengagement.
        </div>""", unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="section-header">Sessions by Weekday</div>', unsafe_allow_html=True)
        weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wd = sessions.groupby("session_weekday").size().reindex(weekday_order).reset_index()
        wd.columns = ["Weekday","Sessions"]
        wd["is_weekend"] = wd["Weekday"].isin(["Saturday","Sunday"])

        fig4 = px.bar(wd, x="Weekday", y="Sessions",
                      color="is_weekend",
                      color_discrete_map={True:"#a371f7", False:"#58a6ff"},
                      labels={"Sessions":"Sessions"})
        fig4.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), showlegend=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <b>Weekend sessions are 50% higher</b> than weekday average (Fri–Sun ~39k each vs
        ~26k Mon–Thu). This creates a clear window for weekend-targeted push notifications
        and playlist promotions.
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Device analysis
    st.markdown('<div class="section-header">Listening by Device Type</div>', unsafe_allow_html=True)
    dev_agg = sess_e.groupby("device_type").agg(
        sessions=("session_id","count"),
        avg_listen=("listen_seconds","mean"),
        skip_rate=("skipped","mean"),
        revenue=("estimated_revenue_usd","sum")
    ).round(2).reset_index().sort_values("sessions", ascending=False)

    fig5 = px.scatter(dev_agg, x="avg_listen", y="skip_rate",
                      size="sessions", color="device_type",
                      text="device_type",
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      labels={"avg_listen":"Avg Listen Duration (s)","skip_rate":"Skip Rate"})
    fig5.update_traces(textposition="top center")
    fig5.update_layout(
        height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"), showlegend=False,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: USER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 User Segments":
    st.markdown("# 👥 User Segment Analysis")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Sessions & Skip Rate by Age Group</div>', unsafe_allow_html=True)
        age_agg = sess_e.groupby("age_group", observed=True).agg(
            sessions=("session_id","count"),
            skip_rate=("skipped","mean"),
            revenue=("estimated_revenue_usd","sum"),
            avg_listen=("listen_seconds","mean")
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=age_agg["age_group"].astype(str), y=age_agg["sessions"],
            name="Sessions", marker_color="#58a6ff", opacity=0.85
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=age_agg["age_group"].astype(str), y=age_agg["skip_rate"]*100,
            name="Skip Rate %", mode="lines+markers",
            line=dict(color="#f85149", width=2), marker=dict(size=9)
        ), secondary_y=True)
        fig.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d", title="Sessions"),
            yaxis2=dict(title="Skip Rate %"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        The <b>55+ cohort</b> generates 115,455 sessions — nearly 4× any other group —
        but skips at 12.9%, suggesting high loyalty. The <b>&lt;25 cohort</b> has the
        highest skip rate (14.1%), indicating content relevance gaps for younger users.
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Subscription Tier Distribution</div>', unsafe_allow_html=True)
        tier_counts = users["subscription_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier","Users"]

        fig2 = px.pie(tier_counts, values="Users", names="Tier", hole=0.6,
                      color_discrete_sequence=["#3fb950","#58a6ff","#a371f7"])
        fig2.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Tier table
        tier_stats = sessions.groupby("subscription_tier").agg(
            sessions=("session_id","count"),
            avg_listen=("listen_seconds","mean"),
            skip_rate=("skipped","mean"),
            revenue=("estimated_revenue_usd","sum")
        ).round(2).reset_index()
        tier_stats.columns = ["Tier","Sessions","Avg Listen(s)","Skip Rate","Revenue($)"]
        st.dataframe(
            tier_stats.set_index("Tier").style.format({
                "Skip Rate":"{:.1%}", "Revenue($)":"${:.2f}", "Avg Listen(s)":"{:.0f}s"
            }),
            use_container_width=True
        )

    st.divider()

    # Free-to-paid conversion analysis
    st.markdown('<div class="section-header">Free-to-Paid Conversion Funnel</div>', unsafe_allow_html=True)

    upgrades_from_free = subs[(subs["event_type"]=="upgrade") & (subs["from_tier"]=="Free")]
    free_users = users[users["subscription_tier"]=="Free"]["user_id"]
    free_sess = sessions[sessions["subscription_tier"]=="Free"]
    free_user_vol = free_sess.groupby("user_id").size()

    quartiles = pd.qcut(free_user_vol, q=4, labels=["Q1 Low","Q2 Mid-Low","Q3 Mid-High","Q4 Heavy"])
    free_quartile_df = pd.DataFrame({"user_id": free_user_vol.index, "quartile": quartiles, "sessions": free_user_vol.values})
    upgraded_ids = set(upgrades_from_free["user_id"])
    free_quartile_df["upgraded"] = free_quartile_df["user_id"].isin(upgraded_ids)

    conv_by_quartile = free_quartile_df.groupby("quartile", observed=True).agg(
        users=("user_id","count"),
        upgraded=("upgraded","sum")
    ).reset_index()
    conv_by_quartile["conv_rate"] = conv_by_quartile["upgraded"] / conv_by_quartile["users"]

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=conv_by_quartile["quartile"].astype(str), y=conv_by_quartile["users"],
                          name="Total Free Users", marker_color="#30363d"), secondary_y=False)
    fig3.add_trace(go.Bar(x=conv_by_quartile["quartile"].astype(str), y=conv_by_quartile["upgraded"],
                          name="Upgraded", marker_color="#3fb950"), secondary_y=False)
    fig3.add_trace(go.Scatter(x=conv_by_quartile["quartile"].astype(str),
                              y=conv_by_quartile["conv_rate"]*100,
                              name="Conv Rate %", mode="lines+markers",
                              line=dict(color="#d29922",width=2), marker=dict(size=10)),
                  secondary_y=True)
    fig3.update_layout(
        barmode="overlay", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#21262d", title="Users"),
        yaxis2=dict(title="Conversion Rate %"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GEOGRAPHIC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Geographic Analysis":
    st.markdown("# 🌍 Geographic Analysis")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Sessions by Country</div>', unsafe_allow_html=True)
        country_stats = sess_e.groupby("country_name").agg(
            sessions=("session_id","count"),
            revenue=("estimated_revenue_usd","sum"),
            skip_rate=("skipped","mean")
        ).sort_values("sessions", ascending=False).reset_index()

        fig = px.bar(country_stats, x="country_name", y="sessions",
                     color="revenue",
                     color_continuous_scale="Blues",
                     labels={"country_name":"","sessions":"Sessions","revenue":"Revenue ($)"})
        fig.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), coloraxis_showscale=False,
            xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Tier Mix by Country</div>', unsafe_allow_html=True)
        tier_country = sess_e.groupby(["country_name","subscription_tier"]).size().reset_index(name="sessions")
        tier_pct = tier_country.pivot_table(index="country_name", columns="subscription_tier", values="sessions", fill_value=0)
        tier_pct_norm = tier_pct.div(tier_pct.sum(axis=1), axis=0).reset_index()

        fig2 = go.Figure()
        colors = {"Free":"#f85149","Premium":"#58a6ff","Family":"#3fb950"}
        for tier in ["Free","Premium","Family"]:
            if tier in tier_pct_norm.columns:
                fig2.add_trace(go.Bar(
                    x=tier_pct_norm["country_name"],
                    y=tier_pct_norm[tier],
                    name=tier,
                    marker_color=colors[tier]
                ))
        fig2.update_layout(
            barmode="stack", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Genre over-indexing heatmap
    st.markdown('<div class="section-header">Country × Genre Over-Index Heatmap</div>', unsafe_allow_html=True)
    cg = sess_e.groupby(["country_name","genre_name"]).size().unstack(fill_value=0)
    global_share = cg.sum() / cg.sum().sum()
    cg_pct = cg.div(cg.sum(axis=1), axis=0)
    over_index = (cg_pct / global_share).round(2)

    fig3 = px.imshow(over_index,
                     color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=1.0,
                     zmin=0.3, zmax=2.5,
                     aspect="auto",
                     labels=dict(color="Over-Index"))
    fig3.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"),
        coloraxis_colorbar=dict(tickcolor="#8b949e", title=""),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <b>Key geographic affinities</b>: Reggae in South Africa (2.3×), Hip-Hop in US (2.0×),
    Latin in Brazil (2.0×), Electronic in Sweden (1.8×), Jazz in France (1.3×).
    These signals should drive geo-targeted playlist curation and artist promotion budgets.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONTENT & CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎵 Content & Catalogue":
    st.markdown("# 🎵 Content & Catalogue Analysis")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Revenue by Genre</div>', unsafe_allow_html=True)
        genre_rev = sess_e.groupby("genre_name").agg(
            sessions=("session_id","count"),
            revenue=("estimated_revenue_usd","sum"),
            skip_rate=("skipped","mean")
        ).sort_values("revenue", ascending=False).reset_index()

        fig = px.bar(genre_rev, x="genre_name", y="revenue",
                     color="skip_rate", color_continuous_scale="RdYlGn_r",
                     labels={"genre_name":"","revenue":"Revenue ($)","skip_rate":"Skip Rate"},
                     text=genre_rev["revenue"].apply(lambda x: f"${x:.0f}"))
        fig.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"),
            xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d"),
            coloraxis_colorbar=dict(title="Skip Rate", tickformat=".0%"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Algo vs Non-Algo Tracks</div>', unsafe_allow_html=True)
        sess_track = sessions.merge(tracks[["track_id","is_algorithmic_recommendation"]], on="track_id")
        algo_stats = sess_track.groupby("is_algorithmic_recommendation").agg(
            sessions=("session_id","count"),
            skip_rate=("skipped","mean"),
            avg_listen=("listen_seconds","mean"),
            revenue=("estimated_revenue_usd","sum"),
            discovery=("new_artist_discovered","mean")
        ).reset_index()
        algo_stats["label"] = algo_stats["is_algorithmic_recommendation"].map({True:"Algorithmic",False:"Non-Algorithmic"})

        fig2 = go.Figure()
        metrics = ["skip_rate","avg_listen","discovery"]
        labels  = ["Skip Rate","Avg Listen(s)","Discovery Rate"]
        colors  = ["#58a6ff","#3fb950"]
        for i, row in algo_stats.iterrows():
            vals = [row["skip_rate"]*100, row["avg_listen"]/10, row["discovery"]*100]
            fig2.add_trace(go.Bar(
                name=row["label"],
                x=labels, y=vals,
                marker_color=colors[i]
            ))
        fig2.update_layout(
            barmode="group", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class="insight-box">
        Algorithmic tracks drive <b>52.5% of sessions</b> and generate slightly higher revenue ($447 vs $405),
        with comparable skip rates (13.2% vs 12.7%). Algorithm performance is strong —
        explore expanding recommendation surface area.
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Top artists
    st.markdown('<div class="section-header">Top 15 Artists by Revenue</div>', unsafe_allow_html=True)
    sess_artist = sessions.merge(artists[["artist_id","artist_name","home_country"]], on="artist_id")
    top_artists = sess_artist.groupby(["artist_name","home_country"]).agg(
        sessions=("session_id","count"),
        revenue=("estimated_revenue_usd","sum"),
        skip_rate=("skipped","mean")
    ).sort_values("revenue", ascending=False).head(15).reset_index()

    fig3 = px.scatter(top_artists, x="sessions", y="revenue",
                      size="revenue", color="home_country",
                      text="artist_name",
                      labels={"sessions":"Sessions","revenue":"Revenue ($)","home_country":"Country"},
                      color_discrete_sequence=px.colors.qualitative.Bold)
    fig3.update_traces(textposition="top center")
    fig3.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FRAUD & RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Fraud & Risk":
    st.markdown("# ⚠️ Fraud & Risk Analysis")
    st.divider()

    fraud_sess = sessions[sessions["user_id"].isin(fraud_ids)]
    legit_sess = sessions[~sessions["user_id"].isin(fraud_ids)]

    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Fraud Users", f"{len(fraud_ids):,}", f"{len(fraud_ids)/len(users):.0%} of users", delta_color="inverse")
    with col2: st.metric("Fraud Sessions", f"{len(fraud_sess):,}", f"{len(fraud_sess)/len(sessions):.0%} of sessions", delta_color="inverse")
    with col3: st.metric("Fraud Revenue Share", f"{fraud_sess['estimated_revenue_usd'].sum()/sessions['estimated_revenue_usd'].sum():.0%}", delta_color="inverse")
    with col4: st.metric("Fraud Avg Listen", f"{fraud_sess['listen_seconds'].mean():.0f}s", f"vs {legit_sess['listen_seconds'].mean():.0f}s legit")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Session Volume: Fraud vs Legit</div>', unsafe_allow_html=True)
        fraud_monthly = fraud_sess.groupby("year_month").size().reset_index(name="sessions")
        legit_monthly = legit_sess.groupby("year_month").size().reset_index(name="sessions")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fraud_monthly["year_month"], y=fraud_monthly["sessions"],
                                 name="Fraud", fill="tozeroy", line=dict(color="#f85149"),
                                 fillcolor="rgba(248,81,73,0.15)"))
        fig.add_trace(go.Scatter(x=legit_monthly["year_month"], y=legit_monthly["sessions"],
                                 name="Legitimate", fill="tozeroy", line=dict(color="#3fb950"),
                                 fillcolor="rgba(63,185,80,0.15)"))
        fig.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#21262d", showticklabels=False),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Listen Duration Distribution</div>', unsafe_allow_html=True)
        sample_fraud = fraud_sess["listen_seconds"].sample(min(5000,len(fraud_sess)), random_state=42)
        sample_legit = legit_sess["listen_seconds"].sample(min(5000,len(legit_sess)), random_state=42)

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=sample_fraud, name="Fraud", nbinsx=40,
                                    marker_color="#f85149", opacity=0.7))
        fig2.add_trace(go.Histogram(x=sample_legit, name="Legitimate", nbinsx=40,
                                    marker_color="#3fb950", opacity=0.7))
        fig2.update_layout(
            barmode="overlay", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#21262d", title="Listen Seconds"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""<div class="danger-box">
    <b>🚨 Critical: Fraud cluster behaviour profile</b><br>
    • <b>326 sessions/user</b> vs 140 for legitimate users — 2.3× higher volume<br>
    • <b>73s avg listen time</b> vs 137s for legitimate — artificially short, just enough for royalty trigger (≥30s)<br>
    • <b>Skip rate 9.1%</b> vs 22% — paradoxically lower skip rate suggests automated/scripted playback<br>
    • <b>67.7% of session revenue</b> originates from flagged users — royalty manipulation risk is material<br>
    • <b>Recommended action</b>: Exclude fraud cluster from royalty calculations and investigate bot detection rules
    </div>""", unsafe_allow_html=True)

    # Fraud by tier
    st.markdown('<div class="section-header">Fraud Distribution by Tier</div>', unsafe_allow_html=True)
    fraud_tier = fraud_sess.groupby("subscription_tier").size().reset_index(name="sessions")
    legit_tier = legit_sess.groupby("subscription_tier").size().reset_index(name="sessions")
    fraud_tier["type"] = "Fraud"
    legit_tier["type"] = "Legitimate"
    combined = pd.concat([fraud_tier, legit_tier])

    fig3 = px.bar(combined, x="subscription_tier", y="sessions", color="type",
                  barmode="group",
                  color_discrete_map={"Fraud":"#f85149","Legitimate":"#3fb950"})
    fig3.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#21262d"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Churn Prediction":
    st.markdown("# 🔮 Churn Signals & Prediction")
    st.divider()

    churn_events = subs[subs["event_type"]=="churn"][["user_id","event_ts"]].copy()
    churn_events.columns = ["user_id","churn_date"]

    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Total Churn Events", "635")
    with col2: st.metric("Users Churned", "453", "47.1% of base", delta_color="inverse")
    with col3: st.metric("Avg Sessions Drop (pre-30d)", "-14%", "vs prior 30 days")
    with col4: st.metric("Skip Rate Rise (pre-30d)", "+2.3pp", "warning signal")

    st.divider()

    # Pre-churn behavior
    st.markdown('<div class="section-header">Pre-Churn Engagement Signal</div>', unsafe_allow_html=True)

    session_churn = sessions.merge(churn_events, on="user_id")
    session_churn["days_before"] = (session_churn["churn_date"] - session_churn["listen_start_ts"]).dt.days

    bins = [0,14,30,45,60,90]
    labels_b = ["0-14d","15-30d","31-45d","46-60d","61-90d"]
    session_churn["window"] = pd.cut(session_churn["days_before"], bins=bins, labels=labels_b)
    pre_churn_agg = session_churn[session_churn["days_before"]>=0].groupby("window", observed=True).agg(
        sessions=("session_id","count"),
        skip_rate=("skipped","mean"),
        avg_listen=("listen_seconds","mean")
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=pre_churn_agg["window"].astype(str), y=pre_churn_agg["sessions"],
                         name="Sessions", marker_color="#58a6ff"), secondary_y=False)
    fig.add_trace(go.Scatter(x=pre_churn_agg["window"].astype(str), y=pre_churn_agg["skip_rate"]*100,
                             name="Skip Rate %", mode="lines+markers",
                             line=dict(color="#f85149",width=2), marker=dict(size=9)),
                  secondary_y=True)
    fig.update_layout(
        height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", title="Days Before Churn"),
        yaxis=dict(gridcolor="#21262d", title="Sessions"),
        yaxis2=dict(title="Skip Rate %"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Churn by Tier</div>', unsafe_allow_html=True)
        churn_tier = subs[subs["event_type"]=="churn"].groupby("from_tier").size().reset_index(name="churns")
        fig2 = px.pie(churn_tier, values="churns", names="from_tier", hole=0.55,
                      color_discrete_sequence=["#f85149","#d29922","#58a6ff"])
        fig2.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Churn Events by Month (2024)</div>', unsafe_allow_html=True)
        churn_2024 = subs[(subs["event_type"]=="churn") & (subs["year"]==2024)]\
                         .groupby("year_month").size().reset_index(name="churns")
        fig3 = px.bar(churn_2024, x="year_month", y="churns",
                      color="churns", color_continuous_scale="Reds")
        fig3.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"), coloraxis_showscale=False,
            xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">🎯 Churn Early Warning — Recommended Intervention Rules</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="warning-box">
        <b>Rule 1 — Session Drop Alert</b><br>
        Trigger if a user's weekly sessions drop ≥30% for 2 consecutive weeks.
        Affects ~12% of active base per quarter.
        <br><br><i>Action: Push personalised "We miss you" playlist</i>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="warning-box">
        <b>Rule 2 — Skip Rate Spike</b><br>
        Trigger if 7-day rolling skip rate exceeds 25% (vs platform avg 12.9%).
        Indicates content relevance breakdown.
        <br><br><i>Action: Refresh recommendation engine seed tracks</i>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="warning-box">
        <b>Rule 3 — Free-tier Stagnation</b><br>
        Free users with ≥50 sessions who haven't converted in 90 days.
        869 conversions show this segment is receptive.
        <br><br><i>Action: Targeted discount trial (1-month 50% off Premium)</i>
        </div>""", unsafe_allow_html=True)
