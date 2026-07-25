"""BI Automation Dashboard — read-only view over pipeline artifacts.

This app never trains a model, never reads raw data, and never recomputes a
business metric. It renders what src/pipeline.py has already written to
data/processed/. That separation is what keeps the deployed footprint small
and guarantees the dashboard and the weekly report can never disagree.

Run from the repo root:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Streamlit puts the script's own directory on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest import load_config  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

ACCENT = "#2563eb"
WARN = "#dc2626"
GOOD = "#16a34a"

st.set_page_config(
    page_title="BI Automation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- loaders
# NOTE: every cached function below takes only hashable scalar arguments.
# Never decorate a function that takes a DataFrame — Streamlit must hash the
# whole frame to build the cache key, and that hash path is unstable on
# Arrow-backed string columns.
@st.cache_data(show_spinner=False)
def _config() -> dict:
    return load_config(ROOT / "config" / "config.yaml")


@st.cache_data(show_spinner=False)
def _parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / name)


@st.cache_data(show_spinner=False)
def _json(name: str) -> dict:
    with (PROCESSED / name).open() as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _narrative() -> dict | None:
    path = REPORTS / "latest_narrative.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def money(v: float, cur: str) -> str:
    if abs(v) >= 1_000_000:
        return f"{cur} {v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"{cur} {v / 1_000:,.1f}k"
    return f"{cur} {v:,.0f}"


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas StringDtype columns to object before Arrow serialization.

    Streamlit converts DataFrames to Arrow for transport to the browser. On
    this stack (macOS ARM, pandas 2.3.3 / pyarrow 25.0.0) that conversion
    segfaults inside pandas' StringArray.__arrow_array__ for some frames.
    Plain object columns take a different, stable conversion path.

    Every DataFrame handed to st.dataframe() must pass through here.
    """
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.StringDtype):
            out[col] = out[col].astype(object)
    return out


# --------------------------------------------------- presentation helpers
def assign_segments(df: pd.DataFrame) -> pd.Series:
    """RFM tiers for display only.

    Quartile-rank recency, frequency and monetary (1-4 each, higher = better),
    sum to a 3-12 score, cut into four named tiers. Deliberately a presentation
    concern, not a modelling one — the churn model has its own ranking.

    Not cached: this is a sub-millisecond computation on ~4k rows, and caching
    it would force Streamlit to hash the input DataFrame on every rerun.
    """
    r = pd.qcut(df["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    m = pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    total = r + f + m
    return pd.cut(
        total,
        bins=[0, 5, 7, 9, 12],
        labels=["Hibernating", "Needs attention", "Loyal", "Champions"],
    ).astype(str)


# ------------------------------------------------------------------ data
cfg = _config()
CUR = cfg["business_rules"]["currency"]
ROLE = cfg["report"]["recipient_role"]

daily = _parquet("daily_series.parquet")
region_daily = _parquet("daily_by_region.parquet")
baseline = _parquet("daily_baseline.parquet")
anomalies = _parquet("anomalies.parquet")
products = _parquet("product_summary.parquet")
scores = _parquet("churn_scores.parquet")
quality = _json("quality_report.json")
metrics = _json("model_metrics.json")

scores = scores.assign(segment=assign_segments(scores))
snapshot = metrics["risk"]["scoring_snapshot"]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Filters")

d_min = daily["date"].min().date()
d_max = daily["date"].max().date()
date_range = st.sidebar.date_input(
    "Trading period", value=(d_min, d_max), min_value=d_min, max_value=d_max
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = pd.Timestamp(d_min), pd.Timestamp(d_max)

all_regions = sorted(region_daily["region"].dropna().unique())
picked = st.sidebar.multiselect(
    "Region", options=all_regions, default=[],
    help="Leave empty for all regions. Applies to the Executive and Customers tabs.",
)
regions = picked or all_regions
region_filtered = bool(picked)

segments = st.sidebar.multiselect(
    "Customer segment",
    options=["Champions", "Loyal", "Needs attention", "Hibernating"],
    default=[],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Currency: {CUR}  \n"
    f"Report audience: {ROLE}  \n"
    f"Churn window: {cfg['business_rules']['churn_window_days']} days"
)
st.sidebar.caption(
    "All values read from pipeline artifacts. "
    "Nothing on this page is computed from raw data."
)

# ---------------------------------------------------------- apply filters
mask = (region_daily["date"] >= start) & (region_daily["date"] <= end)
rd = region_daily[mask & region_daily["region"].isin(regions)]
period = rd.groupby("date", as_index=False).agg(
    revenue=("revenue", "sum"), orders=("orders", "sum"), units=("units", "sum")
)

risk_pool = scores.copy()
if region_filtered:
    risk_pool = risk_pool[risk_pool["region"].isin(regions)]
if segments:
    risk_pool = risk_pool[risk_pool["segment"].isin(segments)]
at_risk = risk_pool[risk_pool["at_risk"]]

# ------------------------------------------------------------------- head
st.title("Retail Performance & Customer Risk")
st.caption(
    f"Automated pipeline · {quality['source']['rows_read']:,} source rows · "
    f"{quality['source']['date_min'][:10]} to {quality['source']['date_max'][:10]}"
)

tab_exec, tab_cust, tab_prod, tab_qual = st.tabs(
    ["Executive", "Customers", "Products", "Data Quality"]
)

# -------------------------------------------------------------- EXECUTIVE
with tab_exec:
    st.subheader("Trading period")
    st.caption(f"{start.date()} to {end.date()} · {len(period):,} trading days")

    rev = float(period["revenue"].sum())
    ords = int(period["orders"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", money(rev, CUR))
    c2.metric("Orders", f"{ords:,}")
    c3.metric("Average order value", f"{CUR} {rev / ords:,.2f}" if ords else "—")

    st.divider()
    st.subheader("Customer risk")
    st.caption(
        f"Fixed at the scoring snapshot of {snapshot}. "
        "Responds to the region and segment filters, but **not** to the date "
        "filter — the model scores one point in time, so a historical date "
        "range cannot change who is at risk today."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Active customers (TTM)", f"{len(risk_pool):,}")
    c2.metric("Flagged at risk", f"{len(at_risk):,}")
    c3.metric("TTM revenue at risk", money(float(at_risk["revenue_365d"].sum()), CUR))

    st.divider()
    st.subheader("Revenue trend")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=period["date"], y=period["revenue"], mode="lines",
            name="Daily revenue", line=dict(color=ACCENT, width=1.4),
        )
    )

    if not region_filtered:
        b = baseline[(baseline["date"] >= start) & (baseline["date"] <= end)]
        fig.add_trace(
            go.Scatter(
                x=b["date"], y=b["expected_revenue"], mode="lines",
                name="28-day expected",
                line=dict(color="#94a3b8", width=1, dash="dot"),
            )
        )
        a = anomalies[(anomalies["date"] >= start) & (anomalies["date"] <= end)]
        for direction, colour in (("spike", GOOD), ("drop", WARN)):
            sub = a[a["direction"] == direction]
            fig.add_trace(
                go.Scatter(
                    x=sub["date"], y=sub["revenue"], mode="markers",
                    name=f"{direction.title()}s ({len(sub)})",
                    marker=dict(color=colour, size=9, symbol="circle-open",
                                line=dict(width=2)),
                )
            )
    else:
        st.caption(
            "Anomaly baseline is computed on total daily revenue, so it is "
            "hidden while a region filter is active rather than shown against "
            "a population it was not fitted to."
        )

    fig.update_layout(
        template="plotly_white", height=380, hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title=f"Revenue ({CUR})", xaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("This week's summary")
    narrative = _narrative()
    if narrative:
        st.info(narrative.get("text", ""))
        st.caption(
            f"Generated {narrative.get('generated_at', 'unknown')} · "
            "model-written from pipeline figures only"
        )
    else:
        st.info(
            "Narrative report not generated yet. Run `python -m src.pipeline "
            "--stage report` once Session 4 is built."
        )

# -------------------------------------------------------------- CUSTOMERS
with tab_cust:
    st.subheader("Segments")
    st.caption(
        "RFM quartile tiers, shown for orientation. The at-risk flag comes "
        "from the churn model and is independent of these tiers."
    )

    seg = (
        risk_pool.groupby("segment")
        .agg(customers=("customer_id", "size"),
             ttm_revenue=("revenue_365d", "sum"),
             at_risk=("at_risk", "sum"))
        .reindex(["Champions", "Loyal", "Needs attention", "Hibernating"])
        .dropna(how="all")
        .reset_index()
    )
    if len(seg):
        cols = st.columns(len(seg))
        for col, row in zip(cols, seg.itertuples()):
            col.metric(row.segment, f"{int(row.customers):,}",
                       f"{int(row.at_risk)} at risk", delta_color="inverse")
    else:
        st.info("No customers match the current filters.")

    if len(risk_pool):
        fig = px.scatter(
            risk_pool, x="recency_days", y="frequency",
            size="monetary", color="churn_probability",
            color_continuous_scale="RdYlGn_r", size_max=28,
            hover_data={"customer_id": True, "revenue_365d": ":,.0f",
                        "segment": True},
            labels={"recency_days": "Days since last order",
                    "frequency": "Orders", "churn_probability": "Churn prob."},
        )
        fig.update_layout(template="plotly_white", height=440,
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("At-risk customers")
    st.caption(
        f"Top decile by churn probability, ranked by revenue at stake · "
        f"snapshot {snapshot}"
    )

    if len(at_risk):
        table = (
            at_risk.sort_values("revenue_365d", ascending=False)
            [["risk_rank", "customer_id", "segment", "region", "churn_probability",
              "recency_days", "frequency", "revenue_365d", "monetary"]]
            .rename(columns={
                "risk_rank": "Rank", "customer_id": "Customer", "segment": "Segment",
                "region": "Region", "churn_probability": "Churn prob.",
                "recency_days": "Days silent", "frequency": "Orders",
                "revenue_365d": f"TTM revenue ({CUR})",
                "monetary": f"Lifetime ({CUR})",
            })
        )
        st.dataframe(
            arrow_safe(table), width="stretch", hide_index=True, height=340,
            column_config={
                "Churn prob.": st.column_config.ProgressColumn(
                    "Churn prob.", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                f"TTM revenue ({CUR})": st.column_config.NumberColumn(format="%.0f"),
                f"Lifetime ({CUR})": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.download_button(
            "Download call list (CSV)",
            table.to_csv(index=False).encode(),
            file_name=f"at_risk_customers_{snapshot}.csv",
            mime="text/csv",
        )
    else:
        st.info("No at-risk customers match the current filters.")

    st.divider()
    st.subheader("Customer drill-down")
    if len(at_risk):
        choice = st.selectbox(
            "Customer", at_risk.sort_values("risk_rank")["customer_id"].tolist()
        )
        row = at_risk[at_risk["customer_id"] == choice].iloc[0]
        med = risk_pool.median(numeric_only=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn probability", f"{row['churn_probability']:.1%}")
        c2.metric("Days since last order", f"{int(row['recency_days'])}",
                  f"{int(row['recency_days'] - med['recency_days'])} vs median",
                  delta_color="inverse")
        c3.metric("Orders", f"{int(row['frequency'])}",
                  f"{int(row['frequency'] - med['frequency'])} vs median")
        c4.metric("TTM revenue", money(float(row["revenue_365d"]), CUR))
        st.caption(
            f"Usual gap between orders: {row['avg_gap_days']:.0f} days · "
            f"currently {row['gap_ratio']:.1f}x that gap · "
            f"return rate {row['return_rate']:.1%} · region {row['region']}"
        )
    else:
        st.info("No at-risk customers match the current filters.")

# --------------------------------------------------------------- PRODUCTS
with tab_prod:
    st.caption("Product view is global — the region and date filters do not apply here.")

    top_n = st.slider("Products shown", 5, 30, 15)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top by revenue")
        best = products.head(top_n).sort_values("revenue")
        fig = px.bar(best, x="revenue", y="product_name", orientation="h",
                     labels={"revenue": f"Revenue ({CUR})", "product_name": ""})
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(template="plotly_white", height=460,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Highest return rates")
        st.caption("Products with at least 100 orders")
        worst = (
            products[products["orders"] >= 100]
            .nlargest(top_n, "return_rate").sort_values("return_rate")
        )
        fig = px.bar(worst, x="return_rate", y="product_name", orientation="h",
                     labels={"return_rate": "Return rate", "product_name": ""})
        fig.update_traces(marker_color=WARN)
        fig.update_layout(template="plotly_white", height=460,
                          margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Revenue concentration")

    p = products.sort_values("revenue", ascending=False).reset_index(drop=True)
    p["cum_share"] = p["revenue"].cumsum() / p["revenue"].sum()
    p["rank_share"] = (p.index + 1) / len(p)
    n80 = int((p["cum_share"] < 0.8).sum()) + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["rank_share"], y=p["cum_share"], mode="lines",
                             line=dict(color=ACCENT, width=2),
                             name="Cumulative revenue"))
    fig.add_hline(y=0.8, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(template="plotly_white", height=340,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Share of catalogue",
                      yaxis_title="Share of revenue",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{n80:,} of {len(p):,} products ({n80 / len(p):.1%} of the catalogue) "
        "generate 80% of revenue."
    )

# ----------------------------------------------------------- DATA QUALITY
with tab_qual:
    st.subheader("Pipeline")
    st.caption(f"Last run {quality['generated_at'][:19].replace('T', ' ')} UTC")

    out, k = quality["outputs"], quality["kpis"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source rows", f"{quality['source']['rows_read']:,}")
    c2.metric("Retained", f"{out['sales_rows']:,}", f"{out['retention_pct']}%")
    c3.metric("Returns isolated", f"{out['returns_rows']:,}")
    c4.metric("Net revenue", money(k["net_revenue"], CUR))

    st.markdown("**Cleaning rules applied**")
    rules = pd.DataFrame(quality["rules_applied"])[
        ["rule", "description", "rows_before", "rows_dropped", "pct_dropped"]
    ].rename(columns={
        "rule": "Rule", "description": "What it does",
        "rows_before": "Rows in", "rows_dropped": "Dropped", "pct_dropped": "%",
    })
    st.dataframe(arrow_safe(rules), width="stretch", hide_index=True)

    st.markdown("**Null rates after cleaning**")
    nulls = (
        pd.Series(quality["null_rates_after_cleaning"])
        .sort_values(ascending=False)
        .rename("null rate")
        .to_frame()
    )
    st.dataframe(arrow_safe(nulls).style.format("{:.2%}"), width="stretch")

    st.divider()
    st.subheader("Model")

    val = metrics["model"]["validation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC", f"{val['roc_auc']:.3f}")
    c2.metric("Precision @ top 10%", f"{val['precision_at_decile']:.1%}")
    c3.metric("Lift", f"{val['lift_at_decile']}x")
    c4.metric("Validation snapshot", metrics["model"]["validation_snapshot"])

    imp = (
        pd.DataFrame(metrics["model"]["feature_importance"])
        .head(10).sort_values("importance")
    )
    fig = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std",
                 labels={"importance": "Permutation importance (AUC drop)",
                         "feature": ""})
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(template="plotly_white", height=360,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Seasonality of churn")
    seas = pd.DataFrame(metrics["seasonality"])
    fig = px.line(seas, x="label_window_ends", y="churn_rate", markers=True,
                  labels={"label_window_ends": "90-day window ends",
                          "churn_rate": "Churn rate (%)"})
    fig.update_traces(line_color=WARN)
    fig.update_layout(template="plotly_white", height=320,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    lo, hi = seas["churn_rate"].min(), seas["churn_rate"].max()
    st.warning(
        f"**Churn is seasonal — a post-Christmas problem.** Across the training "
        f"snapshots, 90-day churn ranges from {lo:.1f}% to {hi:.1f}%, peaking for "
        "windows that fall after the December gift peak. Customers active into "
        "the Christmas rush lapse at a markedly higher rate than the autumn "
        "cohort, which makes a January reactivation campaign the "
        "highest-leverage retention action."
    )
    st.caption(metrics["reporting_note"])