"""Weekly narrative report: deterministic metrics, LLM-written prose.

Design contract
---------------
Every number is computed here, in Python, from pipeline artifacts. The model
receives ONLY the finished metrics dict and is instructed to use no figure that
is not in it. It never sees raw data and performs no arithmetic. If the API is
unavailable the pipeline degrades to a deterministic summary rather than
failing — an unattended weekly job must not break the build.

Grounding is necessary but not sufficient. A model given correct figures will
still misdescribe what they mean: inventing causes, or recasting an observed
figure as a forecast. Two mechanisms address that, in order of reliability:

1. Semantic context in the dict (order_concentration.affects,
   customer_risk.basis) so the model is not left to infer meaning.
2. Withholding (see for_narrative) for anything the model must not report at
   all. Prompt instructions to "omit or caveat these" proved unreliable across
   three iterations; removing the field is deterministic.

Revenue reporting
-----------------
Gross revenue can be dominated by a single large order that is later cancelled.
Cancellations live in returns.parquet, so gross would show growth that net does
not. The report leads with NET revenue and flags any period in which one order
exceeds CONCENTRATION_THRESHOLD_PCT of gross.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROCESSED_DIR = Path("data/processed")
TEMPLATE_DIR = Path("templates")

# One order above this share of gross revenue makes period-on-period
# comparison misleading and is surfaced explicitly.
CONCENTRATION_THRESHOLD_PCT = 15.0

# Metrics computed on gross revenue or unit volume, and therefore distorted
# when a single order dominates the period.
CONCENTRATION_AFFECTED = [
    "gross_revenue",
    "units",
    "average_order_value",
    "returns_value",
    "return_rate_pct",
    "top_products_gaining",
]

# Distorted, but retained in the model's view because the concentration cannot
# be explained without them.
NARRATIVE_RETAIN = {"gross_revenue"}

load_dotenv()


# --------------------------------------------------------------- helpers
def _delta_pct(current: float, previous: float,
               min_base: float | None = None) -> float | None:
    """Percent change, or None when no meaningful baseline exists.

    min_base guards against percentages computed on a near-zero baseline,
    which are arithmetically correct and practically meaningless (a returns
    line moving from -3k to -173k is not "-5215%" in any useful sense).
    """
    if previous in (0, None) or pd.isna(previous):
        return None
    if min_base is not None and abs(previous) < min_base:
        return None
    return round((current - previous) / abs(previous) * 100, 1)


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
            col: str = "date") -> pd.DataFrame:
    return df[(df[col] >= start) & (df[col] <= end)]


def _metric_block(cur: pd.DataFrame, prev: pd.DataFrame, col: str,
                  agg: str = "sum", min_base: float | None = None) -> dict:
    c = float(getattr(cur[col], agg)()) if len(cur) else 0.0
    p = float(getattr(prev[col], agg)()) if len(prev) else 0.0
    return {"current": round(c, 2), "previous": round(p, 2),
            "delta_pct": _delta_pct(c, p, min_base)}


def _order_concentration(cur_sales: pd.DataFrame, cur_returns: pd.DataFrame,
                         gross: float) -> dict:
    """Largest single order in the period and whether it was likely cancelled.

    The cancellation link is heuristic: this dataset carries no explicit
    reference from a C-invoice back to the order it reverses, so we match on
    magnitude within the same period. Stated as 'likely' for that reason.
    """
    if not len(cur_sales) or gross <= 0:
        return {"flagged": False}

    orders = (
        cur_sales.groupby("transaction_id", observed=True)
        .agg(revenue=("line_revenue", "sum"),
             product=("product_name", "first"),
             customer=("customer_id", "first"),
             date=("date", "min"))
        .sort_values("revenue", ascending=False)
    )
    top = orders.iloc[0]
    share = float(top["revenue"]) / gross * 100
    if share < CONCENTRATION_THRESHOLD_PCT:
        return {"flagged": False, "largest_order_share_pct": round(share, 1)}

    likely_cancelled = False
    if len(cur_returns):
        magnitudes = (
            cur_returns.groupby("transaction_id", observed=True)["line_revenue"]
            .sum().abs()
        )
        likely_cancelled = bool(
            ((magnitudes - float(top["revenue"])).abs()
             / float(top["revenue"]) < 0.01).any()
        )

    return {
        "flagged": True,
        "invoice": str(orders.index[0]),
        "revenue": round(float(top["revenue"]), 2),
        "largest_order_share_pct": round(share, 1),
        "product": str(top["product"]),
        "customer_id": str(top["customer"]),
        "date": pd.Timestamp(top["date"]).strftime("%Y-%m-%d"),
        "likely_cancelled": likely_cancelled,
        # Named so the model does not invent trading explanations for movements
        # that are arithmetic consequences of this one order.
        "affects": CONCENTRATION_AFFECTED,
        "note": (
            "A single order accounts for an unusually large share of gross "
            "revenue this period. A matching cancellation of the same value "
            "was found, so net revenue is the reliable figure. Movements in "
            "the metrics listed under 'affects' are consequences of this "
            "order, not of trading."
            if likely_cancelled else
            "A single order accounts for an unusually large share of gross "
            "revenue this period; movements in the metrics listed under "
            "'affects' should be read with that in mind."
        ),
    }


# ------------------------------------------------------- metrics assembly
def compute_metrics(config: dict) -> dict:
    """Build the complete, deterministic metrics dict. No model involved."""
    rep = config["report"]
    days = rep["comparison_days"]
    top_n = rep["top_movers"]
    currency = config["business_rules"]["currency"]

    daily = pd.read_parquet(PROCESSED_DIR / "daily_series.parquet")
    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    returns = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    anomalies = pd.read_parquet(PROCESSED_DIR / "anomalies.parquet")
    scores = pd.read_parquet(PROCESSED_DIR / "churn_scores.parquet")
    with (PROCESSED_DIR / "model_metrics.json").open() as f:
        model_metrics = json.load(f)

    end = daily["date"].max()
    cur_start = end - pd.Timedelta(days=days - 1)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)

    cur_daily = _window(daily, cur_start, end)
    prev_daily = _window(daily, prev_start, prev_end)

    gross = _metric_block(cur_daily, prev_daily, "revenue")
    orders = _metric_block(cur_daily, prev_daily, "orders")
    units = _metric_block(cur_daily, prev_daily, "units")

    # Returns baseline is often tiny; suppress the percentage rather than
    # print a four-digit one.
    returns_block = _metric_block(
        cur_daily, prev_daily, "returns_value",
        min_base=0.05 * abs(gross["current"]) if gross["current"] else None,
    )

    net_cur = gross["current"] + returns_block["current"]
    net_prev = gross["previous"] + returns_block["previous"]
    net = {"current": round(net_cur, 2), "previous": round(net_prev, 2),
           "delta_pct": _delta_pct(net_cur, net_prev)}

    rate_cur = (abs(returns_block["current"]) / gross["current"] * 100
                if gross["current"] else 0.0)
    rate_prev = (abs(returns_block["previous"]) / gross["previous"] * 100
                 if gross["previous"] else 0.0)
    return_rate = {"current": round(rate_cur, 1), "previous": round(rate_prev, 1),
                   "delta_pp": round(rate_cur - rate_prev, 1)}

    cur_aov = gross["current"] / orders["current"] if orders["current"] else 0.0
    prev_aov = gross["previous"] / orders["previous"] if orders["previous"] else 0.0

    # Distinct customers must be counted, not summed across days.
    cur_sales = _window(sales, cur_start, end + pd.Timedelta(days=1))
    prev_sales = _window(sales, prev_start, prev_end + pd.Timedelta(days=1))
    cur_returns = _window(returns, cur_start, end + pd.Timedelta(days=1))
    cur_cust = int(cur_sales["customer_id"].nunique())
    prev_cust = int(prev_sales["customer_id"].nunique())

    concentration = _order_concentration(cur_sales, cur_returns, gross["current"])

    # Product movers, current window vs previous.
    def _by_product(frame: pd.DataFrame) -> pd.Series:
        if not len(frame):
            return pd.Series(dtype="float64")
        return frame.groupby("product_id", observed=True)["line_revenue"].sum()

    cur_p, prev_p = _by_product(cur_sales), _by_product(prev_sales)
    names = (
        sales.groupby("product_id", observed=True)["product_name"].last()
        if len(sales) else pd.Series(dtype="object")
    )
    movers = (
        pd.DataFrame({"current": cur_p, "previous": prev_p})
        .fillna(0.0)
        .assign(change=lambda d: d["current"] - d["previous"])
    )
    # Require meaningful presence in one window to exclude one-off noise.
    movers = movers[(movers["current"] >= 100) | (movers["previous"] >= 100)]

    def _movers(ascending: bool) -> list[dict]:
        sub = movers.sort_values("change", ascending=ascending).head(top_n)
        return [
            {
                "product": str(names.get(pid, pid)),
                "current": round(float(r["current"]), 2),
                "previous": round(float(r["previous"]), 2),
                "change": round(float(r["change"]), 2),
            }
            for pid, r in sub.iterrows()
        ]

    new_anomalies = [
        {
            "date": a["date"].strftime("%Y-%m-%d"),
            "direction": a["direction"],
            "revenue": round(float(a["revenue"]), 2),
            "expected_revenue": round(float(a["expected_revenue"]), 2),
            "z_score": round(float(a["z_score"]), 2),
        }
        for _, a in _window(anomalies, cur_start, end).iterrows()
    ]

    at_risk = scores[scores["at_risk"]].sort_values("revenue_365d", ascending=False)
    risk_block = model_metrics["risk"]

    return {
        "currency": currency,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": {
            "start": cur_start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "days": days,
            "trading_days": len(cur_daily),
        },
        "previous_period": {
            "start": prev_start.strftime("%Y-%m-%d"),
            "end": prev_end.strftime("%Y-%m-%d"),
            "trading_days": len(prev_daily),
        },
        "headline": {
            "net_revenue": net,
            "gross_revenue": gross,
            "orders": orders,
            "units": units,
            "average_order_value": {
                "current": round(cur_aov, 2),
                "previous": round(prev_aov, 2),
                "delta_pct": _delta_pct(cur_aov, prev_aov),
            },
            "active_customers": {
                "current": cur_cust,
                "previous": prev_cust,
                "delta_pct": _delta_pct(cur_cust, prev_cust),
            },
            "returns_value": returns_block,
            "return_rate_pct": return_rate,
        },
        "order_concentration": concentration,
        "anomalies_in_period": new_anomalies,
        "top_products_gaining": _movers(ascending=False),
        "top_products_declining": _movers(ascending=True),
        "customer_risk": {
            "scoring_snapshot": risk_block["scoring_snapshot"],
            "at_risk_customers": risk_block["at_risk_customers"],
            "customers_scored": risk_block["customers_scored"],
            "revenue_at_risk_ttm": risk_block["revenue_at_risk_ttm"],
            "share_of_ttm_revenue_pct": risk_block["share_of_ttm_revenue_pct"],
            # Sourced from models.py so the caveat cannot drift from the
            # methodology that produced the number.
            "basis": model_metrics.get("reporting_note", ""),
            "top_at_risk": [
                {
                    "customer_id": str(r["customer_id"]),
                    "region": str(r["region"]),
                    "ttm_revenue": round(float(r["revenue_365d"]), 2),
                    "days_since_last_order": int(r["recency_days"]),
                }
                for _, r in at_risk.head(top_n).iterrows()
            ],
        },
    }


def for_narrative(metrics: dict) -> dict:
    """Strip metrics that a single dominant order has made meaningless.

    Prompt instructions to "omit or caveat these" proved unreliable across
    three iterations — the model follows hard rules (never invent a number,
    never state a cause) far better than judgment calls. Removing the figures
    from its input is deterministic; asking nicely is not.

    The HTML table renders from the full dict, so the reader still sees every
    row. Only the model's view is narrowed.
    """
    conc = metrics.get("order_concentration", {})
    if not conc.get("flagged"):
        return metrics

    out = copy.deepcopy(metrics)
    withheld: list[str] = []

    for key in conc.get("affects", []):
        if key in NARRATIVE_RETAIN:
            continue
        if key in out["headline"]:
            out["headline"].pop(key)
            withheld.append(key)
        elif key == "top_products_gaining":
            out["top_products_gaining"] = []
            withheld.append(key)

    out["withheld_from_narrative"] = {
        "fields": withheld,
        "reason": (
            f"Distorted by invoice {conc.get('invoice', 'unknown')}; omitted so "
            "they are not reported as trading performance. Gross revenue is "
            "retained so the concentration itself can be explained."
        ),
    }
    return out


# ------------------------------------------------------------- narrative
SYSTEM_PROMPT = (
    "You are a business analyst writing a {frequency} performance summary for a "
    "{recipient_role}. They will read this in under a minute and need to know "
    "what to act on.\n\n"
    "GROUNDING (absolute):\n"
    "- Use ONLY figures present in the JSON. Never infer, estimate, extrapolate, "
    "or calculate a number that is not there.\n"
    "- If a figure is null or absent, omit that point entirely.\n"
    "- Write currency as '{currency} 1,234' — the code, never a symbol.\n"
    "- Round to whole units. Do not write pence or cents.\n"
    "- Never explain WHY a number moved. The data contains no causes. Do not "
    "write phrases like 'reflecting fewer but higher-value transactions' or "
    "'driven by' or 'due to'. State what changed, not why.\n\n"
    "SELECTION (this is the hard part):\n"
    "- Do NOT list every metric. Choose the three or four that matter and "
    "explain them. A recitation of the JSON is a failure.\n"
    "- Lead with net revenue and its change. Net is what the business earned.\n"
    "- If order_concentration.flagged is true, explain it concretely in your "
    "second or third sentence: name the invoice, the amount, its share of "
    "gross, and whether it was cancelled. Do not write vague phrases like "
    "'one order distorts this figure'.\n"
    "- If withheld_from_narrative is present, those fields were removed "
    "because that order made them meaningless. Do not mention them or "
    "speculate about them.\n"
    "- Close with customer risk. Describe revenue_at_risk_ttm exactly as "
    "customer_risk.basis defines it: revenue these customers have ALREADY "
    "generated over the past year, used to size the exposure. It is NOT a "
    "forecast, NOT expected loss, and NOT 'potential lost revenue'. Say the "
    "customers are ranked by churn probability.\n"
    "- You may end with one short action sentence, but only if it follows "
    "directly from a figure you have already stated.\n\n"
    "STYLE:\n"
    "- Tone: {tone}. Plain business language. No jargon, no hedging.\n"
    "- At most {max_words} words. Continuous prose. No headings, no bullets, "
    "no preamble, no sign-off.\n"
    "- Do not reference 'the report', 'the data', or 'the JSON'. Write as if "
    "stating facts about the business."
)


def _fallback_narrative(m: dict) -> str:
    """Deterministic summary used when the model is unavailable.

    Uses only fields that survive for_narrative(), so it works on either the
    full or the filtered dict.
    """
    cur = m["currency"]
    h = m["headline"]
    net = h["net_revenue"]
    d = net["delta_pct"]
    direction = "up" if (d or 0) > 0 else "down"
    parts = [
        f"Net revenue for {m['period']['start']} to {m['period']['end']} was "
        f"{cur} {net['current']:,.0f}"
        + (f", {direction} {abs(d):.1f}% on the previous period."
           if d is not None else "."),
    ]

    conc = m["order_concentration"]
    if conc.get("flagged"):
        parts.append(
            f"Gross revenue of {cur} {h['gross_revenue']['current']:,.0f} includes "
            f"a single order of {cur} {conc['revenue']:,.0f} "
            f"({conc['largest_order_share_pct']}% of the period)"
            + (", which was subsequently cancelled." if conc["likely_cancelled"]
               else "; treat gross growth with caution.")
        )

    parts.append(
        f"Orders totalled {h['orders']['current']:,.0f} from "
        f"{h['active_customers']['current']:,} active customers."
    )

    n = len(m["anomalies_in_period"])
    if n:
        parts.append(f"{n} daily revenue anomal{'y was' if n == 1 else 'ies were'} "
                     "flagged in the period.")

    risk = m["customer_risk"]
    parts.append(
        f"{risk['at_risk_customers']:,} customers ranked highest for churn risk "
        f"generated {cur} {risk['revenue_at_risk_ttm']:,.0f} over the past year, "
        f"{risk['share_of_ttm_revenue_pct']}% of trailing-twelve-month revenue."
    )
    return " ".join(parts)


def generate_narrative(metrics: dict, config: dict) -> dict:
    """Return {text, model, grounded, generated_at}.

    Expects the filtered dict from for_narrative(), not the full one.
    """
    rep = config["report"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not os.getenv("OPENAI_API_KEY"):
        return {"text": _fallback_narrative(metrics),
                "model": "deterministic-fallback",
                "grounded": False, "generated_at": now,
                "note": "OPENAI_API_KEY not set; deterministic summary used."}

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=rep["model"],
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        frequency=rep["frequency"],
                        recipient_role=rep["recipient_role"],
                        tone=rep["tone"],
                        max_words=rep["max_words"],
                        currency=metrics["currency"],
                    ),
                },
                {"role": "user", "content": json.dumps(metrics, indent=2)},
            ],
            temperature=0.2,
        )
        return {"text": response.choices[0].message.content.strip(),
                "model": rep["model"], "grounded": True, "generated_at": now}
    except Exception as exc:  # noqa: BLE001 — never break an unattended job
        return {"text": _fallback_narrative(metrics),
                "model": "deterministic-fallback",
                "grounded": False, "generated_at": now,
                "note": f"LLM call failed ({type(exc).__name__}); fallback used."}


# ---------------------------------------------------------------- render
def render_html(metrics: dict, narrative: dict, config: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["money"] = lambda v: f"{v:,.0f}"
    env.filters["signed"] = lambda v: f"{v:+.1f}%" if v is not None else "n/a"
    return env.get_template("report.html").render(
        m=metrics, narrative=narrative, cfg=config,
    )


def run(config: dict) -> dict:
    out_dir = Path(config["report"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(config)
    # Model sees the filtered view; the HTML table renders the full one.
    narrative = generate_narrative(for_narrative(metrics), config)
    html = render_html(metrics, narrative, config)

    stamp = metrics["period"]["end"]
    html_path = out_dir / f"weekly_report_{stamp}.html"
    html_path.write_text(html, encoding="utf-8")

    # The dashboard reads this one; keep the key names stable.
    with (out_dir / "latest_narrative.json").open("w", encoding="utf-8") as f:
        json.dump(narrative, f, indent=2)
    with (out_dir / "latest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "html_path": html_path.as_posix(),
        "grounded": narrative["grounded"],
        "model": narrative["model"],
        "period": f"{metrics['period']['start']} to {metrics['period']['end']}",
        "words": len(narrative["text"].split()),
        "concentration_flagged": metrics["order_concentration"].get("flagged", False),
        "note": narrative.get("note"),
    }


if __name__ == "__main__":
    from src.ingest import load_config

    result = run(load_config())
    print(f"\nPeriod   : {result['period']}")
    print(f"Model    : {result['model']} (grounded: {result['grounded']})")
    print(f"Words    : {result['words']}")
    print(f"Concentr.: {'FLAGGED' if result['concentration_flagged'] else 'normal'}")
    print(f"Written  : {result['html_path']}")
    if result["note"]:
        print(f"Note     : {result['note']}")
    print()