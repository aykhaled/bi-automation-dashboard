"""Clean canonical transactional data and emit a data quality report.

Every filter is applied via CleaningLog.apply(), so the quality report is a
by-product of cleaning rather than a separate, drift-prone reconstruction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingest import load_config, load_raw

PROCESSED_DIR = Path("data/processed")


@dataclass
class RuleResult:
    name: str
    description: str
    rows_before: int
    rows_after: int

    def to_dict(self) -> dict:
        dropped = self.rows_before - self.rows_after
        pct = (dropped / self.rows_before * 100) if self.rows_before else 0.0
        return {
            "rule": self.name,
            "description": self.description,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_dropped": dropped,
            "pct_dropped": round(pct, 3),
        }


class CleaningLog:
    def __init__(self) -> None:
        self.rules: list[RuleResult] = []

    def apply(self, df: pd.DataFrame, name: str, description: str,
              keep_mask: pd.Series) -> pd.DataFrame:
        before = len(df)
        out = df[keep_mask]
        self.rules.append(RuleResult(name, description, before, len(out)))
        return out

    def record(self, name: str, description: str, before: int, after: int) -> None:
        self.rules.append(RuleResult(name, description, before, after))

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.rules]


def clean(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, CleaningLog]:
    """Return (sales, returns, log)."""
    rules = config["business_rules"]
    prefix = rules["cancellation_prefix"]
    pattern = rules["product_code_pattern"]
    price_floor = rules.get("min_order_value", 0)
    log = CleaningLog()

    if rules.get("drop_exact_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        log.record(
            "drop_exact_duplicates",
            "Identical rows repeated in the source extract.",
            before, len(df),
        )

    # Cancellations are a KPI, not noise — separate rather than drop.
    is_cancellation = df["transaction_id"].str.startswith(prefix, na=False)
    returns = df[is_cancellation].copy()
    sales = df[~is_cancellation].copy()
    log.record(
        "split_cancellations",
        f"Invoices prefixed '{prefix}' routed to the returns frame.",
        len(df), len(sales),
    )

    is_product = sales["product_id"].str.match(pattern, na=False)
    sales = log.apply(
        sales, "drop_non_product_lines",
        "Postage, discounts, bank charges and manual adjustments.",
        is_product,
    )

    sales = log.apply(
        sales, "drop_non_positive_quantity",
        "Sales rows must have quantity greater than zero.",
        sales["quantity"] > 0,
    )

    sales = log.apply(
        sales, "drop_non_positive_price",
        f"Unit price must exceed {price_floor}.",
        sales["unit_price"] > price_floor,
    )

    if rules.get("drop_missing_customer", False):
        sales = log.apply(
            sales, "drop_missing_customer",
            "Guest-checkout rows excluded by configuration.",
            sales["customer_id"].notna(),
        )

    returns = returns[returns["product_id"].str.match(pattern, na=False)]

    for frame in (sales, returns):
        frame["line_revenue"] = frame["quantity"] * frame["unit_price"]

    # Retained, not dropped: needed for revenue totals, excluded from churn.
    sales["has_customer_id"] = sales["customer_id"].notna()

    return sales.reset_index(drop=True), returns.reset_index(drop=True), log


def build_quality_report(sales, returns, log, ingest_stats, config) -> dict:
    gross = float(sales["line_revenue"].sum())
    returned = float(returns["line_revenue"].sum())  # negative
    identified = sales["has_customer_id"].mean() if len(sales) else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "currency": config["business_rules"]["currency"],
        "source": {
            # as_posix() so the artifact is byte-identical whether the pipeline
            # ran on macOS, Windows or the Linux deploy target. A Windows path
            # rendering in the deployed Data Quality tab looks like a mistake.
            "file": Path(ingest_stats["source_file"]).as_posix(),
            "rows_read": ingest_stats["rows_read"],
            "date_min": ingest_stats["date_min"],
            "date_max": ingest_stats["date_max"],
            "unparsed_dates": ingest_stats["date_parse_failures"],
        },
        "rules_applied": log.to_list(),
        "outputs": {
            "sales_rows": len(sales),
            "returns_rows": len(returns),
            "retention_pct": round(len(sales) / ingest_stats["rows_read"] * 100, 2),
        },
        "kpis": {
            "gross_revenue": round(gross, 2),
            "returns_value": round(returned, 2),
            "net_revenue": round(gross + returned, 2),
            "return_rate_pct": round(abs(returned) / gross * 100, 2) if gross else 0.0,
            "orders": int(sales["transaction_id"].nunique()),
            "customers": int(sales["customer_id"].nunique()),
            "products": int(sales["product_id"].nunique()),
            "rows_with_customer_pct": round(float(identified) * 100, 2),
        },
        "null_rates_after_cleaning": {
            c: round(float(sales[c].isna().mean()), 4) for c in sales.columns
        },
    }


def write_artifacts(sales, returns, report) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(PROCESSED_DIR / "sales.parquet", index=False)
    returns.to_parquet(PROCESSED_DIR / "returns.parquet", index=False)
    with (PROCESSED_DIR / "quality_report.json").open("w") as f:
        json.dump(report, f, indent=2)


def print_summary(report: dict) -> None:
    cur = report["currency"]
    print("\n" + "=" * 68)
    print("DATA QUALITY REPORT")
    print("=" * 68)
    src = report["source"]
    print(f"Source        : {src['file']}")
    print(f"Rows read     : {src['rows_read']:,}")
    print(f"Date range    : {src['date_min'][:10]} -> {src['date_max'][:10]}")

    print(f"\n{'Rule':<28}{'Before':>12}{'Dropped':>12}{'%':>8}")
    print("-" * 68)
    for r in report["rules_applied"]:
        print(f"{r['rule']:<28}{r['rows_before']:>12,}"
              f"{r['rows_dropped']:>12,}{r['pct_dropped']:>7.2f}%")

    out, k = report["outputs"], report["kpis"]
    print("-" * 68)
    print(f"Sales rows retained : {out['sales_rows']:,} "
          f"({out['retention_pct']}% of source)")
    print(f"Returns rows        : {out['returns_rows']:,}")
    print(f"\nGross revenue  : {cur} {k['gross_revenue']:,.2f}")
    print(f"Returns        : {cur} {k['returns_value']:,.2f}")
    print(f"Net revenue    : {cur} {k['net_revenue']:,.2f}  "
          f"(return rate {k['return_rate_pct']}%)")
    print(f"Orders / customers / products : "
          f"{k['orders']:,} / {k['customers']:,} / {k['products']:,}")
    print(f"Rows with a customer ID       : {k['rows_with_customer_pct']}%")
    print("=" * 68 + "\n")


def run(config: dict) -> dict:
    df, ingest_stats = load_raw(config)
    sales, returns, log = clean(df, config)
    report = build_quality_report(sales, returns, log, ingest_stats, config)
    write_artifacts(sales, returns, report)
    return report


if __name__ == "__main__":
    print_summary(run(load_config()))