"""Load raw transactional data and normalise it to a canonical schema.

This is the ONLY module aware of client-specific column names. Everything
downstream consumes the canonical schema defined in CANONICAL_COLUMNS.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

CANONICAL_COLUMNS = [
    "transaction_id", "date", "customer_id", "product_id",
    "product_name", "quantity", "unit_price", "region",
]

# Without these the pipeline cannot run at all.
REQUIRED = ["transaction_id", "date", "quantity", "unit_price"]


class SchemaError(Exception):
    """Raised when the configured column mapping does not match the source."""


def load_config(path: str | Path = "config/config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with path.open() as f:
        return yaml.safe_load(f)


def _validate_mapping(source_path: Path, mapping: dict) -> None:
    """Check the header before loading a million rows."""
    header = pd.read_csv(source_path, nrows=0).columns.tolist()

    missing = {
        canonical: raw
        for canonical, raw in mapping.items()
        if canonical in REQUIRED and raw not in header
    }
    if missing:
        raise SchemaError(
            "Column mapping does not match the source file.\n"
            f"  Missing (canonical -> configured name): {missing}\n"
            f"  Columns actually present: {header}\n"
            "  Fix the 'columns:' block in config/config.yaml."
        )


def _normalise_customer_id(s: pd.Series) -> pd.Series:
    """Nulls make this float64 on read; we want a clean nullable string."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64").astype("string")
    return s.astype("string").str.strip().replace({"": pd.NA})


def load_raw(config: dict) -> tuple[pd.DataFrame, dict]:
    """Return (canonical dataframe, ingest stats)."""
    source_path = Path(config["source"]["path"])
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source data not found: {source_path.resolve()}\n"
            "Download online_retail_II.csv into data/raw/ (see README)."
        )

    mapping = config["columns"]
    _validate_mapping(source_path, mapping)

    # Force identifier columns to string so 'C489449' and 489449 don't mix.
    dtype_overrides = {
        mapping[c]: "string"
        for c in ("transaction_id", "product_id")
        if c in mapping
    }

    df = pd.read_csv(source_path, dtype=dtype_overrides)
    rows_read = len(df)

    # Keep only mapped columns, then rename to canonical.
    present = {raw: canonical for canonical, raw in mapping.items() if raw in df.columns}
    df = df[list(present)].rename(columns=present)

    # Type coercion on the canonical schema.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_parse_failures = int(df["date"].isna().sum())

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    if "customer_id" in df.columns:
        df["customer_id"] = _normalise_customer_id(df["customer_id"])

    for col in ("product_name", "region"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    stats = {
        "source_file": str(source_path),
        "rows_read": rows_read,
        "columns_mapped": sorted(present.values()),
        "columns_missing_optional": sorted(set(CANONICAL_COLUMNS) - set(df.columns)),
        "date_parse_failures": date_parse_failures,
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "null_rate": {c: round(float(df[c].isna().mean()), 4) for c in df.columns},
    }
    return df, stats


if __name__ == "__main__":
    cfg = load_config()
    frame, ingest_stats = load_raw(cfg)

    print(f"\nRows read       : {ingest_stats['rows_read']:,}")
    print(f"Date range      : {ingest_stats['date_min']} -> {ingest_stats['date_max']}")
    print(f"Unparsed dates  : {ingest_stats['date_parse_failures']:,}")
    print(f"Missing optional: {ingest_stats['columns_missing_optional'] or 'none'}")
    print("\nNull rates:")
    for col, rate in ingest_stats["null_rate"].items():
        print(f"  {col:<16} {rate:>7.2%}")

    prefix = cfg["business_rules"]["cancellation_prefix"]
    cancels = frame["transaction_id"].str.startswith(prefix, na=False).sum()
    print(f"\nCancellation rows ('{prefix}' prefix): {cancels:,}")