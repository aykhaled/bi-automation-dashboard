"""Churn scoring and revenue anomaly detection.

Reporting stance: this model is used as a RANKING system, not a calibrated
probability system. The training panel spans ten seasonal positions with a
27-point churn spread (42.5% to 69.4%), so absolute probabilities drift with
the scoring date. Ranking is stable under that drift; expected values are not.
Revenue at risk is therefore reported as observed trailing-12-month revenue
carried by the top-decile cohort, not as sum(p * monetary).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features import FEATURE_COLUMNS, PROCESSED_DIR

CATEGORICAL = ["region"]
MODEL_FEATURES = FEATURE_COLUMNS + ["month_sin", "month_cos"]


def _prepare(df: pd.DataFrame, categories=None):
    """Build the design matrix. Seasonal encoding is cyclical so December and
    January are adjacent rather than 11 units apart."""
    month = df["snapshot_date"].dt.month
    X = df[FEATURE_COLUMNS].copy()
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)

    region = df["region"].astype("string").fillna("UNKNOWN")
    if categories is None:
        X["region"] = region.astype("category")
        categories = X["region"].cat.categories
    else:
        # Reuse training categories so unseen regions become NaN, which the
        # model handles natively, rather than silently shifting encodings.
        X["region"] = pd.Categorical(region, categories=categories)
    return X, categories


def _rank_metrics(y: np.ndarray, p: np.ndarray, decile: float) -> dict:
    k = max(1, int(round(len(p) * decile)))
    top = np.argsort(-p)[:k]
    base = float(y.mean())
    precision = float(y[top].mean())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "base_churn_rate": round(base, 4),
        "top_decile_n": k,
        "precision_at_decile": round(precision, 4),
        "recall_at_decile": round(float(y[top].sum() / y.sum()), 4),
        "lift_at_decile": round(precision / base, 3) if base else None,
        # Recorded for transparency, deliberately not used commercially.
        "mean_predicted_prob": round(float(p.mean()), 4),
        "actual_churn_rate": round(base, 4),
        "calibration_gap": round(float(p.mean()) - base, 4),
    }


def train_churn_model(train: pd.DataFrame, val: pd.DataFrame, config: dict):
    m = config["modeling"]
    decile = m["at_risk_decile"]

    X_train, categories = _prepare(train)
    X_val, _ = _prepare(val, categories)
    y_train = train["churned"].to_numpy()
    y_val = val["churned"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=m["max_iter"],
        learning_rate=m["learning_rate"],
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,  # internal random split would mix snapshots
        categorical_features="from_dtype",
        random_state=m["random_state"],
    )
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]

    metrics = {
        "train": _rank_metrics(y_train, p_train, decile),
        "validation": _rank_metrics(y_val, p_val, decile),
        "validation_snapshot": str(val["snapshot_date"].iloc[0].date()),
        "n_train_rows": len(train),
        "n_train_snapshots": int(train["snapshot_date"].nunique()),
    }

    perm = permutation_importance(
        model, X_val, y_val, scoring="roc_auc",
        n_repeats=5, random_state=m["random_state"], n_jobs=-1,
    )
    metrics["feature_importance"] = (
        pd.DataFrame({
            "feature": X_val.columns,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        })
        .sort_values("importance", ascending=False)
        .round(5)
        .to_dict("records")
    )
    return model, categories, metrics


def score_customers(model, categories, scoring: pd.DataFrame, config: dict):
    decile = config["modeling"]["at_risk_decile"]
    X, _ = _prepare(scoring, categories)

    out = scoring.copy()
    out["churn_probability"] = model.predict_proba(X)[:, 1]
    out["risk_rank"] = out["churn_probability"].rank(ascending=False, method="first").astype(int)

    threshold = out["churn_probability"].quantile(1 - decile)
    out["at_risk"] = out["churn_probability"] >= threshold

    at_risk = out[out["at_risk"]]
    summary = {
        "scoring_snapshot": str(out["snapshot_date"].iloc[0].date()),
        "customers_scored": len(out),
        "at_risk_customers": int(len(at_risk)),
        "probability_threshold": round(float(threshold), 4),
        # Headline figure: observed revenue, ranking-based.
        "revenue_at_risk_ttm": round(float(at_risk["revenue_365d"].sum()), 2),
        "revenue_at_risk_lifetime": round(float(at_risk["monetary"].sum()), 2),
        "ttm_revenue_all_customers": round(float(out["revenue_365d"].sum()), 2),
        "share_of_ttm_revenue_pct": round(
            float(at_risk["revenue_365d"].sum() / out["revenue_365d"].sum() * 100), 2
        ),
        "median_recency_days_at_risk": int(at_risk["recency_days"].median()),
    }

    cols = [
        "customer_id", "churn_probability", "risk_rank", "at_risk", "region",
        "recency_days", "frequency", "monetary", "revenue_365d", "revenue_90d",
        "avg_gap_days", "gap_ratio", "return_rate", "snapshot_date",
    ]
    return out[cols].sort_values("risk_rank"), summary


def detect_anomalies(daily: pd.DataFrame, config: dict):
    a = config["anomaly"]
    d = daily.sort_values("date").reset_index(drop=True).copy()

    # shift(1) excludes the current day from its own baseline; without it a
    # large day inflates the mean it is being compared against.
    roll = d["revenue"].shift(1).rolling(a["window_days"], min_periods=a["min_periods"])
    d["expected_revenue"] = roll.mean()
    d["baseline_std"] = roll.std()
    d["z_score"] = (d["revenue"] - d["expected_revenue"]) / d["baseline_std"]
    d["is_anomaly"] = d["z_score"].abs() > a["z_threshold"]

    anomalies = d[d["is_anomaly"].fillna(False)].copy()
    anomalies["direction"] = np.where(anomalies["z_score"] > 0, "spike", "drop")
    anomalies["deviation"] = anomalies["revenue"] - anomalies["expected_revenue"]

    cols = ["date", "revenue", "expected_revenue", "z_score", "direction", "deviation", "orders"]
    return d, anomalies[cols].sort_values("date")


def seasonality_table(train: pd.DataFrame) -> list[dict]:
    """The strongest business finding so far — churn is a post-Christmas problem."""
    s = (
        train.groupby("snapshot_date")
        .agg(customers=("churned", "size"), churn_rate=("churned", "mean"))
        .reset_index()
    )
    s["label_window_ends"] = s["snapshot_date"] + pd.Timedelta(days=90)
    s["churn_rate"] = (s["churn_rate"] * 100).round(1)
    s["snapshot_date"] = s["snapshot_date"].dt.strftime("%Y-%m-%d")
    s["label_window_ends"] = s["label_window_ends"].dt.strftime("%Y-%m-%d")
    return s.to_dict("records")


def run(config: dict) -> dict:
    train = pd.read_parquet(PROCESSED_DIR / "train_panel.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_panel.parquet")
    scoring = pd.read_parquet(PROCESSED_DIR / "scoring_frame.parquet")
    daily = pd.read_parquet(PROCESSED_DIR / "daily_series.parquet")

    model, categories, metrics = train_churn_model(train, val, config)
    scores, risk_summary = score_customers(model, categories, scoring, config)
    daily_baseline, anomalies = detect_anomalies(daily, config)

    scores.to_parquet(PROCESSED_DIR / "churn_scores.parquet", index=False)
    anomalies.to_parquet(PROCESSED_DIR / "anomalies.parquet", index=False)
    daily_baseline.to_parquet(PROCESSED_DIR / "daily_baseline.parquet", index=False)

    report = {
        "model": metrics,
        "risk": risk_summary,
        "seasonality": seasonality_table(train),
        "anomalies": {
            "n_detected": len(anomalies),
            "n_spikes": int((anomalies["direction"] == "spike").sum()),
            "n_drops": int((anomalies["direction"] == "drop").sum()),
            "trading_days_evaluated": int(daily_baseline["z_score"].notna().sum()),
        },
        "reporting_note": (
            "Revenue at risk is the observed trailing-12-month revenue of the "
            "top-decile cohort by churn probability. Not a probability-weighted "
            "expectation: seasonal base-rate variation makes absolute "
            "probabilities unreliable while preserving rank order."
        ),
    }
    with (PROCESSED_DIR / "model_metrics.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    from src.ingest import load_config

    cfg = load_config()
    r = run(cfg)
    cur = cfg["business_rules"]["currency"]
    tr, va, risk = r["model"]["train"], r["model"]["validation"], r["risk"]

    print("\n" + "=" * 62)
    print("CHURN MODEL")
    print("=" * 62)
    print(f"{'':<22}{'train':>12}{'validation':>14}")
    for key, label in [("roc_auc", "ROC AUC"),
                       ("average_precision", "Avg precision"),
                       ("base_churn_rate", "Base churn rate"),
                       ("precision_at_decile", "Precision @ top 10%"),
                       ("recall_at_decile", "Recall @ top 10%"),
                       ("lift_at_decile", "Lift @ top 10%")]:
        print(f"{label:<22}{tr[key]:>12.3f}{va[key]:>14.3f}")
    print(f"{'Calibration gap':<22}{tr['calibration_gap']:>12.3f}"
          f"{va['calibration_gap']:>14.3f}")

    print("\nTop features (permutation importance on validation):")
    for f in r["model"]["feature_importance"][:8]:
        print(f"  {f['feature']:<20} {f['importance']:>8.4f}  ±{f['std']:.4f}")

    print("\n" + "=" * 62)
    print("REVENUE AT RISK")
    print("=" * 62)
    print(f"Scoring snapshot   : {risk['scoring_snapshot']}")
    print(f"At-risk customers  : {risk['at_risk_customers']:,} "
          f"of {risk['customers_scored']:,}")
    print(f"TTM revenue at risk: {cur} {risk['revenue_at_risk_ttm']:,.2f} "
          f"({risk['share_of_ttm_revenue_pct']}% of TTM revenue)")
    print(f"Lifetime value     : {cur} {risk['revenue_at_risk_lifetime']:,.2f}")

    a = r["anomalies"]
    print(f"\nAnomalies: {a['n_detected']} "
          f"({a['n_spikes']} spikes, {a['n_drops']} drops) "
          f"over {a['trading_days_evaluated']} evaluated days\n")