"""Pipeline orchestrator: ingest -> clean -> features -> models -> report.

Run a single stage or the full chain:

    python -m src.pipeline --stage clean
    python -m src.pipeline --stage all
    python -m src.pipeline --stage all --config config/client_acme.yaml

Stages are cumulative when running 'all': each reads the artifacts written by
the previous one from data/processed/.
"""
from __future__ import annotations

import argparse
import time

from src.ingest import load_config, load_raw
from src.clean import run as run_clean, print_summary
from src.features import run as run_features
from src.models import run as run_models
from src.report import run as run_report

STAGES = ("ingest", "clean", "features", "models", "report", "all")


def main() -> int:
    parser = argparse.ArgumentParser(description="BI automation pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    stage = args.stage
    config = load_config(args.config)
    currency = config["business_rules"]["currency"]
    started = time.perf_counter()

    if stage == "ingest":
        _, stats = load_raw(config)
        print(f"Rows read : {stats['rows_read']:,}")
        print(f"Date range: {stats['date_min'][:10]} -> {stats['date_max'][:10]}")
        return 0

    if stage in ("clean", "all"):
        print_summary(run_clean(config))
        if stage == "clean":
            return 0

    if stage in ("features", "all"):
        s = run_features(config)
        print(f"Features  : {s['train_rows']:,} train rows across "
              f"{s['train_snapshots']} snapshots | "
              f"{s['scoring_rows']:,} customers scored | "
              f"{s['trading_days']:,} trading days")
        if stage == "features":
            return 0

    if stage in ("models", "all"):
        r = run_models(config)
        val = r["model"]["validation"]
        risk = r["risk"]
        print(f"Model     : AUC {val['roc_auc']:.3f} | "
              f"lift @ top decile {val['lift_at_decile']}x")
        print(f"At risk   : {risk['at_risk_customers']:,} customers carrying "
              f"{currency} {risk['revenue_at_risk_ttm']:,.0f} TTM revenue "
              f"({risk['share_of_ttm_revenue_pct']}%)")
        print(f"Anomalies : {r['anomalies']['n_detected']} flagged")
        if stage == "models":
            return 0

    if stage in ("report", "all"):
        rep = run_report(config)
        print(f"Report    : {rep['period']} | {rep['words']} words | "
              f"{rep['model']}"
              f"{' | concentration flagged' if rep['concentration_flagged'] else ''}")
        print(f"Written   : {rep['html_path']}")
        if rep["note"]:
            print(f"Note      : {rep['note']}")
        if stage == "report":
            return 0

    print(f"\nPipeline completed in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())