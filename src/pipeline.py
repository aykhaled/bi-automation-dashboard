"""Pipeline orchestrator: ingest -> clean -> features -> models -> report."""
from __future__ import annotations

import argparse
import sys

from src.ingest import load_config, load_raw
from src.clean import run as run_clean, print_summary

STAGES = ("ingest", "clean", "features", "models", "report", "all")


def main() -> int:
    parser = argparse.ArgumentParser(description="BI automation pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stage == "ingest":
        _, stats = load_raw(config)
        print(f"Rows read: {stats['rows_read']:,}")
        return 0

    if args.stage in ("clean", "all"):
        print_summary(run_clean(config))
        if args.stage == "clean":
            return 0

    print(f"Stage '{args.stage}' not implemented yet.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())