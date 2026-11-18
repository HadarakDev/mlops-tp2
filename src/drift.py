#!/usr/bin/env python
"""Judge the wines arriving today against the wines the model learned on.

    python src/drift.py                   judge data/current.csv, write the report
    python src/drift.py --max-share 0.2   exit 1 if too many columns have drifted

Reference: data/prepared/train.csv -- what the model learned on.
Current:   data/current.csv        -- whatever tools/harvest.py last wrote.

The threshold is the usual one for the PSI: above 0.2, something moved.
Evidently applies it to every column, and writes a report a human can read.
"""

print("loading libraries...", flush=True)

import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE = Path("data/prepared/train.csv")
CURRENT = Path("data/current.csv")
REPORT = Path("reports/drift.html")

# Below 0.1 nothing moved, above 0.2 something did.
PSI_THRESHOLD = 0.2

# The label is not a feature: the wines arriving today do not carry it.
TARGET = "good"


def load(path):
    if not path.exists():
        sys.exit(
            f"{path} is missing.\n"
            "Run the pipeline first:        dvc repro\n"
            "then write a batch of wines:   python tools/harvest.py"
        )
    return pd.read_csv(path).drop(columns=[TARGET], errors="ignore")


def verdict(snapshot):
    """Pull the numbers out of the report: how many columns drifted, and which."""
    count, share, drifted = 0, 0.0, []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        kind = config["type"].rsplit(":", 1)[-1]
        if kind == "DriftedColumnsCount":
            count, share = metric["value"]["count"], metric["value"]["share"]
        elif kind == "ValueDrift" and metric["value"] > config["threshold"]:
            drifted.append((config["column"], metric["value"]))
    return int(count), share, sorted(drifted, key=lambda pair: -pair[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-share",
        type=float,
        default=None,
        help="exit 1 if more than this share of the columns have drifted",
    )
    args = parser.parse_args()

    reference, current = load(REFERENCE), load(CURRENT)
    columns = list(reference.columns)
    definition = DataDefinition(numerical_columns=columns)

    snapshot = Report(
        metrics=[DataDriftPreset(method="psi", threshold=PSI_THRESHOLD)]
    ).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT))

    count, share, drifted = verdict(snapshot)

    print(f"reference: {REFERENCE} ({len(reference)} wines)")
    print(f"current:   {CURRENT} ({len(current)} wines)")
    print(f"\n{count} of {len(columns)} columns have drifted ({100 * share:.0f}%)")
    for column, psi in drifted:
        print(f"  {column:22s} {psi:.3f}")
    print(f"\nreport: {REPORT}")

    if args.max_share is not None and share > args.max_share:
        sys.exit(
            f"\ndrift: {100 * share:.0f}% of the columns moved, "
            f"more than the {100 * args.max_share:.0f}% allowed.\n"
            "The wines arriving are no longer the wines the model learned on."
        )


if __name__ == "__main__":
    main()
