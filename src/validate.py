"""Validate a wine CSV against src/schema.py, and say so in one line.

Used by the CI, which has no access to the DVC remote: it validates
data/source/, the only copy of the dataset that Git itself carries.

    python src/validate.py                       # data/raw/winequality-red.csv
    python src/validate.py --path data/source/winequality-red.csv
"""

print("loading libraries...", flush=True)

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaErrors

from schema import wine_schema


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/raw/winequality-red.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        sys.exit(f"{path} is missing.")

    df = pd.read_csv(path, sep=";")
    try:
        wine_schema.validate(df, lazy=True)
    except SchemaErrors as errors:
        failures = errors.failure_cases[["column", "check", "failure_case"]]
        print(f"INVALID: {path} -- {len(failures)} failing values", file=sys.stderr)
        print(failures.to_string(index=False), file=sys.stderr)
        sys.exit(1)

    print(f"VALID: {path} -- {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
