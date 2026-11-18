#!/usr/bin/env python
"""Send wines to the running API, so that its metrics have something to show.

    python tools/traffic.py              400 wines from the test set
    python tools/traffic.py --n 500      more of them
    python tools/traffic.py --drift      wines from a warmer harvest

--drift does not corrupt anything: the wines stay plausible. They are simply
not the wines the model was trained on, which is exactly what happens to a
model left in production for a year.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

TEST = Path("data/prepared/test.csv")
URL = "http://127.0.0.1:8000/predict"

# A warmer year: more alcohol, slightly less acidity, lighter wines.
DRIFT = {"alcohol": +0.7, "volatile acidity": -0.15, "density": -0.0014}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    if not TEST.exists():
        sys.exit(f"{TEST} is missing. Run the pipeline first:    dvc repro")

    df = pd.read_csv(TEST).drop(columns=["good"])
    wines = df.sample(n=args.n, replace=True, random_state=0).reset_index(drop=True)

    if args.drift:
        for column, shift in DRIFT.items():
            wines[column] = wines[column] + shift
        print(f"sending {args.n} wines from a warmer harvest")
    else:
        print(f"sending {args.n} wines")

    good = 0
    for _, wine in wines.iterrows():
        try:
            answer = requests.post(args.url, json=wine.to_dict(), timeout=5)
            answer.raise_for_status()
        except requests.exceptions.ConnectionError:
            sys.exit(
                f"nothing is listening on {args.url}.\n"
                "Start the API first:    uvicorn src.serve:app"
            )
        good += answer.json()["good"]

    print(f"{good} of {args.n} predicted good ({100 * good / args.n:.0f}%)")
    print("now reload http://127.0.0.1:8000/metrics")


if __name__ == "__main__":
    main()
