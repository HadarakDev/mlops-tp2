"""Split the raw wine dataset into a training set and a test set.

Reads  data/raw/winequality-red.csv
Writes data/prepared/train.csv and data/prepared/test.csv

Parameters come from params.yaml, so that DVC can see them change.
"""

print("loading libraries...", flush=True)

import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/winequality-red.csv")
OUT = Path("data/prepared")


def load_params():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    return params["test_size"], params["random_state"]


def main():
    if not RAW.exists():
        sys.exit(
            f"{RAW} is missing.\n"
            "Put the dataset in place first:\n"
            "    cp data/source/winequality-red.csv data/raw/\n"
            "or, if DVC already tracks it:  dvc checkout"
        )

    test_size, random_state = load_params()
    df = pd.read_csv(RAW, sep=";")
    df["good"] = (df["quality"] >= 6).astype(int)
    df = df.drop(columns=["quality"])

    train, test = train_test_split(
        df, test_size=test_size, stratify=df["good"], random_state=random_state
    )

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"{len(df)} rows -> {len(train)} train, {len(test)} test")


if __name__ == "__main__":
    main()
