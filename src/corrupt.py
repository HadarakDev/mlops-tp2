"""Put impossible values into the raw dataset, the way a broken sensor would.

    python src/corrupt.py                        # pH only
    python src/corrupt.py pH alcohol density     # one value per column named
    python src/corrupt.py --so2                  # free SO2 above total SO2

Deterministic: every student corrupts the same row, with the same values.
Undo it with:  ./get-data.sh
"""

import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/winequality-red.csv")
ROW = 7            # the row a failing sensor wrote

# One impossible value per column, so that a corrupted file always breaks the
# schema's bounds rather than merely looking odd.
IMPOSSIBLE = {
    "pH": 42.0,                   # a pH of 42 does not exist
    "alcohol": 99.0,              # 99 % alcohol is not wine
    "density": 5.0,               # five times the density of water
    "fixed acidity": 500.0,
    "volatile acidity": 50.0,
    "citric acid": 50.0,
    "residual sugar": 500.0,
    "chlorides": 50.0,
    "free sulfur dioxide": 5000.0,
    "total sulfur dioxide": 9000.0,
    "sulphates": 50.0,
    "quality": 99,                # the scale stops at 10
}


FREE, TOTAL = "free sulfur dioxide", "total sulfur dioxide"


def break_so2(df):
    """Free SO2 just above total SO2 -- both values stay inside their own bounds.

    Column bounds cannot catch this one: each value is plausible on its own.
    Only a check that receives the whole table can.
    """
    before, total = df.loc[ROW, FREE], df.loc[ROW, TOTAL]
    df.loc[ROW, FREE] = total + 5.0
    print(f"corrupt: row {ROW}, {FREE} {before} -> {total + 5.0} (total is {total})")
    print("corrupt: both values are still within their bounds -- "
          "only a check on the whole table can see this one")


def main():
    args = sys.argv[1:]
    so2 = "--so2" in args
    columns = [a for a in args if a != "--so2"]
    if not columns and not so2:
        columns = ["pH"]
    if not RAW.exists():
        sys.exit(f"{RAW} is missing.")

    df = pd.read_csv(RAW, sep=";")
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        sys.exit(
            f"corrupt: no such column: {', '.join(unknown)}\n"
            f"available: {', '.join(df.columns)}"
        )

    for column in columns:
        value = IMPOSSIBLE.get(column, 999.0)
        before = df.loc[ROW, column]
        df.loc[ROW, column] = value
        print(f"corrupt: row {ROW}, {column} {before} -> {value}")

    if so2:
        break_so2(df)

    df.to_csv(RAW, sep=";", index=False)
    if columns:
        print("corrupt: the dataset is now out of range -- run the pipeline again")
    else:
        print("corrupt: the dataset still passes every bound -- run the pipeline again")


if __name__ == "__main__":
    main()
