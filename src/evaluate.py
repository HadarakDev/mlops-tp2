"""Score the trained model on the test set and write metrics.json.

Reads  models/model.pkl and data/prepared/test.csv
Writes metrics.json
"""

print("loading libraries...", flush=True)

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL = Path("models/model.pkl")
TEST = Path("data/prepared/test.csv")
METRICS = Path("metrics.json")


def main():
    for path in (MODEL, TEST):
        if not path.exists():
            sys.exit(f"{path} is missing: run the steps before this one.")

    model = joblib.load(MODEL)
    df = pd.read_csv(TEST)
    X = df.drop(columns=["good"])
    y = df["good"]
    predicted = model.predict(X)

    metrics = {
        "f1": round(float(f1_score(y, predicted)), 4),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "test_rows": int(len(df)),
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"f1 = {metrics['f1']}  accuracy = {metrics['accuracy']}")
    print(f"wrote {METRICS}")


if __name__ == "__main__":
    main()
