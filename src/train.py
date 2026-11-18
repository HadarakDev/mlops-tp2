"""Train a RandomForest on the prepared training set.

Reads  data/prepared/train.csv   (produced by src/prepare.py)
Writes models/model.pkl
Logs the run to MLflow.

Parameters come from params.yaml, so that DVC can see them change.
"""

print("loading libraries...", flush=True)

import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

TRAIN = Path("data/prepared/train.csv")
MODEL = Path("models/model.pkl")


def load_params():
    return yaml.safe_load(open("params.yaml"))["train"]


def main():
    if not TRAIN.exists():
        sys.exit(
            f"{TRAIN} is missing: the training set has not been prepared.\n"
            "Run the preparation step first:\n"
            "    python src/prepare.py"
        )

    params = load_params()
    df = pd.read_csv(TRAIN)
    X = df.drop(columns=["good"])
    y = df["good"]

    print(f"{len(X)} rows, training {params['n_estimators']} trees "
          f"(max_depth={params['max_depth']})...", flush=True)

    model = RandomForestClassifier(random_state=42, **params)

    mlflow.set_experiment("wine-quality")
    with mlflow.start_run():
        mlflow.log_params(params)
        model.fit(X, y)
        # MLflow 3 serialises sklearn models with skops, which refuses
        # tree-based models unless their node type is declared trusted.
        mlflow.sklearn.log_model(model, name="model",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"saved {MODEL}")


if __name__ == "__main__":
    main()
