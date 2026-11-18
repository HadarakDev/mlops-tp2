"""Serve the trained model over HTTP, and expose what it is doing.

Start it with:
    uvicorn src.serve:app --reload

Then:
    POST /predict   one wine, one prediction
    GET  /metrics   what Prometheus reads

The model is read from models/model.pkl, so run the pipeline first:
    dvc repro
"""

import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

MODEL = Path("models/model.pkl")
REFERENCE = Path("data/prepared/train.csv")

# How many predictions we keep in memory to measure drift, and how often we
# recompute it. A real service would do this in a nightly job; keeping it here
# makes the drift visible while you watch.
WINDOW = 300
EVERY = 50
PSI_ALERT = 0.2

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
# The three columns we watch for drift -- the ones a bad harvest would move.
WATCHED = ["alcohol", "volatile acidity", "density"]

# ---------------------------------------------------------------- metrics
# A Counter only goes up (calls served). A Histogram sorts values into buckets
# (how long a call took). A Gauge goes up and down (how far the data has moved).

##### cycle 5, etape 2 : declarez ici le compteur d'appels, l'histogramme de
##### latence et le compteur de predictions par label


##### cycle 5, etape 3 : declarez ici la jauge de derive, avec un label "column"


app = FastAPI(title="wine-api")
app.mount("/metrics", make_asgi_app())


class Wine(BaseModel):
    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float

    model_config = {"populate_by_name": True}


def load_model():
    if not MODEL.exists():
        raise RuntimeError(
            f"{MODEL} is missing: the pipeline has not produced a model yet.\n"
            "Run it first:    dvc repro"
        )
    return joblib.load(MODEL)


model = load_model()
reference = pd.read_csv(REFERENCE)
seen = deque(maxlen=WINDOW)  # the last WINDOW wines we were asked about


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 5) -> float:
    """Population Stability Index between a reference sample and a live one.

    Both samples are cut along the quantiles of the reference. The further the
    live shares are from the reference shares, the higher the index.
    Rule of thumb: below 0.1 nothing moved, above 0.2 something did.

    Five slices, not ten: on a window of 300 wines, ten slices are thin enough
    that sampling alone pushes the index to 0.18 -- almost the alert threshold,
    with nothing having moved at all.
    """
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def refresh_drift():
    """Compare the wines we have just seen to the ones we trained on."""
    if len(seen) < WINDOW:
        return
    live = pd.DataFrame(seen)
    for column in WATCHED:
        value = psi(reference[column].to_numpy(), live[column].to_numpy())
        ##### cycle 5, etape 3 : publiez ici la valeur dans la jauge


@app.post("/predict")
def predict(wine: Wine):
    start = time.perf_counter()

    row = {name: getattr(wine, name.replace(" ", "_")) for name in FEATURES}
    prediction = int(model.predict(pd.DataFrame([row])[FEATURES])[0])
    label = "good" if prediction == 1 else "ordinary"

    seen.append(row)
    if len(seen) % EVERY == 0:
        refresh_drift()

    ##### cycle 5, etape 2 : comptez l'appel, la prediction, et observez la duree
    ##### (la duree, c'est time.perf_counter() - start)

    return {"good": prediction, "label": label}


@app.get("/health")
def health():
    return {"status": "ok", "seen": len(seen), "window": WINDOW}
