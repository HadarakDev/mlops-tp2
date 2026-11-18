"""Serve the trained model over HTTP.

Start it with:
    uvicorn src.serve:app --reload

Then:
    POST /predict   one wine, one prediction
    GET  /metrics   what Prometheus reads
    GET  /health    is anybody there

The model is read from models/model.pkl, so run the pipeline first:
    dvc repro
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pandera.errors import SchemaErrors
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from src.schema import wine_schema

MODEL = Path("models/model.pkl")

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]

# The schema of cycle 3 checks the raw dataset, which carries the grade. A wine
# sent to the API does not: same bounds, without that one column.
input_schema = wine_schema.remove_columns(["quality"])

# ---------------------------------------------------------------- metrics
# A Counter only goes up (calls served). A Histogram sorts values into buckets
# (how long a call took).

REQUESTS = Counter("wine_requests", "Calls served")
LATENCY = Histogram(
    "wine_request_duration_seconds", "Time spent answering a call",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
PREDICTIONS = Counter("wine_predictions", "Predictions served", ["label"])


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


@app.post("/predict")
def predict(wine: Wine):
    start = time.perf_counter()

    row = {name: getattr(wine, name.replace(" ", "_")) for name in FEATURES}
    df = pd.DataFrame([row])[FEATURES]

    try:
        input_schema.validate(df, lazy=True)
    except SchemaErrors as errors:
        cases = errors.failure_cases[["column", "check", "failure_case"]]
        raise HTTPException(status_code=422, detail=cases.to_dict("records"))

    prediction = int(model.predict(df)[0])
    label = "good" if prediction == 1 else "ordinary"

    REQUESTS.inc()
    PREDICTIONS.labels(label).inc()
    LATENCY.observe(time.perf_counter() - start)

    return {"good": prediction, "label": label}


@app.get("/health")
def health():
    return {"status": "ok"}
