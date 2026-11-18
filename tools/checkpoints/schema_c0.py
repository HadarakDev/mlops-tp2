"""What we expect of the raw wine dataset, before anything is trained on it.

Almost complete: three bounds are missing, and strict is still False -- so an
unexpected column would slip through. Fill both in at cycle 3, then have
src/prepare.py check the data against it.
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Physico-chemical ranges, widened from the dataset's own min/max so that a
# plausible new batch passes, and a sensor failure does not.
wine_schema = DataFrameSchema(
    {
        "fixed acidity": Column(float, Check.in_range(3.0, 17.0)),
        "volatile acidity": Column(float, Check.in_range(0.0, 2.0)),
        "citric acid": Column(float, Check.in_range(0.0, 1.5)),
        "residual sugar": Column(float, Check.in_range(0.0, 20.0)),
        "chlorides": Column(float, Check.in_range(0.0, 1.0)),
        "free sulfur dioxide": Column(float, Check.in_range(0.0, 100.0)),
        "total sulfur dioxide": Column(float, Check.in_range(0.0, 300.0)),
        "density": Column(float, Check.in_range(0.98, 1.01)),
        "sulphates": Column(float, Check.in_range(0.0, 2.5)),

        ##### cycle 3, etape 1 : trois bornes manquent. regardez les min/max du
        ##### fichier, puis prenez-les plus larges -- un vin hors du lot doit
        ##### passer, un capteur en panne non.
        "pH": Column(float, Check.in_range(X, X)),
        "alcohol": Column(float, Check.in_range(X, X)),
        "quality": Column(int, Check.in_range(X, X)),
    },
    strict=False,   ##### cycle 3, etape 1 : a passer a True
    coerce=True,
)
