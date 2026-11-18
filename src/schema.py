"""What we expect of the raw wine dataset, before anything is trained on it.

Checked by src/prepare.py: a run stops here rather than training on bad data.
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
        "pH": Column(float, Check.in_range(2.5, 4.5)),
        "sulphates": Column(float, Check.in_range(0.0, 2.5)),
        "alcohol": Column(float, Check.in_range(8.0, 16.0)),
        "quality": Column(int, Check.in_range(0, 10)),
    },
    strict=True,       # no unexpected column
    coerce=True,
)
