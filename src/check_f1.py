"""Cycle 4: fail the CI when the model is too weak.

    python src/check_f1.py

Reads metrics.json, written by the evaluate stage. A CI step that exits with a
non-zero code turns the whole job red.
"""
import json
import sys

THRESHOLD = 0.75   # the TP model scores about 0.76

with open("metrics.json") as f:
    f1 = json.load(f)["f1"]

print(f"f1 = {f1}  (threshold {THRESHOLD})")

sys.exit(0 if f1 >= THRESHOLD else 1)
