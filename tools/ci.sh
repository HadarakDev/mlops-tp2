#!/usr/bin/env bash
#
# The same steps as .github/workflows/ci.yml, run on your own machine.
# For anyone without a GitHub account -- or to check before pushing.
#
set -e
cd "$(dirname "$0")/.."

echo "== 1/3  validate the raw data"
python src/validate.py --path data/source/winequality-red.csv

echo "== 2/3  run the pipeline"
dvc repro --force

echo "== 3/3  show the metrics"
dvc metrics show

echo "CI: green"
