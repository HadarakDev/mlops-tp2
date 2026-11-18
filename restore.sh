#!/usr/bin/env bash
#
# ./restore.sh N   brings this repository back to the state expected at the end
#                  of cycle N (N = 1..5), after putting your own work aside on a
#                  branch named mine-cN.
#
# Safe to run twice, and safe to run on a clean repository.
#
set -euo pipefail

N="${1:-}"
case "$N" in
  1 | 2 | 3 | 4 | 5) ;;
  *)
    echo "usage: ./restore.sh <1|2|3|4|5>" >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")"
TAG="checkpoint-$N"
PYTHON="${PYTHON:-python}"
SOURCE_CSV="data/source/winequality-red.csv"
RAW_CSV="data/raw/winequality-red.csv"

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "restore: tag $TAG not found -- is this the course repository?" >&2
  exit 1
fi

# --------------------------------------------------------------- 1. your work
BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
BRANCH="${BRANCH:-main}"

git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "WIP before restore to $TAG"
  echo "restore: work in progress committed"
fi
if git rev-parse -q --verify HEAD >/dev/null; then
  git branch -f "mine-c$N" HEAD
  echo "restore: your work is kept on branch mine-c$N"
fi

# ---------------------------------------------------------------- 2. the code
git checkout -q -B "$BRANCH" "$TAG"
echo "restore: code back to $TAG (branch $BRANCH)"

# ---------------------------------------------------------------- 3. the data
# The raw dataset is tracked by DVC. data/source/ is the copy Git carries, so a
# restore never depends on the state of anyone's cache or remote.
mkdir -p data/raw
cp "$SOURCE_CSV" "$RAW_CSV"
if [ -d .dvc ]; then
  dvc add -q "$RAW_CSV"
  git checkout -q -- "$RAW_CSV.dvc" 2>/dev/null || true
  dvc checkout -q --force "$RAW_CSV.dvc" 2>/dev/null || true
fi
echo "restore: $RAW_CSV back to $(($(wc -l <"$RAW_CSV") - 1)) rows"

# ------------------------------------------------------------ 4. the pipeline
# Three stages, a few seconds: rebuilding is more honest than shipping outputs,
# and it leaves dvc.lock, metrics.json and one MLflow run in agreement.
if [ -f dvc.yaml ]; then
  rm -f mlflow.db
  dvc repro --force -q
  echo "restore: pipeline rebuilt -- $(${PYTHON} -c 'import json;print(json.load(open("metrics.json"))["f1"])' 2>/dev/null || echo "metrics.json missing")"
fi

echo "restore: done -- you are at $TAG"
