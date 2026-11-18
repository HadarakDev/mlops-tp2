#!/usr/bin/env bash
#
# ./get-data.sh   puts the dataset where DVC expects it, and records it.
#
# In a real project this would be a single `dvc pull`: the bytes would come
# down from the shared storage. Here the storage is a folder on one machine
# (~/dvc-store), which nobody else can read -- so the repository carries a
# second copy of the CSV in data/source/, and this script puts it back.
#
set -euo pipefail
cd "$(dirname "$0")"

SOURCE="data/source/winequality-red.csv"
TARGET="data/raw/winequality-red.csv"

command -v dvc >/dev/null || {
  echo "dvc is not on your PATH -- did you activate the environment?" >&2
  exit 1
}

mkdir -p data/raw
cp "$SOURCE" "$TARGET"
dvc commit -q -f

echo "$TARGET: $(($(wc -l <"$TARGET") - 1)) rows"
dvc status
