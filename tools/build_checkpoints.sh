#!/usr/bin/env bash
#
# Teacher tool. Rebuilds the whole Git history of the reference repository and
# the five checkpoint-N tags from the files in the working tree.
#
#     PATH=/path/to/venv/bin:$PATH ./tools/build_checkpoints.sh
#
# It rewrites .git, so anything not in the working tree is lost.
#
set -euo pipefail
cd "$(dirname "$0")/.."

for tool in git dvc python; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done

export MLFLOW_DISABLE_AGENT_HINT=1

rm -rf .git .dvc .dvcignore dvc.yaml dvc.lock metrics.json mlflow.db mlartifacts
rm -rf data/prepared models/model.pkl data/raw/winequality-red.csv \
       data/raw/winequality-red.csv.dvc data/raw/.gitignore .github \
       data/.gitignore models/.gitignore

git init -q -b main
# .git vient d'etre supprime : le remote aussi. On le remet tout de suite,
# sinon le push d'apres se plaint que "origin" n'existe pas.
git remote add origin https://github.com/HadarakDev/mlops-tp2.git
git config user.name "Nicolas Roche"
git config user.email "nicolasroche.contact@gmail.com"

# ---- the state a student clones: the TP 1 is behind them, the pipeline is not
#      written yet. DVC is initialised, the remote is declared, the dataset is
#      tracked -- src/schema.py is an empty schema, and there is no dvc.yaml.
#      The two scripts the cycles rewrite are taken from tools/checkpoints, so
#      that a rebuild does not depend on what the working tree happens to hold.
cp tools/checkpoints/prepare_c0.py src/prepare.py
cp tools/checkpoints/schema_c0.py src/schema.py
cp tools/checkpoints/serve_c0.py src/serve.py
cp tools/checkpoints/train_c0.py src/train.py       # the two settings in the code
cp tools/checkpoints/params_c0.yaml params.yaml     # and no train: section yet
cp tools/checkpoints/dvc_c0.yaml dvc.yaml           # the shape of a stage, nothing more
cp data/source/winequality-red.csv data/raw/
dvc init -q
dvc remote add -d -q store '~/dvc-store'
dvc add -q data/raw/winequality-red.csv
git add -A
git commit -q -m "Three scripts, a dataset under DVC, and no pipeline yet"

git checkout -q -b reference

# ---- cycle 1: the pipeline itself, and the settings pulled out of the code
cp tools/checkpoints/train_c1.py src/train.py
cp tools/checkpoints/params_c1.yaml params.yaml
cp tools/checkpoints/dvc_c2.yaml dvc.yaml      # prepare + train + evaluate
dvc repro -q
git add -A
git commit -q -m "Declare the three stages in dvc.yaml"
git tag -f checkpoint-1

# ---- cycle 2: nothing new in the files, but a second run to compare against.
#      max_depth 4 -> 8 -> 4 leaves two commits with different metrics, so that
#      `dvc metrics diff` has something to say at the checkpoint.
sed -i '' 's/max_depth: 4/max_depth: 8/' params.yaml
dvc repro -q
git commit -q -am "Try a deeper forest"
sed -i '' 's/max_depth: 8/max_depth: 4/' params.yaml
dvc repro -q
git commit -q -am "Back to a depth of 4"
git tag -f checkpoint-2

# ---- cycle 3: the schema, and prepare.py checking the data against it
cp tools/checkpoints/schema_c3.py src/schema.py
cp tools/checkpoints/prepare_c3.py src/prepare.py
dvc repro -q
git commit -q -am "Validate the raw dataset before preparing it"
git tag -f checkpoint-3

# ---- cycle 4: the workflow that runs on every pull request
mkdir -p .github/workflows
cp tools/checkpoints/ci_c4.yml .github/workflows/ci.yml
git add -A
git commit -q -m "Run the validation and the pipeline on every pull request"
git tag -f checkpoint-4

# ---- cycle 5: the model served over HTTP, and instrumented
cp tools/checkpoints/serve_c5.py src/serve.py
git add -A
git commit -q -m "Count the calls, time them, and measure drift against the training set"
git tag -f checkpoint-5

# ---- back to the state a student clones
git checkout -q main
git branch -D reference >/dev/null
rm -rf dvc.lock metrics.json data/prepared models/model.pkl .github mlflow.db mlartifacts
git checkout -q -- dvc.yaml
cp data/source/winequality-red.csv data/raw/

git log --oneline --all --decorate
git status --short
echo "build: done"
