# TP 2 — Pipelines, validation, CI, surveillance

Trois scripts, un jeu de données sous DVC, et aucun pipeline : c'est à vous de
l'écrire.

    git clone <ce depot> tp2 && cd tp2
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ./get-data.sh

Les scripts, dans l'ordre où ils doivent tourner :

| Script | Lit | Écrit |
|---|---|---|
| `src/prepare.py` | `data/raw/winequality-red.csv` | `data/prepared/{train,test}.csv` |
| `src/train.py` | `data/prepared/train.csv` | `models/model.pkl` |
| `src/evaluate.py` | le modèle et `data/prepared/test.csv` | `metrics.json` |

Et, pour le dernier cycle, le modèle servi plutôt qu'entraîné :

| Fichier | Rôle |
|---|---|
| `src/serve.py` | une API qui charge `models/model.pkl` et répond sur `/predict` |
| `tools/traffic.py` | envoie des vins à l'API, normaux ou issus d'une année plus chaude |
| `monitoring/` | Prometheus et Grafana, à lancer seulement si vous avez Docker |

`./get-data.sh` remet le CSV là où DVC l'attend : dans un vrai projet ce serait
un `dvc pull`, mais le stockage déclaré ici est un dossier personnel que personne
d'autre ne peut lire — d'où la copie que Git transporte dans `data/source/`.

Les réglages sont dans `params.yaml`. `src/schema.py` est un schéma Pandera vide
— il accepte tout, donc il ne protège de rien.

En cas de blocage : `./restore.sh N` ramène le dépôt à l'état attendu en fin de
cycle N, après avoir mis votre travail de côté sur une branche `mine-cN`.
Il rejoue le pipeline chez vous : vos empreintes sont les vôtres, pas les miennes.
