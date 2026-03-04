# football-data-platform

Plateforme data football avec ingestion API, pipeline ETL, warehouse PostgreSQL, controles de qualite et dashboard Streamlit.

## Prerequisites

- Python 3.11
- Docker + Docker Compose
- `pip` (pour les outils locaux)
- Token `football-data.org` uniquement si tu utilises les modes live (`api`/`hybrid`)

## Quickstart (pour un ami)

1. Cloner le repo puis entrer dans le dossier.
2. Copier le fichier d environnement.
3. Choisir le mode de data (`csv`, `mock`, `api`, `hybrid`).
4. Lancer la stack.

```powershell
Copy-Item .env.example .env
pip install -r requirements.txt
.\start.ps1
```

URL dashboard: `http://localhost:9001`

## Choix du mode de donnees

- `DATA_MODE=mock`: demo locale sans API.
- `DATA_MODE=csv`: historique depuis `data/raw/*_cleaned.csv`.
- `DATA_MODE=api`: live uniquement via `football-data.org`.
- `DATA_MODE=hybrid`: historique CSV + live API (recommande si token dispo).

Variables importantes:

- `FOOTBALL_DATA_TOKEN`: requis pour `api` et `hybrid`.
- `LIVE_COMPETITION_CODES`: competitions live chargees.
- `INCREMENTAL=true`: limite la fenetre API via `INCREMENTAL_DAYS`.
- `DB_*` / `DATABASE_URL`: connexion PostgreSQL.

## Token football-data.org

Chaque personne doit utiliser son propre token.

1. Creer un compte sur `https://www.football-data.org`.
2. Recuperer le token API depuis le dashboard/profile du compte.
3. Le mettre dans `.env`:

```env
FOOTBALL_DATA_TOKEN=ton_token_ici
DATA_MODE=hybrid
```

Sans token:

- utiliser `DATA_MODE=csv` ou `DATA_MODE=mock`.

## Lancement manuel Docker (alternative a `start.ps1`)

```powershell
docker compose up -d postgres
make init
docker compose run --rm pipeline python -m src.run_pipeline
docker compose up -d dashboard
```

## Pages dashboard

- `Overview`: KPI globaux, classement, courbe de position, calendrier.
- `Team`: forme 5/10, split domicile/exterieur, calendrier, courbe.
- `Players`: effectif du club filtre.
- `Live Leagues`: lecture multi-ligues.
- `Europe`: suivi UEFA (classement, calendrier, phases).
- `Monitoring`: runs pipeline, volumes, qualite.

## Securite (important)

Projet adapte a un usage local/dev. Tel quel, ce n est pas un setup production internet.

Points OK:

- `.env` est ignore par git.
- Le token est lu via variables d environnement (pas hardcode dans le code).

Points a durcir avant exposition reseau:

- Changer les identifiants PostgreSQL par defaut (`football/football`).
- Ne pas exposer PostgreSQL publiquement (`5432`) hors local.
- Mettre une authentification devant Streamlit (reverse proxy + auth).
- Ajouter TLS si acces distant.
- Ne jamais partager ni commiter un vrai token.

## Tests et qualite

```powershell
pytest -q
pre-commit run --all-files
```

## Commandes utiles

```powershell
docker compose up -d
docker compose down
make migrate
make study-fbref
make study-fbref-docker
```
