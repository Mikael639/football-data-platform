# Import manuel FBref (Étude Joueurs)

Dépose ton fichier ici :

- `data/study/fbref_input/player_match_manual.csv`

Puis génère les datasets d'étude :

- local : `python -m src.study_fbref` avec `FBREF_STUDY_SOURCE=manual_csv`
- docker : `docker compose run --rm -e FBREF_STUDY_SOURCE=manual_csv pipeline python -m src.study_fbref`

## Colonnes attendues (minimum)

Le script accepte un CSV "canonique" au grain `joueur x match`.

Colonnes minimales obligatoires :

- `player_name`
- `team_name`
- `date_id` (`YYYY-MM-DD`)
- `minutes`

Colonnes fortement recommandées (pour une meilleure étude) :

- `season_start` (ex: `2024` pour la saison `2024-2025`)
- `match_id`
- `player_id`
- `team_id`
- `player_key` (id texte stable, ex: `vinicius_junior|real_madrid`)
- `position` (`FW`, `MF`, `DF`, `GK`)
- `position_group` (`FWD`, `MID`, `DEF`, `GK`)
- `is_starting` (`1/0`, `true/false`)
- `goals`
- `assists`
- `shots`
- `passes`
- `pass_accuracy` (`0.81` ou `81%`)

Référence :

- `data/study/fbref_templates/player_match_manual_template.csv`
