from typing import Dict, Any, List
from sqlalchemy import text
from sqlalchemy.engine import Engine

def _executemany(engine: Engine, sql: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(text(sql), rows)
    return len(rows)

def load_all(engine: Engine, data: Dict[str, Any]) -> int:
    loaded = 0

    loaded += _executemany(engine, """
        INSERT INTO dim_date (date_id, year, month, day)
        VALUES (:date_id, :year, :month, :day)
        ON CONFLICT (date_id) DO NOTHING
    """, data["dim_date"])

    loaded += _executemany(engine, """
        INSERT INTO dim_team (team_id, team_name, country)
        VALUES (:team_id, :team_name, :country)
        ON CONFLICT (team_id) DO UPDATE
        SET team_name = EXCLUDED.team_name,
            country = EXCLUDED.country
    """, data["dim_team"])

    loaded += _executemany(engine, """
        INSERT INTO dim_competition (competition_id, competition_name, country)
        VALUES (:competition_id, :competition_name, :country)
        ON CONFLICT (competition_id) DO UPDATE
        SET competition_name = EXCLUDED.competition_name,
            country = EXCLUDED.country
    """, data["dim_competition"])

    loaded += _executemany(engine, """
        INSERT INTO dim_player (player_id, full_name, position, nationality, birth_date, team_id)
        VALUES (:player_id, :full_name, :position, :nationality, :birth_date, :team_id)
        ON CONFLICT (player_id) DO UPDATE
        SET full_name = EXCLUDED.full_name,
            position = EXCLUDED.position,
            nationality = EXCLUDED.nationality,
            birth_date = EXCLUDED.birth_date,
            team_id = EXCLUDED.team_id
    """, data["dim_player"])

    loaded += _executemany(engine, """
        INSERT INTO fact_match (match_id, date_id, competition_id, home_team_id, away_team_id, home_score, away_score)
        VALUES (:match_id, :date_id, :competition_id, :home_team_id, :away_team_id, :home_score, :away_score)
        ON CONFLICT (match_id) DO UPDATE
        SET date_id = EXCLUDED.date_id,
            competition_id = EXCLUDED.competition_id,
            home_team_id = EXCLUDED.home_team_id,
            away_team_id = EXCLUDED.away_team_id,
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score
    """, data["fact_match"])

    loaded += _executemany(engine, """
        INSERT INTO fact_player_match_stats
        (match_id, player_id, minutes, goals, assists, shots, passes, pass_accuracy)
        VALUES (:match_id, :player_id, :minutes, :goals, :assists, :shots, :passes, :pass_accuracy)
        ON CONFLICT (match_id, player_id) DO UPDATE
        SET minutes = EXCLUDED.minutes,
            goals = EXCLUDED.goals,
            assists = EXCLUDED.assists,
            shots = EXCLUDED.shots,
            passes = EXCLUDED.passes,
            pass_accuracy = EXCLUDED.pass_accuracy
    """, data["fact_player_match_stats"])

    return loaded