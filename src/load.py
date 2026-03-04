from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _executemany(engine: Engine, sql: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(text(sql), rows)
    return len(rows)


def _delete_standings_snapshot_scopes(engine: Engine, scopes: list[tuple[int, int]]) -> None:
    if not scopes:
        return
    with engine.begin() as conn:
        for competition_id, season in scopes:
            conn.execute(
                text(
                    """
                    DELETE FROM fact_standings_snapshot
                    WHERE competition_id = :competition_id
                      AND season = :season
                    """
                ),
                {"competition_id": competition_id, "season": season},
            )


def cleanup_legacy_fact_rows_for_csv(engine: Engine) -> dict[str, int]:
    with engine.begin() as conn:
        deleted_player_stats = conn.execute(
            text(
                """
                DELETE FROM fact_player_match_stats
                WHERE match_id IN (
                    SELECT match_id
                    FROM fact_match
                    WHERE season IS NULL
                )
                """
            )
        ).rowcount
        deleted_matches = conn.execute(
            text(
                """
                DELETE FROM fact_match
                WHERE season IS NULL
                """
            )
        ).rowcount
    return {
        "deleted_player_match_stats": int(deleted_player_stats or 0),
        "deleted_matches": int(deleted_matches or 0),
    }


def _fetch_existing_team_ids(engine: Engine) -> dict[str, int]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT team_id, team_name FROM dim_team")).mappings().all()
    return {
        " ".join(str(row["team_name"]).strip().split()): int(row["team_id"])
        for row in rows
        if row.get("team_name") is not None
    }


def _remap_team_ids_to_existing(engine: Engine, data: dict[str, Any]) -> None:
    existing_by_name = _fetch_existing_team_ids(engine)
    id_map: dict[int, int] = {}

    for row in data.get("dim_team", []):
        normalized_name = " ".join(str(row.get("team_name") or "").strip().split())
        existing_id = existing_by_name.get(normalized_name)
        incoming_id = row.get("team_id")
        if existing_id is None or incoming_id is None:
            continue
        if int(existing_id) != int(incoming_id):
            id_map[int(incoming_id)] = int(existing_id)
            row["team_id"] = int(existing_id)

    if not id_map:
        return

    for row in data.get("fact_match", []):
        if row.get("home_team_id") in id_map:
            row["home_team_id"] = id_map[row["home_team_id"]]
        if row.get("away_team_id") in id_map:
            row["away_team_id"] = id_map[row["away_team_id"]]

    for row in data.get("dim_player", []):
        if row.get("team_id") in id_map:
            row["team_id"] = id_map[row["team_id"]]

    for row in data.get("fact_standings_snapshot", []):
        if row.get("team_id") in id_map:
            row["team_id"] = id_map[row["team_id"]]

    unique_teams: dict[int, dict[str, Any]] = {}
    for row in data.get("dim_team", []):
        unique_teams[int(row["team_id"])] = row
    data["dim_team"] = list(unique_teams.values())


def load_all(engine: Engine, data: dict[str, Any]) -> int:
    _remap_team_ids_to_existing(engine, data)

    loaded = 0

    loaded += _executemany(
        engine,
        """
        INSERT INTO dim_date (date_id, year, month, day)
        VALUES (:date_id, :year, :month, :day)
        ON CONFLICT (date_id) DO NOTHING
    """,
        data["dim_date"],
    )

    for row in data["dim_team"]:
        row.setdefault("crest_url", None)
        row.setdefault("short_name", None)

    loaded += _executemany(
        engine,
        """
        INSERT INTO dim_team (team_id, team_name, country, crest_url, short_name)
        VALUES (:team_id, :team_name, :country, :crest_url, :short_name)
        ON CONFLICT (team_id) DO UPDATE
        SET team_name = EXCLUDED.team_name,
            country = EXCLUDED.country,
            crest_url = EXCLUDED.crest_url,
            short_name = EXCLUDED.short_name
    """,
        data["dim_team"],
    )

    loaded += _executemany(
        engine,
        """
        INSERT INTO dim_competition (competition_id, competition_name, country)
        VALUES (:competition_id, :competition_name, :country)
        ON CONFLICT (competition_id) DO UPDATE
        SET competition_name = EXCLUDED.competition_name,
            country = EXCLUDED.country
    """,
        data["dim_competition"],
    )

    for row in data["dim_player"]:
        row.setdefault("photo_url", None)

    loaded += _executemany(
        engine,
        """
        INSERT INTO dim_player (player_id, full_name, position, nationality, birth_date, photo_url, team_id)
        VALUES (:player_id, :full_name, :position, :nationality, :birth_date, :photo_url, :team_id)
        ON CONFLICT (player_id) DO UPDATE
        SET full_name = EXCLUDED.full_name,
            position = EXCLUDED.position,
            nationality = EXCLUDED.nationality,
            birth_date = EXCLUDED.birth_date,
            photo_url = EXCLUDED.photo_url,
            team_id = EXCLUDED.team_id
    """,
        data["dim_player"],
    )

    for row in data["fact_match"]:
        row.setdefault("status", None)
        row.setdefault("matchday", None)
        row.setdefault("stage", None)
        row.setdefault("group_name", None)
        row.setdefault("kickoff_utc", None)
        row.setdefault("season", None)

    loaded += _executemany(
        engine,
        """
        INSERT INTO fact_match (
            match_id,
            date_id,
            competition_id,
            home_team_id,
            away_team_id,
            status,
            matchday,
            stage,
            group_name,
            kickoff_utc,
            season,
            home_score,
            away_score
        )
        VALUES (
            :match_id,
            :date_id,
            :competition_id,
            :home_team_id,
            :away_team_id,
            :status,
            :matchday,
            :stage,
            :group_name,
            :kickoff_utc,
            :season,
            :home_score,
            :away_score
        )
        ON CONFLICT (match_id) DO UPDATE
        SET date_id = EXCLUDED.date_id,
            competition_id = EXCLUDED.competition_id,
            home_team_id = EXCLUDED.home_team_id,
            away_team_id = EXCLUDED.away_team_id,
            status = EXCLUDED.status,
            matchday = EXCLUDED.matchday,
            stage = EXCLUDED.stage,
            group_name = EXCLUDED.group_name,
            kickoff_utc = EXCLUDED.kickoff_utc,
            season = EXCLUDED.season,
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score
    """,
        data["fact_match"],
    )

    loaded += _executemany(
        engine,
        """
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
    """,
        data["fact_player_match_stats"],
    )

    loaded += _executemany(
        engine,
        """
        INSERT INTO fact_standings_snapshot (
            competition_id,
            season,
            matchday,
            team_id,
            position,
            points,
            played_games,
            won,
            draw,
            lost,
            goals_for,
            goals_against,
            goal_difference,
            snapshot_ts
        )
        VALUES (
            :competition_id,
            :season,
            :matchday,
            :team_id,
            :position,
            :points,
            :played_games,
            :won,
            :draw,
            :lost,
            :goals_for,
            :goals_against,
            :goal_difference,
            :snapshot_ts
        )
        ON CONFLICT (competition_id, season, matchday, team_id) DO UPDATE
        SET position = EXCLUDED.position,
            points = EXCLUDED.points,
            played_games = EXCLUDED.played_games,
            won = EXCLUDED.won,
            draw = EXCLUDED.draw,
            lost = EXCLUDED.lost,
            goals_for = EXCLUDED.goals_for,
            goals_against = EXCLUDED.goals_against,
            goal_difference = EXCLUDED.goal_difference,
            snapshot_ts = EXCLUDED.snapshot_ts
    """,
        data.get("fact_standings_snapshot", []),
    )

    return loaded


def load_standings_snapshot(
    engine: Engine,
    rows: list[dict[str, Any]],
    scopes: list[tuple[int, int]] | None = None,
) -> int:
    _delete_standings_snapshot_scopes(engine, scopes or [])
    return _executemany(
        engine,
        """
        INSERT INTO fact_standings_snapshot (
            competition_id,
            season,
            matchday,
            team_id,
            position,
            points,
            played_games,
            won,
            draw,
            lost,
            goals_for,
            goals_against,
            goal_difference,
            snapshot_ts
        )
        VALUES (
            :competition_id,
            :season,
            :matchday,
            :team_id,
            :position,
            :points,
            :played_games,
            :won,
            :draw,
            :lost,
            :goals_for,
            :goals_against,
            :goal_difference,
            :snapshot_ts
        )
        ON CONFLICT (competition_id, season, matchday, team_id) DO UPDATE
        SET position = EXCLUDED.position,
            points = EXCLUDED.points,
            played_games = EXCLUDED.played_games,
            won = EXCLUDED.won,
            draw = EXCLUDED.draw,
            lost = EXCLUDED.lost,
            goals_for = EXCLUDED.goals_for,
            goals_against = EXCLUDED.goals_against,
            goal_difference = EXCLUDED.goal_difference,
            snapshot_ts = EXCLUDED.snapshot_ts
    """,
        rows,
    )
