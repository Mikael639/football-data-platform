from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from sqlalchemy import text

from src.config import get_settings
from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger("backfill_openfootball_laliga")

OPENFOOTBALL_URL_TEMPLATE = "https://raw.githubusercontent.com/openfootball/football.json/master/{season_code}/es.1.json"
COMPETITION_ID = 2014


TEAM_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "Athletic Club": ("Athletic Club",),
    "CA Osasuna": ("CA Osasuna", "Osasuna"),
    "CD Leganes": ("CD Leganes", "Leganes"),
    "Club Atletico de Madrid": ("Club Atletico de Madrid", "Atletico Madrid"),
    "Cadiz CF": ("Cadiz CF", "Cadiz"),
    "Deportivo Alaves": ("Deportivo Alaves", "Alaves"),
    "Elche CF": ("Elche CF", "Elche"),
    "FC Barcelona": ("FC Barcelona", "Barcelona"),
    "Getafe CF": ("Getafe CF", "Getafe"),
    "Girona FC": ("Girona FC", "Girona"),
    "Granada CF": ("Granada CF", "Granada"),
    "Levante UD": ("Levante UD", "Levante"),
    "RC Celta de Vigo": ("RC Celta de Vigo", "Celta Vigo"),
    "RCD Espanyol de Barcelona": ("RCD Espanyol de Barcelona", "Espanyol"),
    "RCD Mallorca": ("RCD Mallorca", "Mallorca"),
    "Rayo Vallecano de Madrid": ("Rayo Vallecano de Madrid", "Rayo Vallecano"),
    "Real Betis Balompie": ("Real Betis Balompie", "Real Betis"),
    "Real Madrid CF": ("Real Madrid CF", "Real Madrid"),
    "Real Sociedad de Futbol": ("Real Sociedad de Futbol", "Real Sociedad"),
    "Real Valladolid CF": ("Real Valladolid CF", "Valladolid"),
    "SD Eibar": ("SD Eibar", "Eibar"),
    "SD Huesca": ("SD Huesca", "Huesca"),
    "Sevilla FC": ("Sevilla FC", "Sevilla"),
    "UD Almeria": ("UD Almeria", "Almeria"),
    "UD Las Palmas": ("UD Las Palmas", "Las Palmas"),
    "Valencia CF": ("Valencia CF", "Valencia"),
    "Villarreal CF": ("Villarreal CF", "Villarreal"),
}


def _normalize_team_name(value: str) -> str:
    text_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text_value = re.sub(r"[^a-zA-Z0-9 ]+", " ", text_value).lower()
    return " ".join(text_value.split())


NORMALIZED_TEAM_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    _normalize_team_name(key): value for key, value in TEAM_NAME_ALIASES.items()
}


def _season_code(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _load_dim_team_index() -> dict[str, list[int]]:
    settings = get_settings()
    engine = get_engine(settings=settings)
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT team_id, team_name FROM dim_team")).mappings().all()

    by_name: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        team_name = row.get("team_name")
        team_id = row.get("team_id")
        if team_name is None or team_id is None:
            continue
        by_name[_normalize_team_name(str(team_name))].append(int(team_id))

    for key in by_name:
        by_name[key] = sorted(set(by_name[key]))
    return dict(by_name)


def _resolve_team_ids(dim_team_index: dict[str, list[int]], season_teams: set[str]) -> dict[str, int]:
    team_ids: dict[str, int] = {}
    unresolved: list[str] = []

    for source_name in sorted(season_teams):
        normalized_source = _normalize_team_name(source_name)
        aliases = NORMALIZED_TEAM_NAME_ALIASES.get(normalized_source, (source_name,))
        chosen_id: int | None = None
        for candidate in aliases:
            normalized_candidate = _normalize_team_name(candidate)
            ids = dim_team_index.get(normalized_candidate, [])
            if ids:
                chosen_id = int(ids[0])
                break
        if chosen_id is None:
            unresolved.append(source_name)
        else:
            team_ids[source_name] = chosen_id

    if unresolved:
        raise RuntimeError(f"Unable to map team names to dim_team IDs: {', '.join(unresolved)}")
    return team_ids


@dataclass
class TeamStats:
    played_games: int = 0
    won: int = 0
    draw: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


def _parse_matchday(value: Any) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Invalid matchday label: {value!r}")
    return int(match.group(1))


def _snapshot_ts_from_date(date_value: str) -> datetime:
    parsed = datetime.strptime(date_value, "%Y-%m-%d")
    return datetime(parsed.year, parsed.month, parsed.day, 12, 0, 0, tzinfo=timezone.utc)


def _build_rows_for_season(*, start_year: int, dim_team_index: dict[str, list[int]]) -> list[dict[str, Any]]:
    season_code = _season_code(start_year)
    url = OPENFOOTBALL_URL_TEMPLATE.format(season_code=season_code)
    payload = requests.get(url, timeout=45).json()
    matches = payload.get("matches", [])
    if not matches:
        raise RuntimeError(f"No matches returned for season {season_code} from {url}")

    season_teams: set[str] = set()
    for match in matches:
        season_teams.add(str(match["team1"]))
        season_teams.add(str(match["team2"]))
    team_ids = _resolve_team_ids(dim_team_index, season_teams)

    by_matchday: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        score = (match.get("score") or {}).get("ft")
        if not isinstance(score, list) or len(score) != 2:
            continue
        home_score = int(score[0])
        away_score = int(score[1])
        entry = {
            "matchday": _parse_matchday(match.get("round")),
            "date": str(match["date"]),
            "home_team": str(match["team1"]),
            "away_team": str(match["team2"]),
            "home_score": home_score,
            "away_score": away_score,
        }
        by_matchday[int(entry["matchday"])].append(entry)

    stats = {team: TeamStats() for team in sorted(season_teams)}
    rows: list[dict[str, Any]] = []

    for matchday in sorted(by_matchday):
        day_matches = sorted(by_matchday[matchday], key=lambda row: (row["date"], row["home_team"], row["away_team"]))
        for match in day_matches:
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_score = int(match["home_score"])
            away_score = int(match["away_score"])

            home_stats = stats[home_team]
            away_stats = stats[away_team]

            home_stats.played_games += 1
            away_stats.played_games += 1
            home_stats.goals_for += home_score
            home_stats.goals_against += away_score
            away_stats.goals_for += away_score
            away_stats.goals_against += home_score

            if home_score > away_score:
                home_stats.won += 1
                away_stats.lost += 1
                home_stats.points += 3
            elif home_score < away_score:
                away_stats.won += 1
                home_stats.lost += 1
                away_stats.points += 3
            else:
                home_stats.draw += 1
                away_stats.draw += 1
                home_stats.points += 1
                away_stats.points += 1

        sorted_teams = sorted(
            season_teams,
            key=lambda team: (
                -stats[team].points,
                -stats[team].goal_difference,
                -stats[team].goals_for,
                team,
            ),
        )
        snapshot_ts = _snapshot_ts_from_date(day_matches[-1]["date"])
        for index, team in enumerate(sorted_teams, start=1):
            team_stats = stats[team]
            rows.append(
                {
                    "competition_id": COMPETITION_ID,
                    "season": int(start_year),
                    "matchday": int(matchday),
                    "team_id": int(team_ids[team]),
                    "position": int(index),
                    "points": int(team_stats.points),
                    "played_games": int(team_stats.played_games),
                    "won": int(team_stats.won),
                    "draw": int(team_stats.draw),
                    "lost": int(team_stats.lost),
                    "goals_for": int(team_stats.goals_for),
                    "goals_against": int(team_stats.goals_against),
                    "goal_difference": int(team_stats.goal_difference),
                    "snapshot_ts": snapshot_ts,
                }
            )

    logger.info(
        "Built standings rows from openfootball season=%s matchdays=%s rows=%s",
        season_code,
        len(by_matchday),
        len(rows),
    )
    return rows


def _write_rows(rows: list[dict[str, Any]], seasons: list[int]) -> int:
    if not rows:
        return 0
    settings = get_settings()
    engine = get_engine(settings=settings)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM fact_standings_snapshot
                WHERE competition_id = :competition_id
                  AND season = ANY(:seasons)
                """
            ),
            {"competition_id": COMPETITION_ID, "seasons": [int(value) for value in seasons]},
        )
        conn.execute(
            text(
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
                """
            ),
            rows,
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LaLiga standings snapshots from openfootball historical data.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024],
        help="Season start years, e.g. 2020 2021 2022.",
    )
    args = parser.parse_args()
    seasons = sorted(set(int(value) for value in args.seasons))

    dim_team_index = _load_dim_team_index()
    rows: list[dict[str, Any]] = []
    for season_start in seasons:
        rows.extend(_build_rows_for_season(start_year=season_start, dim_team_index=dim_team_index))
    loaded = _write_rows(rows, seasons)
    logger.info("Openfootball standings backfill completed seasons=%s rows=%s", seasons, loaded)


if __name__ == "__main__":
    main()
