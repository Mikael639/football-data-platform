from __future__ import annotations

from datetime import datetime
from typing import Any

Record = dict[str, Any]
Payload = dict[str, Any]
TableRows = list[Record]
TransformedData = dict[str, TableRows]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float_ratio(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        parsed = float(value)
        return max(0.0, min(parsed, 1.0))
    except Exception:
        return 0.0


def _build_dim_date(date_values: set[str]) -> TableRows:
    dim_dates: TableRows = []
    for date_value in sorted(date_values):
        parsed = datetime.strptime(date_value, "%Y-%m-%d")
        dim_dates.append(
            {
                "date_id": date_value,
                "year": parsed.year,
                "month": parsed.month,
                "day": parsed.day,
            }
        )
    return dim_dates


def _extract_mock_squad_players(payload: Payload, players: dict[int, Record]) -> None:
    for player in payload.get("squad_players", []):
        player_id = _safe_int(player.get("player_id"))
        if player_id == 0:
            continue

        players[player_id] = {
            "player_id": player_id,
            "full_name": player.get("full_name"),
            "position": player.get("position"),
            "nationality": player.get("nationality"),
            "birth_date": player.get("birth_date"),
            "photo_url": player.get("photo_url"),
            "team_id": _safe_int(player.get("team_id")),
        }


def _build_mock_competition_row(fixture: Record) -> Record:
    competition_id = _safe_int(fixture.get("competition_id"))
    return {
        "competition_id": competition_id,
        "competition_name": fixture.get("competition_name"),
        "country": None,
    }


def _update_mock_teams(home_team: Record, away_team: Record, teams: dict[int, Record]) -> None:
    for team in (home_team, away_team):
        team_id = _safe_int(team.get("team_id"))
        teams[team_id] = {
            "team_id": team_id,
            "team_name": team.get("team_name"),
            "country": team.get("country"),
        }


def _build_mock_match_row(fixture: Record, home_team: Record, away_team: Record) -> Record:
    score = fixture.get("score", {})
    return {
        "match_id": _safe_int(fixture.get("match_id")),
        "date_id": fixture.get("date"),
        "competition_id": _safe_int(fixture.get("competition_id")),
        "home_team_id": _safe_int(home_team.get("team_id")),
        "away_team_id": _safe_int(away_team.get("team_id")),
        "home_score": _safe_int(score.get("home")),
        "away_score": _safe_int(score.get("away")),
    }


def _build_mock_player_stats(
    fixture: Record,
    players: dict[int, Record],
) -> TableRows:
    player_match_stats: TableRows = []

    for player_stats in fixture.get("player_stats", []):
        player = player_stats.get("player", {})
        player_id = _safe_int(player.get("player_id"))

        players[player_id] = {
            "player_id": player_id,
            "full_name": player.get("full_name"),
            "position": player.get("position"),
            "nationality": player.get("nationality"),
            "birth_date": player.get("birth_date"),
            "photo_url": player.get("photo_url", None),
            "team_id": _safe_int(player.get("team_id")),
        }

        stats = player_stats.get("stats", {})
        minutes = _safe_int(stats.get("minutes"))
        if minutes > 130:
            raise ValueError(f"Invalid minutes value detected: {minutes}")

        player_match_stats.append(
            {
                "match_id": _safe_int(fixture.get("match_id")),
                "player_id": player_id,
                "minutes": minutes,
                "goals": _safe_int(stats.get("goals")),
                "assists": _safe_int(stats.get("assists")),
                "shots": _safe_int(stats.get("shots")),
                "passes": _safe_int(stats.get("passes")),
                "pass_accuracy": _safe_float_ratio(stats.get("pass_accuracy")),
            }
        )

    return player_match_stats


def transform(payload: Payload) -> TransformedData:
    fixtures = payload.get("fixtures", [])

    date_values: set[str] = set()
    teams: dict[int, Record] = {}
    competitions: dict[int, Record] = {}
    players: dict[int, Record] = {}
    matches: TableRows = []
    player_match_stats: TableRows = []

    _extract_mock_squad_players(payload, players)

    for fixture in fixtures:
        date_value = fixture.get("date")
        if date_value:
            date_values.add(date_value)

        competition = _build_mock_competition_row(fixture)
        competitions[competition["competition_id"]] = competition

        home_team = fixture.get("home_team", {})
        away_team = fixture.get("away_team", {})
        _update_mock_teams(home_team, away_team, teams)

        matches.append(_build_mock_match_row(fixture, home_team, away_team))
        player_match_stats.extend(_build_mock_player_stats(fixture, players))

    return {
        "dim_date": _build_dim_date(date_values),
        "dim_team": list(teams.values()),
        "dim_competition": list(competitions.values()),
        "dim_player": list(players.values()),
        "fact_match": matches,
        "fact_player_match_stats": player_match_stats,
    }


def _build_competition_rows(competition_meta: Record, competition_code: str) -> dict[int, Record]:
    competition_id = _safe_int(competition_meta.get("id"), 1)
    return {
        competition_id: {
            "competition_id": competition_id,
            "competition_name": competition_meta.get("name") or competition_code,
            "country": (competition_meta.get("area") or {}).get("name"),
        }
    }


def _build_team_row(team: Record) -> Record:
    return {
        "team_id": int(team["id"]),
        "team_name": team.get("name") or team.get("shortName"),
        "country": (team.get("area") or {}).get("name"),
    }


def _load_competition_teams(competition_teams: list[Record], teams: dict[int, Record]) -> None:
    for team in competition_teams:
        if team.get("id") is None:
            continue
        team_id = int(team["id"])
        teams[team_id] = _build_team_row(team)


def _load_players_from_squads(
    squads_by_team: list[Record],
    teams: dict[int, Record],
    players: dict[int, Record],
) -> None:
    for squad_entry in squads_by_team:
        team_meta = squad_entry.get("team", {}) or {}
        team_id_raw = team_meta.get("id")
        if team_id_raw is None:
            continue

        team_id = int(team_id_raw)
        teams.setdefault(
            team_id,
            {
                "team_id": team_id,
                "team_name": team_meta.get("name") or team_meta.get("shortName"),
                "country": (team_meta.get("area") or {}).get("name"),
            },
        )

        for player in squad_entry.get("squad", []) or []:
            if player.get("id") is None:
                continue
            player_id = int(player["id"])
            players[player_id] = {
                "player_id": player_id,
                "full_name": player.get("name"),
                "position": player.get("position"),
                "nationality": player.get("nationality"),
                "birth_date": player.get("dateOfBirth"),
                "photo_url": None,
                "team_id": team_id,
            }


def _load_players_from_legacy_squad(payload: Payload, players: dict[int, Record]) -> None:
    legacy_team_id = _safe_int((payload.get("team") or {}).get("id"))
    for player in payload.get("squad", []):
        if player.get("id") is None or legacy_team_id == 0:
            continue

        player_id = int(player["id"])
        players[player_id] = {
            "player_id": player_id,
            "full_name": player.get("name"),
            "position": player.get("position"),
            "nationality": player.get("nationality"),
            "birth_date": player.get("dateOfBirth"),
            "photo_url": None,
            "team_id": legacy_team_id,
        }


def _update_match_teams(home_team: Record, away_team: Record, teams: dict[int, Record]) -> None:
    for team in (home_team, away_team):
        if team.get("id") is None:
            continue
        team_id = int(team["id"])
        teams[team_id] = {
            "team_id": team_id,
            "team_name": team.get("name"),
            "country": None,
        }


def _build_football_data_match_row(
    match: Record,
    competition_id: int,
    date_values: set[str],
) -> Record:
    utc_date = match.get("utcDate")
    date_id = utc_date[:10] if utc_date else None
    if date_id:
        date_values.add(date_id)

    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})
    score = match.get("score", {}).get("fullTime", {})

    return {
        "match_id": int(match["id"]),
        "date_id": date_id,
        "competition_id": competition_id,
        "home_team_id": int(home_team["id"]),
        "away_team_id": int(away_team["id"]),
        "home_score": score.get("home"),
        "away_score": score.get("away"),
    }


def transform_football_data(payload: Payload) -> TransformedData:
    matches = payload.get("matches", [])
    competition_meta = payload.get("competition", {}) or {}
    competition_teams = payload.get("teams", [])
    competition_code = payload.get("competition_code", "PD")

    date_values: set[str] = set()
    teams: dict[int, Record] = {}
    players: dict[int, Record] = {}
    competitions = _build_competition_rows(competition_meta, competition_code)
    competition_id = next(iter(competitions))

    _load_competition_teams(competition_teams, teams)
    _load_players_from_squads(payload.get("squads_by_team", []), teams, players)
    _load_players_from_legacy_squad(payload, players)

    fact_matches: TableRows = []
    for match in matches:
        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})
        _update_match_teams(home_team, away_team, teams)
        fact_matches.append(_build_football_data_match_row(match, competition_id, date_values))

    return {
        "dim_date": _build_dim_date(date_values),
        "dim_team": list(teams.values()),
        "dim_competition": list(competitions.values()),
        "dim_player": list(players.values()),
        "fact_match": fact_matches,
        "fact_player_match_stats": [],
    }


def count_loaded(transformed: TransformedData) -> int:
    return sum(len(rows) for rows in transformed.values())
