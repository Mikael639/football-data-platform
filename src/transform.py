from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime, timezone
from typing import Any

Record = dict[str, Any]
Payload = dict[str, Any]
TableRows = list[Record]
TransformedData = dict[str, TableRows]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _safe_float_ratio(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        parsed = float(value)
        return max(0.0, min(parsed, 1.0))
    except Exception:
        return 0.0


def _parse_utc_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _season_label_from_date_value(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        match_date = value.date()
    elif isinstance(value, date_cls):
        match_date = value
    else:
        text = str(value)
        try:
            match_date = datetime.fromisoformat(text[:10]).date()
        except ValueError:
            return None
    if match_date.month >= 7:
        return f"{match_date.year}-{match_date.year + 1}"
    return f"{match_date.year - 1}-{match_date.year}"


def _season_label_from_start_year(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    try:
        start_year = int(value)
    except (TypeError, ValueError):
        return None
    return f"{start_year}-{start_year + 1}"


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


def _build_mock_team_row(team: Record) -> Record:
    return {
        "team_id": _safe_int(team.get("team_id")),
        "team_name": team.get("team_name"),
        "country": team.get("country"),
        "crest_url": team.get("crest_url"),
        "short_name": team.get("short_name"),
    }


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


def _build_mock_competition_row(fixture: Record, payload: Payload) -> Record:
    competition_id = _safe_int(fixture.get("competition_id"))
    return {
        "competition_id": competition_id,
        "competition_name": fixture.get("competition_name") or payload.get("competition"),
        "country": payload.get("country"),
    }


def _update_mock_teams(home_team: Record, away_team: Record, teams: dict[int, Record]) -> None:
    for team in (home_team, away_team):
        team_id = _safe_int(team.get("team_id"))
        teams[team_id] = _build_mock_team_row(team)


def _build_mock_match_row(fixture: Record, payload: Payload, home_team: Record, away_team: Record) -> Record:
    score = fixture.get("score", {})
    return {
        "match_id": _safe_int(fixture.get("match_id")),
        "date_id": fixture.get("date"),
        "competition_id": _safe_int(fixture.get("competition_id")),
        "home_team_id": _safe_int(home_team.get("team_id")),
        "away_team_id": _safe_int(away_team.get("team_id")),
        "status": fixture.get("status"),
        "matchday": fixture.get("matchday"),
        "kickoff_utc": _parse_utc_datetime(fixture.get("kickoff_utc") or fixture.get("utc_date")),
        "season": fixture.get("season")
        or payload.get("season")
        or _season_label_from_date_value(fixture.get("date")),
        "home_score": _safe_int(score.get("home")),
        "away_score": _safe_int(score.get("away")),
    }


def _build_mock_player_stats(fixture: Record, players: dict[int, Record]) -> TableRows:
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
            "photo_url": player.get("photo_url"),
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


def _extract_mock_standings(
    payload: Payload,
    competition_id: int,
    extracted_at: datetime | None,
) -> TableRows:
    standings_payload = payload.get("standings") or {}
    table = standings_payload.get("table", [])
    season = _safe_int(standings_payload.get("season"), _safe_int(payload.get("season")))
    matchday = standings_payload.get("matchday")
    snapshot_ts = extracted_at or _parse_utc_datetime(standings_payload.get("snapshot_ts"))

    rows: TableRows = []
    for row in table:
        rows.append(
            {
                "competition_id": competition_id,
                "season": season,
                "matchday": matchday,
                "team_id": _safe_int(row.get("team_id")),
                "position": row.get("position"),
                "points": row.get("points"),
                "played_games": row.get("played_games"),
                "won": row.get("won"),
                "draw": row.get("draw"),
                "lost": row.get("lost"),
                "goals_for": row.get("goals_for"),
                "goals_against": row.get("goals_against"),
                "goal_difference": row.get("goal_difference"),
                "snapshot_ts": snapshot_ts,
            }
        )
    return rows


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

        competition = _build_mock_competition_row(fixture, payload)
        competitions[competition["competition_id"]] = competition

        home_team = fixture.get("home_team", {})
        away_team = fixture.get("away_team", {})
        _update_mock_teams(home_team, away_team, teams)

        matches.append(_build_mock_match_row(fixture, payload, home_team, away_team))
        player_match_stats.extend(_build_mock_player_stats(fixture, players))

    competition_id = next(iter(competitions)) if competitions else _safe_int(payload.get("competition_id"), 1)
    extracted_at = _parse_utc_datetime(payload.get("extracted_at_utc"))

    return {
        "dim_date": _build_dim_date(date_values),
        "dim_team": list(teams.values()),
        "dim_competition": list(competitions.values()),
        "dim_player": list(players.values()),
        "fact_match": matches,
        "fact_player_match_stats": player_match_stats,
        "fact_standings_snapshot": _extract_mock_standings(payload, competition_id, extracted_at),
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
        "crest_url": team.get("crest"),
        "short_name": team.get("shortName"),
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
                "crest_url": team_meta.get("crest"),
                "short_name": team_meta.get("shortName"),
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
            "team_name": team.get("name") or team.get("shortName"),
            "country": None,
            "crest_url": team.get("crest"),
            "short_name": team.get("shortName"),
        }


def _build_football_data_match_row(
    match: Record,
    competition_id: int,
    date_values: set[str],
    season_label: str | None,
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
        "status": match.get("status"),
        "matchday": match.get("matchday"),
        "kickoff_utc": _parse_utc_datetime(utc_date),
        "season": season_label or _season_label_from_date_value(date_id),
        "home_score": score.get("home"),
        "away_score": score.get("away"),
    }


def _extract_standings_snapshot(
    payload: Payload,
    competition_id: int,
    extracted_at: datetime | None,
) -> TableRows:
    standings_payload = payload.get("standings") or {}
    standings_blocks = standings_payload.get("standings", [])
    season = _safe_int(payload.get("season"))
    season_meta = standings_payload.get("season", {}) or {}
    matchday = season_meta.get("currentMatchday") or payload.get("standings_matchday")
    snapshot_ts = extracted_at or _parse_utc_datetime(payload.get("extracted_at_utc"))

    rows: TableRows = []
    for standing in standings_blocks:
        if standing.get("type") != "TOTAL":
            continue
        for row in standing.get("table", []) or []:
            team = row.get("team", {}) or {}
            team_id = _safe_int(team.get("id"))
            if team_id == 0:
                continue
            rows.append(
                {
                    "competition_id": competition_id,
                    "season": season,
                    "matchday": matchday,
                    "team_id": team_id,
                    "position": row.get("position"),
                    "points": row.get("points"),
                    "played_games": row.get("playedGames"),
                    "won": row.get("won"),
                    "draw": row.get("draw"),
                    "lost": row.get("lost"),
                    "goals_for": row.get("goalsFor"),
                    "goals_against": row.get("goalsAgainst"),
                    "goal_difference": row.get("goalDifference"),
                    "snapshot_ts": snapshot_ts,
                }
            )
    return rows


def transform_football_data(payload: Payload) -> TransformedData:
    matches = payload.get("matches", [])
    competition_meta = payload.get("competition", {}) or {}
    competition_teams = payload.get("teams", [])
    competition_code = payload.get("competition_code", "PD")
    season_label = _season_label_from_start_year(payload.get("season"))

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
        fact_matches.append(_build_football_data_match_row(match, competition_id, date_values, season_label))

    extracted_at = _parse_utc_datetime(payload.get("extracted_at_utc"))

    return {
        "dim_date": _build_dim_date(date_values),
        "dim_team": list(teams.values()),
        "dim_competition": list(competitions.values()),
        "dim_player": list(players.values()),
        "fact_match": fact_matches,
        "fact_player_match_stats": [],
        "fact_standings_snapshot": _extract_standings_snapshot(payload, competition_id, extracted_at),
    }


def transform_csv_to_tables(payload: Payload) -> TransformedData:
    competition_meta = payload.get("competition", {}) or {}
    competition_code = payload.get("competition_code", "PD")
    competition_id = _safe_int(competition_meta.get("id"), 1)
    competition_rows = _build_competition_rows(competition_meta, competition_code)

    date_values: set[str] = set()
    teams: dict[int, Record] = {}
    fact_matches: TableRows = []

    for team in payload.get("teams", []):
        team_id = _safe_int(team.get("id"))
        if team_id == 0:
            continue
        teams[team_id] = {
            "team_id": team_id,
            "team_name": team.get("name") or team.get("shortName"),
            "country": (team.get("area") or {}).get("name"),
            "crest_url": team.get("crest"),
            "short_name": team.get("shortName"),
        }

    for match in payload.get("match_candidates", []):
        date_id = str(match.get("date_id"))
        if date_id:
            date_values.add(date_id)
        fact_matches.append(
            {
                "match_id": _safe_int(match.get("match_id")),
                "date_id": date_id,
                "competition_id": competition_id,
                "home_team_id": _safe_int(match.get("home_team_id")),
                "away_team_id": _safe_int(match.get("away_team_id")),
                "status": match.get("status"),
                "matchday": match.get("matchday"),
                "kickoff_utc": _parse_utc_datetime(match.get("kickoff_utc")),
                "season": match.get("season") or _season_label_from_date_value(date_id),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
            }
        )

    return {
        "dim_date": _build_dim_date(date_values),
        "dim_team": list(teams.values()),
        "dim_competition": list(competition_rows.values()),
        "dim_player": [],
        "fact_match": fact_matches,
        "fact_player_match_stats": [],
        "fact_standings_snapshot": [],
    }


def _normalize_team_name(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _team_row_priority(row: Record) -> tuple[int, int, int]:
    return (
        1 if row.get("crest_url") else 0,
        1 if row.get("country") else 0,
        1 if row.get("short_name") else 0,
    )


def merge_transformed_data(*datasets: TransformedData) -> TransformedData:
    merged: TransformedData = {
        "dim_date": [],
        "dim_team": [],
        "dim_competition": [],
        "dim_player": [],
        "fact_match": [],
        "fact_player_match_stats": [],
        "fact_standings_snapshot": [],
    }
    if not datasets:
        return merged

    all_team_rows: list[Record] = []
    for dataset in datasets:
        all_team_rows.extend(dataset.get("dim_team", []))

    canonical_team_by_name: dict[str, Record] = {}
    team_id_map: dict[int, int] = {}
    for row in all_team_rows:
        team_id = _safe_int(row.get("team_id"))
        if team_id == 0:
            continue
        normalized_name = _normalize_team_name(row.get("team_name"))
        if not normalized_name:
            canonical_team_by_name[f"id::{team_id}"] = row.copy()
            team_id_map[team_id] = team_id
            continue
        existing = canonical_team_by_name.get(normalized_name)
        if existing is None or _team_row_priority(row) >= _team_row_priority(existing):
            canonical_team = row.copy()
            if existing is not None:
                team_id_map[_safe_int(existing.get("team_id"))] = team_id
            canonical_team_by_name[normalized_name] = canonical_team
        else:
            canonical_team = existing
        team_id_map[team_id] = _safe_int(canonical_team.get("team_id"))

    merged["dim_team"] = list(canonical_team_by_name.values())

    def remap_team_id(value: Any) -> Any:
        team_id = _safe_int(value)
        return team_id_map.get(team_id, team_id) if team_id else value

    dedupe_keys = {
        "dim_date": "date_id",
        "dim_competition": "competition_id",
        "dim_player": "player_id",
        "fact_match": "match_id",
    }
    tuple_keys = {
        "fact_player_match_stats": ("match_id", "player_id"),
        "fact_standings_snapshot": ("competition_id", "season", "matchday", "team_id"),
    }

    for table_name, key_name in dedupe_keys.items():
        deduped: dict[Any, Record] = {}
        for dataset in datasets:
            for row in dataset.get(table_name, []):
                record = row.copy()
                if "team_id" in record:
                    record["team_id"] = remap_team_id(record.get("team_id"))
                if "home_team_id" in record:
                    record["home_team_id"] = remap_team_id(record.get("home_team_id"))
                if "away_team_id" in record:
                    record["away_team_id"] = remap_team_id(record.get("away_team_id"))
                deduped[record[key_name]] = record
        merged[table_name] = list(deduped.values())

    for table_name, key_names in tuple_keys.items():
        deduped: dict[tuple[Any, ...], Record] = {}
        for dataset in datasets:
            for row in dataset.get(table_name, []):
                record = row.copy()
                if "team_id" in record:
                    record["team_id"] = remap_team_id(record.get("team_id"))
                if "home_team_id" in record:
                    record["home_team_id"] = remap_team_id(record.get("home_team_id"))
                if "away_team_id" in record:
                    record["away_team_id"] = remap_team_id(record.get("away_team_id"))
                deduped[tuple(record[key] for key in key_names)] = record
        merged[table_name] = list(deduped.values())

    return merged


def count_loaded(transformed: TransformedData) -> int:
    return sum(len(rows) for rows in transformed.values())
