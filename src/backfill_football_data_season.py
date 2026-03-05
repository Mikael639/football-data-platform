from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

import requests

from src.config import get_settings
from src.extract import _fd_get, _fetch_standings_payload, DEFAULT_BASE_URL
from src.load import load_all
from src.transform import transform_football_data
from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger("backfill_football_data_season")


def extract_competition_season(competition_code: str, season_start: int) -> dict[str, Any]:
    settings = get_settings()
    token = (settings.football_data_token or "").strip()
    if not token:
        raise RuntimeError("FOOTBALL_DATA_TOKEN is required for backfill.")

    base_url = settings.football_data_base_url or DEFAULT_BASE_URL

    teams_payload = _fd_get(
        f"/competitions/{competition_code}/teams",
        token,
        base_url,
        params={"season": season_start},
    )
    teams = teams_payload.get("teams", [])
    if not teams:
        raise RuntimeError(f"No teams returned for competition={competition_code}, season={season_start}")

    squads_by_team: list[dict[str, Any]] = []
    squad_fetch_errors: list[dict[str, Any]] = []
    squad_rate_limited = False
    for team in teams:
        team_id = team.get("id")
        if team_id is None:
            continue
        if squad_rate_limited:
            squads_by_team.append({"team": team, "squad": []})
            continue
        try:
            team_payload = _fd_get(f"/teams/{int(team_id)}", token, base_url)
            team_meta = team_payload.get("team") or team
            squads_by_team.append({"team": team_meta, "squad": team_payload.get("squad", [])})
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            squads_by_team.append({"team": team, "squad": []})
            squad_fetch_errors.append({"team_id": int(team_id), "status_code": status_code})
            if status_code == 429:
                squad_rate_limited = True

    matches_payload = _fd_get(
        f"/competitions/{competition_code}/matches",
        token,
        base_url,
        params={"season": season_start},
    )
    matches = matches_payload.get("matches", [])
    standings_payload = _fetch_standings_payload(competition_code, season_start, token, base_url)

    return {
        "source": "football-data.org",
        "extracted_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "season": season_start,
        "competition_code": competition_code,
        "competition": teams_payload.get("competition", {}),
        "teams": teams,
        "squads_by_team": squads_by_team,
        "matches": matches,
        "standings": standings_payload,
        "standings_matchday": (standings_payload.get("season") or {}).get("currentMatchday"),
        "squad_fetch_errors": squad_fetch_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill one football-data.org competition season into local warehouse.")
    parser.add_argument("--competition", default="CL", help="Competition code, e.g. CL, EL, PD.")
    parser.add_argument("--season", type=int, required=True, help="Season start year, e.g. 2024 for 2024-2025.")
    args = parser.parse_args()

    payload = extract_competition_season(args.competition, args.season)
    transformed = transform_football_data(payload)

    engine = get_engine(settings=get_settings())
    loaded = load_all(engine, transformed)
    logger.info(
        "Backfill done | competition=%s season=%s extracted_matches=%s loaded_rows=%s",
        args.competition,
        args.season,
        len(payload.get("matches", [])),
        loaded,
    )


if __name__ == "__main__":
    main()

