import json
import time
from datetime import datetime, date
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import requests

from src.config import Settings, get_settings

RAW_PATH = Path("data/raw/fixtures_mock.json")
DEFAULT_BASE_URL = "https://api.football-data.org/v4"


# -----------------------
# Mock extractor
# -----------------------
def extract_from_mock() -> dict[str, Any]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Mock file not found: {RAW_PATH}")
    with RAW_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_extracted(payload: dict[str, Any]) -> int:
    # Mock payload
    if "fixtures" in payload:
        return len(payload.get("fixtures", []))
    # football-data payload
    if "matches" in payload:
        return len(payload.get("matches", []))
    return 0


# -----------------------
# football-data helpers
# -----------------------
def _fd_get(
    path: str,
    token: str,
    base_url: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    url = f"{base_url}{path}"
    max_retries = 3

    for attempt in range(max_retries + 1):
        r = requests.get(url, headers={"X-Auth-Token": token}, params=params, timeout=30)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()

        if attempt >= max_retries:
            r.raise_for_status()

        retry_after_raw = r.headers.get("Retry-After")
        try:
            retry_after = int(retry_after_raw) if retry_after_raw else 0
        except ValueError:
            retry_after = 0

        wait_seconds = retry_after if retry_after > 0 else min(5 * (attempt + 1), 30)
        time.sleep(wait_seconds)

    raise RuntimeError("Unexpected retry loop exit in _fd_get")


def current_season_start_year(today: date | None = None) -> int:
    """
    LaLiga season start year rule of thumb (starts around Aug).
    If month >= 7 (July+) => season start is current year, else previous year.
    """
    if today is None:
        today = datetime.utcnow().date()
    return today.year if today.month >= 7 else today.year - 1


def resolved_current_season_start_year() -> int:
    """
    Always use the current season start year derived from today's date.
    This prevents stale .env values (e.g. previous season) from leaking into live extracts.
    """
    return current_season_start_year()


def calculate_incremental_window(
    days: int,
    today: date | None = None,
) -> dict[str, str]:
    current_day = today or datetime.utcnow().date()
    window_start = current_day - timedelta(days=days)
    window_end = current_day + timedelta(days=1)
    return {
        "dateFrom": window_start.isoformat(),
        "dateTo": window_end.isoformat(),
    }


def _build_match_query_params(settings: Settings, season: int, today: date | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"season": season}
    if settings.incremental:
        params.update(calculate_incremental_window(settings.incremental_days, today=today))
    return params


def _fetch_standings_payload(
    competition_code: str,
    season: int,
    token: str,
    base_url: str,
) -> dict[str, Any]:
    return _fd_get(
        f"/competitions/{competition_code}/standings",
        token,
        base_url,
        params={"season": season},
    )


# -----------------------
# football-data extract (all LaLiga clubs, current season)
# -----------------------
def extract_football_data_laliga_all_clubs(
    settings: Settings | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or get_settings()
    token = resolved_settings.football_data_token
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = resolved_settings.competition_code
    season = resolved_current_season_start_year()
    base_url = resolved_settings.football_data_base_url or DEFAULT_BASE_URL
    match_params = _build_match_query_params(resolved_settings, season, today=today)

    # 1) Get all teams in the competition for the current season
    teams_payload = _fd_get(
        f"/competitions/{competition_code}/teams",
        token,
        base_url,
        params={"season": season},
    )
    teams = teams_payload.get("teams", [])
    if not teams:
        raise RuntimeError(f"No teams returned for competition={competition_code}, season={season}")

    # 2) Get the squad for each club (one request per club)
    squads_by_team = []
    squad_fetch_errors = []
    squad_rate_limited = False
    for t in teams:
        tid = t.get("id")
        if tid is None:
            continue
        if squad_rate_limited:
            squads_by_team.append({"team": t, "squad": []})
            continue
        try:
            team_payload = _fd_get(f"/teams/{int(tid)}", token, base_url)
            team_meta = team_payload.get("team") or t
            squads_by_team.append(
                {
                    "team": team_meta,
                    "squad": team_payload.get("squad", []),
                }
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            squads_by_team.append({"team": t, "squad": []})
            squad_fetch_errors.append({"team_id": int(tid), "status_code": status_code})
            if status_code == 429:
                squad_rate_limited = True

    # 3) Get all competition matches for the current season
    matches_payload = _fd_get(
        f"/competitions/{competition_code}/matches",
        token,
        base_url,
        params=match_params,
    )
    matches = matches_payload.get("matches", [])
    standings_payload = _fetch_standings_payload(competition_code, season, token, base_url)

    return {
        "source": "football-data.org",
        "extracted_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "season": season,
        "competition_code": competition_code,
        "competition": teams_payload.get("competition", {}),
        "teams": teams,
        "squads_by_team": squads_by_team,
        "matches": matches,
        "standings": standings_payload,
        "standings_matchday": (standings_payload.get("season") or {}).get("currentMatchday"),
        "squad_fetch_errors": squad_fetch_errors,
        "incremental_window": calculate_incremental_window(resolved_settings.incremental_days, today=today)
        if resolved_settings.incremental
        else None,
    }


def extract_football_data_laliga_real_madrid(
    settings: Settings | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible name kept for existing imports.
    Current behavior returns all LaLiga clubs for the current season.
    """
    return extract_football_data_laliga_all_clubs(settings=settings, today=today)


def extract_laliga_standings_current_season(settings: Settings | None = None) -> dict[str, Any]:
    resolved_settings = settings or get_settings()
    token = resolved_settings.football_data_token
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = resolved_settings.competition_code

    season = resolved_current_season_start_year()
    base_url = resolved_settings.football_data_base_url or DEFAULT_BASE_URL

    standings_payload = _fd_get(
        f"/competitions/{competition_code}/standings",
        token,
        base_url,
        params={"season": season},
    )

    return {
        "source": "football-data.org",
        "season": season,
        "competition_code": competition_code,
        "standings": standings_payload,
    }
