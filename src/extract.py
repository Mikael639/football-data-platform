import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, date

RAW_PATH = Path("data/raw/fixtures_mock.json")
BASE_URL = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")


# -----------------------
# Mock extractor
# -----------------------
def extract_from_mock() -> Dict[str, Any]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Mock file not found: {RAW_PATH}")
    with RAW_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_extracted(payload: Dict[str, Any]) -> int:
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
def _fd_get(path: str, token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
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


# -----------------------
# football-data extract (all LaLiga clubs, current season)
# -----------------------
def extract_football_data_laliga_all_clubs() -> Dict[str, Any]:
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")
    season = resolved_current_season_start_year()

    # 1) Get all teams in the competition for the current season
    teams_payload = _fd_get(f"/competitions/{competition_code}/teams", token, params={"season": season})
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
            team_payload = _fd_get(f"/teams/{int(tid)}", token)
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
        params={"season": season},
    )
    matches = matches_payload.get("matches", [])

    return {
        "source": "football-data.org",
        "season": season,
        "competition_code": competition_code,
        "competition": teams_payload.get("competition", {}),
        "teams": teams,
        "squads_by_team": squads_by_team,
        "matches": matches,
        "squad_fetch_errors": squad_fetch_errors,
    }


def extract_football_data_laliga_real_madrid() -> Dict[str, Any]:
    """
    Backward-compatible name kept for existing imports.
    Current behavior returns all LaLiga clubs for the current season.
    """
    return extract_football_data_laliga_all_clubs()


def extract_laliga_standings_current_season() -> Dict[str, Any]:
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

    season = resolved_current_season_start_year()

    standings_payload = _fd_get(
        f"/competitions/{competition_code}/standings",
        token,
        params={"season": season},
    )

    return {
        "source": "football-data.org",
        "season": season,
        "competition_code": competition_code,
        "standings": standings_payload,
    }
