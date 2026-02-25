import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, date

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
    r = requests.get(url, headers={"X-Auth-Token": token}, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def current_season_start_year(today: date | None = None) -> int:
    """
    LaLiga season start year rule of thumb (starts around Aug).
    If month >= 7 (July+) => season start is current year, else previous year.
    """
    if today is None:
        today = datetime.utcnow().date()
    return today.year if today.month >= 7 else today.year - 1


# -----------------------
# football-data extract (Real Madrid LaLiga current season, +- 60 days window)
# -----------------------
def extract_football_data_laliga_real_madrid() -> Dict[str, Any]:
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")
    team_name = os.getenv("FOOTBALL_DATA_TEAM_NAME", "Real Madrid").strip().lower()

    season_env = os.getenv("FOOTBALL_DATA_SEASON")
    season = int(season_env) if season_env else current_season_start_year()

    # 1) Find Real Madrid team_id from competition teams
    teams_payload = _fd_get(f"/competitions/{competition_code}/teams", token, params={"season": season})
    teams = teams_payload.get("teams", [])

    def is_team(t: Dict[str, Any]) -> bool:
        name = (t.get("name") or "").strip().lower()
        short = (t.get("shortName") or "").strip().lower()
        tla = (t.get("tla") or "").strip().lower()
        return team_name in name or team_name in short or team_name == tla

    rm = next((t for t in teams if is_team(t)), None)
    if not rm:
        raise RuntimeError(f"Team not found in competition teams list: {team_name}")

    team_id = int(rm["id"])

    # 2) Get squad
    team_payload = _fd_get(f"/teams/{team_id}", token)
    squad = team_payload.get("squad", [])

    # 3) Get matches in time window: last 60 days to next 60 days
    today = datetime.utcnow().date()
    date_from = (today - timedelta(days=60)).isoformat()
    date_to = (today + timedelta(days=60)).isoformat()

    matches_payload = _fd_get(
        f"/competitions/{competition_code}/matches",
        token,
        params={"season": season, "dateFrom": date_from, "dateTo": date_to},
    )
    matches = matches_payload.get("matches", [])

    rm_matches = []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        if int(home.get("id", -1)) == team_id or int(away.get("id", -1)) == team_id:
            rm_matches.append(m)

    return {
        "source": "football-data.org",
        "season": season,
        "competition_code": competition_code,
        "team": rm,
        "squad": squad,
        "matches": rm_matches,
        "window": {"dateFrom": date_from, "dateTo": date_to},
    }


def extract_laliga_standings_current_season() -> Dict[str, Any]:
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

    season_env = os.getenv("FOOTBALL_DATA_SEASON")
    season = int(season_env) if season_env else current_season_start_year()

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