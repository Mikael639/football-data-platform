import json
import re
import time
from datetime import datetime, date
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from src.config import Settings, get_settings
from src.utils.logger import get_logger

RAW_PATH = Path("data/raw/fixtures_mock.json")
CSV_GLOB = "data/raw/*_cleaned.csv"
DEFAULT_BASE_URL = "https://api.football-data.org/v4"
CSV_MIN_COLUMNS = {"Date", "Comp", "Round", "Venue", "Result", "Squad", "Opponent"}
logger = get_logger("extract")


# -----------------------
# Mock extractor
# -----------------------
def extract_from_mock() -> dict[str, Any]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Mock file not found: {RAW_PATH}")
    with RAW_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_extracted(payload: dict[str, Any]) -> int:
    if "csv_payload" in payload or "api_payload" in payload or "api_payloads" in payload:
        total = 0
        if "csv_payload" in payload:
            total += count_extracted(payload["csv_payload"])
        if "api_payload" in payload:
            total += count_extracted(payload["api_payload"])
        if "api_payloads" in payload:
            total += sum(count_extracted(item) for item in payload["api_payloads"])
        return total
    # Mock payload
    if "fixtures" in payload:
        return len(payload.get("fixtures", []))
    if "match_candidates" in payload:
        return len(payload.get("match_candidates", []))
    # football-data payload
    if "matches" in payload:
        return len(payload.get("matches", []))
    return 0


def _find_cleaned_header_row(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    marker = "Date,Day,Comp,Round,Venue,Result,Squad,Opponent"
    for index, line in enumerate(lines[:40]):
        if line.strip().startswith(marker):
            return index
    raise ValueError(
        f"Could not find match log header in {path.name}. "
        "Expected a row starting with 'Date,Day,Comp,Round,Venue,Result,Squad,Opponent'."
    )


def _normalize_team_name(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_competition(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if text.lower() == "la liga":
        return "La Liga"
    return text


def _normalize_venue(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"home", "domicile"}:
        return "Home"
    if text in {"away", "exterieur", "extérieur"}:
        return "Away"
    return text.title() if text else ""


def _season_label_from_date(match_date: date) -> str:
    if match_date.month >= 7:
        return f"{match_date.year}-{match_date.year + 1}"
    return f"{match_date.year - 1}-{match_date.year}"


def _season_label_from_filename(path: Path) -> str | None:
    match = re.search(r"(20\d{2})[_-](20\d{2})", path.stem)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def _parse_round(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def _normalize_result_text(value: Any) -> str:
    text = str(value or "").strip()
    return (
        text.replace("â€“", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("  ", " ")
    )


def _parse_score(result_text: str, venue: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)\s*-\s*(\d+)", result_text)
    if not match:
        return None, None
    left_score = int(match.group(1))
    right_score = int(match.group(2))
    if venue == "Home":
        return left_score, right_score
    if venue == "Away":
        return right_score, left_score
    return left_score, right_score


def _result_implies_finished(result_text: str) -> bool:
    if not result_text:
        return False
    if re.search(r"(\d+)\s*-\s*(\d+)", result_text):
        return True
    return bool(re.match(r"^[WDL]\b", result_text.upper()))


def _stable_int_id(value: str, modulus: int = 2_000_000_000) -> int:
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def _load_cleaned_matchlog(path: Path) -> pd.DataFrame:
    header_row = _find_cleaned_header_row(path)
    frame = pd.read_csv(path, skiprows=header_row)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = sorted(CSV_MIN_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"CSV {path.name} is missing required columns: {missing}")

    frame = frame.copy()
    frame["__source_file"] = path.name
    frame["__season_hint"] = _season_label_from_filename(path)
    return frame


def _build_side_match_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        competition = _normalize_competition(row.get("Comp"))
        if competition != "La Liga":
            continue
        squad = _normalize_team_name(row.get("Squad"))
        opponent = _normalize_team_name(row.get("Opponent"))
        venue = _normalize_venue(row.get("Venue"))
        if not competition or not squad or not opponent or venue not in {"Home", "Away"}:
            continue

        match_date = pd.to_datetime(row.get("Date"), errors="coerce")
        if pd.isna(match_date):
            continue

        match_date_iso = match_date.date().isoformat()
        season = row.get("__season_hint") or _season_label_from_date(match_date.date())
        round_value = _parse_round(row.get("Round"))
        result_text = _normalize_result_text(row.get("Result"))
        home_score, away_score = _parse_score(result_text, venue)
        status = "FINISHED" if _result_implies_finished(result_text) or home_score is not None else "SCHEDULED"
        home_team = squad if venue == "Home" else opponent
        away_team = opponent if venue == "Home" else squad

        rows.append(
            {
                "season": season,
                "competition": competition,
                "date_id": match_date_iso,
                "candidate_key": "|".join([season, competition, match_date_iso, squad, opponent, venue]),
                "match_key": "|".join([season, competition, match_date_iso, home_team, away_team]),
                "squad": squad,
                "opponent": opponent,
                "venue": venue,
                "home_team": home_team,
                "away_team": away_team,
                "matchday": round_value,
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "source_file": row.get("__source_file"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "competition",
                "date_id",
                "candidate_key",
                "match_key",
                "squad",
                "opponent",
                "venue",
                "home_team",
                "away_team",
                "matchday",
                "status",
                "home_score",
                "away_score",
                "source_file",
            ]
        )
    return pd.DataFrame(rows)


def _mode_with_warning(series: pd.Series, *, field_name: str, key: str) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    counts = non_null.astype(str).value_counts()
    if len(counts) > 1:
        logger.warning("CSV divergence on %s for key=%s values=%s", field_name, key, counts.to_dict())
    top_value = counts.index[0]
    sample = non_null[non_null.astype(str) == top_value].iloc[0]
    return sample


def _dedupe_match_candidates(side_candidates: pd.DataFrame, competition_code: str) -> list[dict[str, Any]]:
    if side_candidates.empty:
        return []

    deduped_matches: list[dict[str, Any]] = []
    for match_key, group in side_candidates.groupby("match_key", dropna=False):
        home_team = group["home_team"].iloc[0]
        away_team = group["away_team"].iloc[0]
        date_id = group["date_id"].iloc[0]
        season = group["season"].iloc[0]
        competition = group["competition"].iloc[0]
        home_score = _mode_with_warning(group["home_score"], field_name="home_score", key=str(match_key))
        away_score = _mode_with_warning(group["away_score"], field_name="away_score", key=str(match_key))
        matchday = _mode_with_warning(group["matchday"], field_name="matchday", key=str(match_key))
        status = _mode_with_warning(group["status"], field_name="status", key=str(match_key)) or "SCHEDULED"

        deduped_matches.append(
            {
                "match_id": _stable_int_id(str(match_key)),
                "season": season,
                "competition": competition,
                "competition_code": competition_code,
                "date_id": date_id,
                "kickoff_utc": f"{date_id}T12:00:00Z",
                "status": status,
                "matchday": int(matchday) if matchday is not None else None,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score) if home_score is not None else None,
                "away_score": int(away_score) if away_score is not None else None,
            }
        )
    return sorted(deduped_matches, key=lambda item: (item["season"], item["date_id"], item["match_id"]))


def extract_csv(
    settings: Settings | None = None,
    raw_glob: str = CSV_GLOB,
) -> dict[str, Any]:
    resolved_settings = settings or get_settings()
    files = sorted(Path(".").glob(raw_glob))
    if not files:
        raise FileNotFoundError(f"No CSV files found with pattern: {raw_glob}")

    frames = [_load_cleaned_matchlog(path) for path in files]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise ValueError("No rows available after reading cleaned CSV files.")

    side_candidates = _build_side_match_candidates(merged)
    if side_candidates.empty:
        raise ValueError("No valid match candidates could be extracted from the cleaned CSV files.")

    team_names = sorted(set(side_candidates["home_team"].tolist()) | set(side_candidates["away_team"].tolist()))
    team_rows = [
        {
            "id": _stable_int_id(f"team::{team_name}"),
            "name": team_name,
            "shortName": team_name,
            "area": {"name": "Spain"},
            "crest": None,
        }
        for team_name in team_names
    ]
    team_id_map = {team["name"]: team["id"] for team in team_rows}
    match_candidates = _dedupe_match_candidates(side_candidates, resolved_settings.competition_code)
    for match in match_candidates:
        match["home_team_id"] = team_id_map[match["home_team"]]
        match["away_team_id"] = team_id_map[match["away_team"]]
    competition_name = "La Liga" if resolved_settings.competition_code == "PD" else resolved_settings.competition_code

    return {
        "source": "csv",
        "competition_code": resolved_settings.competition_code,
        "competition": {
            "id": 2014 if resolved_settings.competition_code == "PD" else _stable_int_id(resolved_settings.competition_code),
            "name": competition_name,
            "area": {"name": "Spain" if resolved_settings.competition_code == "PD" else None},
        },
        "extracted_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "match_candidates": match_candidates,
        "teams": team_rows,
        "csv_files": [path.name for path in files],
    }


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
def extract_football_data_competition(
    settings: Settings | None = None,
    competition_code: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or get_settings()
    token = resolved_settings.football_data_token
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN in environment")

    competition_code = competition_code or resolved_settings.competition_code
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


def extract_football_data_live_competitions(
    settings: Settings | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    resolved_settings = settings or get_settings()
    return [
        extract_football_data_competition(settings=resolved_settings, competition_code=competition_code, today=today)
        for competition_code in resolved_settings.live_competition_codes
    ]


def extract_football_data_laliga_all_clubs(
    settings: Settings | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible helper kept for existing imports.
    Returns the payload for the primary competition configured in COMPETITION_CODE.
    """
    return extract_football_data_competition(settings=settings, competition_code=(settings or get_settings()).competition_code, today=today)


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
