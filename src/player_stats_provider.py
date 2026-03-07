from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger("player_stats_provider")

FBREF_LEAGUE_BY_COMPETITION_CODE = {
    "PD": "ESP-La Liga",
    "PL": "ENG-Premier League",
    "SA": "ITA-Serie A",
    "BL1": "GER-Bundesliga",
    "FL1": "FRA-Ligue 1",
}


def _safe_int(value: Any) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0
    return int(parsed)


def _parse_pass_accuracy(value: Any) -> float:
    text = str(value or "").strip().replace("%", "")
    if not text:
        return 0.0
    try:
        parsed = float(text)
    except ValueError:
        return 0.0
    if parsed > 1:
        parsed = parsed / 100.0
    return max(0.0, min(parsed, 1.0))


def _stable_int_id(value: str, modulus: int = 2_000_000_000) -> int:
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def _canonical_team_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text).lower()
    text = " ".join(text.split())
    for token in ("fc ", "cf ", "ac ", "as ", "rc "):
        if text.startswith(token):
            text = text[len(token) :]
    for token in (" fc", " cf"):
        if text.endswith(token):
            text = text[: -len(token)]
    return text.strip()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) > 0
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "start", "starter", "started"}


def _first_available_column(df: pd.DataFrame, *candidates: str) -> str | None:
    if df is None or df.empty:
        return None
    normalized: dict[str, str] = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        normalized[key] = str(col)
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _resolve_team_id(team_name: str, team_id_by_key: dict[str, int]) -> int | None:
    key = _canonical_team_key(team_name)
    if not key:
        return None
    if key in team_id_by_key:
        return int(team_id_by_key[key])
    compact_key = key.replace(" ", "")
    for known_key, known_id in team_id_by_key.items():
        if compact_key == known_key.replace(" ", ""):
            return int(known_id)
    return None


def _build_team_index(teams: list[dict[str, Any]]) -> dict[str, int]:
    team_id_by_key: dict[str, int] = {}
    for team in teams:
        team_id = _safe_int(team.get("id"))
        if team_id == 0:
            continue
        for name_key in (team.get("name"), team.get("shortName"), team.get("tla"), team.get("team_name")):
            canonical = _canonical_team_key(name_key)
            if canonical:
                team_id_by_key.setdefault(canonical, team_id)
    return team_id_by_key


def _build_matches_by_date_team(matches: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for match in matches:
        date_id = str(match.get("utcDate") or match.get("date_id") or "")[:10]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_id):
            continue
        home_team_id = _safe_int((match.get("homeTeam") or {}).get("id") or match.get("home_team_id"))
        away_team_id = _safe_int((match.get("awayTeam") or {}).get("id") or match.get("away_team_id"))
        if home_team_id > 0:
            grouped.setdefault((date_id, home_team_id), []).append(match)
        if away_team_id > 0:
            grouped.setdefault((date_id, away_team_id), []).append(match)
    return grouped


def _resolve_match_id(
    *,
    row_match_id: Any,
    row_date_id: Any,
    row_team_id: Any,
    row_team_name: Any,
    team_id_by_key: dict[str, int],
    matches_by_date_team: dict[tuple[str, int], list[dict[str, Any]]],
    row_minutes: Any,
) -> tuple[int | None, int | None]:
    direct_match_id = _safe_int(row_match_id)
    if direct_match_id > 0:
        resolved_team_id = _safe_int(row_team_id)
        if resolved_team_id == 0 and row_team_name:
            maybe_team_id = _resolve_team_id(str(row_team_name), team_id_by_key)
            resolved_team_id = int(maybe_team_id) if maybe_team_id is not None else 0
        return direct_match_id, (resolved_team_id if resolved_team_id > 0 else None)

    date_id = str(row_date_id or "")[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_id):
        return None, None

    resolved_team_id = _safe_int(row_team_id)
    if resolved_team_id == 0 and row_team_name:
        maybe_team_id = _resolve_team_id(str(row_team_name), team_id_by_key)
        if maybe_team_id is not None:
            resolved_team_id = int(maybe_team_id)
    if resolved_team_id == 0:
        return None, None

    candidates = matches_by_date_team.get((date_id, resolved_team_id), [])
    if not candidates:
        return None, None
    if len(candidates) == 1:
        match_id = _safe_int(candidates[0].get("id") or candidates[0].get("match_id"))
        return (match_id if match_id > 0 else None), resolved_team_id

    has_minutes = _safe_int(row_minutes) > 0
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if (str(item.get("status") or "") == "FINISHED" and has_minutes) else 1,
            _safe_int(item.get("id") or item.get("match_id")),
        ),
    )
    match_id = _safe_int(ranked[0].get("id") or ranked[0].get("match_id"))
    return (match_id if match_id > 0 else None), resolved_team_id


def _dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[int, int], dict[str, Any]] = {}
    for item in rows:
        key = (int(item["match_id"]), int(item["player_id"]))
        existing = deduped.get(key)
        if existing is None or int(item["minutes"]) > int(existing.get("minutes", 0)):
            deduped[key] = item
    return list(deduped.values())


def _normalize_candidate_row(
    *,
    source: str,
    season_start: int,
    row: dict[str, Any],
    team_id_by_key: dict[str, int],
    matches_by_date_team: dict[tuple[str, int], list[dict[str, Any]]],
    key_map: dict[str, str | None],
) -> dict[str, Any] | None:
    player_name = " ".join(str(row.get(key_map["player_name"] or "", "")).strip().split())
    if not player_name:
        return None
    player_id_raw = row.get(key_map["player_id"] or "")
    player_id = _safe_int(player_id_raw)
    if player_id == 0:
        player_key = re.sub(r"[^a-z0-9]+", "_", player_name.casefold()).strip("_")
        player_id = _stable_int_id(f"{source}_player::{player_key}")

    match_id, resolved_team_id = _resolve_match_id(
        row_match_id=row.get(key_map["match_id"] or ""),
        row_date_id=row.get(key_map["date_id"] or ""),
        row_team_id=row.get(key_map["team_id"] or ""),
        row_team_name=row.get(key_map["team_name"] or ""),
        team_id_by_key=team_id_by_key,
        matches_by_date_team=matches_by_date_team,
        row_minutes=row.get(key_map["minutes"] or ""),
    )
    if match_id is None:
        return None

    position_raw = row.get(key_map["position"] or "")
    team_name = " ".join(str(row.get(key_map["team_name"] or "", "")).strip().split())
    return {
        "match_id": int(match_id),
        "player_id": int(player_id),
        "player_name": player_name,
        "position": (str(position_raw).strip() or None) if position_raw is not None else None,
        "is_starting": _coerce_bool(row.get(key_map["is_starting"] or "")),
        "minutes": max(0, min(_safe_int(row.get(key_map["minutes"] or "")), 130)),
        "goals": max(0, _safe_int(row.get(key_map["goals"] or ""))),
        "assists": max(0, _safe_int(row.get(key_map["assists"] or ""))),
        "shots": max(0, _safe_int(row.get(key_map["shots"] or ""))),
        "passes": max(0, _safe_int(row.get(key_map["passes"] or ""))),
        "pass_accuracy": _parse_pass_accuracy(row.get(key_map["pass_accuracy"] or "")),
        "team_id": int(resolved_team_id) if resolved_team_id is not None else None,
        "team_name": team_name,
        "season": f"{int(season_start)}-{int(season_start) + 1}",
        "source": source,
    }


def _extract_fbref_player_match_candidates(
    *,
    competition_code: str,
    season_start: int,
    teams: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    league_code = FBREF_LEAGUE_BY_COMPETITION_CODE.get(str(competition_code).upper())
    if not league_code:
        logger.info("FBref enrichment skipped for competition=%s (no mapping).", competition_code)
        return []

    try:
        import soccerdata as sd  # type: ignore
    except ImportError:
        logger.warning("FBref enrichment skipped: soccerdata dependency missing.")
        return []

    try:
        fbref = sd.FBref(
            leagues=league_code,
            seasons=[int(season_start)],
            proxy=(None if not os.getenv("FBREF_PROXY") else os.getenv("FBREF_PROXY")),
            no_cache=str(os.getenv("FBREF_NO_CACHE", "false")).strip().lower() in {"1", "true", "yes", "y"},
        )
        raw_stats = fbref.read_player_match_stats(stat_type="summary").reset_index()
    except Exception:
        logger.exception("FBref enrichment failed for competition=%s season=%s.", competition_code, season_start)
        return []

    if raw_stats is None or raw_stats.empty:
        return []

    frame = raw_stats.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join([str(part) for part in col if str(part) not in {"", "None"}]).strip("_")
            for col in frame.columns.to_flat_index()
        ]
    frame.columns = [str(col).strip() for col in frame.columns]

    key_map = {
        "match_id": None,
        "date_id": _first_available_column(frame, "date", "date_id", "match_date", "game"),
        "team_id": None,
        "team_name": _first_available_column(frame, "team", "squad"),
        "player_id": None,
        "player_name": _first_available_column(frame, "player", "player_name"),
        "position": _first_available_column(frame, "pos", "position"),
        "is_starting": _first_available_column(frame, "starts", "start"),
        "minutes": _first_available_column(frame, "min", "minutes"),
        "goals": _first_available_column(frame, "gls", "goals"),
        "assists": _first_available_column(frame, "ast", "assists"),
        "shots": _first_available_column(frame, "sh", "shots"),
        "passes": _first_available_column(frame, "cmp", "passes", "passescompleted"),
        "pass_accuracy": _first_available_column(frame, "cmp_pct", "passaccuracy"),
    }
    if not key_map["player_name"] or not key_map["team_name"] or not key_map["date_id"]:
        logger.warning("FBref enrichment skipped: missing required columns in output.")
        return []

    team_id_by_key = _build_team_index(teams)
    matches_by_date_team = _build_matches_by_date_team(matches)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in frame.iterrows():
        normalized = _normalize_candidate_row(
            source="fbref",
            season_start=season_start,
            row=row.to_dict(),
            team_id_by_key=team_id_by_key,
            matches_by_date_team=matches_by_date_team,
            key_map=key_map,
        )
        if normalized is None:
            skipped += 1
            continue
        rows.append(normalized)
    deduped = _dedupe_candidates(rows)
    logger.info(
        "FBref enrichment done competition=%s season=%s raw=%s mapped=%s deduped=%s skipped=%s",
        competition_code,
        season_start,
        len(frame),
        len(rows),
        len(deduped),
        skipped,
    )
    return deduped


def _extract_custom_http_player_match_candidates(
    *,
    competition_code: str,
    season_start: int,
    teams: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    token: str | None,
    base_url: str | None,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    if not base_url:
        logger.warning("custom_http enrichment skipped: PLAYER_STATS_BASE_URL is empty.")
        return []

    url = f"{base_url.rstrip('/')}/player-match-stats"
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-Key"] = token

    try:
        response = requests.get(
            url,
            params={"competition_code": competition_code, "season_start": int(season_start)},
            headers=headers,
            timeout=max(5, int(timeout_sec)),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.exception(
            "custom_http enrichment failed competition=%s season=%s url=%s",
            competition_code,
            season_start,
            url,
        )
        return []

    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, dict):
        raw_rows = (
            payload.get("data")
            or payload.get("results")
            or payload.get("items")
            or payload.get("player_stats")
            or []
        )
    else:
        raw_rows = []
    if not isinstance(raw_rows, list) or not raw_rows:
        return []

    team_id_by_key = _build_team_index(teams)
    matches_by_date_team = _build_matches_by_date_team(matches)
    key_map = {
        "match_id": "match_id",
        "date_id": "date_id",
        "team_id": "team_id",
        "team_name": "team_name",
        "player_id": "player_id",
        "player_name": "player_name",
        "position": "position",
        "is_starting": "is_starting",
        "minutes": "minutes",
        "goals": "goals",
        "assists": "assists",
        "shots": "shots",
        "passes": "passes",
        "pass_accuracy": "pass_accuracy",
    }
    aliases = {
        "match_id": ("match_id", "fixture_id", "game_id"),
        "date_id": ("date_id", "date", "match_date"),
        "team_id": ("team_id",),
        "team_name": ("team_name", "team", "squad"),
        "player_id": ("player_id",),
        "player_name": ("player_name", "player"),
        "position": ("position", "pos"),
        "is_starting": ("is_starting", "starts", "start"),
        "minutes": ("minutes", "min"),
        "goals": ("goals", "gls"),
        "assists": ("assists", "ast"),
        "shots": ("shots", "sh"),
        "passes": ("passes", "cmp"),
        "pass_accuracy": ("pass_accuracy", "pass_accuracy_pct", "cmp_pct"),
    }

    normalized_rows: list[dict[str, Any]] = []
    skipped = 0
    for item in raw_rows:
        if not isinstance(item, dict):
            skipped += 1
            continue
        source_row: dict[str, Any] = {}
        for canonical, keys in aliases.items():
            for key in keys:
                if key in item:
                    source_row[key_map[canonical]] = item.get(key)
                    break
        normalized = _normalize_candidate_row(
            source="custom_http",
            season_start=season_start,
            row=source_row,
            team_id_by_key=team_id_by_key,
            matches_by_date_team=matches_by_date_team,
            key_map=key_map,
        )
        if normalized is None:
            skipped += 1
            continue
        normalized_rows.append(normalized)

    deduped = _dedupe_candidates(normalized_rows)
    logger.info(
        "custom_http enrichment done competition=%s season=%s raw=%s mapped=%s deduped=%s skipped=%s",
        competition_code,
        season_start,
        len(raw_rows),
        len(normalized_rows),
        len(deduped),
        skipped,
    )
    return deduped


def fetch_player_match_candidates(
    *,
    provider: str,
    competition_code: str,
    season_start: int,
    teams: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    token: str | None = None,
    base_url: str | None = None,
    timeout_sec: int = 30,
) -> list[dict[str, Any]]:
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider == "fbref":
        return _extract_fbref_player_match_candidates(
            competition_code=competition_code,
            season_start=season_start,
            teams=teams,
            matches=matches,
        )
    if normalized_provider == "custom_http":
        return _extract_custom_http_player_match_candidates(
            competition_code=competition_code,
            season_start=season_start,
            teams=teams,
            matches=matches,
            token=token,
            base_url=base_url,
            timeout_sec=timeout_sec,
        )
    logger.warning("Unknown player stats provider=%s, enrichment skipped.", provider)
    return []
