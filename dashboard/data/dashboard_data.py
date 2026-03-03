from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.config import get_settings
from src.utils.db import get_engine as build_engine

MATCH_DATE_EXPR = "COALESCE((m.kickoff_utc AT TIME ZONE 'UTC')::date, m.date_id)"
SEASON_START_EXPR = (
    f"CASE WHEN EXTRACT(MONTH FROM {MATCH_DATE_EXPR}) >= 7 "
    f"THEN EXTRACT(YEAR FROM {MATCH_DATE_EXPR})::int "
    f"ELSE EXTRACT(YEAR FROM {MATCH_DATE_EXPR})::int - 1 END"
)
DEFAULT_CACHE_TTL = 300
KNOWN_COMPETITION_NAMES = {
    2002: "Bundesliga",
    2014: "Primera Division",
    2015: "Ligue 1",
    2019: "Serie A",
    2021: "Premier League",
}

TEAM_NAME_EQUIVALENTS = {
    "fc barcelona": "barcelona",
    "barcelone": "barcelona",
    "real madrid cf": "real madrid",
    "club atletico de madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "atletico madrid": "atletico madrid",
    "atletico": "atletico madrid",
    "ca osasuna": "osasuna",
    "real sociedad de futbol": "real sociedad",
    "valencia cf": "valencia",
    "villarreal cf": "villarreal",
    "sevilla fc": "sevilla",
    "real betis balompie": "real betis",
    "rayo vallecano de madrid": "rayo vallecano",
    "deportivo alaves": "alaves",
    "rcd espanyol de barcelona": "espanyol",
    "rcd mallorca": "mallorca",
    "ud las palmas": "las palmas",
    "real valladolid cf": "valladolid",
    "elche cf": "elche",
    "girona fc": "girona",
    "getafe cf": "getafe",
    "cd leganes": "leganes",
    "levante ud": "levante",
    "rc celta de vigo": "celta vigo",
}

TEAM_NAME_PREFERRED_LABELS = {
    "alaves": "Alaves",
    "atletico madrid": "Atletico Madrid",
    "barcelona": "Barcelona",
    "celta vigo": "Celta Vigo",
    "elche": "Elche",
    "espanyol": "Espanyol",
    "getafe": "Getafe",
    "girona": "Girona",
    "las palmas": "Las Palmas",
    "leganes": "Leganes",
    "mallorca": "Mallorca",
    "osasuna": "Osasuna",
    "rayo vallecano": "Rayo Vallecano",
    "real betis": "Real Betis",
    "real madrid": "Real Madrid",
    "real sociedad": "Real Sociedad",
    "sevilla": "Sevilla",
    "valencia": "Valencia",
    "valladolid": "Valladolid",
    "villarreal": "Villarreal",
}


@dataclass(frozen=True)
class DashboardFilters:
    competition_id: int | None
    season: str | None
    team_id: int | None
    date_start: str | None
    date_end: str | None


def get_engine():
    return build_engine(settings=get_settings())


def _competition_name_sql(alias: str = "competition_id") -> str:
    cases = " ".join(
        f"WHEN {alias} = {competition_id} THEN '{competition_name}'"
        for competition_id, competition_name in sorted(KNOWN_COMPETITION_NAMES.items())
    )
    return f"CASE {cases} ELSE CAST({alias} AS TEXT) END"


def _normalize_competition_display_name(value: Any) -> str:
    text_value = str(value or "").strip()
    if text_value in {"Primera Division", "Primera División", "PD"}:
        return "LaLiga"
    return text_value


def enrich_live_enabled() -> bool:
    value = os.getenv("ENRICH_LIVE", "false").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def current_season_start_year_dash() -> int:
    today = datetime.utcnow().date()
    return today.year if today.month >= 7 else today.year - 1


def current_season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def current_season_bounds(start_year: int) -> tuple[str, str]:
    season_start = date(start_year, 7, 1)
    season_end = date(start_year + 1, 6, 30)
    return season_start.isoformat(), season_end.isoformat()


def _normalize_team_alias(name: str | None) -> str:
    text_value = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    text_value = re.sub(r"[^a-zA-Z0-9 ]+", " ", text_value).lower()
    text_value = " ".join(text_value.split())
    return TEAM_NAME_EQUIVALENTS.get(text_value, text_value)


def _choose_team_display_row(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["crest_rank"] = ranked["crest_url"].fillna("").ne("").astype(int)
    ranked["name_length"] = ranked["team_name"].astype(str).str.len()
    ranked["short_name_rank"] = ranked["short_name"].fillna("").ne("").astype(int)
    ranked = ranked.sort_values(
        ["crest_rank", "short_name_rank", "name_length", "team_name", "team_id"],
        ascending=[False, False, True, True, True],
    )
    return ranked.iloc[0]


def _preferred_team_label(canonical_key: str, alias_names: list[str], fallback_name: str) -> str:
    preferred = TEAM_NAME_PREFERRED_LABELS.get(canonical_key)
    if preferred:
        return preferred

    cleaned_aliases = sorted({str(value).strip() for value in alias_names if str(value).strip()}, key=len)
    if cleaned_aliases:
        return cleaned_aliases[0]
    return fallback_name


def _group_team_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["team_id", "team_name", "short_name", "crest_url", "alias_team_ids", "alias_names"])

    working = df.copy()
    working["canonical_team_key"] = working["team_name"].map(_normalize_team_alias)
    grouped_rows: list[dict[str, Any]] = []
    for _, group in working.groupby("canonical_team_key", sort=True):
        display_row = _choose_team_display_row(group)
        alias_ids = sorted(group["team_id"].astype(int).unique().tolist())
        alias_names = sorted(group["team_name"].astype(str).unique().tolist())
        canonical_key = str(display_row["canonical_team_key"])
        grouped_rows.append(
            {
                "team_id": int(display_row["team_id"]),
                "team_name": _preferred_team_label(canonical_key, alias_names, str(display_row["team_name"])),
                "short_name": display_row["short_name"],
                "crest_url": display_row["crest_url"],
                "alias_team_ids": alias_ids,
                "alias_names": alias_names,
                "canonical_team_key": canonical_key,
            }
        )
    return pd.DataFrame(grouped_rows).sort_values("team_name").reset_index(drop=True)


def _build_sql_in_clause(values: list[int], prefix: str, params: dict[str, Any]) -> str:
    placeholders: list[str] = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = int(value)
        placeholders.append(f":{key}")
    return ", ".join(placeholders)


def describe_season_source(season: str | None) -> str:
    if season in {None, "", "Toutes"}:
        return "Source: historique charge + saison courante live"
    return (
        "Source probable: flux live de la saison courante (football-data.org)"
        if season == current_season_label(current_season_start_year_dash())
        else "Source probable: historique consolide"
    )


def _season_start_from_label(season: str | None) -> int | None:
    if season in {None, "", "Toutes"}:
        return None
    text_value = str(season).strip()
    if "-" in text_value:
        text_value = text_value.split("-", 1)[0]
    try:
        return int(text_value)
    except ValueError:
        return None


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def fact_match_has_season_column() -> bool:
    query = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'fact_match'
      AND column_name = 'season'
    LIMIT 1
    """
    return not _read_sql(query).empty


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def fact_match_has_non_null_seasons(competition_id: int | None = None) -> bool:
    if not fact_match_has_season_column():
        return False
    query = """
    SELECT 1
    FROM fact_match
    WHERE (:competition_id IS NULL OR competition_id = :competition_id)
      AND NULLIF(TRIM(season), '') IS NOT NULL
    LIMIT 1
    """
    return not _read_sql(query, {"competition_id": competition_id}).empty


def build_match_where_clause(filters: DashboardFilters) -> tuple[str, dict[str, Any]]:
    clauses = ["1 = 1"]
    params: dict[str, Any] = {}

    if filters.competition_id is not None:
        clauses.append("m.competition_id = :competition_id")
        params["competition_id"] = int(filters.competition_id)
    if filters.season is not None and fact_match_has_season_column():
        clauses.append("m.season = :season")
        params["season"] = str(filters.season)
    elif fact_match_has_non_null_seasons(filters.competition_id):
        clauses.append("NULLIF(TRIM(m.season), '') IS NOT NULL")
    if filters.team_id is not None:
        team_ids = get_team_alias_ids(filters.team_id, filters.competition_id, filters.season)
        in_clause = _build_sql_in_clause(team_ids, "team_id", params)
        clauses.append(f"(m.home_team_id IN ({in_clause}) OR m.away_team_id IN ({in_clause}))")
    if filters.date_start is not None:
        clauses.append(f"{MATCH_DATE_EXPR} >= :date_start")
        params["date_start"] = filters.date_start
    if filters.date_end is not None:
        clauses.append(f"{MATCH_DATE_EXPR} <= :date_end")
        params["date_end"] = filters.date_end

    return " AND ".join(clauses), params


def build_perspective_table(df_matches: pd.DataFrame, team_id: int | None = None) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "team_name",
                "opponent_id",
                "opponent_name",
                "date_dt",
                "kickoff_utc",
                "match_id",
                "venue",
                "status",
                "matchday",
                "goals_for",
                "goals_against",
                "result",
                "points",
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, row in df_matches.iterrows():
        views = []
        if team_id is None or int(row["home_team_id"]) == int(team_id):
            views.append(
                {
                    "team_id": int(row["home_team_id"]),
                    "team_name": row["home_team"],
                    "opponent_id": int(row["away_team_id"]),
                    "opponent_name": row["away_team"],
                    "venue": "Home",
                    "goals_for": row["home_score"],
                    "goals_against": row["away_score"],
                }
            )
        if team_id is None or int(row["away_team_id"]) == int(team_id):
            views.append(
                {
                    "team_id": int(row["away_team_id"]),
                    "team_name": row["away_team"],
                    "opponent_id": int(row["home_team_id"]),
                    "opponent_name": row["home_team"],
                    "venue": "Away",
                    "goals_for": row["away_score"],
                    "goals_against": row["home_score"],
                }
            )

        for view in views:
            goals_for = view["goals_for"]
            goals_against = view["goals_against"]
            if pd.isna(goals_for) or pd.isna(goals_against):
                result = None
                points = 0
            elif int(goals_for) > int(goals_against):
                result = "W"
                points = 3
            elif int(goals_for) == int(goals_against):
                result = "D"
                points = 1
            else:
                result = "L"
                points = 0

            rows.append(
                {
                    **view,
                    "date_dt": row["date_dt"],
                    "kickoff_utc": row["kickoff_utc"],
                    "match_id": row["match_id"],
                    "status": row["status"],
                    "matchday": row["matchday"],
                    "result": result,
                    "points": points,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["date_dt", "match_id"]).reset_index(drop=True)


def compute_team_kpis(df: pd.DataFrame, team_id: int):
    played = wins = draws = losses = gf = ga = 0
    for _, row in df.iterrows():
        hs = row["home_score"]
        aw = row["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        played += 1
        if int(row["home_team_id"]) == team_id:
            gf += int(hs)
            ga += int(aw)
            if hs > aw:
                wins += 1
            elif hs == aw:
                draws += 1
            else:
                losses += 1
        else:
            gf += int(aw)
            ga += int(hs)
            if aw > hs:
                wins += 1
            elif aw == hs:
                draws += 1
            else:
                losses += 1

    points = wins * 3 + draws
    return played, wins, draws, losses, gf, ga, points


def build_team_match_view(df_matches: pd.DataFrame, team_id: int) -> pd.DataFrame:
    perspective = build_perspective_table(df_matches, team_id=team_id)
    if perspective.empty:
        return pd.DataFrame(columns=["date_dt", "match_id", "venue", "opponent", "GF", "GA", "Result", "Points"])

    played = perspective.dropna(subset=["result"]).copy()
    if played.empty:
        return pd.DataFrame(columns=["date_dt", "match_id", "venue", "opponent", "GF", "GA", "Result", "Points"])

    played = played.rename(
        columns={
            "opponent_name": "opponent",
            "goals_for": "GF",
            "goals_against": "GA",
            "result": "Result",
            "points": "Points",
        }
    )
    played["venue"] = played["venue"].map({"Home": "Domicile", "Away": "Exterieur"})
    played["GoalsTotal"] = played["GF"] + played["GA"]
    played["CumulativePoints"] = played["Points"].cumsum()
    return played[
        ["date_dt", "match_id", "venue", "opponent", "GF", "GA", "Result", "Points", "GoalsTotal", "CumulativePoints"]
    ]


def build_local_league_table(df_matches: pd.DataFrame) -> pd.DataFrame:
    played = df_matches.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        return pd.DataFrame(columns=["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"])

    table: dict[str, dict[str, int]] = {}

    def ensure(team_name: str):
        if team_name not in table:
            table[team_name] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
        return table[team_name]

    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aw = int(row["away_score"])

        h = ensure(home)
        a = ensure(away)
        h["P"] += 1
        a["P"] += 1
        h["GF"] += hs
        h["GA"] += aw
        a["GF"] += aw
        a["GA"] += hs

        if hs > aw:
            h["W"] += 1
            h["Pts"] += 3
            a["L"] += 1
        elif hs < aw:
            a["W"] += 1
            a["Pts"] += 3
            h["L"] += 1
        else:
            h["D"] += 1
            a["D"] += 1
            h["Pts"] += 1
            a["Pts"] += 1

    out = [{"Team": team, **stats, "GD": stats["GF"] - stats["GA"]} for team, stats in table.items()]
    return pd.DataFrame(out).sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True]).reset_index(drop=True)


def _read_sql(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(text(query), engine, params=params or {})


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_competitions() -> pd.DataFrame:
    query = f"""
    WITH known_competitions AS (
      SELECT DISTINCT m.competition_id
      FROM fact_match m
      WHERE m.competition_id IS NOT NULL
    )
    SELECT
      kc.competition_id,
      COALESCE(c.competition_name, {_competition_name_sql("kc.competition_id")}) AS competition_name
    FROM known_competitions kc
    LEFT JOIN dim_competition c ON c.competition_id = kc.competition_id
    ORDER BY competition_name
    """
    df = _read_sql(query)
    if df.empty:
        settings = get_settings()
        return pd.DataFrame(
            [{"competition_id": None, "competition_name": _normalize_competition_display_name(settings.competition_code or "Competition")}]
        )
    df["competition_name"] = df["competition_name"].map(_normalize_competition_display_name)
    return df


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_seasons(competition_id: int | None = None) -> pd.DataFrame:
    if fact_match_has_season_column():
        query = """
        SELECT DISTINCT NULLIF(TRIM(m.season), '') AS season
        FROM fact_match m
        WHERE (:competition_id IS NULL OR m.competition_id = :competition_id)
          AND NULLIF(TRIM(m.season), '') IS NOT NULL
        ORDER BY season DESC
        """
        df = _read_sql(query, {"competition_id": competition_id})
        if not df.empty:
            return df
    return pd.DataFrame(columns=["season"])


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_team_alias_groups(competition_id: int | None = None, season: str | None = None) -> pd.DataFrame:
    where_clause, params = build_match_where_clause(
        DashboardFilters(
            competition_id=competition_id,
            season=season,
            team_id=None,
            date_start=None,
            date_end=None,
        )
    )
    scoped_query = f"""
    WITH scoped_matches AS (
        SELECT DISTINCT home_team_id AS team_id FROM fact_match m WHERE {where_clause}
        UNION
        SELECT DISTINCT away_team_id AS team_id FROM fact_match m WHERE {where_clause}
    )
    SELECT t.team_id, t.team_name, t.short_name, t.crest_url
    FROM dim_team t
    JOIN scoped_matches s ON s.team_id = t.team_id
    ORDER BY t.team_name
    """
    scoped_df = _read_sql(scoped_query, params)
    if not scoped_df.empty:
        return _group_team_aliases(scoped_df)
    fallback = _read_sql("SELECT team_id, team_name, short_name, crest_url FROM dim_team ORDER BY team_name")
    return _group_team_aliases(fallback)


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_teams(competition_id: int | None = None, season: str | None = None) -> pd.DataFrame:
    return get_team_alias_groups(competition_id, season)


def get_team_alias_ids(team_id: int, competition_id: int | None = None, season: str | None = None) -> list[int]:
    try:
        groups = get_team_alias_groups(competition_id, season)
    except Exception:
        return [int(team_id)]
    if groups.empty:
        return [int(team_id)]
    matched = groups[groups["team_id"].astype(int) == int(team_id)]
    if matched.empty:
        return [int(team_id)]
    alias_ids = matched.iloc[0]["alias_team_ids"]
    return [int(value) for value in alias_ids] if isinstance(alias_ids, list) else [int(team_id)]


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_date_bounds(
    competition_id: int | None = None,
    season: str | None = None,
    team_id: int | None = None,
) -> dict[str, Any]:
    where_clause, params = build_match_where_clause(
        DashboardFilters(
            competition_id=competition_id,
            season=season,
            team_id=team_id,
            date_start=None,
            date_end=None,
        )
    )
    query = f"""
    SELECT
      MIN({MATCH_DATE_EXPR}) AS min_date,
      MAX({MATCH_DATE_EXPR}) AS max_date,
      MAX(CASE WHEN m.kickoff_utc IS NOT NULL THEN 1 ELSE 0 END) AS has_kickoff
    FROM fact_match m
    WHERE {where_clause}
    """
    df = _read_sql(query, params)
    if df.empty or pd.isna(df.iloc[0]["min_date"]) or pd.isna(df.iloc[0]["max_date"]):
        season_start = _season_start_from_label(season)
        if season_start is not None:
            season_start_date, season_end_date = current_season_bounds(season_start)
            return {
                "min_date": season_start_date,
                "max_date": season_end_date,
                "has_kickoff": False,
            }
        return {"min_date": None, "max_date": None, "has_kickoff": False}
    row = df.iloc[0]
    return {
        "min_date": str(row["min_date"]),
        "max_date": str(row["max_date"]),
        "has_kickoff": bool(row["has_kickoff"]),
    }


def get_default_date_range(bounds: dict[str, Any]) -> tuple[str | None, str | None]:
    min_date = bounds.get("min_date")
    max_date = bounds.get("max_date")
    if not min_date or not max_date:
        return None, None
    if not bounds.get("has_kickoff"):
        return min_date, max_date

    max_dt = datetime.fromisoformat(str(max_date)).date()
    min_dt = datetime.fromisoformat(str(min_date)).date()
    start_dt = max(min_dt, max_dt - timedelta(days=29))
    return start_dt.isoformat(), max_dt.isoformat()


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_matches(
    competition_id: int | None = None,
    season: str | None = None,
    team_id: int | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    season_expr = (
        f"COALESCE(NULLIF(TRIM(m.season), ''), CONCAT({SEASON_START_EXPR}, '-', {SEASON_START_EXPR} + 1))"
        if fact_match_has_season_column()
        else f"CONCAT({SEASON_START_EXPR}, '-', {SEASON_START_EXPR} + 1)"
    )
    where_clause, params = build_match_where_clause(
        DashboardFilters(
            competition_id=competition_id,
            season=season,
            team_id=team_id,
            date_start=date_start,
            date_end=date_end,
        )
    )
    query = f"""
    SELECT
      m.match_id,
      m.competition_id,
      COALESCE(c.competition_name, {_competition_name_sql("m.competition_id")}) AS competition_name,
      {season_expr} AS season,
      {MATCH_DATE_EXPR} AS match_date,
      m.date_id,
      m.kickoff_utc,
      m.status,
      m.matchday,
      m.home_team_id,
      ht.team_name AS home_team,
      ht.short_name AS home_short_name,
      ht.crest_url AS home_crest_url,
      m.away_team_id,
      at.team_name AS away_team,
      at.short_name AS away_short_name,
      at.crest_url AS away_crest_url,
      m.home_score,
      m.away_score
    FROM fact_match m
    JOIN dim_team ht ON ht.team_id = m.home_team_id
    JOIN dim_team at ON at.team_id = m.away_team_id
    LEFT JOIN dim_competition c ON c.competition_id = m.competition_id
    WHERE {where_clause}
    ORDER BY COALESCE(m.kickoff_utc, m.date_id::timestamp) DESC NULLS LAST, m.match_id DESC
    """
    df = _read_sql(query, params)
    if df.empty:
        return df
    df["competition_name"] = df["competition_name"].map(_normalize_competition_display_name)
    df["date_dt"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    return df


def get_kpis(
    competition_id: int | None = None,
    season: str | None = None,
    team_id: int | None = None,
    date_range: tuple[str | None, str | None] | None = None,
) -> dict[str, Any]:
    date_start, date_end = date_range or (None, None)
    matches = get_matches(competition_id, season, team_id, date_start, date_end)
    perspective = build_perspective_table(matches, team_id=team_id)
    played = perspective.dropna(subset=["result"]).copy()

    if played.empty:
        return {
            "matches": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "win_rate": None,
            "points": 0,
        }

    wins = int((played["result"] == "W").sum())
    goals_for = int(played["goals_for"].sum())
    goals_against = int(played["goals_against"].sum())
    matches_played = int(len(played))
    return {
        "matches": matches_played,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "win_rate": round((wins / matches_played) * 100, 1) if matches_played > 0 else None,
        "points": int(played["points"].sum()),
    }


def split_recent_and_upcoming_matches(
    matches: pd.DataFrame,
    recent_limit: int = 10,
    upcoming_limit: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matches is None or matches.empty:
        return pd.DataFrame(), pd.DataFrame()

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    working = matches.copy()
    working["kickoff_sort"] = (
        pd.to_datetime(working["kickoff_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    )
    working["kickoff_sort"] = working["kickoff_sort"].fillna(pd.to_datetime(working["date_dt"], errors="coerce"))

    played_mask = working["status"].fillna("").eq("FINISHED") | (
        working["home_score"].notna() & working["away_score"].notna() & (working["kickoff_sort"] < today)
    )
    recent = working[played_mask].sort_values(["kickoff_sort", "match_id"], ascending=[False, False]).head(recent_limit)
    upcoming = working[~played_mask].sort_values(["kickoff_sort", "match_id"], ascending=[True, True]).head(upcoming_limit)
    return recent.reset_index(drop=True), upcoming.reset_index(drop=True)


def get_recent_matches(
    competition_id: int | None = None,
    season: str | None = None,
    team_id: int | None = None,
    date_range: tuple[str | None, str | None] | None = None,
    recent_limit: int = 10,
    upcoming_limit: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_start, date_end = date_range or (None, None)
    matches = get_matches(competition_id, season, team_id, date_start, date_end)
    return split_recent_and_upcoming_matches(matches, recent_limit=recent_limit, upcoming_limit=upcoming_limit)


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_current_standings(
    competition_id: int | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    season_start = _season_start_from_label(season)
    query = """
    WITH scoped AS (
      SELECT *
      FROM fact_standings_snapshot
      WHERE (:competition_id IS NULL OR competition_id = :competition_id)
        AND (:season_start IS NULL OR season = :season_start)
    ),
    latest_season AS (
      SELECT competition_id, MAX(season) AS season
      FROM scoped
      GROUP BY competition_id
    ),
    latest AS (
      SELECT s.competition_id, s.season, MAX(s.matchday) AS matchday
      FROM scoped s
      JOIN latest_season ls
        ON ls.competition_id = s.competition_id
       AND ls.season = s.season
      GROUP BY s.competition_id, s.season
    ),
    previous_matchday AS (
      SELECT s.competition_id, s.season, MAX(s.matchday) AS previous_matchday
      FROM scoped s
      JOIN latest l
        ON l.competition_id = s.competition_id
       AND l.season = s.season
      WHERE l.matchday IS NOT NULL
        AND s.matchday < l.matchday
      GROUP BY s.competition_id, s.season
    )
    SELECT
      s.competition_id,
      s.season,
      s.matchday,
      s.team_id,
      t.team_name,
      t.short_name,
      t.crest_url,
      s.position,
      s.points,
      s.played_games,
      s.won,
      s.draw,
      s.lost,
      s.goals_for,
      s.goals_against,
      s.goal_difference,
      prev.position AS previous_position,
      CASE
        WHEN prev.position IS NULL OR s.position IS NULL THEN 0
        ELSE prev.position - s.position
      END AS position_delta
    FROM scoped s
    JOIN latest l
      ON l.competition_id = s.competition_id
     AND l.season = s.season
     AND ((l.matchday IS NULL AND s.matchday IS NULL) OR l.matchday = s.matchday)
    LEFT JOIN previous_matchday pm
      ON pm.competition_id = s.competition_id
     AND pm.season = s.season
    LEFT JOIN scoped prev
      ON prev.competition_id = s.competition_id
     AND prev.season = s.season
     AND prev.team_id = s.team_id
     AND ((pm.previous_matchday IS NULL AND prev.matchday IS NULL) OR prev.matchday = pm.previous_matchday)
    JOIN dim_team t ON t.team_id = s.team_id
    ORDER BY s.position NULLS LAST, t.team_name
    """
    return _read_sql(query, {"competition_id": competition_id, "season_start": season_start})


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_standings_curve(
    competition_id: int | None = None,
    season: str | None = None,
    team_id: int | None = None,
) -> pd.DataFrame:
    season_start = _season_start_from_label(season)
    team_filter_clause = ""
    params = {"competition_id": competition_id, "season_start": season_start}
    if team_id is not None:
        team_ids = get_team_alias_ids(team_id, competition_id, season)
        in_clause = _build_sql_in_clause(team_ids, "curve_team_id", params)
        team_filter_clause = f" AND s.team_id IN ({in_clause})"
    query = """
    WITH scoped AS (
      SELECT *
      FROM fact_standings_snapshot
      WHERE (:competition_id IS NULL OR competition_id = :competition_id)
        AND (:season_start IS NULL OR season = :season_start)
    ),
    latest_season AS (
      SELECT competition_id, MAX(season) AS season
      FROM scoped
      GROUP BY competition_id
    )
    SELECT
      s.competition_id,
      s.season,
      s.matchday,
      s.team_id,
      t.team_name,
      t.short_name,
      t.crest_url,
      s.position,
      s.points,
      s.played_games,
      s.won,
      s.draw,
      s.lost,
      s.goals_for,
      s.goals_against,
      s.goal_difference
    FROM scoped s
    JOIN latest_season ls
      ON ls.competition_id = s.competition_id
     AND ls.season = s.season
    JOIN dim_team t ON t.team_id = s.team_id
    WHERE 1 = 1
    """ + team_filter_clause + """
    ORDER BY s.matchday, s.position, t.team_name
    """
    df = _read_sql(query, params)
    if df.empty or team_id is not None:
        return df

    latest_matchday = df["matchday"].dropna().max()
    if pd.isna(latest_matchday):
        return df.head(0)
    top_team_ids = (
        df[df["matchday"] == latest_matchday]
        .sort_values(["position", "team_name"])
        .head(5)["team_id"]
        .astype(int)
        .tolist()
    )
    return df[df["team_id"].isin(top_team_ids)].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_team_meta(team_id: int, competition_id: int | None = None, season: str | None = None) -> dict[str, Any] | None:
    groups = get_team_alias_groups(competition_id, season)
    if groups.empty:
        return None
    matched = groups[groups["team_id"].astype(int) == int(team_id)]
    if matched.empty:
        return None
    row = matched.iloc[0].to_dict()
    row["country"] = "Spain"
    return row


def get_home_away_split(
    competition_id: int | None = None,
    season: str | None = None,
    team_id: int | None = None,
    date_range: tuple[str | None, str | None] | None = None,
) -> pd.DataFrame:
    if team_id is None:
        return pd.DataFrame()
    date_start, date_end = date_range or (None, None)
    matches = get_matches(competition_id, season, team_id, date_start, date_end)
    perspective = build_perspective_table(matches, team_id=team_id)
    played = perspective.dropna(subset=["result"]).copy()
    if played.empty:
        return pd.DataFrame()

    split = (
        played.groupby("venue", as_index=False)
        .agg(
            Matches=("match_id", "count"),
            Points=("points", "sum"),
            GoalsFor=("goals_for", "sum"),
            GoalsAgainst=("goals_against", "sum"),
        )
        .sort_values("venue")
    )
    split["PPM"] = (split["Points"] / split["Matches"]).round(2)
    return split


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_players(team_id: int | None = None) -> pd.DataFrame:
    alias_clause = ""
    params: dict[str, Any] = {}
    if team_id is not None:
        team_ids = get_team_alias_ids(team_id)
        in_clause = _build_sql_in_clause(team_ids, "player_team_id", params)
        alias_clause = f" AND p.team_id IN ({in_clause})"
    query = """
    SELECT
      p.player_id,
      p.full_name,
      p.position,
      p.nationality,
      p.birth_date,
      p.team_id,
      t.team_name
    FROM dim_player p
    LEFT JOIN dim_team t ON t.team_id = p.team_id
    WHERE 1 = 1
    """ + alias_clause + """
    ORDER BY t.team_name, p.full_name
    """
    return _read_sql(query, params)


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_live_league_tables() -> dict[str, pd.DataFrame]:
    query = f"""
    WITH latest AS (
      SELECT s.competition_id, MAX(s.season) AS season
      FROM fact_standings_snapshot s
      GROUP BY s.competition_id
    ),
    latest_matchday AS (
      SELECT s.competition_id, s.season, MAX(s.matchday) AS matchday
      FROM fact_standings_snapshot s
      JOIN latest l
        ON l.competition_id = s.competition_id
       AND l.season = s.season
      GROUP BY s.competition_id, s.season
    ),
    previous_matchday AS (
      SELECT s.competition_id, s.season, MAX(s.matchday) AS previous_matchday
      FROM fact_standings_snapshot s
      JOIN latest_matchday lm
        ON lm.competition_id = s.competition_id
       AND lm.season = s.season
      WHERE lm.matchday IS NOT NULL
        AND s.matchday < lm.matchday
      GROUP BY s.competition_id, s.season
    )
    SELECT
      s.competition_id,
      COALESCE(c.competition_name, {_competition_name_sql("s.competition_id")}) AS competition_name,
      s.season,
      s.matchday,
      s.team_id,
      t.team_name,
      t.short_name,
      t.crest_url,
      s.position,
      s.points,
      s.played_games,
      s.won,
      s.draw,
      s.lost,
      s.goals_for,
      s.goals_against,
      s.goal_difference,
      prev.position AS previous_position,
      CASE
        WHEN prev.position IS NULL OR s.position IS NULL THEN 0
        ELSE prev.position - s.position
      END AS position_delta
    FROM fact_standings_snapshot s
    JOIN latest_matchday lm
      ON lm.competition_id = s.competition_id
     AND lm.season = s.season
     AND ((lm.matchday IS NULL AND s.matchday IS NULL) OR lm.matchday = s.matchday)
    LEFT JOIN previous_matchday pm
      ON pm.competition_id = s.competition_id
     AND pm.season = s.season
    LEFT JOIN fact_standings_snapshot prev
      ON prev.competition_id = s.competition_id
     AND prev.season = s.season
     AND prev.team_id = s.team_id
     AND ((pm.previous_matchday IS NULL AND prev.matchday IS NULL) OR prev.matchday = pm.previous_matchday)
    LEFT JOIN dim_competition c ON c.competition_id = s.competition_id
    JOIN dim_team t ON t.team_id = s.team_id
    ORDER BY competition_name, s.position, t.team_name
    """
    df = _read_sql(query)
    if df.empty:
        return {}
    df["competition_name"] = df["competition_name"].map(_normalize_competition_display_name)
    grouped: dict[str, pd.DataFrame] = {}
    for competition_name, group in df.groupby("competition_name", sort=True):
        grouped[str(competition_name)] = group.reset_index(drop=True)
    return grouped


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_live_league_form(competition_id: int, season_start: int, limit: int = 5) -> dict[int, list[str]]:
    season_label = current_season_label(int(season_start))
    query = """
    SELECT
      m.match_id,
      m.home_team_id,
      m.away_team_id,
      m.home_score,
      m.away_score,
      m.status,
      COALESCE(m.kickoff_utc, m.date_id::timestamp) AS match_ts
    FROM fact_match m
    WHERE m.competition_id = :competition_id
      AND m.season = :season
      AND m.status = 'FINISHED'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY match_ts DESC, m.match_id DESC
    """
    df = _read_sql(query, {"competition_id": int(competition_id), "season": season_label})
    if df.empty:
        return {}

    forms: dict[int, list[str]] = {}
    for _, row in df.iterrows():
        home_team_id = int(row["home_team_id"])
        away_team_id = int(row["away_team_id"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        if home_team_id not in forms:
            forms[home_team_id] = []
        if away_team_id not in forms:
            forms[away_team_id] = []

        if len(forms[home_team_id]) < limit:
            forms[home_team_id].append("W" if home_score > away_score else "D" if home_score == away_score else "L")
        if len(forms[away_team_id]) < limit:
            forms[away_team_id].append("W" if away_score > home_score else "D" if away_score == home_score else "L")

        if all(len(values) >= limit for values in forms.values()):
            continue

    return forms


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def fetch_laliga_teams_live(competition_code: str, season_start_year: int):
    competitions = get_competitions()
    competition_id = None
    if not competitions.empty:
        matched = competitions[competitions["competition_name"].astype(str).str.contains(competition_code, case=False, na=False)]
        if not matched.empty and matched.iloc[0]["competition_id"] is not None:
            competition_id = int(matched.iloc[0]["competition_id"])
    teams = get_teams(competition_id=competition_id, season=current_season_label(season_start_year))
    if teams.empty:
        return None, "No teams available in local database"
    return teams[["team_id", "team_name"]].copy(), None


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def fetch_live_team_squad(team_id: int):
    players = get_players(team_id)
    if players.empty:
        return pd.DataFrame(), "No players available in local database"
    return players, None


def upsert_players_to_db(engine_obj, players_df: pd.DataFrame) -> int:
    if players_df is None or players_df.empty:
        return 0

    rows = []
    for _, row in players_df.iterrows():
        rows.append(
            {
                "player_id": int(row["player_id"]),
                "full_name": row.get("full_name"),
                "position": row.get("position"),
                "nationality": row.get("nationality"),
                "birth_date": row.get("birth_date"),
                "team_id": int(row["team_id"]) if pd.notna(row.get("team_id")) else None,
            }
        )

    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO dim_player (player_id, full_name, position, nationality, birth_date, team_id)
                VALUES (:player_id, :full_name, :position, :nationality, :birth_date, :team_id)
                ON CONFLICT (player_id) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    position = EXCLUDED.position,
                    nationality = EXCLUDED.nationality,
                    birth_date = EXCLUDED.birth_date,
                    team_id = EXCLUDED.team_id
                """
            ),
            rows,
        )
    return len(rows)


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_pipeline_runs(limit: int = 20) -> pd.DataFrame:
    query = """
    SELECT run_id, started_at, ended_at, status, extracted_count, loaded_count, error_message, metrics_jsonb, volumes_jsonb
    FROM pipeline_run_log
    ORDER BY started_at DESC
    LIMIT :limit
    """
    df = _read_sql(query, {"limit": limit})
    if df.empty:
        return df

    def _parse_json(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None or value == "":
            return {}
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}

    df["metrics"] = df["metrics_jsonb"].map(_parse_json)
    df["volumes"] = df["volumes_jsonb"].map(_parse_json)
    df["duration_ms"] = df["metrics"].map(lambda item: item.get("total_duration_ms"))
    df["run_id"] = df["run_id"].astype(str)
    return df


@st.cache_data(show_spinner=False, ttl=DEFAULT_CACHE_TTL)
def get_dq_checks(run_id: str | None = None, limit: int = 200) -> pd.DataFrame:
    query = """
    SELECT run_id, check_name, status, severity, metric_value, threshold, details, created_at
    FROM data_quality_check
    WHERE (:run_id IS NULL OR run_id = CAST(:run_id AS UUID))
    ORDER BY created_at DESC
    LIMIT :limit
    """
    df = _read_sql(query, {"run_id": run_id, "limit": limit})
    if not df.empty and "run_id" in df.columns:
        df["run_id"] = df["run_id"].astype(str)
    return df
