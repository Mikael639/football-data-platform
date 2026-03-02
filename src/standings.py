from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.logger import get_logger

logger = get_logger("standings")


@dataclass(frozen=True)
class StandingsComputationResult:
    rows: list[dict[str, Any]]
    scopes: list[tuple[int, int]]
    ignored_null_matchday_count: int


def _season_start_from_match_row(row: pd.Series) -> int | None:
    season = row.get("season")
    if season is not None and str(season).strip():
        text_value = str(season).strip().split("-", 1)[0]
        try:
            return int(text_value)
        except ValueError:
            pass

    date_value = row.get("match_date")
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.year if parsed.month >= 7 else parsed.year - 1)


def _build_team_result_rows(matches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, match in matches_df.iterrows():
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        rows.append(
            {
                "competition_id": int(match["competition_id"]),
                "season": int(match["season_start"]),
                "matchday": int(match["matchday"]),
                "team_id": int(match["home_team_id"]),
                "team_name": str(match["home_team_name"]),
                "played_games": 1,
                "won": 1 if home_score > away_score else 0,
                "draw": 1 if home_score == away_score else 0,
                "lost": 1 if home_score < away_score else 0,
                "goals_for": home_score,
                "goals_against": away_score,
                "goal_difference": home_score - away_score,
                "points": 3 if home_score > away_score else 1 if home_score == away_score else 0,
                "snapshot_ts": match["kickoff_utc"],
            }
        )
        rows.append(
            {
                "competition_id": int(match["competition_id"]),
                "season": int(match["season_start"]),
                "matchday": int(match["matchday"]),
                "team_id": int(match["away_team_id"]),
                "team_name": str(match["away_team_name"]),
                "played_games": 1,
                "won": 1 if away_score > home_score else 0,
                "draw": 1 if away_score == home_score else 0,
                "lost": 1 if away_score < home_score else 0,
                "goals_for": away_score,
                "goals_against": home_score,
                "goal_difference": away_score - home_score,
                "points": 3 if away_score > home_score else 1 if away_score == home_score else 0,
                "snapshot_ts": match["kickoff_utc"],
            }
        )
    return pd.DataFrame(rows)


def build_standings_rows(matches_df: pd.DataFrame) -> StandingsComputationResult:
    if matches_df is None or matches_df.empty:
        return StandingsComputationResult(rows=[], scopes=[], ignored_null_matchday_count=0)

    working = matches_df.copy()
    working["season_start"] = working.apply(_season_start_from_match_row, axis=1)
    working["matchday"] = pd.to_numeric(working["matchday"], errors="coerce")
    working["home_score"] = pd.to_numeric(working["home_score"], errors="coerce")
    working["away_score"] = pd.to_numeric(working["away_score"], errors="coerce")
    working["kickoff_utc"] = pd.to_datetime(working["kickoff_utc"], errors="coerce", utc=True)
    working["match_date"] = pd.to_datetime(working["match_date"], errors="coerce")

    finished = working[
        working["status"].fillna("").eq("FINISHED")
        & working["home_score"].notna()
        & working["away_score"].notna()
        & working["season_start"].notna()
    ].copy()

    if finished.empty:
        return StandingsComputationResult(rows=[], scopes=[], ignored_null_matchday_count=0)

    ignored_null_matchday_count = int(finished["matchday"].isna().sum())
    if ignored_null_matchday_count:
        logger.warning("Ignoring %s finished matches for standings because matchday is NULL", ignored_null_matchday_count)
    finished = finished[finished["matchday"].notna()].copy()
    if finished.empty:
        return StandingsComputationResult(rows=[], scopes=[], ignored_null_matchday_count=ignored_null_matchday_count)

    team_results = _build_team_result_rows(finished)
    numeric_columns = [
        "played_games",
        "won",
        "draw",
        "lost",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
    ]

    aggregated = (
        team_results.groupby(
            ["competition_id", "season", "matchday", "team_id", "team_name"],
            as_index=False,
        )
        .agg(
            played_games=("played_games", "sum"),
            won=("won", "sum"),
            draw=("draw", "sum"),
            lost=("lost", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            goal_difference=("goal_difference", "sum"),
            points=("points", "sum"),
            snapshot_ts=("snapshot_ts", "max"),
        )
    )

    standings_rows: list[dict[str, Any]] = []
    scopes: list[tuple[int, int]] = []
    for (competition_id, season), scope_df in aggregated.groupby(["competition_id", "season"], sort=True):
        scopes.append((int(competition_id), int(season)))
        matchdays = sorted(scope_df["matchday"].astype(int).unique().tolist())
        teams = scope_df[["team_id", "team_name"]].drop_duplicates().sort_values(["team_name", "team_id"])
        team_names = {int(row["team_id"]): str(row["team_name"]) for _, row in teams.iterrows()}

        reindexed_rows: list[pd.DataFrame] = []
        for team_id, team_name in team_names.items():
            team_scope = scope_df[scope_df["team_id"] == team_id].set_index("matchday")
            team_scope = team_scope.reindex(matchdays).reset_index()
            team_scope["competition_id"] = int(competition_id)
            team_scope["season"] = int(season)
            team_scope["team_id"] = int(team_id)
            team_scope["team_name"] = team_name
            team_scope["snapshot_ts"] = pd.to_datetime(team_scope["snapshot_ts"], errors="coerce", utc=True)
            team_scope["snapshot_ts"] = team_scope["snapshot_ts"].ffill()
            for column in numeric_columns:
                team_scope[column] = pd.to_numeric(team_scope[column], errors="coerce").fillna(0).astype(int)
            reindexed_rows.append(team_scope)

        full_scope = pd.concat(reindexed_rows, ignore_index=True)
        full_scope = full_scope.sort_values(["team_id", "matchday"]).reset_index(drop=True)
        cumulative = full_scope.groupby("team_id", as_index=False, sort=False)[numeric_columns].cumsum()
        full_scope[numeric_columns] = cumulative

        for matchday in matchdays:
            matchday_rows = full_scope[full_scope["matchday"] == matchday].copy()
            matchday_rows = matchday_rows.sort_values(
                ["points", "goal_difference", "goals_for", "team_name"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            matchday_rows["position"] = matchday_rows.index + 1
            snapshot_ts = pd.to_datetime(matchday_rows["snapshot_ts"], errors="coerce", utc=True).max()
            if pd.isna(snapshot_ts):
                snapshot_ts = pd.Timestamp(datetime.now(timezone.utc))

            for _, row in matchday_rows.iterrows():
                standings_rows.append(
                    {
                        "competition_id": int(row["competition_id"]),
                        "season": int(row["season"]),
                        "matchday": int(row["matchday"]),
                        "team_id": int(row["team_id"]),
                        "position": int(row["position"]),
                        "points": int(row["points"]),
                        "played_games": int(row["played_games"]),
                        "won": int(row["won"]),
                        "draw": int(row["draw"]),
                        "lost": int(row["lost"]),
                        "goals_for": int(row["goals_for"]),
                        "goals_against": int(row["goals_against"]),
                        "goal_difference": int(row["goal_difference"]),
                        "snapshot_ts": snapshot_ts.to_pydatetime(),
                    }
                )

    return StandingsComputationResult(
        rows=standings_rows,
        scopes=scopes,
        ignored_null_matchday_count=ignored_null_matchday_count,
    )


def load_matches_for_standings(
    engine: Engine,
    scopes: list[tuple[int, str | None]] | None = None,
) -> pd.DataFrame:
    conditions = [
        "m.status = 'FINISHED'",
        "m.home_score IS NOT NULL",
        "m.away_score IS NOT NULL",
    ]
    params: dict[str, Any] = {}
    if scopes:
        scope_clauses: list[str] = []
        for index, (competition_id, season_label) in enumerate(scopes):
            comp_key = f"competition_id_{index}"
            params[comp_key] = int(competition_id)
            if season_label:
                season_key = f"season_{index}"
                params[season_key] = str(season_label)
                scope_clauses.append(f"(m.competition_id = :{comp_key} AND m.season = :{season_key})")
            else:
                scope_clauses.append(f"(m.competition_id = :{comp_key})")
        conditions.append(f"({' OR '.join(scope_clauses)})")

    query = f"""
    SELECT
      m.match_id,
      m.competition_id,
      m.season,
      COALESCE((m.kickoff_utc AT TIME ZONE 'UTC')::date, m.date_id) AS match_date,
      m.kickoff_utc,
      m.status,
      m.matchday,
      m.home_team_id,
      ht.team_name AS home_team_name,
      m.away_team_id,
      at.team_name AS away_team_name,
      m.home_score,
      m.away_score
    FROM fact_match m
    JOIN dim_team ht ON ht.team_id = m.home_team_id
    JOIN dim_team at ON at.team_id = m.away_team_id
    WHERE {' AND '.join(conditions)}
    ORDER BY m.competition_id, m.season, m.matchday NULLS LAST, match_date, m.match_id
    """
    return pd.read_sql(text(query), engine, params=params)


def compute_standings_snapshot(
    engine: Engine,
    scopes: list[tuple[int, str | None]] | None = None,
) -> StandingsComputationResult:
    matches_df = load_matches_for_standings(engine, scopes=scopes)
    return build_standings_rows(matches_df)
