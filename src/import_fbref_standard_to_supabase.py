from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.study_fbref import _normalize_text, _position_group, build_progression, season_label
from src.utils.logger import get_logger

logger = get_logger("fbref_standard_supabase")


RAW_PATTERN = re.compile(r"laliga_(\d{4})_(\d{4})_standard\.csv$", re.IGNORECASE)


def _parse_int(v: Any) -> int | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.replace("\xa0", " ").replace(",", "").strip()
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _parse_float(v: Any) -> float | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.replace("\xa0", " ").replace(",", "").strip()
    # Preserve decimal points; commas in these files are thousands separators.
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None


def _extract_nation_code(v: Any) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).replace("\xa0", " ").strip()
    if not s:
        return None
    parts = [p for p in s.split(" ") if p]
    if not parts:
        return None
    code = parts[-1].upper()
    return code if 2 <= len(code) <= 3 else None


def _read_fbref_standard_csv(path: Path) -> pd.DataFrame:
    season_match = RAW_PATTERN.search(path.name)
    if not season_match:
        raise ValueError(f"Nom de fichier inattendu (saison introuvable): {path.name}")
    season_start = int(season_match.group(1))

    df = pd.read_csv(path, sep=";", skiprows=1, dtype=str, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    # Drop duplicate header rows reinserted in the middle of some exports.
    if "RK" in df.columns:
        df = df[df["RK"].astype(str).str.strip().ne("RK")]
    if "Joueur" in df.columns:
        df = df[df["Joueur"].notna() & df["Joueur"].astype(str).str.strip().ne("")]
        df = df[df["Joueur"].astype(str).str.strip().ne("Joueur")]

    # Keep only rows with numeric rank.
    df["rk"] = df["RK"].apply(_parse_int)
    df = df[df["rk"].notna()].copy()
    df["season_start"] = season_start
    df["source_file"] = path.name

    # Raw text cleanup.
    for c in ["Joueur", "Nation", "Pos", "Effectif"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

    # Numeric conversions from French export.
    numeric_map = {
        "Âge": "age",
        "Né": "birth_year",
        "MP": "matches_played",
        "Débuts": "starts",
        "Min": "minutes_total",
        "Années 90": "nineties",
        "GLS": "goals_total",
        "Ast": "assists_total",
        "G+A": "ga_total",
        "G-PK": "goals_non_pk_total",
        "PK": "pk_goals",
        "PKatt": "pk_attempts",
        "CrdY": "yellow_cards",
        "CrdR": "red_cards",
        "GLS.1": "goals_p90",
        "Ast.1": "assists_p90",
        "G+A.1": "ga_p90",
        "G-PK.1": "goals_non_pk_p90",
        "G+A-PK": "ga_non_pk_p90",
    }
    for src, dst in numeric_map.items():
        if src in df.columns:
            parser = _parse_float if dst in {"nineties", "goals_p90", "assists_p90", "ga_p90", "goals_non_pk_p90", "ga_non_pk_p90"} else _parse_int
            df[dst] = df[src].apply(parser)
        else:
            df[dst] = None

    df["player_name"] = df["Joueur"].astype(str).str.strip()
    df["team_name"] = df["Effectif"].astype(str).str.strip()
    df["nation_raw"] = df.get("Nation")
    df["nation_code"] = df["nation_raw"].apply(_extract_nation_code)
    df["position_raw"] = df.get("Pos").fillna("").astype(str).str.upper()
    df["position_group"] = df["position_raw"].apply(_position_group)

    # Stable player identity across seasons: use name + birth year (better than name+team for transfers).
    df["player_key"] = (
        df["player_name"].map(_normalize_text)
        + "|"
        + df["birth_year"].fillna(0).astype(int).astype(str)
    )

    keep_cols = [
        "season_start",
        "source_file",
        "rk",
        "player_name",
        "nation_raw",
        "nation_code",
        "position_raw",
        "position_group",
        "team_name",
        "age",
        "birth_year",
        "matches_played",
        "starts",
        "minutes_total",
        "nineties",
        "goals_total",
        "assists_total",
        "ga_total",
        "goals_non_pk_total",
        "pk_goals",
        "pk_attempts",
        "yellow_cards",
        "red_cards",
        "goals_p90",
        "assists_p90",
        "ga_p90",
        "goals_non_pk_p90",
        "ga_non_pk_p90",
        "player_key",
    ]
    return df[keep_cols].reset_index(drop=True)


def _build_player_season_from_standard(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    out = raw_df.copy()
    # Stable IDs across all seasons.
    player_keys = sorted(out["player_key"].dropna().unique().tolist())
    player_id_map = {k: i + 1 for i, k in enumerate(player_keys)}
    team_names = sorted(out["team_name"].fillna("").astype(str).unique().tolist())
    team_id_map = {k: i + 1 for i, k in enumerate(team_names)}

    out["player_id"] = out["player_key"].map(player_id_map).astype(int)
    out["team_id"] = out["team_name"].map(team_id_map).astype(int)
    out["starts"] = out["starts"].fillna(0).astype(int)
    out["matches_played"] = out["matches_played"].fillna(0).astype(int)
    out["sub_apps"] = (out["matches_played"] - out["starts"]).clip(lower=0).astype(int)
    out["minutes_total"] = out["minutes_total"].fillna(0).astype(int)

    for c in [
        "goals_total",
        "assists_total",
        "ga_total",
        "goals_non_pk_total",
        "pk_goals",
        "pk_attempts",
        "yellow_cards",
        "red_cards",
        "goals_p90",
        "assists_p90",
        "ga_p90",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # Not available in standard season export used here (kept for schema/dashboard compatibility).
    out["shots_total"] = 0
    out["passes_total"] = 0
    out["pass_acc_mean"] = 0.0
    out["shots_p90"] = 0.0
    out["passes_p90"] = 0.0
    out["goals_non_pk_total"] = out["goals_non_pk_total"].fillna(0).astype(int)
    out["pk_goals_total"] = out["pk_goals"].fillna(0).astype(int)
    out["pk_attempts_total"] = out["pk_attempts"].fillna(0).astype(int)
    out["yellow_cards_total"] = out["yellow_cards"].fillna(0).astype(int)
    out["red_cards_total"] = out["red_cards"].fillna(0).astype(int)
    out["eligible_600"] = out["minutes_total"] >= 600
    out["eligible_900"] = out["minutes_total"] >= 900
    out["source_mode"] = "manual_standard_csv"
    out["clubs_count"] = 1
    out["clubs_list"] = out["team_name"].fillna("").astype(str)
    out["is_multi_club_season"] = False

    cols = [
        "season_start",
        "player_id",
        "player_key",
        "player_name",
        "team_id",
        "team_name",
        "position_group",
        "position_raw",
        "clubs_count",
        "clubs_list",
        "is_multi_club_season",
        "nation_code",
        "age",
        "birth_year",
        "matches_played",
        "starts",
        "sub_apps",
        "minutes_total",
        "goals_total",
        "assists_total",
        "ga_total",
        "goals_non_pk_total",
        "pk_goals_total",
        "pk_attempts_total",
        "yellow_cards_total",
        "red_cards_total",
        "shots_total",
        "passes_total",
        "pass_acc_mean",
        "goals_p90",
        "assists_p90",
        "ga_p90",
        "shots_p90",
        "passes_p90",
        "eligible_600",
        "eligible_900",
        "source_mode",
        "source_file",
    ]
    out = out[cols].copy()

    # Some players have multiple rows in the same season (transfers / multiple clubs).
    # Aggregate to one row per player-season and keep the club with the most minutes as the display club.
    if out.duplicated(subset=["season_start", "player_id"]).any():
        out = out.sort_values(["season_start", "player_id", "minutes_total"], ascending=[True, True, False]).copy()
        primary_team = out.drop_duplicates(subset=["season_start", "player_id"], keep="first")[
            ["season_start", "player_id", "team_id", "team_name", "position_group", "position_raw", "nation_code", "age", "birth_year", "source_file"]
        ].copy()

        def _pick_first(series: pd.Series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else None

        grouped = (
            out.groupby(["season_start", "player_id", "player_key", "player_name"], dropna=False)
            .agg(
                matches_played=("matches_played", "sum"),
                starts=("starts", "sum"),
                sub_apps=("sub_apps", "sum"),
                minutes_total=("minutes_total", "sum"),
                goals_total=("goals_total", "sum"),
                assists_total=("assists_total", "sum"),
                ga_total=("ga_total", "sum"),
                goals_non_pk_total=("goals_non_pk_total", "sum"),
                pk_goals_total=("pk_goals_total", "sum"),
                pk_attempts_total=("pk_attempts_total", "sum"),
                yellow_cards_total=("yellow_cards_total", "sum"),
                red_cards_total=("red_cards_total", "sum"),
                shots_total=("shots_total", "sum"),
                passes_total=("passes_total", "sum"),
                pass_acc_mean=("pass_acc_mean", "mean"),
            )
            .reset_index()
        )
        clubs_meta = (
            out.groupby(["season_start", "player_id"], dropna=False)
            .agg(
                clubs_count=("team_name", lambda s: int(pd.Series(s).dropna().astype(str).nunique())),
                clubs_list=("team_name", lambda s: " | ".join(sorted(pd.Series(s).dropna().astype(str).unique().tolist()))),
            )
            .reset_index()
        )
        clubs_meta["is_multi_club_season"] = clubs_meta["clubs_count"] > 1

        mins = grouped["minutes_total"].replace(0, np.nan)
        grouped["goals_p90"] = (grouped["goals_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
        grouped["assists_p90"] = (grouped["assists_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
        grouped["ga_p90"] = (grouped["ga_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
        grouped["shots_p90"] = (grouped["shots_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
        grouped["passes_p90"] = (grouped["passes_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
        grouped["eligible_600"] = grouped["minutes_total"] >= 600
        grouped["eligible_900"] = grouped["minutes_total"] >= 900
        grouped["source_mode"] = "manual_standard_csv"

        out = grouped.merge(primary_team, on=["season_start", "player_id"], how="left")
        out = out.merge(clubs_meta, on=["season_start", "player_id"], how="left")
        out = out[
            [
                "season_start",
                "player_id",
                "player_key",
                "player_name",
                "team_id",
                "team_name",
                "position_group",
                "position_raw",
                "clubs_count",
                "clubs_list",
                "is_multi_club_season",
                "nation_code",
                "age",
                "birth_year",
                "matches_played",
                "starts",
                "sub_apps",
                "minutes_total",
                "goals_total",
                "assists_total",
                "ga_total",
                "goals_non_pk_total",
                "pk_goals_total",
                "pk_attempts_total",
                "yellow_cards_total",
                "red_cards_total",
                "shots_total",
                "passes_total",
                "pass_acc_mean",
                "goals_p90",
                "assists_p90",
                "ga_p90",
                "shots_p90",
                "passes_p90",
                "eligible_600",
                "eligible_900",
                "source_mode",
                "source_file",
            ]
        ].copy()
    out["clubs_count"] = pd.to_numeric(out["clubs_count"], errors="coerce").fillna(1).astype(int)
    out["clubs_list"] = out["clubs_list"].fillna(out["team_name"]).astype(str)
    out["is_multi_club_season"] = out["is_multi_club_season"].fillna(False).astype(bool)

    return out


def _empty_regularity_frame() -> pd.DataFrame:
    cols = [
        "season_start",
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position_group",
        "minutes_total",
        "matches_played",
        "ga_p90_mean",
        "ga_p90_std",
        "shots_p90_mean",
        "shots_p90_std",
        "passes_p90_mean",
        "passes_p90_std",
        "ga_p90_cv",
        "shots_p90_cv",
        "passes_p90_cv",
        "stability_proxy",
        "perf_z",
        "stab_z",
        "regularity_score",
        "regularity_rank_pos",
        "podium",
        "source_mode",
        "note",
    ]
    return pd.DataFrame(columns=cols)


def _build_meta(player_season: pd.DataFrame, progression: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(player_season["season_start"].dropna().astype(int).unique().tolist())
    now = datetime.now(timezone.utc)
    meta = {
        "dataset_name": "fbref_study",
        "league": "La Liga",
        "source_mode": "manual_standard_csv",
        "seasons_start_years": json.dumps(seasons),
        "season_labels": json.dumps([season_label(s) for s in seasons]),
        "files": json.dumps(
            {
                "raw_standard_rows": int(len(raw_df)),
                "player_season_rows": int(len(player_season)),
                "progression_rows": int(len(progression)),
                "regularity_rows": 0,
            }
        ),
        "notes": "CSV FBref standard saison (agrege). Regularite match-par-match indisponible dans cette source.",
        # Multi-club seasons are aggregated to player-season totals; display club is the club with most minutes.
        "generated_at_utc": now,
        "updated_at_utc": now,
    }
    return pd.DataFrame([meta])


def _get_engine():
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("STUDY_SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL (ou STUDY_SUPABASE_DB_URL) est requis pour l'import Supabase.")
    return create_engine(db_url, pool_pre_ping=True)


def _apply_schema_if_requested(engine_obj) -> None:
    apply_schema = (os.getenv("SUPABASE_IMPORT_APPLY_SCHEMA") or "true").strip().lower() in {"1", "true", "yes", "y"}
    if not apply_schema:
        return
    sql_path = Path(os.getenv("SUPABASE_IMPORT_SCHEMA_SQL") or "sql/05_supabase_fbref_study.sql")
    if not sql_path.exists():
        logger.warning("Schema SQL introuvable: %s (skip)", sql_path)
        return
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
    with engine_obj.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("Schema SQL applique: %s", sql_path)


def _append_table(engine_obj, table_name: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.copy().to_sql(table_name, engine_obj, schema="public", if_exists="append", index=False, chunksize=1000, method="multi")


def _upsert_meta(engine_obj, meta_df: pd.DataFrame) -> None:
    if meta_df is None or meta_df.empty:
        return
    row = meta_df.iloc[0].to_dict()
    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO public.study_fbref_meta
                (dataset_name, league, source_mode, seasons_start_years, season_labels, files, notes, generated_at_utc, updated_at_utc)
                VALUES
                (:dataset_name, :league, :source_mode, CAST(:seasons_start_years AS jsonb), CAST(:season_labels AS jsonb),
                 CAST(:files AS jsonb), :notes, :generated_at_utc, :updated_at_utc)
                ON CONFLICT (dataset_name) DO UPDATE SET
                    league = EXCLUDED.league,
                    source_mode = EXCLUDED.source_mode,
                    seasons_start_years = EXCLUDED.seasons_start_years,
                    season_labels = EXCLUDED.season_labels,
                    files = EXCLUDED.files,
                    notes = EXCLUDED.notes,
                    generated_at_utc = EXCLUDED.generated_at_utc,
                    updated_at_utc = EXCLUDED.updated_at_utc
                """
            ),
            row,
        )


def main() -> None:
    glob_pattern = os.getenv("FBREF_STANDARD_GLOB") or "data/raw/laliga_*_standard.csv"
    min_minutes = int(os.getenv("FBREF_STUDY_MIN_MINUTES") or "600")
    files = [Path(p) for p in sorted(Path().glob(glob_pattern.replace("\\", "/")))]
    if not files:
        raise RuntimeError(f"Aucun fichier CSV trouve avec le motif: {glob_pattern}")

    logger.info("Import FBref standard -> Supabase start | files=%s | min_minutes=%s", [f.name for f in files], min_minutes)

    raw_frames = [_read_fbref_standard_csv(path) for path in files]
    raw_df = pd.concat(raw_frames, ignore_index=True)
    player_season = _build_player_season_from_standard(raw_df)
    progression = build_progression(player_season, min_minutes=min_minutes)
    regularity = _empty_regularity_frame()
    if not regularity.empty:
        regularity["source_mode"] = "manual_standard_csv"
    meta_df = _build_meta(player_season, progression, raw_df)

    if not progression.empty:
        progression = progression.copy()
        progression["source_mode"] = "manual_standard_csv"

    engine_obj = _get_engine()
    _apply_schema_if_requested(engine_obj)

    with engine_obj.begin() as conn:
        conn.execute(text("DELETE FROM public.study_fbref_standard_season_raw"))
        conn.execute(text("DELETE FROM public.study_fbref_player_season"))
        conn.execute(text("DELETE FROM public.study_fbref_regularity"))
        conn.execute(text("DELETE FROM public.study_fbref_progression"))
        conn.execute(text("DELETE FROM public.study_fbref_meta WHERE dataset_name = 'fbref_study'"))

    _append_table(engine_obj, "study_fbref_standard_season_raw", raw_df)
    _append_table(engine_obj, "study_fbref_player_season", player_season)
    _append_table(engine_obj, "study_fbref_regularity", regularity)
    _append_table(engine_obj, "study_fbref_progression", progression)
    _upsert_meta(engine_obj, meta_df)

    logger.info(
        "Import FBref standard -> Supabase done | raw=%s player_season=%s progression=%s regularity=%s",
        len(raw_df),
        len(player_season),
        len(progression),
        len(regularity),
    )


if __name__ == "__main__":
    main()
