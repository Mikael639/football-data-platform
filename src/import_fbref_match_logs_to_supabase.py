from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from src.import_fbref_standard_to_supabase import _append_table, _apply_schema_if_requested, _get_engine
from src.study_fbref import (
    _ensure_manual_match_columns,
    add_match_features,
    build_player_season,
    build_progression,
    build_regularity,
)
from src.utils.logger import get_logger

logger = get_logger("fbref_matchlogs_supabase")


def _norm(value: object) -> str:
    if value is None:
        return ""
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def _parse_season_filter() -> list[int] | None:
    single = os.getenv("FBREF_MATCHLOG_SEASON_START")
    multi = os.getenv("FBREF_MATCHLOG_SEASONS")
    values: list[int] = []
    if single:
        try:
            values.append(int(single))
        except ValueError:
            raise RuntimeError(f"FBREF_MATCHLOG_SEASON_START invalide: {single!r}")
    if multi:
        for part in multi.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(int(part))
            except ValueError:
                raise RuntimeError(f"FBREF_MATCHLOG_SEASONS invalide: {multi!r}")
    uniq = sorted(set(values))
    return uniq or None


def _load_existing_player_season(engine_obj, seasons: Iterable[int]) -> pd.DataFrame:
    seasons = sorted(set(int(s) for s in seasons))
    if not seasons:
        return pd.DataFrame()
    existing = _load_existing_table(engine_obj, "study_fbref_player_season")
    if existing.empty or "season_start" not in existing.columns:
        return pd.DataFrame()
    return existing[existing["season_start"].astype(int).isin(seasons)][
        ["season_start", "player_id", "player_key", "player_name", "team_id", "team_name", "position_group"]
    ].copy()


def _load_existing_table(engine_obj, table_name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(text(f"SELECT * FROM public.{table_name}"), engine_obj)
    except Exception:
        return pd.DataFrame()


def _align_ids_with_existing(df_match: pd.DataFrame, existing_player_season: pd.DataFrame) -> pd.DataFrame:
    if df_match.empty or existing_player_season.empty:
        return df_match

    out = df_match.copy()
    out["__player_norm"] = out["player_name"].astype(str).map(_norm)
    out["__team_norm"] = out["team_name"].astype(str).map(_norm)

    eps = existing_player_season.copy()
    eps["__player_norm"] = eps["player_name"].astype(str).map(_norm)
    eps["__team_norm"] = eps["team_name"].astype(str).map(_norm)

    # Exact (season + player + team) mapping first.
    exact = eps.drop_duplicates(subset=["season_start", "__player_norm", "__team_norm"]).copy()
    exact_map = {
        (int(r["season_start"]), r["__player_norm"], r["__team_norm"]): (int(r["player_id"]), int(r["team_id"]), r.get("player_key"))
        for _, r in exact.iterrows()
    }

    # Fallback by season + player only, but only when unique in existing season.
    fallback_counts = (
        eps.groupby(["season_start", "__player_norm"], dropna=False)["player_id"].nunique().reset_index(name="n")
    )
    fallback_unique = eps.merge(
        fallback_counts[fallback_counts["n"] == 1][["season_start", "__player_norm"]],
        on=["season_start", "__player_norm"],
        how="inner",
    )
    fallback_unique = fallback_unique.drop_duplicates(subset=["season_start", "__player_norm"])
    fallback_map = {
        (int(r["season_start"]), r["__player_norm"]): (int(r["player_id"]), int(r["team_id"]), r.get("player_key"))
        for _, r in fallback_unique.iterrows()
    }

    matched_exact = matched_fallback = 0
    new_player_ids: list[int] = []
    max_existing_player_id = int(pd.to_numeric(eps["player_id"], errors="coerce").fillna(0).max()) if not eps.empty else 0
    next_player_id = max_existing_player_id + 1

    new_team_ids: list[int] = []
    max_existing_team_id = int(pd.to_numeric(eps["team_id"], errors="coerce").fillna(0).max()) if not eps.empty else 0
    next_team_id = max_existing_team_id + 1

    assigned_player_ids: dict[str, int] = {}
    assigned_team_ids: dict[str, int] = {}

    for idx, row in out.iterrows():
        season_key = int(row["season_start"])
        pnorm = row["__player_norm"]
        tnorm = row["__team_norm"]
        exact_key = (season_key, pnorm, tnorm)
        fallback_key = (season_key, pnorm)
        mapping = exact_map.get(exact_key)
        if mapping:
            matched_exact += 1
        else:
            mapping = fallback_map.get(fallback_key)
            if mapping:
                matched_fallback += 1
        if mapping:
            pid, tid, pkey = mapping
            out.at[idx, "player_id"] = pid
            out.at[idx, "team_id"] = tid
            if pkey:
                out.at[idx, "player_key"] = pkey
            new_player_ids.append(pid)
            new_team_ids.append(tid)
            continue

        # Unmatched: assign stable IDs within this import (same player/team across rows keeps same ID).
        player_assign_key = str(row.get("player_key") or pnorm or row.get("player_name") or "")
        if player_assign_key not in assigned_player_ids:
            assigned_player_ids[player_assign_key] = next_player_id
            next_player_id += 1
        team_assign_key = tnorm or str(row.get("team_name") or "")
        if team_assign_key not in assigned_team_ids:
            assigned_team_ids[team_assign_key] = next_team_id
            next_team_id += 1

        out.at[idx, "player_id"] = assigned_player_ids[player_assign_key]
        out.at[idx, "team_id"] = assigned_team_ids[team_assign_key]
        new_player_ids.append(int(out.at[idx, "player_id"]))
        new_team_ids.append(int(out.at[idx, "team_id"]))

    logger.info(
        "ID alignment done | rows=%s exact=%s fallback=%s unmatched=%s",
        len(out),
        matched_exact,
        matched_fallback,
        len(out) - matched_exact - matched_fallback,
    )
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").fillna(0).astype(int)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").fillna(0).astype(int)
    return out.drop(columns=["__player_norm", "__team_norm"], errors="ignore")


def _merge_replace_seasons(existing_df: pd.DataFrame, replacement_df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    seasons_set = {int(s) for s in seasons}
    if existing_df is None or existing_df.empty:
        return replacement_df.copy()
    out = existing_df.copy()
    if "season_start" in out.columns:
        out = out[~out["season_start"].astype(int).isin(seasons_set)].copy()
    return pd.concat([out, replacement_df], ignore_index=True, sort=False)


def _prepare_player_match_table(df_match: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = df_match.copy()
    out["source_mode"] = "manual_matchlog_csv"
    out["source_file"] = source_file
    for c in [
        "played_flag",
        "start_flag",
        "sub_in_flag",
        "ga",
        "goals_p90_match",
        "assists_p90_match",
        "ga_p90_match",
        "shots_p90_match",
        "passes_p90_match",
    ]:
        if c not in out.columns:
            out[c] = 0
    # keep only columns present in schema order
    cols = [
        "season_start",
        "match_id",
        "date_id",
        "competition",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "player_key",
        "position",
        "position_group",
        "is_starting",
        "minutes",
        "goals",
        "assists",
        "shots",
        "passes",
        "pass_accuracy",
        "played_flag",
        "start_flag",
        "sub_in_flag",
        "ga",
        "goals_p90_match",
        "assists_p90_match",
        "ga_p90_match",
        "shots_p90_match",
        "passes_p90_match",
        "source_mode",
        "source_file",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols].copy()


def _recompute_progression_with_existing(engine_obj, replacement_player_season: pd.DataFrame, seasons: list[int], min_minutes: int) -> pd.DataFrame:
    existing = _load_existing_table(engine_obj, "study_fbref_player_season")
    merged = _merge_replace_seasons(existing, replacement_player_season, seasons)
    if merged.empty:
        return pd.DataFrame()
    prog = build_progression(merged, min_minutes=min_minutes)
    if not prog.empty:
        prog = prog.copy()
        prog["source_mode"] = "hybrid_standard_plus_matchlogs"
    return prog


def _build_meta_patch(engine_obj, seasons: list[int], player_match_rows: int, regularity_rows: int) -> None:
    # Keep it lightweight: patch only notes/updated_at/files if meta row exists.
    files_patch = {
        "player_match_rows_last_import": int(player_match_rows),
        "regularity_rows_last_import": int(regularity_rows),
        "matchlog_seasons_last_import": [int(s) for s in seasons],
    }
    note = (
        "Import match logs FBref manuel applique sur certaines saisons (player_match + regularity + player_season partiel). "
        "Progression recalculee sur le dataset fusionne."
    )
    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE public.study_fbref_meta
                SET notes = COALESCE(notes, '') || CASE WHEN COALESCE(notes,'') = '' THEN '' ELSE ' ' END || :note,
                    files = COALESCE(files, '{}'::jsonb) || CAST(:files_patch AS jsonb),
                    updated_at_utc = NOW()
                WHERE dataset_name = 'fbref_study'
                """
            ),
            {"note": note, "files_patch": json.dumps(files_patch)},
        )


def _strip_managed_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["ingested_at_utc", "updated_at_utc", "raw_id"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def main() -> None:
    manual_match_csv = Path(os.getenv("FBREF_MATCHLOG_CSV") or "data/study/fbref_input/player_match_manual.csv")
    league_label = os.getenv("FBREF_MATCHLOG_LEAGUE_LABEL") or "La Liga (FBref manuel match logs)"
    min_minutes = int(os.getenv("FBREF_STUDY_MIN_MINUTES") or "600")
    season_filter = _parse_season_filter()

    if not manual_match_csv.exists():
        raise FileNotFoundError(f"CSV match logs introuvable: {manual_match_csv}")

    logger.info(
        "Import FBref match logs -> Supabase start | csv=%s | min_minutes=%s | season_filter=%s",
        manual_match_csv,
        min_minutes,
        season_filter,
    )

    raw = pd.read_csv(manual_match_csv)
    df_match = _ensure_manual_match_columns(raw, league_label=league_label)
    if season_filter:
        df_match = df_match[df_match["season_start"].astype(int).isin(season_filter)].copy()
    if df_match.empty:
        raise RuntimeError("Aucune ligne match-log apres filtrage saison.")
    seasons = sorted(df_match["season_start"].dropna().astype(int).unique().tolist())

    engine_obj = _get_engine()
    _apply_schema_if_requested(engine_obj)

    existing_ps = _load_existing_player_season(engine_obj, seasons)
    df_match = _align_ids_with_existing(df_match, existing_ps)
    df_match = add_match_features(df_match)

    player_season = build_player_season(df_match)
    regularity = build_regularity(df_match, min_minutes=min_minutes)
    progression_all = _recompute_progression_with_existing(engine_obj, player_season, seasons, min_minutes=min_minutes)

    if not player_season.empty:
        player_season = player_season.copy()
        player_season["source_mode"] = "manual_matchlog_csv"
        # columns expected by study_fbref_player_season schema but absent in match-log build
        defaults = {
            "player_key": None,
            "position_raw": None,
            "clubs_count": 1,
            "clubs_list": None,
            "is_multi_club_season": False,
            "nation_code": None,
            "age": None,
            "birth_year": None,
            "goals_non_pk_total": 0,
            "pk_goals_total": 0,
            "pk_attempts_total": 0,
            "yellow_cards_total": 0,
            "red_cards_total": 0,
            "source_file": str(manual_match_csv.name),
        }
        for c, default in defaults.items():
            if c not in player_season.columns:
                player_season[c] = default
        if "clubs_list" in player_season.columns:
            player_season["clubs_list"] = player_season["clubs_list"].fillna(player_season.get("team_name", ""))
        if "player_key" in player_season.columns:
            player_season["player_key"] = player_season["player_key"].fillna(
                player_season["player_name"].astype(str).map(_norm)
            )
        ordered_cols = [
            "season_start", "player_id", "player_key", "player_name", "team_id", "team_name",
            "position_group", "position_raw", "clubs_count", "clubs_list", "is_multi_club_season",
            "nation_code", "age", "birth_year", "matches_played", "starts", "sub_apps", "minutes_total",
            "goals_total", "assists_total", "ga_total", "goals_non_pk_total", "pk_goals_total", "pk_attempts_total",
            "yellow_cards_total", "red_cards_total", "shots_total", "passes_total", "pass_acc_mean",
            "goals_p90", "assists_p90", "ga_p90", "shots_p90", "passes_p90", "eligible_600", "eligible_900",
            "source_mode", "source_file",
        ]
        player_season = player_season[[c for c in ordered_cols if c in player_season.columns]].copy()

    if not regularity.empty:
        regularity = regularity.copy()
        regularity["source_mode"] = "manual_matchlog_csv"
        if "note" not in regularity.columns:
            regularity["note"] = None
    player_match_table = _prepare_player_match_table(df_match, source_file=manual_match_csv.name)

    existing_match = _load_existing_table(engine_obj, "study_fbref_player_match")
    merged_match = _merge_replace_seasons(existing_match, player_match_table, seasons)
    existing_reg = _load_existing_table(engine_obj, "study_fbref_regularity")
    merged_reg = _merge_replace_seasons(existing_reg, regularity, seasons)
    existing_ps_all = _load_existing_table(engine_obj, "study_fbref_player_season")
    merged_ps = _merge_replace_seasons(existing_ps_all, player_season, seasons)

    with engine_obj.begin() as conn:
        conn.execute(text("DELETE FROM public.study_fbref_player_match"))
        conn.execute(text("DELETE FROM public.study_fbref_regularity"))
        conn.execute(text("DELETE FROM public.study_fbref_player_season"))
        conn.execute(text("DELETE FROM public.study_fbref_progression"))

    _append_table(engine_obj, "study_fbref_player_match", _strip_managed_cols(merged_match))
    _append_table(engine_obj, "study_fbref_regularity", _strip_managed_cols(merged_reg))
    _append_table(engine_obj, "study_fbref_player_season", _strip_managed_cols(merged_ps))
    _append_table(engine_obj, "study_fbref_progression", _strip_managed_cols(progression_all))
    try:
        _build_meta_patch(engine_obj, seasons, len(player_match_table), len(regularity))
    except Exception:
        logger.exception("Meta patch skipped after match-log import")

    logger.info(
        "Import FBref match logs -> Supabase done | seasons=%s | player_match(new=%s total=%s) regularity(new=%s total=%s) player_season(new=%s total=%s) progression(total=%s)",
        seasons,
        len(player_match_table),
        len(merged_match),
        len(regularity),
        len(merged_reg),
        len(player_season),
        len(merged_ps),
        len(progression_all),
    )


if __name__ == "__main__":
    main()
