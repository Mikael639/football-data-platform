from datetime import date

import numpy as np
import pandas as pd


def current_season_start_year_dash_local() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_label_from_start(start_year: int) -> str:
    return f"{int(start_year)}-{int(start_year) + 1}"


def season_picker_label_from_start(start_year: int) -> str:
    start_year = int(start_year)
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def study_expected_complete_seasons(n: int = 3) -> list[int]:
    current_start = current_season_start_year_dash_local()
    return [current_start - i for i in range(n, 0, -1)]


def add_podium_icons_generic(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df is None or df.empty or label_col not in df.columns:
        return df
    out = df.copy()
    icons = ["🥇", "🥈", "🥉"]
    for idx, icon in enumerate(icons):
        if idx >= len(out):
            break
        out.iloc[idx, out.columns.get_loc(label_col)] = f"{icon} {out.iloc[idx][label_col]}"
    return out


def build_study_leaders_scope(
    season_df: pd.DataFrame,
    selected_season: int | None,
    selected_pos: str = "Tous",
) -> pd.DataFrame:
    if season_df is None or season_df.empty:
        return pd.DataFrame()

    df = season_df.copy()
    if "season_start" in df.columns and selected_season is not None:
        df = df[df["season_start"].astype(int) == int(selected_season)].copy()
    if selected_pos != "Tous" and "position_group" in df.columns:
        df = df[df["position_group"] == selected_pos].copy()
    if df.empty:
        return df

    numeric_defaults = [
        "goals_total",
        "assists_total",
        "ga_total",
        "goals_non_pk_total",
        "pk_goals_total",
        "pk_attempts_total",
        "yellow_cards_total",
        "red_cards_total",
        "minutes_total",
        "matches_played",
        "starts",
    ]
    for c in numeric_defaults:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "clubs_count" not in df.columns:
        df["clubs_count"] = 1
    df["clubs_count"] = pd.to_numeric(df["clubs_count"], errors="coerce").fillna(1).astype(int)
    if "clubs_list" not in df.columns:
        df["clubs_list"] = df.get("team_name", "").astype(str)
    if "is_multi_club_season" not in df.columns:
        df["is_multi_club_season"] = False
    df["is_multi_club_season"] = df["is_multi_club_season"].fillna(False).astype(bool)

    if selected_season is not None:
        return df

    df = df.sort_values(["player_id", "minutes_total"], ascending=[True, False]).copy()
    primary = df.drop_duplicates(subset=["player_id"], keep="first")[
        ["player_id", "player_name", "team_name", "position_group"]
    ].copy()
    grouped = (
        df.groupby(["player_id"], dropna=False)
        .agg(
            minutes_total=("minutes_total", "sum"),
            matches_played=("matches_played", "sum"),
            starts=("starts", "sum"),
            goals_total=("goals_total", "sum"),
            assists_total=("assists_total", "sum"),
            ga_total=("ga_total", "sum"),
            goals_non_pk_total=("goals_non_pk_total", "sum"),
            pk_goals_total=("pk_goals_total", "sum"),
            pk_attempts_total=("pk_attempts_total", "sum"),
            yellow_cards_total=("yellow_cards_total", "sum"),
            red_cards_total=("red_cards_total", "sum"),
            clubs_count=("team_name", lambda s: int(pd.Series(s).dropna().astype(str).nunique())),
            clubs_list=("team_name", lambda s: " | ".join(sorted(pd.Series(s).dropna().astype(str).unique().tolist()))),
            seasons_count=("season_start", lambda s: int(pd.Series(s).dropna().astype(int).nunique())),
        )
        .reset_index()
    )
    grouped["is_multi_club_season"] = grouped["clubs_count"] > 1
    grouped = grouped.merge(primary, on="player_id", how="left")
    grouped["ga_total"] = grouped.get("ga_total", grouped["goals_total"] + grouped["assists_total"])
    mins = grouped["minutes_total"].replace(0, np.nan)
    grouped["goals_p90"] = (grouped["goals_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped["assists_p90"] = (grouped["assists_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped["ga_p90"] = (grouped["ga_total"] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)
    return grouped
