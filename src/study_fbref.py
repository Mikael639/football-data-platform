from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("study_fbref")

CANONICAL_PLAYER_MATCH_COLUMNS = [
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
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def current_season_start_year(today: datetime | None = None) -> int:
    dt = (today or datetime.utcnow()).date()
    return dt.year if dt.month >= 7 else dt.year - 1


def last_completed_season_start_years(num_seasons: int = 3) -> list[int]:
    current_start = current_season_start_year()
    return [current_start - i for i in range(1, num_seasons + 1)][::-1]


def season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "__".join([str(x) for x in tup if x is not None and str(x) != ""]).strip("_")
            for tup in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def _find_col(df: pd.DataFrame, *, exact: Iterable[str] = (), contains: Iterable[str] = ()) -> str | None:
    if df is None or df.empty:
        return None
    normalized = {_normalize_text(c): c for c in df.columns}
    for e in exact:
        key = _normalize_text(e)
        if key in normalized:
            return normalized[key]

    contains_norm = [_normalize_text(x) for x in contains if x]
    for col in df.columns:
        n = _normalize_text(col)
        if all(token in n for token in contains_norm):
            return col
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    def _to_bool(v):
        if pd.isna(v):
            return False
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(int(v))
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes", "y", "start", "starter", "started"}

    return series.apply(_to_bool).astype(bool)


def _coerce_numeric(series: pd.Series | None, default=0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _parse_pass_accuracy(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")

    def _parse(v):
        if pd.isna(v):
            return 0.0
        s = str(v).strip().replace("%", "")
        try:
            num = float(s)
        except ValueError:
            return 0.0
        if num > 1:
            num = num / 100.0
        return max(0.0, min(1.0, num))

    return series.apply(_parse).astype(float)


def _position_group(pos: Any) -> str:
    if pd.isna(pos):
        return "UNK"
    s = str(pos).upper()
    if "GK" in s:
        return "GK"
    if any(x in s for x in ["DF", "CB", "LB", "RB", "WB"]):
        return "DEF"
    if any(x in s for x in ["MF", "DM", "CM", "AM", "LM", "RM"]):
        return "MID"
    if any(x in s for x in ["FW", "ST", "CF", "LW", "RW"]):
        return "FWD"
    return "UNK"


def _derive_season_start_from_date(date_value: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_value, errors="coerce")
    return np.where(dt.dt.month >= 7, dt.dt.year, dt.dt.year - 1)


def _stable_int_ids(values: pd.Series, start_at: int = 1) -> pd.Series:
    vals = values.fillna("").astype(str)
    unique = {v: idx + start_at for idx, v in enumerate(sorted(set(vals.tolist())))}
    return vals.map(unique).astype(int)


def _ensure_manual_match_columns(df: pd.DataFrame, league_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Le CSV manuel de matchs joueurs est vide.")

    out = _flatten_columns(df.copy())
    out.columns = [str(c).strip() for c in out.columns]

    required_min = {"player_name", "team_name", "date_id", "minutes"}
    missing_min = [c for c in required_min if c not in out.columns]
    if missing_min:
        raise ValueError(
            "CSV manuel invalide: colonnes minimales manquantes "
            f"{missing_min}. Colonnes recues: {list(out.columns)}"
        )

    if "season_start" not in out.columns:
        out["season_start"] = _derive_season_start_from_date(out["date_id"])
    else:
        out["season_start"] = pd.to_numeric(out["season_start"], errors="coerce")
    out = out.dropna(subset=["season_start"]).copy()
    out["season_start"] = out["season_start"].astype(int)

    if "competition" not in out.columns:
        out["competition"] = league_label
    else:
        out["competition"] = out["competition"].fillna(league_label).astype(str)

    if "match_id" not in out.columns:
        out["match_id"] = (
            out["season_start"].astype(str)
            + "|"
            + out["date_id"].astype(str)
            + "|"
            + out["team_name"].astype(str)
            + "|"
            + out["player_name"].astype(str)
            + "|"
            + (out.index.astype(str))
        )
    else:
        out["match_id"] = out["match_id"].astype(str)

    if "player_key" not in out.columns:
        out["player_key"] = (
            out["player_name"].fillna("").astype(str).map(_normalize_text)
            + "|"
            + out["team_name"].fillna("").astype(str).map(_normalize_text)
        )
    else:
        out["player_key"] = out["player_key"].astype(str)

    if "player_id" not in out.columns:
        out["player_id"] = _stable_int_ids(out["player_key"])
    else:
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
        missing_player_ids = out["player_id"].isna()
        if missing_player_ids.any():
            out.loc[missing_player_ids, "player_id"] = _stable_int_ids(out.loc[missing_player_ids, "player_key"])
        out["player_id"] = out["player_id"].astype(int)

    if "team_id" not in out.columns:
        out["team_id"] = _stable_int_ids(out["team_name"])
    else:
        out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce")
        missing_team_ids = out["team_id"].isna()
        if missing_team_ids.any():
            out.loc[missing_team_ids, "team_id"] = _stable_int_ids(out.loc[missing_team_ids, "team_name"])
        out["team_id"] = out["team_id"].astype(int)

    if "position" not in out.columns:
        out["position"] = None
    if "position_group" not in out.columns:
        out["position_group"] = out["position"].apply(_position_group)
    else:
        out["position_group"] = out["position_group"].fillna("").astype(str).str.upper()
        out.loc[out["position_group"].isin(["", "NAN"]), "position_group"] = out.loc[
            out["position_group"].isin(["", "NAN"]), "position"
        ].apply(_position_group)

    if "is_starting" not in out.columns:
        out["is_starting"] = False
    else:
        out["is_starting"] = _coerce_bool_series(out["is_starting"])

    for c in ["goals", "assists", "shots", "passes"]:
        if c not in out.columns:
            out[c] = 0
    if "pass_accuracy" not in out.columns:
        out["pass_accuracy"] = 0.0

    out["date_id"] = pd.to_datetime(out["date_id"], errors="coerce").dt.date.astype("string")
    out = out.dropna(subset=["date_id"]).copy()

    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0).clip(lower=0, upper=130).astype(int)
    out["goals"] = pd.to_numeric(out["goals"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["assists"] = pd.to_numeric(out["assists"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["shots"] = pd.to_numeric(out["shots"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["passes"] = pd.to_numeric(out["passes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["pass_accuracy"] = _parse_pass_accuracy(out["pass_accuracy"])

    out["player_name"] = out["player_name"].astype(str)
    out["team_name"] = out["team_name"].astype(str)
    out["competition"] = out["competition"].astype(str)

    for col in CANONICAL_PLAYER_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[CANONICAL_PLAYER_MATCH_COLUMNS].reset_index(drop=True)


def _prepare_fbref_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = _flatten_columns(df.reset_index())
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _normalize_fbref_player_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["season_start", "player_key", "player_name", "team_name", "position", "position_group"]
        )

    raw = _prepare_fbref_raw(df)

    player_col = _find_col(raw, exact=["player", "player_name"], contains=["player"])
    squad_col = _find_col(raw, exact=["squad", "team", "team_name"], contains=["squad"])
    season_col = _find_col(raw, exact=["season"])
    pos_col = _find_col(raw, exact=["pos", "position"], contains=["pos"])
    player_id_col = _find_col(raw, exact=["player_id", "fbref_id"], contains=["player", "id"])

    out = pd.DataFrame()
    out["player_name"] = raw[player_col].astype(str) if player_col else ""
    out["team_name"] = raw[squad_col].astype(str) if squad_col else ""
    if season_col:
        out["season_start"] = pd.to_numeric(raw[season_col], errors="coerce")
    else:
        out["season_start"] = np.nan
    out["position"] = raw[pos_col].astype(str) if pos_col else None
    out["position_group"] = out["position"].apply(_position_group)

    if player_id_col:
        out["player_key"] = raw[player_id_col].astype(str)
    else:
        out["player_key"] = (
            out["player_name"].fillna("").map(_normalize_text) + "|" + out["team_name"].fillna("").map(_normalize_text)
        )

    out = out.dropna(subset=["season_start"]).copy()
    out["season_start"] = out["season_start"].astype(int)
    out = (
        out.sort_values(["season_start", "team_name", "player_name"])
        .drop_duplicates(subset=["season_start", "player_key"], keep="first")
        .reset_index(drop=True)
    )
    return out


def _normalize_fbref_player_match_stats(df: pd.DataFrame, league_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    raw = _prepare_fbref_raw(df)

    player_col = _find_col(raw, exact=["player", "player_name"], contains=["player"])
    player_id_col = _find_col(raw, exact=["player_id", "fbref_id"], contains=["player", "id"])
    team_col = _find_col(raw, exact=["squad", "team", "team_name"], contains=["squad"])
    date_col = _find_col(raw, exact=["date", "match_date"], contains=["date"])
    season_col = _find_col(raw, exact=["season"])
    pos_col = _find_col(raw, exact=["pos", "position"], contains=["pos"])

    starts_col = _find_col(raw, exact=["starts", "start"], contains=["start"])
    minutes_col = (
        _find_col(raw, exact=["minutes", "min"], contains=["minutes"])
        or _find_col(raw, contains=["playing", "time", "min"])
        or _find_col(raw, contains=["performance", "minutes"])
    )
    goals_col = _find_col(raw, exact=["gls", "goals"], contains=["gls"]) or _find_col(raw, contains=["goals"])
    assists_col = _find_col(raw, exact=["ast", "assists"], contains=["ast"]) or _find_col(raw, contains=["assists"])
    shots_col = _find_col(raw, exact=["sh", "shots"], contains=["shots"]) or _find_col(raw, contains=["sh"])
    passes_cmp_col = _find_col(raw, exact=["cmp", "passes_completed"], contains=["passing", "cmp"])
    passes_att_col = _find_col(raw, exact=["att", "passes_attempted"], contains=["passing", "att"])
    passes_col = _find_col(raw, exact=["passes"], contains=["passes"])
    pass_pct_col = (
        _find_col(raw, exact=["cmp_pct", "pass_accuracy", "pass_completion"], contains=["cmp", "pct"])
        or _find_col(raw, contains=["pass", "accuracy"])
    )

    match_id_col = (
        _find_col(raw, exact=["match_id", "game_id"])
        or _find_col(raw, contains=["game", "id"])
        or _find_col(raw, contains=["match", "id"])
    )
    opponent_col = _find_col(raw, exact=["opponent"], contains=["opponent"])
    venue_col = _find_col(raw, exact=["venue"], contains=["venue"])

    out = pd.DataFrame(index=raw.index)
    out["competition"] = league_label
    out["player_name"] = raw[player_col].astype(str) if player_col else "Inconnu"
    out["team_name"] = raw[team_col].astype(str) if team_col else "Club inconnu"
    out["position"] = raw[pos_col].astype(str) if pos_col else None
    out["position_group"] = out["position"].apply(_position_group)

    if player_id_col:
        out["player_key"] = raw[player_id_col].astype(str)
    else:
        out["player_key"] = out["player_name"].map(_normalize_text)

    date_series = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.Series(pd.NaT, index=raw.index)
    out["date_id"] = date_series.dt.date.astype("string")

    if season_col:
        out["season_start"] = pd.to_numeric(raw[season_col], errors="coerce")
    else:
        out["season_start"] = _derive_season_start_from_date(date_series)
    out = out.dropna(subset=["season_start"]).copy()
    out["season_start"] = out["season_start"].astype(int)

    if starts_col:
        out["is_starting"] = _coerce_bool_series(raw[starts_col])
    else:
        out["is_starting"] = False

    out["minutes"] = _coerce_numeric(raw[minutes_col], default=0.0).clip(lower=0, upper=130).astype(int) if minutes_col else 0
    out["goals"] = _coerce_numeric(raw[goals_col], default=0.0).clip(lower=0).astype(int) if goals_col else 0
    out["assists"] = _coerce_numeric(raw[assists_col], default=0.0).clip(lower=0).astype(int) if assists_col else 0
    out["shots"] = _coerce_numeric(raw[shots_col], default=0.0).clip(lower=0).astype(int) if shots_col else 0

    if passes_col:
        passes_series = _coerce_numeric(raw[passes_col], default=0.0)
    elif passes_cmp_col:
        passes_series = _coerce_numeric(raw[passes_cmp_col], default=0.0)
    else:
        passes_series = pd.Series(0, index=raw.index)
    out["passes"] = passes_series.clip(lower=0).astype(int)

    if pass_pct_col:
        out["pass_accuracy"] = _parse_pass_accuracy(raw[pass_pct_col])
    elif passes_cmp_col and passes_att_col:
        cmp_s = _coerce_numeric(raw[passes_cmp_col], default=0.0)
        att_s = _coerce_numeric(raw[passes_att_col], default=0.0).replace(0, np.nan)
        out["pass_accuracy"] = (cmp_s / att_s).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    else:
        out["pass_accuracy"] = 0.0

    if match_id_col:
        out["match_id"] = raw[match_id_col].astype(str)
    else:
        parts = [
            out["season_start"].astype(str),
            out["date_id"].astype(str),
            out["team_name"].fillna(""),
            raw[opponent_col].astype(str) if opponent_col else pd.Series("", index=raw.index),
            raw[venue_col].astype(str) if venue_col else pd.Series("", index=raw.index),
            out["player_key"].astype(str),
        ]
        out["match_id"] = (
            parts[0]
            + "|"
            + parts[1]
            + "|"
            + parts[2]
            + "|"
            + parts[3]
            + "|"
            + parts[4]
            + "|"
            + parts[5]
        )

    out["team_id"] = _stable_int_ids(out["team_name"])
    out["player_id"] = _stable_int_ids(out["player_key"])

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
    ]
    return out[cols].reset_index(drop=True)


def merge_position_metadata(df_match: pd.DataFrame, df_season_meta: pd.DataFrame) -> pd.DataFrame:
    if df_match is None or df_match.empty:
        return df_match
    if df_season_meta is None or df_season_meta.empty:
        return df_match

    out = df_match.copy()
    meta = df_season_meta[["season_start", "player_key", "position", "position_group"]].copy()
    meta = meta.rename(
        columns={"position": "position_meta", "position_group": "position_group_meta"}
    ).drop_duplicates(subset=["season_start", "player_key"])

    out = out.merge(meta, on=["season_start", "player_key"], how="left")
    out["position"] = out["position"].where(out["position"].notna() & (out["position"] != ""), out["position_meta"])
    out["position_group"] = out["position_group"].where(
        out["position_group"].notna() & (out["position_group"] != "UNK"),
        out["position_group_meta"],
    )
    out["position_group"] = out["position_group"].fillna("UNK")
    out = out.drop(columns=["position_meta", "position_group_meta"])
    return out


def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0).clip(lower=0)
    out["goals"] = pd.to_numeric(out["goals"], errors="coerce").fillna(0).clip(lower=0)
    out["assists"] = pd.to_numeric(out["assists"], errors="coerce").fillna(0).clip(lower=0)
    out["shots"] = pd.to_numeric(out["shots"], errors="coerce").fillna(0).clip(lower=0)
    out["passes"] = pd.to_numeric(out["passes"], errors="coerce").fillna(0).clip(lower=0)
    out["pass_accuracy"] = pd.to_numeric(out["pass_accuracy"], errors="coerce").fillna(0).clip(lower=0, upper=1)
    if "is_starting" not in out.columns:
        out["is_starting"] = False
    out["is_starting"] = out["is_starting"].fillna(False).astype(bool)

    out["played_flag"] = (out["minutes"] > 0).astype(int)
    out["start_flag"] = out["is_starting"].astype(int)
    out["sub_in_flag"] = ((out["minutes"] > 0) & (~out["is_starting"])).astype(int)
    out["ga"] = out["goals"] + out["assists"]

    mins = out["minutes"].replace(0, np.nan)
    for col in ["goals", "assists", "ga", "shots", "passes"]:
        out[f"{col}_p90_match"] = (out[col] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)

    out["position_group"] = out["position_group"].fillna("UNK")
    return out


def build_player_season(df_match: pd.DataFrame) -> pd.DataFrame:
    gcols = ["season_start", "player_id", "player_name", "team_id", "team_name", "position_group"]

    agg = (
        df_match.groupby(gcols, dropna=False)
        .agg(
            matches_played=("played_flag", "sum"),
            starts=("start_flag", "sum"),
            sub_apps=("sub_in_flag", "sum"),
            minutes_total=("minutes", "sum"),
            goals_total=("goals", "sum"),
            assists_total=("assists", "sum"),
            ga_total=("ga", "sum"),
            shots_total=("shots", "sum"),
            passes_total=("passes", "sum"),
            pass_acc_mean=("pass_accuracy", "mean"),
        )
        .reset_index()
    )

    mins = agg["minutes_total"].replace(0, np.nan)
    for raw, out_col in [
        ("goals_total", "goals_p90"),
        ("assists_total", "assists_p90"),
        ("ga_total", "ga_p90"),
        ("shots_total", "shots_p90"),
        ("passes_total", "passes_p90"),
    ]:
        agg[out_col] = (agg[raw] * 90 / mins).replace([np.inf, -np.inf], np.nan).fillna(0)

    agg["eligible_600"] = agg["minutes_total"] >= 600
    agg["eligible_900"] = agg["minutes_total"] >= 900
    return agg


def _zscore_grouped(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def build_regularity(df_match: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    base = df_match[df_match["minutes"] > 0].copy()

    gcols = ["season_start", "player_id", "player_name", "team_id", "team_name", "position_group"]
    reg = (
        base.groupby(gcols, dropna=False)
        .agg(
            minutes_total=("minutes", "sum"),
            matches_played=("played_flag", "sum"),
            ga_p90_mean=("ga_p90_match", "mean"),
            ga_p90_std=("ga_p90_match", "std"),
            shots_p90_mean=("shots_p90_match", "mean"),
            shots_p90_std=("shots_p90_match", "std"),
            passes_p90_mean=("passes_p90_match", "mean"),
            passes_p90_std=("passes_p90_match", "std"),
        )
        .reset_index()
    )

    reg = reg[reg["minutes_total"] >= min_minutes].copy()
    if reg.empty:
        reg["podium"] = []
        return reg

    for c in ["ga_p90_std", "shots_p90_std", "passes_p90_std"]:
        reg[c] = reg[c].fillna(0)

    reg["ga_p90_cv"] = np.where(reg["ga_p90_mean"] > 0, reg["ga_p90_std"] / reg["ga_p90_mean"], np.nan)
    reg["shots_p90_cv"] = np.where(reg["shots_p90_mean"] > 0, reg["shots_p90_std"] / reg["shots_p90_mean"], np.nan)
    reg["passes_p90_cv"] = np.where(reg["passes_p90_mean"] > 0, reg["passes_p90_std"] / reg["passes_p90_mean"], np.nan)
    reg["stability_proxy"] = reg[["ga_p90_cv", "shots_p90_cv", "passes_p90_cv"]].mean(axis=1, skipna=True).fillna(0)

    grp = ["season_start", "position_group"]
    reg["perf_z"] = reg.groupby(grp)["ga_p90_mean"].transform(_zscore_grouped)
    reg["stab_z"] = reg.groupby(grp)["stability_proxy"].transform(lambda s: -_zscore_grouped(s))
    reg["regularity_score"] = 0.6 * reg["perf_z"] + 0.4 * reg["stab_z"]

    reg["regularity_rank_pos"] = (
        reg.groupby(grp)["regularity_score"].rank(method="dense", ascending=False).astype(int)
    )
    reg["podium"] = reg["regularity_rank_pos"].map({1: "🥇", 2: "🥈", 3: "🥉"}).fillna("")
    return reg.sort_values(["season_start", "position_group", "regularity_score"], ascending=[False, True, False])


def build_progression(df_player_season: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    cols = [
        "season_start",
        "player_id",
        "player_name",
        "team_name",
        "position_group",
        "minutes_total",
        "goals_p90",
        "assists_p90",
        "ga_p90",
        "shots_p90",
        "passes_p90",
        "pass_acc_mean",
    ]
    s = df_player_season[cols].copy()

    prev = s.copy()
    prev["season_start"] = prev["season_start"] + 1
    prev = prev.rename(columns={c: f"{c}_prev" for c in cols if c not in {"season_start", "player_id"}})

    out = s.merge(prev, on=["season_start", "player_id"], how="inner")
    out = out[(out["minutes_total"] >= min_minutes) & (out["minutes_total_prev"] >= min_minutes)].copy()
    if out.empty:
        out["podium"] = []
        return out

    for c in ["goals_p90", "assists_p90", "ga_p90", "shots_p90", "passes_p90", "pass_acc_mean", "minutes_total"]:
        out[f"delta_{c}"] = out[c] - out[f"{c}_prev"]

    out["progress_score"] = (
        out["delta_ga_p90"].fillna(0) * 0.45
        + out["delta_shots_p90"].fillna(0) * 0.20
        + out["delta_passes_p90"].fillna(0) * 0.15
        + out["delta_pass_acc_mean"].fillna(0) * 0.20
    )
    grp = ["season_start", "position_group"]
    out["progress_rank_pos"] = out.groupby(grp)["progress_score"].rank(method="dense", ascending=False).astype(int)
    out["podium"] = out["progress_rank_pos"].map({1: "🥇", 2: "🥈", 3: "🥉"}).fillna("")
    return out.sort_values(["season_start", "position_group", "progress_score"], ascending=[False, True, False])


def _extract_fbref_raw(leagues: str, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import soccerdata as sd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("soccerdata n'est pas installe. Ajoute la dependance puis relance.") from exc

    logger.info("FBref extract start - league=%s seasons=%s", leagues, seasons)
    fb = sd.FBref(leagues=leagues, seasons=seasons)

    match_stats = fb.read_player_match_stats(stat_type="summary")
    logger.info("FBref player match stats rows: %s", len(match_stats))

    try:
        season_stats = fb.read_player_season_stats(stat_type="standard")
        logger.info("FBref player season stats rows: %s", len(season_stats))
    except Exception:
        logger.exception("FBref player season stats extraction failed; continuing without season metadata")
        season_stats = pd.DataFrame()

    return match_stats, season_stats


def build_fbref_study_outputs(
    *,
    leagues: str = "ESP-La Liga",
    seasons: list[int] | None = None,
    min_minutes: int = 600,
) -> dict[str, pd.DataFrame]:
    seasons = seasons or last_completed_season_start_years(3)
    raw_match, raw_season = _extract_fbref_raw(leagues=leagues, seasons=seasons)

    df_match = _normalize_fbref_player_match_stats(raw_match, league_label="La Liga (FBref)")
    df_season_meta = _normalize_fbref_player_season_stats(raw_season)
    df_match = merge_position_metadata(df_match, df_season_meta)
    df_match = add_match_features(df_match)
    df_player_season = build_player_season(df_match)
    df_regularity = build_regularity(df_match, min_minutes=min_minutes)
    df_progression = build_progression(df_player_season, min_minutes=min_minutes)

    return {
        "player_match": df_match,
        "player_season": df_player_season,
        "regularity": df_regularity,
        "progression": df_progression,
    }


def build_fbref_study_outputs_from_manual_csv(
    *,
    player_match_csv: Path,
    league_label: str = "La Liga (FBref manuel)",
    min_minutes: int = 600,
) -> dict[str, pd.DataFrame]:
    if not player_match_csv.exists():
        raise FileNotFoundError(f"Fichier CSV manuel introuvable: {player_match_csv}")

    logger.info("Manual FBref study import - reading %s", player_match_csv)
    raw = pd.read_csv(player_match_csv)
    df_match = _ensure_manual_match_columns(raw, league_label=league_label)
    df_match = add_match_features(df_match)
    df_player_season = build_player_season(df_match)
    df_regularity = build_regularity(df_match, min_minutes=min_minutes)
    df_progression = build_progression(df_player_season, min_minutes=min_minutes)
    return {
        "player_match": df_match,
        "player_season": df_player_season,
        "regularity": df_regularity,
        "progression": df_progression,
    }


def save_fbref_study_outputs(
    outputs: dict[str, pd.DataFrame],
    out_dir: Path,
    seasons: list[int],
    league: str,
    source_mode: str = "soccerdata",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Saved %s rows to %s", len(df), path)

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "league": league,
        "source_mode": source_mode,
        "seasons_start_years": seasons,
        "season_labels": [season_label(s) for s in seasons],
        "files": {f"{k}.csv": int(len(v)) for k, v in outputs.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", out_dir / "meta.json")


def main() -> None:
    source_mode = (os.getenv("FBREF_STUDY_SOURCE") or "soccerdata").strip().lower()
    league = os.getenv("FBREF_LEAGUE_CODE") or "ESP-La Liga"
    num_past = int(os.getenv("FBREF_NUM_PAST_SEASONS") or "3")
    min_minutes = int(os.getenv("FBREF_STUDY_MIN_MINUTES") or "600")
    out_dir = Path(os.getenv("FBREF_STUDY_OUTDIR") or "data/study/fbref")
    manual_match_csv = Path(os.getenv("FBREF_MANUAL_MATCH_CSV") or "data/study/fbref_input/player_match_manual.csv")

    seasons = last_completed_season_start_years(num_past)
    logger.info(
        "FBref study job start - source=%s league=%s seasons=%s min_minutes=%s out_dir=%s",
        source_mode,
        league,
        seasons,
        min_minutes,
        out_dir,
    )
    if source_mode in {"manual", "manual_csv", "csv"}:
        outputs = build_fbref_study_outputs_from_manual_csv(
            player_match_csv=manual_match_csv,
            league_label=f"{league} (manuel)",
            min_minutes=min_minutes,
        )
        seasons = sorted(outputs["player_match"]["season_start"].dropna().astype(int).unique().tolist())
    elif source_mode in {"soccerdata", "fbref"}:
        outputs = build_fbref_study_outputs(leagues=league, seasons=seasons, min_minutes=min_minutes)
    else:
        raise RuntimeError(
            f"FBREF_STUDY_SOURCE={source_mode!r} non supporte. Utilise 'soccerdata' ou 'manual_csv'."
        )

    save_fbref_study_outputs(outputs, out_dir=out_dir, seasons=seasons, league=league, source_mode=source_mode)
    logger.info("FBref study job done")


if __name__ == "__main__":
    main()
