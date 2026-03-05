from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from src.import_fbref_standard_to_supabase import _get_engine
from src.utils.logger import get_logger

logger = get_logger("import_ucl_csv_supabase")
load_dotenv()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.strip().split()).lower()
    return text


def _as_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("\xa0", " ")
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("\xa0", " ")
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        return float(match.group(0))


def _read_ucl_csv(csv_path: Path) -> pd.DataFrame:
    # First line is a grouped-header row in FBref export format.
    frame = pd.read_csv(csv_path, skiprows=1, dtype=str, encoding="utf-8")
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Rk" not in frame.columns:
        raise RuntimeError(f"CSV format unexpected: missing 'Rk' column in {csv_path}")

    frame = frame[frame["Rk"].astype(str).str.strip().ne("Rk")].copy()
    player_clean = frame["Player"].astype(str).str.strip()
    frame = frame[player_clean.ne("") & player_clean.str.lower().ne("nan")].copy()
    return frame.reset_index(drop=True)


def _build_player_season(df: pd.DataFrame, season_start: int, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    out["season_start"] = int(season_start)
    out["competition"] = "UEFA Champions League"
    out["source_file"] = source_file
    out["player_name"] = df["Player"].astype(str).str.strip()
    out["nation_raw"] = df.get("Nation")
    out["position_raw"] = df.get("Pos")
    out["team_name"] = df.get("Squad")
    out["age"] = df.get("Age").map(_as_int)
    out["birth_year"] = df.get("Born").map(_as_int)
    out["matches_played"] = df.get("MP").map(_as_int)
    out["starts"] = df.get("Starts").map(_as_int)
    out["minutes_total"] = df.get("Min").map(_as_int)
    out["nineties"] = df.get("90s").map(_as_float)
    out["goals_total"] = df.get("Gls").map(_as_int)
    out["assists_total"] = df.get("Ast").map(_as_int)
    out["ga_total"] = df.get("G+A").map(_as_int)
    out["goals_non_pk_total"] = df.get("G-PK").map(_as_int)
    out["pk_goals"] = df.get("PK").map(_as_int)
    out["pk_attempts"] = df.get("PKatt").map(_as_int)
    out["yellow_cards"] = df.get("CrdY").map(_as_int)
    out["red_cards"] = df.get("CrdR").map(_as_int)
    out["goals_p90"] = df.get("Gls.1").map(_as_float)
    out["assists_p90"] = df.get("Ast.1").map(_as_float)
    out["ga_p90"] = df.get("G+A.1").map(_as_float)
    out["goals_non_pk_p90"] = df.get("G-PK.1").map(_as_float)
    out["ga_non_pk_p90"] = df.get("G+A-PK").map(_as_float)
    out["player_key"] = (
        out["player_name"].map(_normalize_text)
        + "|"
        + out["birth_year"].fillna(0).astype(int).astype(str)
        + "|"
        + out["team_name"].fillna("").astype(str).map(_normalize_text)
    )
    out["imported_at_utc"] = datetime.now(timezone.utc)
    return out.reset_index(drop=True)


def _ensure_tables(engine_obj) -> None:
    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.study_ucl_player_season (
                    season_start INT NOT NULL,
                    competition TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    nation_raw TEXT,
                    position_raw TEXT,
                    team_name TEXT,
                    age INT,
                    birth_year INT,
                    matches_played INT,
                    starts INT,
                    minutes_total INT,
                    nineties DOUBLE PRECISION,
                    goals_total INT,
                    assists_total INT,
                    ga_total INT,
                    goals_non_pk_total INT,
                    pk_goals INT,
                    pk_attempts INT,
                    yellow_cards INT,
                    red_cards INT,
                    goals_p90 DOUBLE PRECISION,
                    assists_p90 DOUBLE PRECISION,
                    ga_p90 DOUBLE PRECISION,
                    goals_non_pk_p90 DOUBLE PRECISION,
                    ga_non_pk_p90 DOUBLE PRECISION,
                    player_key TEXT NOT NULL,
                    imported_at_utc TIMESTAMPTZ NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_study_ucl_player_season_season
                    ON public.study_ucl_player_season(season_start)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_study_ucl_player_season_player_key
                    ON public.study_ucl_player_season(player_key)
                """
            )
        )


def main() -> None:
    csv_path = Path(
        os.getenv("UCL_STANDARD_CSV_PATH")
        or r"C:\Users\mikae\Downloads\Feuille de calcul sans titre - Feuille 1.csv"
    )
    if not csv_path.exists():
        raise RuntimeError(f"CSV file not found: {csv_path}")

    season_start = int(os.getenv("UCL_CSV_SEASON_START") or "2024")
    raw_df = _read_ucl_csv(csv_path)
    player_season = _build_player_season(raw_df, season_start=season_start, source_file=csv_path.name)

    engine_obj = _get_engine()
    _ensure_tables(engine_obj)

    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM public.study_ucl_player_season
                WHERE season_start = :season_start
                  AND competition = 'UEFA Champions League'
                """
            ),
            {"season_start": season_start},
        )

    player_season.to_sql(
        "study_ucl_player_season",
        engine_obj,
        schema="public",
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi",
    )

    logger.info(
        "UCL CSV import -> Supabase done | file=%s season_start=%s rows=%s players=%s teams=%s",
        csv_path.name,
        season_start,
        len(player_season),
        int(player_season["player_name"].nunique()) if not player_season.empty else 0,
        int(player_season["team_name"].nunique()) if not player_season.empty else 0,
    )


if __name__ == "__main__":
    main()
