from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.study_fbref import (
    CANONICAL_PLAYER_MATCH_COLUMNS,
    _extract_fbref_raw,
    _normalize_fbref_player_match_stats,
    _normalize_fbref_player_season_stats,
    last_completed_season_start_years,
    merge_position_metadata,
    season_label,
)
from src.utils.logger import get_logger

logger = get_logger("export_fbref_matchlogs_manual_csv")


def _parse_seasons() -> list[int]:
    raw = os.getenv("FBREF_EXPORT_SEASONS", "").strip()
    if raw:
        seasons: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit():
                raise ValueError(
                    "FBREF_EXPORT_SEASONS doit contenir des annees de debut separees par des virgules, "
                    f"ex: 2024 ou 2022,2023,2024. Recu: {raw!r}"
                )
            seasons.append(int(token))
        if not seasons:
            raise ValueError("FBREF_EXPORT_SEASONS est vide apres parsing.")
        return sorted(set(seasons))

    num_past = int(os.getenv("FBREF_EXPORT_NUM_PAST", "1"))
    if num_past <= 0:
        raise ValueError("FBREF_EXPORT_NUM_PAST doit etre > 0.")
    return last_completed_season_start_years(num_past)


def export_matchlogs_manual_csv(
    *,
    leagues: str = "ESP-La Liga",
    seasons: list[int] | None = None,
    out_csv: Path | None = None,
) -> Path:
    seasons = seasons or _parse_seasons()
    out_csv = out_csv or Path(os.getenv("FBREF_EXPORT_OUT_CSV", "data/study/fbref_input/player_match_manual.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "FBref match logs export start - league=%s seasons=%s out=%s",
        leagues,
        seasons,
        out_csv,
    )

    raw_match, raw_season = _extract_fbref_raw(leagues=leagues, seasons=seasons)
    df_match = _normalize_fbref_player_match_stats(raw_match, league_label="La Liga (FBref)")
    df_season_meta = _normalize_fbref_player_season_stats(raw_season)
    df_match = merge_position_metadata(df_match, df_season_meta)

    # Ensure the exported file matches the manual importer contract exactly.
    for col in CANONICAL_PLAYER_MATCH_COLUMNS:
        if col not in df_match.columns:
            df_match[col] = None
    df_out = df_match[CANONICAL_PLAYER_MATCH_COLUMNS].copy()
    df_out = df_out.sort_values(["season_start", "date_id", "team_name", "player_name"]).reset_index(drop=True)

    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(
        "FBref match logs export done - rows=%s players=%s seasons=%s labels=%s file=%s",
        len(df_out),
        int(df_out["player_id"].nunique()) if not df_out.empty else 0,
        int(df_out["season_start"].nunique()) if not df_out.empty else 0,
        [season_label(s) for s in seasons],
        out_csv,
    )
    return out_csv


def main() -> None:
    league = os.getenv("FBREF_EXPORT_LEAGUE", "ESP-La Liga")
    seasons = _parse_seasons()
    out_csv = Path(os.getenv("FBREF_EXPORT_OUT_CSV", "data/study/fbref_input/player_match_manual.csv"))
    path = export_matchlogs_manual_csv(leagues=league, seasons=seasons, out_csv=out_csv)

    # Friendly console summary for the user after local execution.
    df = pd.read_csv(path)
    season_values = sorted(df["season_start"].dropna().astype(int).unique().tolist()) if not df.empty else []
    print(f"CSV genere: {path}")
    print(f"Lignes: {len(df)}")
    print(f"Joueurs uniques: {df['player_id'].nunique() if not df.empty else 0}")
    print(f"Saisons: {season_values}")


if __name__ == "__main__":
    main()
