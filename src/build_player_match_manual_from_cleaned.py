from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from src.study_fbref import CANONICAL_PLAYER_MATCH_COLUMNS, _ensure_manual_match_columns
from src.utils.logger import get_logger

logger = get_logger("build_player_match_manual_from_cleaned")


def _player_name_from_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_cleaned$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("Vinicius_stat", "Vinicius Junior")
    return stem.replace("_", " ").strip()


def _find_header_row(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    marker = "Date,Day,Comp,Round,Venue,Result,Squad,Opponent,Start,Pos,Min"
    for idx, line in enumerate(lines[:30]):
        if line.strip().startswith(marker):
            return idx
    raise ValueError(
        f"Impossible de trouver la ligne d'entete match logs dans {path.name}. "
        "Attendu: ligne commencant par 'Date,Day,Comp,...'"
    )


def _normalize_single_cleaned_csv(path: Path, competition_filter: str) -> pd.DataFrame:
    header_idx = _find_header_row(path)
    raw = pd.read_csv(path, skiprows=header_idx)
    raw.columns = [str(c).strip() for c in raw.columns]

    if "Date" not in raw.columns:
        raise ValueError(f"Colonne 'Date' manquante apres parsing dans {path.name}.")

    player_fallback = _player_name_from_filename(path)
    if "Player" in raw.columns:
        player_series = raw["Player"].fillna("").astype(str).str.strip()
        player_series = player_series.where(player_series != "", player_fallback)
    else:
        player_series = pd.Series(player_fallback, index=raw.index, dtype="object")

    if "Comp" in raw.columns:
        comp_series = raw["Comp"].fillna("").astype(str).str.strip()
        raw = raw.loc[comp_series == competition_filter].copy()
        player_series = player_series.loc[raw.index]
    else:
        raw = raw.copy()
        player_series = player_series.loc[raw.index]

    out = pd.DataFrame(index=raw.index)
    out["player_name"] = player_series
    out["date_id"] = raw.get("Date")
    out["competition"] = raw.get("Comp", competition_filter)
    out["team_name"] = raw.get("Squad")
    out["is_starting"] = raw.get("Start")
    out["position"] = raw.get("Pos")
    out["minutes"] = raw.get("Min")
    out["goals"] = raw.get("Gls")
    out["assists"] = raw.get("Ast")
    out["shots"] = raw.get("Sh")
    out["passes"] = raw.get("Cmp", 0)
    out["pass_accuracy"] = raw.get("Cmp%", 0)
    return out


def build_player_match_manual_csv(
    *,
    input_glob: str = "data/raw/*_cleaned.csv",
    out_csv: Path | None = None,
    competition_filter: str = "La Liga",
    league_label: str = "La Liga (FBref manuel)",
) -> Path:
    out_csv = out_csv or Path("data/study/fbref_input/player_match_manual.csv")
    files = sorted(Path(".").glob(input_glob))
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouve avec le pattern: {input_glob}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            df = _normalize_single_cleaned_csv(file_path, competition_filter=competition_filter)
            frames.append(df)
            logger.info("Loaded cleaned file %s rows=%s", file_path.name, len(df))
        except Exception:
            logger.exception("Echec parsing cleaned file: %s", file_path)
            raise

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise ValueError("Aucune ligne exploitable apres lecture des fichiers cleaned.")

    canonical = _ensure_manual_match_columns(merged, league_label=league_label)
    canonical = canonical[canonical["competition"].astype(str).str.strip() == competition_filter].copy()
    canonical = canonical.sort_values(
        ["season_start", "date_id", "team_name", "player_name", "match_id"]
    ).reset_index(drop=True)
    canonical = canonical[CANONICAL_PLAYER_MATCH_COLUMNS]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(
        "player_match_manual generated rows=%s players=%s seasons=%s file=%s",
        len(canonical),
        int(canonical["player_name"].nunique()) if not canonical.empty else 0,
        int(canonical["season_start"].nunique()) if not canonical.empty else 0,
        out_csv,
    )
    return out_csv


def main() -> None:
    input_glob = os.getenv("FBREF_CLEANED_INPUT_GLOB", "data/raw/*_cleaned.csv")
    out_csv = Path(os.getenv("FBREF_CLEANED_OUT_CSV", "data/study/fbref_input/player_match_manual.csv"))
    competition_filter = os.getenv("FBREF_CLEANED_COMPETITION", "La Liga")
    league_label = os.getenv("FBREF_CLEANED_LEAGUE_LABEL", "La Liga (FBref manuel)")

    out_path = build_player_match_manual_csv(
        input_glob=input_glob,
        out_csv=out_csv,
        competition_filter=competition_filter,
        league_label=league_label,
    )
    df = pd.read_csv(out_path)
    seasons = sorted(df["season_start"].dropna().astype(int).unique().tolist()) if not df.empty else []
    print(f"CSV genere: {out_path}")
    print(f"Lignes: {len(df)}")
    print(f"Joueurs uniques: {df['player_name'].nunique() if not df.empty else 0}")
    print(f"Saisons: {seasons}")


if __name__ == "__main__":
    main()

