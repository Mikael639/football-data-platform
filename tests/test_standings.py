from pathlib import Path

import pandas as pd

from src.config import Settings
from src.extract import extract_csv
from src.standings import build_standings_rows
from src.transform import transform_csv_to_tables


FIXTURE_PATH = Path("tests/fixtures/minimal_matchlog_cleaned.csv")


def _prepare_csv_workspace(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    content = FIXTURE_PATH.read_text(encoding="utf-8")
    (raw_dir / "Player_A_cleaned.csv").write_text(content, encoding="utf-8")
    (raw_dir / "Player_B_cleaned.csv").write_text(content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)


def _matches_dataframe_from_transformed(transformed: dict[str, list[dict]]) -> pd.DataFrame:
    teams = {row["team_id"]: row["team_name"] for row in transformed["dim_team"]}
    rows = []
    for row in transformed["fact_match"]:
        rows.append(
            {
                "match_id": row["match_id"],
                "competition_id": row["competition_id"],
                "season": row["season"],
                "match_date": row["date_id"],
                "kickoff_utc": row["kickoff_utc"],
                "status": row["status"],
                "matchday": row["matchday"],
                "home_team_id": row["home_team_id"],
                "home_team_name": teams[row["home_team_id"]],
                "away_team_id": row["away_team_id"],
                "away_team_name": teams[row["away_team_id"]],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
            }
        )
    return pd.DataFrame(rows)


def test_standings_snapshot_generated_nonempty(tmp_path, monkeypatch):
    _prepare_csv_workspace(tmp_path, monkeypatch)

    payload = extract_csv(settings=Settings.from_env({"DATA_MODE": "csv"}))
    transformed = transform_csv_to_tables(payload)
    result = build_standings_rows(_matches_dataframe_from_transformed(transformed))

    assert result.rows
    assert result.ignored_null_matchday_count == 0
    assert any(row["matchday"] == 5 for row in result.rows)


def test_standings_ranking_tiebreak():
    matches_df = pd.DataFrame(
        [
            {
                "match_id": 1,
                "competition_id": 2014,
                "season": "2020-2021",
                "match_date": "2020-09-01",
                "kickoff_utc": "2020-09-01T12:00:00Z",
                "status": "FINISHED",
                "matchday": 1,
                "home_team_id": 1,
                "home_team_name": "Alpha",
                "away_team_id": 2,
                "away_team_name": "Beta",
                "home_score": 1,
                "away_score": 0,
            },
            {
                "match_id": 2,
                "competition_id": 2014,
                "season": "2020-2021",
                "match_date": "2020-09-01",
                "kickoff_utc": "2020-09-01T12:00:00Z",
                "status": "FINISHED",
                "matchday": 1,
                "home_team_id": 3,
                "home_team_name": "Gamma",
                "away_team_id": 4,
                "away_team_name": "Delta",
                "home_score": 2,
                "away_score": 1,
            },
            {
                "match_id": 3,
                "competition_id": 2014,
                "season": "2020-2021",
                "match_date": "2020-09-08",
                "kickoff_utc": "2020-09-08T12:00:00Z",
                "status": "FINISHED",
                "matchday": 2,
                "home_team_id": 1,
                "home_team_name": "Alpha",
                "away_team_id": 4,
                "away_team_name": "Delta",
                "home_score": 0,
                "away_score": 1,
            },
            {
                "match_id": 4,
                "competition_id": 2014,
                "season": "2020-2021",
                "match_date": "2020-09-08",
                "kickoff_utc": "2020-09-08T12:00:00Z",
                "status": "FINISHED",
                "matchday": 2,
                "home_team_id": 2,
                "home_team_name": "Beta",
                "away_team_id": 3,
                "away_team_name": "Gamma",
                "home_score": 1,
                "away_score": 0,
            },
        ]
    )

    result = build_standings_rows(matches_df)
    md2_rows = [row for row in result.rows if row["matchday"] == 2]

    assert [row["team_id"] for row in md2_rows] == [4, 3, 1, 2]
    assert [row["position"] for row in md2_rows] == [1, 2, 3, 4]
