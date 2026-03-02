from pathlib import Path

from src.config import Settings
from src.extract import extract_csv
from src.transform import transform_csv_to_tables


FIXTURE_PATH = Path("tests/fixtures/minimal_matchlog_cleaned.csv")


def _prepare_csv_workspace(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    content = FIXTURE_PATH.read_text(encoding="utf-8")
    (raw_dir / "Player_A_cleaned.csv").write_text(content, encoding="utf-8")
    (raw_dir / "Player_B_cleaned.csv").write_text(content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)


def test_csv_parsing_minimal_dedupes_player_level_rows(tmp_path, monkeypatch):
    _prepare_csv_workspace(tmp_path, monkeypatch)

    payload = extract_csv(settings=Settings.from_env({"DATA_MODE": "csv"}))

    assert len(payload["csv_files"]) == 2
    assert len(payload["match_candidates"]) == 3


def test_kickoff_utc_not_null_after_csv_transform(tmp_path, monkeypatch):
    _prepare_csv_workspace(tmp_path, monkeypatch)

    payload = extract_csv(settings=Settings.from_env({"DATA_MODE": "csv"}))
    transformed = transform_csv_to_tables(payload)

    assert transformed["fact_match"]
    assert all(row["kickoff_utc"] is not None for row in transformed["fact_match"])
    assert all(row["status"] == "FINISHED" for row in transformed["fact_match"])
    assert all(row["matchday"] is not None for row in transformed["fact_match"])


def test_csv_transform_sets_non_null_season(tmp_path, monkeypatch):
    _prepare_csv_workspace(tmp_path, monkeypatch)

    payload = extract_csv(settings=Settings.from_env({"DATA_MODE": "csv"}))
    transformed = transform_csv_to_tables(payload)

    assert all(row["season"] == "2020-2021" for row in transformed["fact_match"])
