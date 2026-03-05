from datetime import date
from pathlib import Path

import pytest

from src.config import Settings, SettingsError
from src.extract import calculate_incremental_window


def test_settings_validation_rejects_blank_database_values():
    env = {
        "DB_HOST": "",
        "DB_PORT": "5432",
        "DB_NAME": "",
        "DB_USER": "",
        "DB_PASSWORD": "",
    }

    with pytest.raises(SettingsError, match="Missing database configuration"):
        Settings.from_env(env)


def test_settings_supports_legacy_pipeline_mode_alias():
    settings = Settings.from_env({"PIPELINE_MODE": "mock"})

    assert settings.data_mode == "mock"


def test_settings_accepts_csv_mode():
    settings = Settings.from_env({"DATA_MODE": "csv"})

    assert settings.data_mode == "csv"


def test_settings_accepts_hybrid_mode():
    settings = Settings.from_env({"DATA_MODE": "hybrid", "FOOTBALL_DATA_TOKEN": "token"})

    assert settings.data_mode == "hybrid"


def test_settings_parses_live_competition_codes():
    settings = Settings.from_env({"LIVE_COMPETITION_CODES": "PD, PL,SA, BL1 , FL1, CL, EL, UCL"})

    assert settings.live_competition_codes == ("PD", "PL", "SA", "BL1", "FL1", "CL", "EL", "UCL")


def test_settings_default_live_competition_codes_include_uefa():
    settings = Settings.from_env({})

    assert settings.live_competition_codes == ("PD", "PL", "SA", "BL1", "FL1", "CL", "EL", "UCL")


def test_settings_reads_db_password_from_secret_file(tmp_path: Path):
    secret_file = tmp_path / "db_password.txt"
    secret_file.write_text("super-secret\n", encoding="utf-8")

    settings = Settings.from_env(
        {
            "DB_HOST": "db",
            "DB_NAME": "football_dw",
            "DB_USER": "pipeline_rw",
            "DB_PASSWORD_FILE": str(secret_file),
        }
    )

    assert settings.db_password == "super-secret"


def test_settings_raises_when_secret_file_is_missing():
    with pytest.raises(SettingsError, match="could not be read"):
        Settings.from_env(
            {
                "DB_HOST": "db",
                "DB_NAME": "football_dw",
                "DB_USER": "pipeline_rw",
                "DB_PASSWORD_FILE": "/tmp/does-not-exist.txt",
            }
        )


def test_incremental_window_uses_today_minus_days_plus_one():
    window = calculate_incremental_window(days=14, today=date(2026, 3, 2))

    assert window == {
        "dateFrom": "2026-02-16",
        "dateTo": "2026-03-03",
    }
