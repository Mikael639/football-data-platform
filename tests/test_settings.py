from datetime import date

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


def test_incremental_window_uses_today_minus_days_plus_one():
    window = calculate_incremental_window(days=14, today=date(2026, 3, 2))

    assert window == {
        "dateFrom": "2026-02-16",
        "dateTo": "2026-03-03",
    }
