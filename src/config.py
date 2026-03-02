from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping


class SettingsError(ValueError):
    pass


def _first_value(
    source: Mapping[str, str],
    keys: tuple[str, ...],
    default: str | None = None,
    *,
    allow_blank: bool = False,
) -> str | None:
    for key in keys:
        if key not in source:
            continue
        value = source.get(key)
        if value is None:
            continue
        stripped = value.strip()
        if allow_blank:
            return stripped
        if stripped:
            return stripped
    return default


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise SettingsError(f"Invalid boolean value: {value!r}")


def _parse_int(value: str | None, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingsError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise SettingsError(f"{field_name} must be > 0, got {parsed}")
    return parsed


@dataclass(frozen=True)
class Settings:
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "football_dw"
    db_user: str = "football"
    db_password: str = "football"
    database_url_override: str | None = None
    football_data_token: str | None = None
    football_data_base_url: str = "https://api.football-data.org/v4"
    competition_code: str = "PD"
    data_mode: Literal["mock", "api", "csv"] = "api"
    incremental: bool = False
    incremental_days: int = 14
    dq_freshness_days: int = 7
    supabase_db_url: str | None = None
    study_supabase_db_url: str | None = None
    fbref_study_backend: str = "local"

    def __post_init__(self) -> None:
        if self.data_mode not in {"mock", "api", "csv"}:
            raise SettingsError(f"DATA_MODE must be 'mock', 'api' or 'csv', got {self.data_mode!r}")

        required_parts = {
            "DB_HOST": self.db_host,
            "DB_NAME": self.db_name,
            "DB_USER": self.db_user,
            "DB_PASSWORD": self.db_password,
        }
        missing = [field_name for field_name, value in required_parts.items() if not str(value).strip()]
        if missing and not self.database_url_override:
            joined = ", ".join(missing)
            raise SettingsError(f"Missing database configuration: {joined}")

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def validate_for_pipeline(self) -> None:
        if self.data_mode == "api" and not self.football_data_token:
            raise SettingsError(
                "FOOTBALL_DATA_TOKEN is required when DATA_MODE is 'api'. "
                "Set DATA_MODE=mock or DATA_MODE=csv to use local datasets."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        data_mode = (_first_value(source, ("DATA_MODE", "PIPELINE_MODE"), "api") or "api").lower()
        return cls(
            db_host=_first_value(source, ("DB_HOST",), "localhost", allow_blank=True) or "",
            db_port=_parse_int(_first_value(source, ("DB_PORT",), "5432"), field_name="DB_PORT", default=5432),
            db_name=_first_value(source, ("DB_NAME",), "football_dw", allow_blank=True) or "",
            db_user=_first_value(source, ("DB_USER",), "football", allow_blank=True) or "",
            db_password=_first_value(source, ("DB_PASSWORD",), "football", allow_blank=True) or "",
            database_url_override=_first_value(source, ("DATABASE_URL",)),
            football_data_token=_first_value(source, ("FOOTBALL_DATA_TOKEN",)),
            football_data_base_url=(
                _first_value(source, ("FOOTBALL_DATA_BASE_URL",), "https://api.football-data.org/v4")
                or "https://api.football-data.org/v4"
            ),
            competition_code=_first_value(source, ("COMPETITION_CODE", "FOOTBALL_DATA_COMPETITION"), "PD") or "PD",
            data_mode=data_mode,  # type: ignore[arg-type]
            incremental=_parse_bool(_first_value(source, ("INCREMENTAL",), "false"), default=False),
            incremental_days=_parse_int(
                _first_value(source, ("INCREMENTAL_DAYS",), "14"),
                field_name="INCREMENTAL_DAYS",
                default=14,
            ),
            dq_freshness_days=_parse_int(
                _first_value(source, ("DQ_FRESHNESS_DAYS",), "7"),
                field_name="DQ_FRESHNESS_DAYS",
                default=7,
            ),
            supabase_db_url=_first_value(source, ("SUPABASE_DB_URL",)),
            study_supabase_db_url=_first_value(source, ("STUDY_SUPABASE_DB_URL",)),
            fbref_study_backend=_first_value(source, ("FBREF_STUDY_BACKEND",), "local") or "local",
        )


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings.from_env()


def get_settings(env: Mapping[str, str] | None = None) -> Settings:
    if env is None:
        return _cached_settings()
    return Settings.from_env(env)


def clear_settings_cache() -> None:
    _cached_settings.cache_clear()
