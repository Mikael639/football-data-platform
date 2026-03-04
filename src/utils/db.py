import os
from collections.abc import Mapping

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import Settings, get_settings

load_dotenv()


def build_database_url(
    env: Mapping[str, str] | None = None,
    settings: Settings | None = None,
) -> str:
    if settings is not None:
        return settings.database_url
    if env is not None:
        return Settings.from_env(env).database_url
    return get_settings().database_url


def get_engine(
    env: Mapping[str, str] | None = None,
    settings: Settings | None = None,
) -> Engine:
    return create_engine(build_database_url(env=env, settings=settings), pool_pre_ping=True)
