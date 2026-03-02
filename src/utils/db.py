import os
from collections.abc import Mapping

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def build_database_url(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    host = source.get("DB_HOST", "localhost")
    port = source.get("DB_PORT", "5432")
    name = source.get("DB_NAME", "football_dw")
    user = source.get("DB_USER", "football")
    password = source.get("DB_PASSWORD", "football")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def get_engine(env: Mapping[str, str] | None = None) -> Engine:
    return create_engine(build_database_url(env), pool_pre_ping=True)
